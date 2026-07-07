#!/usr/bin/env python3
"""Bisect layer-0 sparse GQA prefill: HF vs vLLM per position and sub-step.

Compares embed, input_layernorm, q/k (slice), flash-attn out, gated attn,
o_proj branch, and full layer-0 output for prompt+t1 (seqlen=7).

Usage:
  python3 gate1_l0_sparse_bisect.py
  MINICPM_SALA_PROMPT='Briefly explain gravity:' python3 gate1_l0_sparse_bisect.py
  MINICPM_SALA_DENSE_EAGER_PREFILL=0 python3 gate1_l0_sparse_bisect.py
"""

from __future__ import annotations

import contextlib
import gc
import os
import subprocess
import sys
import tempfile

import torch

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = os.environ.get("MINICPM_SALA_PROMPT", "Hello, my name is")


def _patch_hf() -> None:
    script = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "remote",
        "patch_hf_transformers_compat.py",
    )
    script = os.path.normpath(script)
    if os.path.isfile(script):
        subprocess.run([sys.executable, script], check=False)


def _make_sparse_prefill_metadata(
    seq_len: int,
    block_size: int,
    dense_len: int,
    device: torch.device,
):
    from vllm.v1.attention.backends.minicpm_sala_sparse import (
        MiniCPMSALASparseAttentionMetadata,
    )

    num_blocks = max(1, (seq_len + block_size - 1) // block_size)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).unsqueeze(
        0
    )
    slot_mapping = torch.arange(seq_len, device=device, dtype=torch.int64)
    return MiniCPMSALASparseAttentionMetadata(
        query_start_loc=torch.tensor([0, seq_len], device=device, dtype=torch.int32),
        seq_lens=torch.tensor([seq_len], device=device, dtype=torch.int32),
        block_table=block_table,
        slot_mapping=slot_mapping,
        dense_len=dense_len,
        page_block_size=block_size,
        num_actual_tokens=seq_len,
        max_query_len=seq_len,
        max_seq_len=seq_len,
    )


def _bind_sparse_kv_cache(attn, block_size: int, seq_len: int) -> None:
    from vllm.v1.attention.backends.minicpm_sala_sparse import (
        MiniCPMSALASparseAttentionBackend,
    )

    num_blocks = max(1, (seq_len + block_size - 1) // block_size)
    shape = MiniCPMSALASparseAttentionBackend.get_kv_cache_shape(
        num_blocks,
        block_size,
        attn.num_kv_heads,
        attn.head_size,
    )
    attn.kv_cache = torch.zeros(shape, device="cuda", dtype=torch.bfloat16)


def _pos_diff(a: torch.Tensor, b: torch.Tensor) -> list[float]:
    d = (a.float() - b.float()).abs()
    if d.dim() == 1:
        return [d.max().item()]
    return [d[i].max().item() for i in range(d.shape[0])]


def _print_stage(name: str, hf: torch.Tensor, vv: torch.Tensor) -> float:
    diffs = _pos_diff(hf, vv)
    peak = max(diffs)
    pos_str = " ".join(f"p{i}={d:.6g}" for i, d in enumerate(diffs))
    print(f"{name:16s} peak={peak:.6g}  {pos_str}", flush=True)
    return peak


def _flash_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Varlen flash on packed (T, H*D) or (T, hidden) GQA layout."""
    from flash_attn import flash_attn_varlen_func

    # Inputs are (T, q_dim), (T, kv_dim) from HF/vLLM linear output.
    # Reshape to (T, n_heads, head_dim) using HF head counts from shapes.
    t = q.shape[0]
    kv_t = k.shape[0]
    assert t == kv_t
    # Infer from common MiniCPM-SALA L0: 16 q heads, 2 kv heads, head_dim=64
    head_dim = 64
    n_q = q.shape[1] // head_dim
    n_kv = k.shape[1] // head_dim
    q4 = q.view(t, n_q, head_dim)
    k4 = k.view(t, n_kv, head_dim)
    v4 = v.view(t, n_kv, head_dim)
    cu = torch.tensor([0, t], device=q.device, dtype=torch.int32)
    o = flash_attn_varlen_func(
        q4,
        k4,
        v4,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=t,
        max_seqlen_k=t,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=True,
    )
    return o.reshape(t, -1)


def hf_l0_traces(ids: list[int]) -> dict[str, torch.Tensor]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    layer0 = model.model.layers[0]
    attn_mod = layer0.self_attn
    pos = torch.arange(len(ids), device="cuda").unsqueeze(0)
    mask = torch.ones(1, len(ids), device="cuda")
    traces: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        emb = model.model.embed_tokens(torch.tensor([ids], device="cuda")) * model.config.scale_emb
        traces["embed"] = emb[0].float().cpu()
        x = layer0.input_layernorm(emb)
        traces["norm"] = x[0].float().cpu()

        q = attn_mod.q_proj(x)
        k = attn_mod.k_proj(x)
        v = attn_mod.v_proj(x)
        traces["q"] = q[0].float().cpu()
        traces["k"] = k[0].float().cpu()
        traces["v"] = v[0].float().cpu()

        scale = getattr(attn_mod, "scale", attn_mod.head_dim**-0.5)
        flash = _flash_varlen(q[0], k[0], v[0], scale)
        traces["flash_raw"] = flash.float().cpu()

        attn_branch, _, _ = attn_mod(
            x, attention_mask=mask, position_ids=pos, use_cache=False
        )
        traces["attn_branch"] = attn_branch[0].float().cpu()

        h0 = layer0(emb, attention_mask=mask, position_ids=pos, use_cache=False)
        h0_t = h0[0] if isinstance(h0, tuple) else h0
        traces["layer0"] = h0_t[0].float().cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return traces


def manual_l0_from_model(
    model: torch.nn.Module,
    vllm_config,
    ids: list[int],
) -> dict[str, torch.Tensor]:
    """Manual-metadata layer0 forward on an already-loaded worker model."""
    from vllm.forward_context import set_forward_context
    from vllm.v1.attention.backends.minicpm_sala_sparse import parse_sparse_config

    seq_len = len(ids)
    ids_t = torch.tensor(ids, device="cuda")
    positions = torch.arange(seq_len, device="cuda", dtype=torch.long)
    traces: dict[str, torch.Tensor] = {}

    block_size = vllm_config.cache_config.block_size
    dense_len = parse_sparse_config(vllm_config.model_config.hf_config).dense_len
    layer0 = model.model.layers[0]
    sparse_attn = layer0.self_attn.attn
    sparse_prefix = sparse_attn.layer_name
    _bind_sparse_kv_cache(sparse_attn, block_size, seq_len)
    sparse_meta = _make_sparse_prefill_metadata(
        seq_len, block_size, dense_len, ids_t.device
    )

    with torch.no_grad():
        emb = model.model.get_input_embeddings(ids_t)
        traces["embed"] = emb.float().cpu()
        sa = layer0.self_attn
        captured: dict[str, torch.Tensor] = {}

        def _attn_hook(_mod, _inp, out):
            captured["attn_branch"] = out.detach().float().cpu()

        handle = sa.register_forward_hook(_attn_hook)
        with set_forward_context(
            attn_metadata={sparse_prefix: sparse_meta},
            vllm_config=vllm_config,
            num_tokens=seq_len,
            slot_mapping={sparse_prefix: sparse_meta.slot_mapping},
        ):
            h0 = layer0(positions, emb)
        handle.remove()
        traces["attn_branch"] = captured["attn_branch"]
        traces["layer0"] = h0.float().cpu()
    return traces


def vllm_l0_traces(ids: list[int]) -> dict[str, torch.Tensor]:
    import vllm.config as vconfig
    from vllm.config import CacheConfig, ModelConfig, VllmConfig
    from vllm.config.device import DeviceConfig
    from vllm.config.load import LoadConfig
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.v1.attention.backends.minicpm_sala_sparse import parse_sparse_config

    seq_len = len(ids)
    ids_t = torch.tensor(ids, device="cuda")
    positions = torch.arange(seq_len, device="cuda", dtype=torch.long)
    traces: dict[str, torch.Tensor] = {}

    model_config = ModelConfig(
        model=WEIGHTS, trust_remote_code=True, dtype="bfloat16", max_model_len=4096
    )
    load_config = LoadConfig()
    cache_config = CacheConfig(block_size=256)
    vllm_config = VllmConfig(
        model_config=model_config,
        load_config=load_config,
        cache_config=cache_config,
        device_config=DeviceConfig(device="cuda"),
    )
    fd, temp = tempfile.mkstemp()
    os.close(fd)
    try:
        with vconfig.set_current_vllm_config(vllm_config, check_compile=False):
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{temp}",
                local_rank=0,
                backend="nccl",
            )
            initialize_model_parallel(1, 1)
            model = get_model_loader(load_config).load_model(
                vllm_config=vllm_config, model_config=model_config
            )
            model.eval().cuda()

            block_size = cache_config.block_size
            dense_len = parse_sparse_config(model_config.hf_config).dense_len
            layer0 = model.model.layers[0]
            sparse_attn = layer0.self_attn.attn
            sparse_prefix = sparse_attn.layer_name
            _bind_sparse_kv_cache(sparse_attn, block_size, seq_len)
            sparse_meta = _make_sparse_prefill_metadata(
                seq_len, block_size, dense_len, ids_t.device
            )

            with torch.no_grad():
                emb = model.model.get_input_embeddings(ids_t)
                traces["embed"] = emb.float().cpu()
                x = layer0.input_layernorm(emb)
                traces["norm"] = x.float().cpu()

                sa = layer0.self_attn
                qkv, _ = sa.qkv_proj(x)
                q, k, v = qkv.split(
                    [sa.q_size, sa.kv_size, sa.kv_size], dim=-1
                )
                traces["q"] = q.float().cpu()
                traces["k"] = k.float().cpu()
                traces["v"] = v.float().cpu()
                flash = _flash_varlen(q, k, v, sa.scaling)
                traces["flash_raw"] = flash.float().cpu()

                captured: dict[str, torch.Tensor] = {}

                def _attn_hook(_mod, _inp, out):
                    captured["attn_branch"] = out.detach().float().cpu()

                handle = sa.register_forward_hook(_attn_hook)
                with set_forward_context(
                    attn_metadata={sparse_prefix: sparse_meta},
                    vllm_config=vllm_config,
                    num_tokens=seq_len,
                    slot_mapping={sparse_prefix: sparse_meta.slot_mapping},
                ):
                    h0 = layer0(positions, emb)
                handle.remove()
                traces["attn_branch"] = captured["attn_branch"]
                traces["layer0"] = h0.float().cpu()

            destroy_model_parallel()
            destroy_distributed_environment()
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp)
    gc.collect()
    torch.cuda.empty_cache()
    return traces


def main() -> int:
    _patch_hf()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    eager = os.environ.get("MINICPM_SALA_DENSE_EAGER_PREFILL", "1")
    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids = tok.encode(PROMPT, add_special_tokens=True)
    hf = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    with torch.no_grad():
        t1 = int(
            hf(
                torch.tensor([ids], device="cuda"),
                attention_mask=torch.ones(1, len(ids), device="cuda"),
            )
            .logits[0, -1]
            .argmax()
        )
    del hf
    gc.collect()
    torch.cuda.empty_cache()

    ids2 = ids + [t1]
    print(
        f"prompt={PROMPT!r} t1={t1} seqlen={len(ids2)} "
        f"DENSE_EAGER_PREFILL={eager}",
        flush=True,
    )

    hf_t = hf_l0_traces(ids2)
    vv_t = vllm_l0_traces(ids2)

    first_stage = None
    for stage in ("embed", "norm", "q", "k", "v", "flash_raw", "attn_branch", "layer0"):
        peak = _print_stage(stage, hf_t[stage], vv_t[stage])
        if first_stage is None and peak > 0.01:
            first_stage = stage

    if first_stage:
        print(f"first_stage_above_0.01={first_stage}", flush=True)
    else:
        print("all_stages_within_0.01=True", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
