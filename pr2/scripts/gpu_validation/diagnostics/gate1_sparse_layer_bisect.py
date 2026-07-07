#!/usr/bin/env python3
"""Bisect sparse layer N (default 9): HF vs vLLM at prompt-only seqlen.

Layer 9 is the second sparse (minicpm4) block; stack bisect showed it as the
first layer with max_abs > 0.05 for Briefly prompt-only.

Usage:
  MINICPM_SALA_LAYER_IDX=9 MINICPM_SALA_PROMPT='Briefly explain gravity:' \\
    python3 gate1_sparse_layer_bisect.py
  MINICPM_SALA_ISOLATE=1  # feed HF L(N-1) hidden to both L(N)
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
PROMPT = os.environ.get("MINICPM_SALA_PROMPT", "Briefly explain gravity:")
LAYER_IDX = int(os.environ.get("MINICPM_SALA_LAYER_IDX", "9"))
ISOLATE = os.environ.get("MINICPM_SALA_ISOLATE", "0") == "1"


def _patch_hf() -> None:
    script = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "remote",
        "patch_hf_transformers_compat.py",
    )
    script = os.path.normpath(script)
    if os.path.isfile(script):
        subprocess.run([sys.executable, script], check=False)


def _make_sparse_prefill_metadata(
    seq_len: int, block_size: int, dense_len: int, device: torch.device
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
        num_blocks, block_size, attn.num_kv_heads, attn.head_size
    )
    attn.kv_cache = torch.zeros(shape, device="cuda", dtype=torch.bfloat16)


def _flash_varlen(q, k, v, scale, *, head_dim, num_q_heads, num_kv_heads):
    from flash_attn import flash_attn_varlen_func

    t = q.shape[0]
    q4 = q.view(t, num_q_heads, head_dim)
    k4 = k.view(t, num_kv_heads, head_dim)
    v4 = v.view(t, num_kv_heads, head_dim)
    cu = torch.tensor([0, t], device=q.device, dtype=torch.int32)
    o = flash_attn_varlen_func(
        q4, k4, v4,
        cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=t, max_seqlen_k=t,
        dropout_p=0.0, softmax_scale=scale, causal=True,
    )
    return o.reshape(t, num_q_heads * head_dim)


def _peak(name: str, hf: torch.Tensor, vv: torch.Tensor) -> float:
    d = (hf.float() - vv.float()).abs()
    peak = d.max().item() if d.numel() else 0.0
    print(f"{name:16s} peak={peak:.6g}", flush=True)
    return peak


def _hf_hidden_before_layer(ids: list[int], layer_idx: int) -> torch.Tensor:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="flash_attention_2",
    ).eval()
    pos = torch.arange(len(ids), device="cuda").unsqueeze(0)
    mask = torch.ones(1, len(ids), device="cuda")
    with torch.no_grad():
        h = model.model.embed_tokens(torch.tensor([ids], device="cuda")) * model.config.scale_emb
        for i in range(layer_idx):
            out = model.model.layers[i](
                h, attention_mask=mask, position_ids=pos, use_cache=False
            )
            h = out[0] if isinstance(out, tuple) else out
        hidden = h[0].detach()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return hidden


def hf_sparse_traces(ids: list[int], layer_idx: int, hidden_in: torch.Tensor | None):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="flash_attention_2",
    ).eval()
    layer = model.model.layers[layer_idx]
    attn_mod = layer.self_attn
    pos = torch.arange(len(ids), device="cuda").unsqueeze(0)
    mask = torch.ones(1, len(ids), device="cuda")
    traces: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        if hidden_in is None:
            h = model.model.embed_tokens(torch.tensor([ids], device="cuda")) * model.config.scale_emb
            for i in range(layer_idx):
                out = model.model.layers[i](h, attention_mask=mask, position_ids=pos, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out
            emb = h
        else:
            emb = hidden_in.unsqueeze(0)
        x = layer.input_layernorm(emb)
        traces["norm"] = x[0, -1].float().cpu()
        q, k, v = attn_mod.q_proj(x), attn_mod.k_proj(x), attn_mod.v_proj(x)
        traces["q"] = q[0, -1].float().cpu()
        traces["k"] = k[0, -1].float().cpu()
        traces["v"] = v[0, -1].float().cpu()
        scale = getattr(attn_mod, "scale", attn_mod.head_dim**-0.5)
        flash = _flash_varlen(
            q[0], k[0], v[0], scale,
            head_dim=int(attn_mod.head_dim),
            num_q_heads=int(attn_mod.num_heads),
            num_kv_heads=int(attn_mod.num_key_value_heads),
        )
        traces["flash_raw"] = flash[-1].float().cpu()
        attn_out, _, _ = attn_mod(x, attention_mask=mask, position_ids=pos, use_cache=False)
        traces["attn_branch"] = attn_out[0, -1].float().cpu()
        out = layer(emb, attention_mask=mask, position_ids=pos, use_cache=False)
        h_out = out[0] if isinstance(out, tuple) else out
        traces["layer_out"] = h_out[0, -1].float().cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return traces


def vllm_sparse_traces(ids: list[int], layer_idx: int, hidden_in: torch.Tensor | None):
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
    from vllm.model_executor.models.minicpm_sala import _dense_o_proj, _dense_qkv_proj
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
        model_config=model_config, load_config=load_config,
        cache_config=cache_config, device_config=DeviceConfig(device="cuda"),
    )
    fd, temp = tempfile.mkstemp()
    os.close(fd)
    try:
        with vconfig.set_current_vllm_config(vllm_config, check_compile=False):
            init_distributed_environment(
                world_size=1, rank=0,
                distributed_init_method=f"file://{temp}",
                local_rank=0, backend="nccl",
            )
            initialize_model_parallel(1, 1)
            model = get_model_loader(load_config).load_model(
                vllm_config=vllm_config, model_config=model_config
            )
            model.eval().cuda()
            block_size = cache_config.block_size
            dense_len = parse_sparse_config(model_config.hf_config).dense_len
            layer = model.model.layers[layer_idx]
            sa = layer.self_attn
            sparse_attn = sa.attn
            sparse_prefix = sparse_attn.layer_name
            _bind_sparse_kv_cache(sparse_attn, block_size, seq_len)
            sparse_meta = _make_sparse_prefill_metadata(
                seq_len, block_size, dense_len, ids_t.device
            )
            with torch.no_grad():
                if hidden_in is None:
                    emb = model.model.get_input_embeddings(ids_t)
                    for i in range(layer_idx):
                        emb = model.model.layers[i](positions, emb)
                else:
                    emb = hidden_in.to(torch.bfloat16)
                x = layer.input_layernorm(emb)
                traces["norm"] = x[-1].float().cpu()
                q, k, v = _dense_qkv_proj(sa.qkv_proj, x, sa.q_size, sa.kv_size)
                traces["q"] = q[-1].float().cpu()
                traces["k"] = k[-1].float().cpu()
                traces["v"] = v[-1].float().cpu()
                flash = _flash_varlen(
                    q, k, v, sa.scaling,
                    head_dim=sa.head_dim, num_q_heads=sa.num_heads,
                    num_kv_heads=sa.num_kv_heads,
                )
                traces["flash_raw"] = flash[-1].float().cpu()
                with set_forward_context(
                    attn_metadata={sparse_prefix: sparse_meta},
                    vllm_config=vllm_config, num_tokens=seq_len,
                    slot_mapping={sparse_prefix: sparse_meta.slot_mapping},
                ):
                    sparse_core = sa.attn(q, k, v)
                if sa.use_output_gate:
                    gate, _ = sa.o_gate(x)
                    gated = sparse_core * torch.sigmoid(gate)
                else:
                    gated = sparse_core
                traces["gated"] = gated[-1].float().cpu()
                o_out = _dense_o_proj(sa.o_proj, gated)
                traces["o_proj_out"] = o_out[-1].float().cpu()
                captured: dict[str, torch.Tensor] = {}

                def _hook(_m, _i, out):
                    captured["attn_branch"] = out[-1].detach().float().cpu()

                handle = sa.register_forward_hook(_hook)
                with set_forward_context(
                    attn_metadata={sparse_prefix: sparse_meta},
                    vllm_config=vllm_config, num_tokens=seq_len,
                    slot_mapping={sparse_prefix: sparse_meta.slot_mapping},
                ):
                    h_out = layer(positions, emb)
                handle.remove()
                traces["attn_branch"] = captured["attn_branch"]
                traces["layer_out"] = h_out[-1].float().cpu()
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
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids = tok.encode(PROMPT, add_special_tokens=True)
    hidden = _hf_hidden_before_layer(ids, LAYER_IDX) if ISOLATE else None
    print(
        f"layer={LAYER_IDX} prompt={PROMPT!r} seqlen={len(ids)} isolate={ISOLATE}",
        flush=True,
    )
    hf_t = hf_sparse_traces(ids, LAYER_IDX, hidden)
    vv_t = vllm_sparse_traces(ids, LAYER_IDX, hidden)
    first = None
    for stage in ("norm", "q", "k", "v", "flash_raw", "gated", "o_proj_out", "attn_branch", "layer_out"):
        if stage not in hf_t or stage not in vv_t:
            continue
        peak = _peak(stage, hf_t[stage], vv_t[stage])
        if first is None and peak > 0.01:
            first = stage
    print(f"first_stage_above_0.01={first}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
