#!/usr/bin/env python3
"""Stage-1: HF _flash_attention_forward_dense vs vLLM sa.attn on identical q/k/v.

No hooks. Captures scale, cu_seqlens, shapes, per-position diff (Briefly seqlen=7).
"""

from __future__ import annotations

import contextlib
import gc
import inspect
import os
import sys
import tempfile

import torch

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = os.environ.get("MINICPM_SALA_PROMPT", "Briefly explain gravity:")


def _pos_peak(a: torch.Tensor, b: torch.Tensor) -> tuple[float, list[float]]:
    d = (a.float().cpu() - b.float().cpu()).abs()
    if d.dim() == 1:
        diffs = [d.max().item()]
    else:
        diffs = [d[i].max().item() for i in range(d.shape[0])]
    return max(diffs), diffs


def _build_ids2():
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    return ids + [t1]


def _hf_qkv_and_dense_flash(ids: list[int]) -> dict:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    layer0 = model.model.layers[0]
    sa = layer0.self_attn
    seqlen = len(ids)
    pos = torch.arange(seqlen, device="cuda").unsqueeze(0)
    mask = torch.ones(1, seqlen, device="cuda")
    out: dict = {}
    with torch.no_grad():
        emb = model.model.embed_tokens(torch.tensor([ids], device="cuda")) * model.config.scale_emb
        x = layer0.input_layernorm(emb)
        q = sa.q_proj(x)
        k = sa.k_proj(x)
        v = sa.v_proj(x)
        out["q_flat"] = q[0].contiguous()
        out["k_flat"] = k[0].contiguous()
        out["v_flat"] = v[0].contiguous()
        head_dim = int(sa.head_dim)
        n_q = int(sa.num_heads)
        n_kv = int(sa.num_key_value_heads)
        out["head_dim"] = head_dim
        out["num_q_heads"] = n_q
        out["num_kv_heads"] = n_kv
        hf_scale = float(getattr(sa, "scale", head_dim**-0.5))
        out["hf_scale"] = hf_scale
        out["expected_scale"] = head_dim**-0.5

        # HF internal layout before dense flash (mirror forward).
        q_len = seqlen
        bsz = 1
        query_states = q.view(bsz, q_len, n_q, head_dim).transpose(1, 2)
        key_states = k.view(bsz, q_len, n_kv, head_dim).transpose(1, 2)
        value_states = v.view(bsz, q_len, n_kv, head_dim).transpose(1, 2)
        out["hf_q_shape"] = tuple(query_states.shape)
        out["hf_k_shape"] = tuple(key_states.shape)

        flash_out = sa._flash_attention_forward_dense(
            query_states,
            key_states,
            value_states,
            mask,
            q_len,
            dropout=0.0,
        )
        # flash_out: (bsz, n_heads, q_len, head_dim) -> reshape like forward
        flash_flat = (
            flash_out.transpose(1, 2).contiguous().view(seqlen, n_q * head_dim)
        )
        out["hf_flash"] = flash_flat.float().cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out


def _vllm_attn_on_qkv(q_flat, k_flat, v_flat, ids_len: int) -> dict:
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

    sys.path.insert(0, os.path.dirname(__file__))
    from gate1_l0_sparse_bisect import _bind_sparse_kv_cache, _make_sparse_prefill_metadata

    out: dict = {}
    model_config = ModelConfig(
        model=WEIGHTS, trust_remote_code=True, dtype="bfloat16", max_model_len=4096
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        load_config=LoadConfig(),
        cache_config=CacheConfig(block_size=256),
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
            model = get_model_loader(LoadConfig()).load_model(
                vllm_config=vllm_config, model_config=model_config
            )
            model.eval().cuda()
            sa = model.model.layers[0].self_attn
            sparse_attn = sa.attn
            impl = sparse_attn.impl
            block_size = vllm_config.cache_config.block_size
            dense_len = parse_sparse_config(model_config.hf_config).dense_len
            _bind_sparse_kv_cache(sparse_attn, block_size, ids_len)
            meta = _make_sparse_prefill_metadata(
                ids_len, block_size, dense_len, q_flat.device
            )
            prefix = sparse_attn.layer_name

            out["vllm_scale"] = float(impl.scale)
            out["cu_seqlens"] = meta.query_start_loc.tolist()
            out["max_query_len"] = meta.max_query_len
            out["max_seq_len"] = meta.max_seq_len
            out["num_actual_tokens"] = meta.num_actual_tokens

            q = q_flat.contiguous()
            k = k_flat.contiguous()
            v = v_flat.contiguous()
            out["vv_q_shape"] = tuple(q.shape)
            out["vv_k_shape"] = tuple(k.shape)

            with torch.no_grad():
                with set_forward_context(
                    attn_metadata={prefix: meta},
                    vllm_config=vllm_config,
                    num_tokens=ids_len,
                    slot_mapping={prefix: meta.slot_mapping},
                ):
                    vv_out = sa.attn(q, k, v)
            # sa.attn returns flat [T, n_heads * head_dim]
            out["vllm_attn"] = vv_out.float().cpu()

            # Direct in-memory dense flash (3D layout as Attention uses internally).
            q3 = q.view(-1, sa.num_heads, sa.head_dim)
            k3 = k.view(-1, sa.num_kv_heads, sa.head_dim)
            v3 = v.view(-1, sa.num_kv_heads, sa.head_dim)
            buf = torch.empty_like(q3)
            with torch.no_grad():
                impl._forward_dense_in_memory_flash(q3, k3, v3, meta, buf)
            out["vllm_dense_flash"] = (
                buf.reshape(ids_len, sa.num_heads * sa.head_dim).float().cpu()
            )

            destroy_model_parallel()
            destroy_distributed_environment()
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp)
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main() -> int:
    sys.path.insert(0, os.path.dirname(__file__))
    from gate1_l0_sparse_bisect import _patch_hf

    _patch_hf()
    ids2 = _build_ids2()
    print(f"prompt={PROMPT!r} seqlen={len(ids2)}", flush=True)

    hf = _hf_qkv_and_dense_flash(ids2)
    vv = _vllm_attn_on_qkv(hf["q_flat"], hf["k_flat"], hf["v_flat"], len(ids2))

    print(f"head_dim={hf['head_dim']} num_q={hf['num_q_heads']} num_kv={hf['num_kv_heads']}", flush=True)
    print(f"hf_scale={hf['hf_scale']:.8g} expected_1_sqrt_d={hf['expected_scale']:.8g}", flush=True)
    print(f"vllm_scale={vv['vllm_scale']:.8g}", flush=True)
    print(f"hf_q_shape={hf['hf_q_shape']} hf_k_shape={hf['hf_k_shape']}", flush=True)
    print(f"vv_q_shape={vv['vv_q_shape']} vv_k_shape={vv['vv_k_shape']}", flush=True)
    print(f"cu_seqlens={vv['cu_seqlens']} max_q={vv['max_query_len']} max_k={vv['max_seq_len']}", flush=True)

    p1, d1 = _pos_peak(hf["hf_flash"], vv["vllm_attn"])
    print(f"hf_dense_flash vs vllm_sa_attn peak={p1:.6g}", flush=True)
    print("  per_pos " + " ".join(f"p{i}={d:.6g}" for i, d in enumerate(d1)), flush=True)

    p2, d2 = _pos_peak(hf["hf_flash"], vv["vllm_dense_flash"])
    print(f"hf_dense_flash vs vllm_in_memory_flash peak={p2:.6g}", flush=True)
    print("  per_pos " + " ".join(f"p{i}={d:.6g}" for i, d in enumerate(d2)), flush=True)

    p3, _ = _pos_peak(vv["vllm_attn"], vv["vllm_dense_flash"])
    print(f"vllm_sa_attn vs vllm_in_memory_flash peak={p3:.6g}", flush=True)

    return 0 if max(p1, p2) < 1e-3 else 1


if __name__ == "__main__":
    sys.exit(main())
