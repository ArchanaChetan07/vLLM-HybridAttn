#!/usr/bin/env python3
"""Isolate fused QKV GEMM vs split for layer-0 dense attention (Briefly seqlen=7).

Hypotheses:
  H1 (fused GEMM): one bf16 matmul accumulates differently than three HF matmuls.
  H2 (split): q/k/v partition of fused output differs from HF head layout.

Usage (CPU or GPU):
  MINICPM_SALA_WEIGHTS=/path/to/MiniCPM-SALA python3 gate1_qkv_isolate.py
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys

import torch
import torch.nn.functional as F

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = os.environ.get("MINICPM_SALA_PROMPT", "Briefly explain gravity:")
DEVICE = os.environ.get("MINICPM_SALA_DEVICE", "cpu")


def _patch_hf() -> None:
    script = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "remote",
        "patch_hf_transformers_compat.py",
    )
    script = os.path.normpath(script)
    if os.path.isfile(script):
        subprocess.run([sys.executable, script], check=False)


def _peak(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def _pos_peaks(a: torch.Tensor, b: torch.Tensor) -> list[float]:
    d = (a.float() - b.float()).abs()
    if d.dim() == 1:
        return [d.max().item()]
    return [d[i].max().item() for i in range(d.shape[0])]


def _build_ids2() -> list[int]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids = tok.encode(PROMPT, add_special_tokens=True)
    hf = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        attn_implementation="flash_attention_2",
    ).eval()
    with torch.no_grad():
        t1 = int(
            hf(
                torch.tensor([ids], device=DEVICE),
                attention_mask=torch.ones(1, len(ids), device=DEVICE),
            )
            .logits[0, -1]
            .argmax()
        )
    del hf
    gc.collect()
    return ids + [t1]


def _hf_qkv(x: torch.Tensor, sa) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        q = sa.q_proj(x)
        k = sa.k_proj(x)
        v = sa.v_proj(x)
    return q[0], k[0], v[0]


def _fused_qkv(qkv_proj, x: torch.Tensor) -> torch.Tensor:
    out, _ = qkv_proj(x)
    return out


def _split_qkv(
    qkv: torch.Tensor, q_size: int, kv_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    return q, k, v


def _three_matmuls(
    qkv_proj,
    x: torch.Tensor,
    q_size: int,
    kv_size: int,
    *,
    fp32: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    w = qkv_proj.weight
    b = qkv_proj.bias
    if fp32:
        xin = x.float()
        w = w.float()
        b = b.float() if b is not None else None
    else:
        xin = x
    q_w, k_w, v_w = w[:q_size], w[q_size : q_size + kv_size], w[q_size + kv_size : q_size + 2 * kv_size]
    if b is not None:
        q_b, k_b, v_b = b[:q_size], b[q_size : q_size + kv_size], b[q_size + kv_size : q_size + 2 * kv_size]
    else:
        q_b = k_b = v_b = None
    q = F.linear(xin, q_w, q_b)
    k = F.linear(xin, k_w, k_b)
    v = F.linear(xin, v_w, v_b)
    if fp32:
        q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
    if q.dim() == 3:
        q, k, v = q[0], k[0], v[0]
    return q, k, v


def _load_vllm_l0():
    import tempfile
    import contextlib

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
    from vllm.model_executor.model_loader import get_model_loader

    model_config = ModelConfig(
        model=WEIGHTS, trust_remote_code=True, dtype="bfloat16", max_model_len=4096
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        load_config=LoadConfig(),
        cache_config=CacheConfig(block_size=256),
        device_config=DeviceConfig(device=DEVICE),
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
                backend="gloo" if DEVICE == "cpu" else "nccl",
            )
            initialize_model_parallel(1, 1)
            model = get_model_loader(LoadConfig()).load_model(
                vllm_config=vllm_config, model_config=model_config
            )
            model.eval()
            if DEVICE != "cpu":
                model.cuda()
            layer0 = model.model.layers[0]
            destroy_model_parallel()
            destroy_distributed_environment()
            return model, layer0
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp)


def main() -> int:
    _patch_hf()
    ids = _build_ids2()
    print(f"prompt={PROMPT!r} seqlen={len(ids)} device={DEVICE}", flush=True)

    from transformers import AutoModelForCausalLM

    hf_model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        attn_implementation="flash_attention_2",
    ).eval()
    hf_sa = hf_model.model.layers[0].self_attn
    with torch.no_grad():
        emb = hf_model.model.embed_tokens(torch.tensor([ids], device=DEVICE))
        emb = emb * hf_model.config.scale_emb
        x_hf = hf_model.model.layers[0].input_layernorm(emb)
    hf_q, hf_k, hf_v = _hf_qkv(x_hf, hf_sa)

    vv_model, layer0 = _load_vllm_l0()
    sa = layer0.self_attn
    with torch.no_grad():
        ids_t = torch.tensor(ids, device=DEVICE)
        emb_v = vv_model.model.get_input_embeddings(ids_t)
        x_vv = layer0.input_layernorm(emb_v).unsqueeze(0)

    qkv_fused = _fused_qkv(sa.qkv_proj, x_vv)
    vv_q_f, vv_k_f, vv_v_f = _split_qkv(qkv_fused[0], sa.q_size, sa.kv_size)

    vv_q3, vv_k3, vv_v3 = _three_matmuls(sa.qkv_proj, x_vv, sa.q_size, sa.kv_size, fp32=False)
    vv_q3f, vv_k3f, vv_v3f = _three_matmuls(
        sa.qkv_proj, x_vv, sa.q_size, sa.kv_size, fp32=True
    )

    print("=== H1 fused GEMM vs HF separate proj ===", flush=True)
    for name, t in [("q", vv_q_f), ("k", vv_k_f), ("v", vv_v_f)]:
        p = _peak(t, hf_q if name == "q" else hf_k if name == "k" else hf_v)
        print(f"  fused+split {name} peak={p:.6g}", flush=True)

    print("=== H1b three bf16 matmuls (same weights, HF order) ===", flush=True)
    for name, t, ref in [("q", vv_q3, hf_q), ("k", vv_k3, hf_k), ("v", vv_v3, hf_v)]:
        p = _peak(t, ref)
        print(f"  three_matmul {name} peak={p:.6g}", flush=True)

    print("=== H2 split-only: fused output vs manual slice ===", flush=True)
    manual = qkv_fused[0]
    q_s, k_s, v_s = manual.split([sa.q_size, sa.kv_size, sa.kv_size], dim=-1)
    print(f"  fused vs manual split q peak={_peak(q_s, vv_q_f):.6g} (expect 0)", flush=True)

    print("=== fp32 three matmuls vs HF ===", flush=True)
    for name, t, ref in [("q", vv_q3f, hf_q), ("k", vv_k3f, hf_k), ("v", vv_v3f, hf_v)]:
        p = _peak(t, ref)
        print(f"  fp32_three {name} peak={p:.6g}", flush=True)

    print("=== VERDICT ===", flush=True)
    fused_k = _peak(vv_k_f, hf_k)
    fused_v = _peak(vv_v_f, hf_v)
    three_k = _peak(vv_k3, hf_k)
    three_v = _peak(vv_v3, hf_v)
    if three_k < fused_k * 0.5 and three_v < fused_v * 0.5:
        print("H1 fused GEMM: IN — three separate matmuls shrink k/v delta", flush=True)
    else:
        print("H1 fused GEMM: OUT or inconclusive", flush=True)
    if _peak(q_s, vv_q_f) == 0:
        print("H2 split layout: OUT — split is exact partition of fused output", flush=True)

    trace_dir = os.path.join(os.path.dirname(__file__), "traces")
    os.makedirs(trace_dir, exist_ok=True)
    trace_path = os.path.join(trace_dir, "qkv_isolate_latest.txt")
    lines = [
        f"prompt={PROMPT!r} seqlen={len(ids)} device={DEVICE}",
        f"fused_k_peak={fused_k:.6g} fused_v_peak={fused_v:.6g}",
        f"three_bf16_k_peak={three_k:.6g} three_bf16_v_peak={three_v:.6g}",
        f"fp32_three_k_peak={_peak(vv_k3f, hf_k):.6g} fp32_three_v_peak={_peak(vv_v3f, hf_v):.6g}",
    ]
    with open(trace_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"trace_written={trace_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
