#!/usr/bin/env python3
"""Real-checkpoint qkv isolation using HF model + fused weight slices (no vLLM load)."""

from __future__ import annotations

import gc
import os
import subprocess
import sys

import torch
import torch.nn.functional as F

WEIGHTS = os.environ.get("MINICPM_SALA_WEIGHTS", "openbmb/MiniCPM-SALA")
PROMPT = os.environ.get("MINICPM_SALA_PROMPT", "Briefly explain gravity:")
DEVICE = os.environ.get("MINICPM_SALA_DEVICE", "cuda")


def _patch_hf() -> None:
    script = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "..",
            "scripts", "remote", "patch_hf_transformers_compat.py",
        )
    )
    if os.path.isfile(script):
        subprocess.run([sys.executable, script], check=False)


def _peak(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def main() -> int:
    _patch_hf()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids = tok.encode(PROMPT, add_special_tokens=True)
    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        attn_implementation="flash_attention_2",
    ).eval()
    with torch.no_grad():
        t1 = int(
            model(
                torch.tensor([ids], device=DEVICE),
                attention_mask=torch.ones(1, len(ids), device=DEVICE),
            )
            .logits[0, -1]
            .argmax()
        )
    ids2 = ids + [t1]
    sa = model.model.layers[0].self_attn
    with torch.no_grad():
        emb = model.model.embed_tokens(torch.tensor([ids2], device=DEVICE))
        emb = emb * model.config.scale_emb
        x = model.model.layers[0].input_layernorm(emb)
        hf_q = sa.q_proj(x)[0]
        hf_k = sa.k_proj(x)[0]
        hf_v = sa.v_proj(x)[0]

        wq = sa.q_proj.weight
        wk = sa.k_proj.weight
        wv = sa.v_proj.weight
        wf = torch.cat([wq, wk, wv], dim=0)
        if sa.q_proj.bias is not None:
            bf = torch.cat([sa.q_proj.bias, sa.k_proj.bias, sa.v_proj.bias], dim=0)
        else:
            bf = None

        x2 = x[0]
        fused = F.linear(x2, wf, bf)
        q_size = hf_q.shape[-1]
        kv_size = hf_k.shape[-1]
        fq, fk, fv = fused.split([q_size, kv_size, kv_size], dim=-1)
        tq, tk, tv = F.linear(x2, wq, sa.q_proj.bias), F.linear(x2, wk, sa.k_proj.bias), F.linear(x2, wv, sa.v_proj.bias)

    print(f"prompt={PROMPT!r} seqlen={len(ids2)} device={DEVICE}", flush=True)
    print("=== vs HF separate proj ===", flush=True)
    print(f"  fused q={_peak(fq, hf_q):.6g} k={_peak(fk, hf_k):.6g} v={_peak(fv, hf_v):.6g}", flush=True)
    print(f"  three q={_peak(tq, hf_q):.6g} k={_peak(tk, hf_k):.6g} v={_peak(tv, hf_v):.6g}", flush=True)
    print("=== VERDICT ===", flush=True)
    if _peak(fk, hf_k) > 1e-6 or _peak(fv, hf_v) > 1e-6:
        print("H1 fused GEMM: IN (fused one-matmul k/v differs from HF three-matmul)", flush=True)
    if _peak(tk, hf_k) < 1e-6 and _peak(tv, hf_v) < 1e-6:
        print("Fix: three separate matmuls restore HF parity on q/k/v", flush=True)
    trace = os.path.join(os.path.dirname(__file__), "traces", "qkv_isolate_latest.txt")
    os.makedirs(os.path.dirname(trace), exist_ok=True)
    with open(trace, "w", encoding="utf-8") as f:
        f.write(
            f"fused_k={_peak(fk, hf_k):.6g} fused_v={_peak(fv, hf_v):.6g}\n"
            f"three_k={_peak(tk, hf_k):.6g} three_v={_peak(tv, hf_v):.6g}\n"
        )
    print(f"trace_written={trace}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
