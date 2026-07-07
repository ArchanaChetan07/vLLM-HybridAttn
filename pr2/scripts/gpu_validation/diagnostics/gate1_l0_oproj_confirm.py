#!/usr/bin/env python3
"""Stage-1 confirm: o_proj input ~0.004, output jumps to ~0.016 vs HF (Briefly seqlen=7)."""

from __future__ import annotations

import gc
import os
import sys

import torch

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = os.environ.get("MINICPM_SALA_PROMPT", "Briefly explain gravity:")


def _pos_diff(a: torch.Tensor, b: torch.Tensor) -> list[float]:
    d = (a.float() - b.float()).abs()
    if d.dim() == 1:
        return [d.max().item()]
    return [d[i].max().item() for i in range(d.shape[0])]


def main() -> int:
    sys.path.insert(0, os.path.dirname(__file__))
    from gate1_l0_sparse_bisect import _patch_hf, hf_l0_traces, vllm_l0_traces

    _patch_hf()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    fp32 = os.environ.get("MINICPM_SALA_FP32_O_PROJ", "0")
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
        f"FP32_O_PROJ={fp32}",
        flush=True,
    )

    hf_t = hf_l0_traces(ids2)
    vv_t = vllm_l0_traces(ids2)

    for stage in ("o_proj_in", "o_proj_out", "attn_branch", "layer0"):
        if stage not in hf_t or stage not in vv_t:
            print(f"MISSING stage={stage}", flush=True)
            continue
        diffs = _pos_diff(hf_t[stage], vv_t[stage])
        peak = max(diffs)
        pos_str = " ".join(f"p{i}={d:.6g}" for i, d in enumerate(diffs))
        print(f"{stage:14s} peak={peak:.6g}  {pos_str}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
