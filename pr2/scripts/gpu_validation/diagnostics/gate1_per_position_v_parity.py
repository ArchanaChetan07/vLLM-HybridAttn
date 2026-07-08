#!/usr/bin/env python3
"""Blocker 2 regression: per-position v parity incremental vs one-shot at seq=21.

Asserts lightning-layer v histories match position-by-position between
incremental decode and one-shot prefill. Fails loud on first mismatch.
"""

from __future__ import annotations

import gc
import os
import sys

import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = os.environ.get("MINICPM_SALA_PROMPT", "Hello, my name is")
STEP = int(os.environ.get("MINICPM_SALA_MISMATCH_STEP", "14"))
TOL = float(os.environ.get("MINICPM_SALA_V_PARITY_TOL", "1e-5"))
LIGHTNING_LAYERS = tuple(
    int(x) for x in os.environ.get("MINICPM_SALA_LIGHTNING_LAYERS", "1,6,9").split(",")
)
HF_GREEDY = [
    2132, 1417, 1523, 7089, 1520, 1606, 5, 1975, 19020, 59324,
    59342, 63, 59377, 59320, 16091, 1525,
]


def _reset_hist(model: torch.nn.Module) -> int:
    for layer in model.model.layers:
        reset = getattr(layer.self_attn, "_reset_qkv_history", None)
        if callable(reset):
            reset()
    return 0


def _read_v_hist(model: torch.nn.Module) -> dict[int, torch.Tensor]:
    out: dict[int, torch.Tensor] = {}
    for idx in LIGHTNING_LAYERS:
        attn = model.model.layers[idx].self_attn
        if getattr(attn, "_qkv_hist_v", None) is not None:
            out[idx] = attn._qkv_hist_v.detach().float().cpu().clone()
    return out


def _first_v_mismatch(
    inc: torch.Tensor, one: torch.Tensor, tol: float
) -> tuple[int, float] | None:
    n = min(inc.shape[0], one.shape[0])
    diffs = (inc[:n] - one[:n]).abs().amax(dim=tuple(range(1, inc.dim())))
    bad = (diffs > tol).nonzero(as_tuple=False)
    if bad.numel() == 0:
        return None
    pos = int(bad[0].item())
    return pos, float(diffs[pos].item())


def main() -> int:
    from transformers import AutoTokenizer

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    prompt_ids = tok.encode(PROMPT, add_special_tokens=True)
    prefix_ids = prompt_ids + HF_GREEDY[:STEP]
    seq_len = len(prefix_ids) + 1
    print(f"prompt_len={len(prompt_ids)} step={STEP} seq_len={seq_len}", flush=True)

    llm = LLM(
        model=WEIGHTS,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
        block_size=256,
        gpu_memory_utilization=0.45,
        enforce_eager=True,
        max_num_seqs=1,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        enable_chunked_prefill=False,
    )

    llm.generate(
        [TokensPrompt(prompt_token_ids=prompt_ids)],
        SamplingParams(temperature=0, max_tokens=STEP + 1),
    )
    inc_v = llm.apply_model(_read_v_hist)[0]

    llm.apply_model(_reset_hist)
    llm.generate(
        [TokensPrompt(prompt_token_ids=prefix_ids)],
        SamplingParams(temperature=0, max_tokens=1),
    )
    one_v = llm.apply_model(_read_v_hist)[0]

    failed = False
    for layer_idx in LIGHTNING_LAYERS:
        iv = inc_v.get(layer_idx)
        ov = one_v.get(layer_idx)
        if iv is None or ov is None:
            print(f"FAIL L{layer_idx}: missing v history", flush=True)
            failed = True
            continue
        n = min(iv.shape[0], ov.shape[0])
        peak = (iv[:n] - ov[:n]).abs().max().item()
        print(f"L{layer_idx}_v_hist peak={peak:.6g} len={n}", flush=True)
        mm = _first_v_mismatch(iv, ov, TOL)
        if mm is not None:
            pos, p = mm
            print(
                f"FAIL L{layer_idx} first_v_mismatch pos={pos} peak={p:.6g}",
                flush=True,
            )
            failed = True
        else:
            print(f"PASS L{layer_idx} per-position v parity (tol={TOL})", flush=True)

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
