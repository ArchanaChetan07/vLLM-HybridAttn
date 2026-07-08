#!/usr/bin/env python3
"""Blocker 2 regression: same-length incremental vs one-shot v parity.

Compares lightning-layer v histories after incremental decode against a
fresh one-shot prefill of the *same* token prefix (prompt + greedy tokens).

Primary assert: last-token v matches (decode output semantics).
Secondary assert: for short online lengths (step <= SHORT_STEP), every
position matches within tol. Longer one-shot prefills can differ from an
accumulated history in early-position FA residua (same pattern as HF
use_cache=True vs False flipping Hello step 14: 17802 vs 16091) — that is
tracked but must not weaken the last-token gate.
"""

from __future__ import annotations

import gc
import os

import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = os.environ.get("MINICPM_SALA_PROMPT", "Hello, my name is")
MAX_STEP = int(os.environ.get("MINICPM_SALA_MISMATCH_STEP", "14"))
SHORT_STEP = int(os.environ.get("MINICPM_SALA_V_PARITY_SHORT_STEP", "7"))
TOL = float(os.environ.get("MINICPM_SALA_V_PARITY_TOL", "1e-5"))
LIGHTNING_LAYERS = tuple(
    int(x)
    for x in os.environ.get("MINICPM_SALA_LIGHTNING_LAYERS", "1,6").split(",")
    if x.strip()
)


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


def _first_mismatch(
    a: torch.Tensor, b: torch.Tensor, tol: float
) -> tuple[int, float] | None:
    n = min(a.shape[0], b.shape[0])
    diffs = (a[:n] - b[:n]).abs().amax(dim=tuple(range(1, a.dim())))
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
    print(
        f"prompt_len={len(prompt_ids)} max_step={MAX_STEP} "
        f"short_step={SHORT_STEP} tol={TOL}",
        flush=True,
    )

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

    failed = False
    steps = sorted({1, max(1, SHORT_STEP), MAX_STEP})
    for step in steps:
        llm.apply_model(_reset_hist)
        gen = llm.generate(
            [TokensPrompt(prompt_token_ids=prompt_ids)],
            SamplingParams(temperature=0, max_tokens=step),
        )[0]
        gen_ids = list(gen.outputs[0].token_ids)
        prefix = prompt_ids + gen_ids
        inc_v = llm.apply_model(_read_v_hist)[0]

        llm.apply_model(_reset_hist)
        llm.generate(
            [TokensPrompt(prompt_token_ids=prefix)],
            SamplingParams(temperature=0, max_tokens=1),
        )
        one_v = llm.apply_model(_read_v_hist)[0]

        print(
            f"--- step={step} seq={len(prefix)} last={gen_ids[-1]} ---",
            flush=True,
        )
        for layer_idx in LIGHTNING_LAYERS:
            iv = inc_v.get(layer_idx)
            ov = one_v.get(layer_idx)
            if iv is None or ov is None:
                print(f"FAIL L{layer_idx}: missing v history", flush=True)
                failed = True
                continue
            n = min(iv.shape[0], ov.shape[0])
            last_peak = (iv[n - 1] - ov[n - 1]).abs().max().item()
            overall = (iv[:n] - ov[:n]).abs().max().item()
            mm = _first_mismatch(iv, ov, TOL)

            if last_peak > TOL:
                print(
                    f"FAIL L{layer_idx} last_token_v peak={last_peak:.6g}",
                    flush=True,
                )
                failed = True
            else:
                print(
                    f"PASS L{layer_idx} last_token_v peak={last_peak:.6g}",
                    flush=True,
                )

            if step <= SHORT_STEP:
                if mm is None:
                    print(
                        f"PASS L{layer_idx} full_v parity peak={overall:.6g} n={n}",
                        flush=True,
                    )
                else:
                    pos, p = mm
                    print(
                        f"FAIL L{layer_idx} early_v pos={pos} peak={p:.6g}",
                        flush=True,
                    )
                    failed = True
            elif mm is not None:
                pos, p = mm
                print(
                    f"NOTE L{layer_idx} early_FA_residua pos={pos} "
                    f"peak={p:.6g} overall={overall:.6g} "
                    f"(not fail: long oneshot vs accumulated hist)",
                    flush=True,
                )

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
