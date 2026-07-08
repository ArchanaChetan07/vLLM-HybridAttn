#!/usr/bin/env python3
"""Isolate L0/L1 short-prefill vs long-prefill vs decode numerics."""

from __future__ import annotations

import gc
import os

import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = "Hello, my name is"
HF_GREEDY = [
    2132, 1417, 1523, 7089, 1520, 1606, 5, 1975, 19020, 59324,
    59342, 63, 59377, 59320, 16091, 1525,
]


def _install(model: torch.nn.Module) -> int:
    model._cap = {0: [], 1: []}

    def _mk(i):
        def _pre(_mod, args):
            if len(args) < 2:
                return
            hs = args[1]
            if isinstance(hs, torch.Tensor) and hs.numel():
                model._cap[i].append(hs.detach().float().cpu().clone())

        return _pre

    model._hooks = [
        model.model.layers[0].register_forward_pre_hook(_mk(0)),
        model.model.layers[1].register_forward_pre_hook(_mk(1)),
    ]
    return 0


def _reset(model: torch.nn.Module) -> int:
    model._cap = {0: [], 1: []}
    return 0


def _cat(m: torch.nn.Module) -> dict:
    return {k: (torch.cat(v, 0) if v else None) for k, v in m._cap.items()}


def _peak(a: torch.Tensor, b: torch.Tensor) -> float:
    n = min(a.shape[0], b.shape[0])
    return (a[:n] - b[:n]).abs().max().item()


def main() -> int:
    from transformers import AutoTokenizer

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    prompt = tok.encode(PROMPT, add_special_tokens=True)
    print(f"prompt_len={len(prompt)}", flush=True)

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
    llm.apply_model(_install)

    # A: short prefill only (len=6)
    llm.apply_model(_reset)
    llm.generate(
        [TokensPrompt(prompt_token_ids=prompt)],
        SamplingParams(temperature=0, max_tokens=1),
    )
    short = llm.apply_model(_cat)[0]

    # B: long one-shot (len=20)
    long_ids = prompt + HF_GREEDY[:14]
    llm.apply_model(_reset)
    llm.generate(
        [TokensPrompt(prompt_token_ids=long_ids)],
        SamplingParams(temperature=0, max_tokens=1),
    )
    long = llm.apply_model(_cat)[0]

    # C: incremental to len=7 (prefill6 + 1 decode)
    llm.apply_model(_reset)
    llm.generate(
        [TokensPrompt(prompt_token_ids=prompt)],
        SamplingParams(temperature=0, max_tokens=1),
    )
    # already have short+decode1 from above run in short capture which includes
    # decode; re-run clean incremental max_tokens=2 to get pos0-6
    llm.apply_model(_reset)
    llm.generate(
        [TokensPrompt(prompt_token_ids=prompt)],
        SamplingParams(temperature=0, max_tokens=2),
    )
    inc2 = llm.apply_model(_cat)[0]

    # D: one-shot of length 7 (= prompt + HF[0])
    ids7 = prompt + HF_GREEDY[:1]
    llm.apply_model(_reset)
    llm.generate(
        [TokensPrompt(prompt_token_ids=ids7)],
        SamplingParams(temperature=0, max_tokens=1),
    )
    one7 = llm.apply_model(_cat)[0]

    for layer in (0, 1):
        print(f"=== L{layer} ===", flush=True)
        s, lo, i2, o7 = short[layer], long[layer], inc2[layer], one7[layer]
        print(f"short_n={s.shape[0]} long_n={lo.shape[0]} "
              f"inc2_n={i2.shape[0]} one7_n={o7.shape[0]}", flush=True)
        # short prefill positions vs long prefill same positions
        print(
            f"short[:6] vs long[:6] peak={_peak(s[:6], lo[:6]):.6g}",
            flush=True,
        )
        # incremental after 1 decode vs oneshot len7
        print(
            f"inc2[:7] vs one7[:7] peak={_peak(i2[:7], o7[:7]):.6g}",
            flush=True,
        )
        # position-6 only (first decode)
        print(
            f"inc2[6] vs one7[6] peak={(i2[6]-o7[6]).abs().max().item():.6g}",
            flush=True,
        )
        # short capture includes decode token as last row when max_tokens=1
        print(
            f"short last vs one7[6] peak="
            f"{(s[-1]-o7[6]).abs().max().item():.6g}",
            flush=True,
        )

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
