#!/usr/bin/env python3
"""Hello 16-token greedy parity vs known HF reference (ISSUE-03b gate)."""

from __future__ import annotations

import os

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = "Hello, my name is"
HF_REF = [
    2132,
    1417,
    1523,
    7089,
    1520,
    1606,
    5,
    1975,
    19020,
    59324,
    59342,
    63,
    59377,
    59320,
    16091,
    1525,
]


def main() -> int:
    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids = tok.encode(PROMPT, add_special_tokens=True)
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
    out = list(
        llm.generate(
            [TokensPrompt(prompt_token_ids=ids)],
            SamplingParams(temperature=0, max_tokens=len(HF_REF)),
        )[0]
        .outputs[0]
        .token_ids
    )
    print("vLLM", out, flush=True)
    print("HF  ", HF_REF, flush=True)
    first_bad = None
    for i, (h, v) in enumerate(zip(HF_REF, out)):
        ok = h == v
        if not ok and first_bad is None:
            first_bad = i
        print(f"idx{i}: {'OK' if ok else f'RED hf={h} vllm={v}'}", flush=True)
    t14 = out[14] == HF_REF[14] if len(out) > 14 else False
    print(
        f"token14: {'GREEN' if t14 else f'RED hf={HF_REF[14]} vllm={out[14]}'}",
        flush=True,
    )
    print(f"first_mismatch_idx={first_bad}", flush=True)
    return 0 if first_bad is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
