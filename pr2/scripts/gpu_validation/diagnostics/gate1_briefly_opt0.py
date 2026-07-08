#!/usr/bin/env python3
"""Briefly t1 with optimization_level=0 (no norm/act fusions)."""

from __future__ import annotations

import os

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = "Briefly explain gravity:"


def main() -> int:
    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids = tok.encode(PROMPT, add_special_tokens=True)
    llm = LLM(
        model=WEIGHTS,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
        block_size=256,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        max_num_seqs=1,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        optimization_level=0,
    )
    t1 = int(
        llm.generate(
            [TokensPrompt(prompt_token_ids=ids)],
            SamplingParams(temperature=0, max_tokens=1),
        )[0]
        .outputs[0]
        .token_ids[0]
    )
    print(f"opt0 t1={t1} expected=1420", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
