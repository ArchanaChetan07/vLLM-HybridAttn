#!/usr/bin/env python3
"""Sweep mamba_cache_mode for Briefly t1 greedy."""

from __future__ import annotations

import gc
import os

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = "Briefly explain gravity:"


def main() -> int:
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    ids = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True).encode(
        PROMPT, add_special_tokens=True
    )
    for mode in ("none", "align", "all"):
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
            mamba_cache_mode=mode,
            enable_chunked_prefill=False,
        )
        t1 = int(
            llm.generate(
                [TokensPrompt(prompt_token_ids=ids)],
                SamplingParams(temperature=0, max_tokens=1),
            )[0]
            .outputs[0]
            .token_ids[0]
        )
        print(f"mode={mode} t1={t1}", flush=True)
        del llm
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
