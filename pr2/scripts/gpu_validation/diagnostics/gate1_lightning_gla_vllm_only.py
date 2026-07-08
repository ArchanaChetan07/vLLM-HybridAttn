#!/usr/bin/env python3
"""vLLM-only incremental steps with L1 GLA NDJSON (no HF model load)."""
from __future__ import annotations

import os

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = os.environ.get("MINICPM_SALA_PROMPT", "Hello, my name is")
MAX_STEP = int(os.environ.get("MINICPM_SALA_MAX_STEP", "15"))
HF_REF_14 = 16091


def main() -> int:
    os.environ.setdefault("MINICPM_SALA_DEBUG_GLA", "1")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    prompt_ids = tok.encode(PROMPT, add_special_tokens=True)
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
    out = llm.generate(
        [TokensPrompt(prompt_token_ids=prompt_ids)],
        SamplingParams(temperature=0, max_tokens=MAX_STEP),
    )[0].outputs[0].token_ids
    for step in range(min(MAX_STEP, len(out))):
        print(
            f"step={step} vllm={out[step]} match_hf14={out[14]==HF_REF_14 if step==14 else 'n/a'}",
            flush=True,
        )
    print(f"idx14_token={out[14] if len(out)>14 else None} hf_ref={HF_REF_14}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
