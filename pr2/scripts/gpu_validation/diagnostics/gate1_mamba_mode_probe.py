#!/usr/bin/env python3
"""Probe token-2 parity vs mamba_cache_mode."""

from __future__ import annotations

import gc
import os
import sys

import torch

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = "Hello, my name is"


def hf_two() -> tuple[int, int]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids = tok.encode(PROMPT, add_special_tokens=True)
    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    with torch.no_grad():
        attn = torch.ones(1, len(ids), device="cuda")
        t1 = int(
            model(
                torch.tensor([ids], device="cuda"), attention_mask=attn
            ).logits[0, -1].argmax()
        )
        ids2 = ids + [t1]
        attn2 = torch.ones(1, len(ids2), device="cuda")
        t2 = int(
            model(
                torch.tensor([ids2], device="cuda"), attention_mask=attn2
            ).logits[0, -1].argmax()
        )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return t1, t2


def vllm_two(mode: str, mamba_cache_dtype: str = "auto") -> list[int]:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer

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
        mamba_cache_mode=mode,
        enable_prefix_caching=False,
        mamba_cache_dtype=mamba_cache_dtype,
    )
    out = llm.generate(
        [TokensPrompt(prompt_token_ids=ids)],
        SamplingParams(temperature=0, max_tokens=2),
    )[0]
    v = list(out.outputs[0].token_ids)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return v


def main() -> int:
    t1, t2 = hf_two()
    print(f"HF two-token greedy: [{t1}, {t2}]", flush=True)
    for mode in ("none",):
        for dtype in ("auto", "float32"):
            try:
                v = vllm_two(mode, mamba_cache_dtype=dtype)
                print(
                    f"vLLM mode={mode!r} mamba_cache_dtype={dtype!r}: "
                    f"{v} match={v == [t1, t2]}",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"vLLM mode={mode!r} mamba_cache_dtype={dtype!r}: FAIL {e}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
