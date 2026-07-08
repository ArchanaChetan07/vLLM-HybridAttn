#!/usr/bin/env python3
"""Find first decode step where incremental state diverges from full prefill."""

from __future__ import annotations

import gc
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = os.environ.get("MINICPM_SALA_PROMPT", "Hello, my name is")
MAX_STEP = int(os.environ.get("MINICPM_SALA_MAX_STEP", "16"))


def hf_greedy(prompt_ids: list[int], steps: int) -> list[int]:
    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    cur = prompt_ids[:]
    out: list[int] = []
    with torch.no_grad():
        for _ in range(steps):
            nxt = int(
                model(
                    torch.tensor([cur], device="cuda"),
                    attention_mask=torch.ones(1, len(cur), device="cuda"),
                )
                .logits[0, -1]
                .argmax()
                .item()
            )
            out.append(nxt)
            cur.append(nxt)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main() -> int:
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    prompt_ids = tok.encode(PROMPT, add_special_tokens=True)
    hf = hf_greedy(prompt_ids, MAX_STEP)

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
        enable_chunked_prefill=False,
    )

    print(f"prompt_len={len(prompt_ids)}", flush=True)
    for step in range(MAX_STEP):
        prefix = prompt_ids + hf[:step]
        inc = int(
            llm.generate(
                [TokensPrompt(prompt_token_ids=prompt_ids)],
                SamplingParams(temperature=0, max_tokens=step + 1),
            )[0]
            .outputs[0]
            .token_ids[step]
        )
        one = int(
            llm.generate(
                [TokensPrompt(prompt_token_ids=prefix)],
                SamplingParams(temperature=0, max_tokens=1),
            )[0]
            .outputs[0]
            .token_ids[0]
        )
        hf_t = hf[step]
        ok = inc == one == hf_t
        print(
            f"step={step} seq={len(prefix)+1} hf={hf_t} incremental={inc} "
            f"one_shot={one} ok={ok}",
            flush=True,
        )
        if not ok:
            break
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
