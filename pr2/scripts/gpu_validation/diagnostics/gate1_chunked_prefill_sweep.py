#!/usr/bin/env python3
"""Quick sweep: chunked prefill on/off vs HF greedy."""

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
MAX_TOKENS = int(os.environ.get("MINICPM_SALA_MAX_TOKENS", "16"))


def hf_greedy(ids: list[int]) -> list[int]:
    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    cur = ids[:]
    out: list[int] = []
    with torch.no_grad():
        for _ in range(MAX_TOKENS):
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
    ids = tok.encode(PROMPT, add_special_tokens=True)
    hf = hf_greedy(ids)
    print(f"prompt={PROMPT!r} len={len(ids)} hf={hf}", flush=True)
    for chunked in (True, False):
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
            enable_chunked_prefill=chunked,
        )
        v = list(
            llm.generate(
                [TokensPrompt(prompt_token_ids=ids)],
                SamplingParams(temperature=0, max_tokens=MAX_TOKENS),
            )[0]
            .outputs[0]
            .token_ids
        )
        match = v == hf
        first = next((i for i, (a, b) in enumerate(zip(hf, v)) if a != b), None)
        print(
            f"chunked_prefill={chunked} match={match} first_mismatch={first} v={v}",
            flush=True,
        )
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
