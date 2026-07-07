#!/usr/bin/env python3
"""Compare HF vs vLLM greedy after prefill on prompt+t1 (no incremental state)."""

from __future__ import annotations

import gc
import os
import sys

import torch

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = os.environ.get("MINICPM_SALA_PROMPT", "Hello, my name is")


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids = tok.encode(PROMPT, add_special_tokens=True)
    print(f"prompt_len={len(ids)} ids={ids}", flush=True)

    hf = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    with torch.no_grad():
        t1 = int(
            hf(
                torch.tensor([ids], device="cuda"),
                attention_mask=torch.ones(1, len(ids), device="cuda"),
            )
            .logits[0, -1]
            .argmax()
        )
        ids2 = ids + [t1]
        logits2 = hf(
            torch.tensor([ids2], device="cuda"),
            attention_mask=torch.ones(1, len(ids2), device="cuda"),
        ).logits[0, -1].float()
        t2 = int(logits2.argmax())
        topv, topi = torch.topk(logits2, 5)
    print(f"HF t1={t1} t2={t2} top5={list(zip(topi.tolist(), topv.tolist()))}", flush=True)
    del hf
    gc.collect()
    torch.cuda.empty_cache()

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
    # Full prefill on prompt+t1, generate one token (no cross-request state).
    out = llm.generate(
        [TokensPrompt(prompt_token_ids=ids2)],
        SamplingParams(temperature=0, max_tokens=1, logprobs=5),
    )[0]
    vt2 = int(out.outputs[0].token_ids[0])
    vlp = out.outputs[0].logprobs[0]
    vtop = sorted(
        ((int(k), float(v.logprob if hasattr(v, "logprob") else v)) for k, v in vlp.items()),
        key=lambda x: -x[1],
    )[:5]
    print(f"vLLM prefill(ids2) gen t2={vt2} top5={vtop} match={vt2 == t2}", flush=True)
    return 0 if vt2 == t2 else 1


if __name__ == "__main__":
    sys.exit(main())
