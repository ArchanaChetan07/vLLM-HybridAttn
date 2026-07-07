#!/usr/bin/env python3
"""Gate 1: HF vs vLLM greedy + sparse-attn guard stats during real LLM.generate."""

from __future__ import annotations

import gc
import os
import subprocess
import sys

import torch

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = "Hello, my name is"


def _patch_hf_compat() -> None:
    script = "/workspace/hybridattn/scripts/remote/patch_hf_transformers_compat.py"
    if os.path.isfile(script):
        subprocess.run([sys.executable, script], check=False)


def hf_greedy() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids = tok.encode(PROMPT, return_tensors="pt").to("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
    greedy = int(logits[0, -1].float().argmax())
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return greedy


def vllm_greedy_with_sparse_trace() -> tuple[int, dict[str, int]]:
    from vllm import LLM, SamplingParams
    from vllm.v1.attention.backends.minicpm_sala_sparse import (
        MiniCPMSALASparseAttentionImpl,
        sequence_sparse_mask,
    )

    stats = {"none_meta": 0, "bad_kv": 0, "dense": 0, "sparse": 0, "mixed": 0}
    orig = MiniCPMSALASparseAttentionImpl.forward

    def traced(self, layer, query, key, value, kv_cache, attn_metadata, output, *a, **kw):
        if kv_cache.ndim < 2:
            stats["bad_kv"] += 1
        elif attn_metadata is None:
            stats["none_meta"] += 1
        else:
            mask = sequence_sparse_mask(attn_metadata.seq_lens, attn_metadata.dense_len)
            if mask.all():
                stats["sparse"] += 1
            elif not mask.any():
                stats["dense"] += 1
            else:
                stats["mixed"] += 1
        return orig(self, layer, query, key, value, kv_cache, attn_metadata, output, *a, **kw)

    MiniCPMSALASparseAttentionImpl.forward = traced
    try:
        llm = LLM(
            model=WEIGHTS,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=4096,
            block_size=256,
            gpu_memory_utilization=0.5,
            enforce_eager=True,
        )
        sp = SamplingParams(temperature=0, max_tokens=1, logprobs=1)
        out = llm.generate([PROMPT], sp)[0]
        greedy = int(out.outputs[0].token_ids[0])
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        return greedy, stats
    finally:
        MiniCPMSALASparseAttentionImpl.forward = orig


def main() -> int:
    _patch_hf_compat()
    hf = hf_greedy()
    vllm, stats = vllm_greedy_with_sparse_trace()
    print(f"HF greedy={hf}")
    print(f"vLLM greedy={vllm}")
    print(f"sparse_attn_stats={stats}")
    match = hf == vllm
    print(f"GREEDY_MATCH={match}", flush=True)
    return 0 if match else 1


if __name__ == "__main__":
    sys.exit(main())
