#!/usr/bin/env python3
import gc
import os
import sys

import torch

W = os.environ.get("MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA")
P = "Hello, my name is"


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(W, trust_remote_code=True)
    ids = tok.encode(P, return_tensors="pt").to("cuda")
    hf = AutoModelForCausalLM.from_pretrained(
        W,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    with torch.no_grad():
        g_hf = int(
            hf(input_ids=ids, attention_mask=torch.ones_like(ids))
            .logits[0, -1]
            .argmax()
        )
    print("HF greedy", g_hf, flush=True)
    del hf
    gc.collect()
    torch.cuda.empty_cache()

    llm = LLM(
        model=W,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
        block_size=256,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
    )
    out = llm.generate([P], SamplingParams(temperature=0, max_tokens=1))[0]
    g_v = int(out.outputs[0].token_ids[0])
    print("vLLM greedy", g_v, "match", g_hf == g_v, flush=True)
    return 0 if g_hf == g_v else 1


if __name__ == "__main__":
    sys.exit(main())
