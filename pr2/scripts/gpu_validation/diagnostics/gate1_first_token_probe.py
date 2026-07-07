#!/usr/bin/env python3
"""Compare HF vs vLLM tokenization and first-step greedy token."""

from __future__ import annotations

import gc
import os
import sys

import torch

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = "Hello, my name is"


def hf_step():
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
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0, -1]
    lp = torch.log_softmax(logits.float(), dim=-1)
    topv, topi = torch.topk(lp, 5)
    greedy = int(topi[0])
    print("HF ids", ids[0].tolist())
    print("HF greedy", greedy, tok.decode([greedy]))
    print("HF top5", [(int(topi[i]), float(topv[i])) for i in range(5)])
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return tok, greedy


def vllm_step(tok):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=WEIGHTS,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
        block_size=256,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
    )
    vtok = llm.get_tokenizer()
    print("vLLM ids", vtok.encode(PROMPT))
    sp = SamplingParams(temperature=0, max_tokens=1, logprobs=5)
    out = llm.generate([PROMPT], sp)[0]
    v_ids = list(out.outputs[0].token_ids)
    v_lps = out.outputs[0].logprobs[0]
    print("vLLM greedy", v_ids[0], vtok.decode([v_ids[0]]))
    top = sorted(v_lps.items(), key=lambda kv: kv[1], reverse=True)[:5]
    print("vLLM top5", [(int(k), float(v)) for k, v in top])
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return v_ids[0]


def main() -> int:
    os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
    import subprocess

    subprocess.run(
        [sys.executable, "/workspace/hybridattn/scripts/remote/patch_hf_transformers_compat.py"],
        check=False,
    )
    tok, hf_greedy = hf_step()
    v_greedy = vllm_step(tok)
    match = hf_greedy == v_greedy
    print(f"FIRST_TOKEN_MATCH={match}", flush=True)
    return 0 if match else 1


if __name__ == "__main__":
    sys.exit(main())
