#!/usr/bin/env python3
"""GPU Step B: HF-reference vs vLLM parity, run SEQUENTIALLY.

Loads the HF reference (trust_remote_code) first, records greedy tokens
and top-k logprobs, frees it, then loads this port under vLLM and
compares. Sequential loading keeps peak memory to one ~19 GB model at a
time (fits a 48 GB card; comfortable on A100 80 GB).

Harness rules learned from the failed 2026-07-07 run (both are baked in
here so they cannot regress):

  1. vLLM receives ``TokensPrompt(prompt_token_ids=...)`` built from the
     SAME ``tokenizer.encode(..., add_special_tokens=True)`` ids HF sees.
     Passing raw strings dropped the BOS token (id 1) and desynced every
     position downstream.
  2. ``LLM.generate`` has no ``prompt_token_ids=`` kwarg on vLLM 0.24 --
     token ids go inside ``TokensPrompt``.

Env:
  MINICPM_SALA_WEIGHTS  path or hub id (default: openbmb/MiniCPM-SALA)
  MINICPM_SALA_LONG=1   also run the >= 8192-token sparse-regime prompt
  PARITY_MAX_TOKENS     greedy continuation length (default 16)
  PARITY_TOPK           logprob set size compared per step (default 5)

Exit 0 = all compared prompts parity-clean. Exit 1 otherwise.
"""

import gc
import os
import sys

import torch

MODEL = os.environ.get("MINICPM_SALA_WEIGHTS", "openbmb/MiniCPM-SALA")
MAX_TOKENS = int(os.environ.get("PARITY_MAX_TOKENS", "16"))
TOPK = int(os.environ.get("PARITY_TOPK", "5"))
RUN_LONG = os.environ.get("MINICPM_SALA_LONG", "0") == "1"

SHORT_PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
    "Briefly explain gravity:",
]


def hf_reference(prompt_ids_list: list[list[int]]) -> list[dict]:
    from transformers import AutoModelForCausalLM

    print(f"[HF] loading {MODEL} (bf16, trust_remote_code) ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).cuda()
    model.eval()

    results = []
    with torch.no_grad():
        for ids in prompt_ids_list:
            input_ids = torch.tensor([ids], device="cuda")
            attn_mask = torch.ones_like(input_ids)
            generated: list[int] = []
            step_topk: list[list[int]] = []
            cur_ids, cur_mask = input_ids, attn_mask
            for _ in range(MAX_TOKENS):
                logits = model(input_ids=cur_ids, attention_mask=cur_mask).logits
                last = logits[0, -1].float()
                top = last.topk(TOPK).indices.tolist()
                nxt = int(last.argmax().item())
                generated.append(nxt)
                step_topk.append(top)
                cur_ids = torch.cat(
                    [cur_ids, torch.tensor([[nxt]], device="cuda")], dim=1
                )
                cur_mask = torch.ones_like(cur_ids)
            results.append({"tokens": generated, "topk": step_topk})

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("[HF] done, model freed.")
    return results


def vllm_port(prompt_ids_list: list[list[int]]) -> list[dict]:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    print(f"[vLLM] loading {MODEL} ...")
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
    )
    params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, logprobs=TOPK)
    # Rule 1: token ids identical to HF's, BOS included, via TokensPrompt.
    prompts = [TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids_list]
    outs = llm.generate(prompts, params)

    results = []
    for out in outs:
        comp = out.outputs[0]
        step_topk = [
            sorted(lp.keys(), key=lambda t: lp[t].rank)[:TOPK] if lp else []
            for lp in (comp.logprobs or [])
        ]
        results.append({"tokens": list(comp.token_ids), "topk": step_topk})
    return results


def main() -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompt_ids = [
        tokenizer.encode(p, add_special_tokens=True) for p in SHORT_PROMPTS
    ]
    labels = list(SHORT_PROMPTS)

    if RUN_LONG:
        # Sparse-regime prompt: >= dense_len (8192) tokens. Repeated text is
        # fine -- this checks numerics, not quality.
        base = tokenizer.encode(
            "The quick brown fox jumps over the lazy dog. ",
            add_special_tokens=False,
        )
        long_ids = [tokenizer.bos_token_id or 1] + base * (8300 // len(base) + 1)
        long_ids = long_ids[:8500]
        prompt_ids.append(long_ids)
        labels.append(f"<long {len(long_ids)} tokens>")

    hf = hf_reference(prompt_ids)
    vl = vllm_port(prompt_ids)

    all_ok = True
    for label, h, v in zip(labels, hf, vl):
        greedy_ok = h["tokens"] == v["tokens"]
        # Logprob check: vLLM's greedy token must be inside HF's top-k at
        # every step (loose but load-bearing; exact-token check above is
        # the strict gate).
        logprobs_ok = all(
            vt in htop for vt, htop in zip(v["tokens"], h["topk"])
        )
        status = "PASS" if greedy_ok and logprobs_ok else "FAIL"
        all_ok &= greedy_ok and logprobs_ok
        print(
            f"[{status}] {label!r}: greedy_ok={greedy_ok} "
            f"logprobs_ok={logprobs_ok}\n"
            f"    HF   tokens: {h['tokens']}\n"
            f"    vLLM tokens: {v['tokens']}"
        )

    print("\nSTEP B " + ("PASS" if all_ok else "FAIL"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
