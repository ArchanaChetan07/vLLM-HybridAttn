#!/usr/bin/env python3
"""Compare HF vs vLLM greedy logits after 1 generated token (token-2 decision)."""

from __future__ import annotations

import gc
import os
import subprocess
import sys

import torch

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
    "Briefly explain gravity:",
]


def _patch_hf() -> None:
    script = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "remote",
        "patch_hf_transformers_compat.py",
    )
    script = os.path.normpath(script)
    if os.path.isfile(script):
        subprocess.run([sys.executable, script], check=False)


def _encode(tok, text: str) -> torch.Tensor:
    ids = tok.encode(text, add_special_tokens=True)
    return torch.tensor([ids], device="cuda")


def hf_two_step(model, ids: torch.Tensor) -> tuple[int, int, torch.Tensor]:
    attn = torch.ones_like(ids)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=attn)
        t1 = int(out.logits[0, -1].argmax().item())
        ids2 = torch.cat(
            [ids, torch.tensor([[t1]], device=ids.device, dtype=ids.dtype)], dim=1
        )
        attn2 = torch.ones_like(ids2)
        out2 = model(input_ids=ids2, attention_mask=attn2)
        logits2 = out2.logits[0, -1].float()
        t2 = int(logits2.argmax().item())
    return t1, t2, logits2.cpu()


def vllm_two_step(tok, prompt: str) -> tuple[int, int, torch.Tensor]:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    ids = tok.encode(prompt, add_special_tokens=True)
    llm = LLM(
        model=WEIGHTS,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
        block_size=256,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        max_num_seqs=1,
    )
    sp1 = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=0)
    out1 = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp1)[0]
    t1 = int(out1.outputs[0].token_ids[0])

    sp2 = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=0)
    ids2 = ids + [t1]
    out2 = llm.generate([TokensPrompt(prompt_token_ids=ids2)], sp2)[0]
    t2 = int(out2.outputs[0].token_ids[0])

    # Logits at last prefill position after prompt+t1 via one more prefill-only pass
    # is not exposed; compare token ids and re-run HF-style logit probe separately.
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return t1, t2, torch.tensor([])


def main() -> int:
    _patch_hf()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()

    print("=== HF two-token greedy ===", flush=True)
    hf_results = {}
    for p in PROMPTS:
        ids = _encode(tok, p)
        t1, t2, logits2 = hf_two_step(model, ids)
        hf_results[p] = (t1, t2, logits2)
        topv, topi = torch.topk(logits2, 5)
        print(f"{p!r}: t1={t1} t2={t2} top5={list(zip(topi.tolist(), topv.tolist()))}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=WEIGHTS,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
        block_size=256,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        max_num_seqs=1,
    )
    print("=== vLLM two-token greedy (fresh prefill per step) ===", flush=True)
    for p in PROMPTS:
        ids = tok.encode(p, add_special_tokens=True)
        t1 = int(
            llm.generate(
                [TokensPrompt(prompt_token_ids=ids)],
                SamplingParams(temperature=0, max_tokens=1),
            )[0]
            .outputs[0]
            .token_ids[0]
        )
        ids2 = ids + [t1]
        t2 = int(
            llm.generate(
                [TokensPrompt(prompt_token_ids=ids2)],
                SamplingParams(temperature=0, max_tokens=1),
            )[0]
            .outputs[0]
            .token_ids[0]
        )
        ht1, ht2, _ = hf_results[p]
        print(
            f"{p!r}: t1={t1} t2={t2} "
            f"hf_t2={ht2} match_t1={t1==ht1} match_t2={t2==ht2}"
        )

    print("=== vLLM single-session 2-token decode ===", flush=True)
    for p in PROMPTS:
        ids = tok.encode(p, add_special_tokens=True)
        out = llm.generate(
            [TokensPrompt(prompt_token_ids=ids)],
            SamplingParams(temperature=0, max_tokens=2),
        )[0]
        v_ids = list(out.outputs[0].token_ids)
        ht1, ht2, _ = hf_results[p]
        print(f"{p!r}: vllm={v_ids} hf=[{ht1},{ht2}] match={v_ids[:2]==[ht1,ht2]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
