#!/usr/bin/env python3
"""Compare positions tensor: incremental decode step14 vs one-shot prefill."""

from __future__ import annotations

import os

import torch

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = "Hello, my name is"
STEP = 14


def _install(model: torch.nn.Module) -> int:
    model._pos: list[torch.Tensor] = []

    def _pre(_mod, args):
        if len(args) >= 1 and isinstance(args[0], torch.Tensor):
            model._pos.append(args[0].detach().cpu().clone())

    model._h = model.model.layers[0].register_forward_pre_hook(_pre)
    return 0


def _read(model: torch.nn.Module) -> list:
    return list(getattr(model, "_pos", []))


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    prompt_ids = tok.encode(PROMPT, add_special_tokens=True)
    hf = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    cur = prompt_ids[:]
    for _ in range(STEP + 1):
        with torch.no_grad():
            nxt = int(hf(torch.tensor([cur], device="cuda")).logits[0, -1].argmax())
        cur.append(nxt)
    del hf

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
    llm.apply_model(_install)
    llm.generate(
        [TokensPrompt(prompt_token_ids=prompt_ids)],
        SamplingParams(temperature=0, max_tokens=STEP + 1),
    )
    inc = llm.apply_model(_read)[0]
    llm.apply_model(lambda m: setattr(m, "_pos", []) or 0)
    llm.generate(
        [TokensPrompt(prompt_token_ids=cur[:-1])],
        SamplingParams(temperature=0, max_tokens=1),
    )
    one = llm.apply_model(_read)[0]
    print(f"inc_last={inc[-1].tolist() if inc else None}", flush=True)
    print(f"one_last={one[-1].tolist() if one else None}", flush=True)
    if inc and one:
        print(f"match={inc[-1].item() == one[-1].item()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
