#!/usr/bin/env python3
"""Compare L0 output: incremental decode vs one-shot prefill at mismatch."""

from __future__ import annotations

import gc
import os

import torch

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = "Hello, my name is"
STEP = 14


def _install(model: torch.nn.Module) -> int:
    model._last: torch.Tensor | None = None

    def _hook(_mod, _inp, out):
        h = out if isinstance(out, torch.Tensor) else out
        if isinstance(h, torch.Tensor) and h.shape[0] == 1:
            model._last = h[-1].detach().float().cpu()

    model._h = model.model.layers[0].register_forward_hook(_hook)
    return 0


def _read(model: torch.nn.Module) -> torch.Tensor | None:
    return getattr(model, "_last", None)


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
    llm.apply_model(_install)
    llm.generate(
        [TokensPrompt(prompt_token_ids=prompt_ids)],
        SamplingParams(temperature=0, max_tokens=STEP + 1),
    )
    inc = llm.apply_model(_read)[0]
    llm2 = LLM(
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
    llm2.apply_model(_install)
    llm2.generate(
        [TokensPrompt(prompt_token_ids=cur[:-1])],
        SamplingParams(temperature=0, max_tokens=1),
    )
    one = llm2.apply_model(_read)[0]
    peak = (inc.float() - one.float()).abs().max().item() if inc is not None and one is not None else -1
    print(f"step={STEP} l0_peak={peak}", flush=True)
    del llm, llm2
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
