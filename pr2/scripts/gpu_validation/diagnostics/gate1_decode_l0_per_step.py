#!/usr/bin/env python3
"""L0 hidden peak per decode step: incremental vs one-shot prefix."""

from __future__ import annotations

import gc
import os

import torch

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = "Hello, my name is"
MAX_STEP = 15


def _install(model: torch.nn.Module) -> int:
    model._last: torch.Tensor | None = None

    def _hook(_mod, _inp, out):
        h = out if isinstance(out, torch.Tensor) else out
        if isinstance(h, torch.Tensor) and h.shape[0] >= 1:
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
    for _ in range(MAX_STEP + 1):
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
        gpu_memory_utilization=0.45,
        enforce_eager=True,
        max_num_seqs=1,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        enable_chunked_prefill=False,
    )
    llm.apply_model(_install)
    for step in range(MAX_STEP + 1):
        llm.apply_model(lambda m: setattr(m, "_last", None) or 0)
        llm.generate(
            [TokensPrompt(prompt_token_ids=prompt_ids)],
            SamplingParams(temperature=0, max_tokens=step + 1),
        )
        inc = llm.apply_model(_read)[0]
        llm.apply_model(lambda m: setattr(m, "_last", None) or 0)
        llm.generate(
            [TokensPrompt(prompt_token_ids=cur[: len(prompt_ids) + step])],
            SamplingParams(temperature=0, max_tokens=1),
        )
        one = llm.apply_model(_read)[0]
        peak = (inc.float() - one.float()).abs().max().item()
        print(f"step={step} l0_peak={peak:.6g}", flush=True)
    del llm
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
