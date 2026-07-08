#!/usr/bin/env python3
"""Trace compute_logits argmax vs sampled token at mismatch step."""

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
    model._trace: list[dict] = []
    orig = model.compute_logits

    def _wrap(hidden_states: torch.Tensor):
        logits = orig(hidden_states)
        if logits is not None:
            row = logits[-1].float()
            model._trace.append(
                {
                    "rows": int(hidden_states.shape[0]),
                    "argmax": int(row.argmax().item()),
                    "top2": row.topk(2).indices.tolist(),
                }
            )
        return logits

    model.compute_logits = _wrap
    return 0


def _read(model: torch.nn.Module) -> list:
    return list(getattr(model, "_trace", []))


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
        gpu_memory_utilization=0.45,
        enforce_eager=True,
        max_num_seqs=1,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        enable_chunked_prefill=False,
    )
    llm.apply_model(_install)
    out_inc = llm.generate(
        [TokensPrompt(prompt_token_ids=prompt_ids)],
        SamplingParams(temperature=0, max_tokens=STEP + 1),
    )[0]
    inc_trace = llm.apply_model(_read)[0]
    inc_tok = int(out_inc.outputs[0].token_ids[STEP])
    llm.apply_model(lambda m: setattr(m, "_trace", []) or 0)
    out_one = llm.generate(
        [TokensPrompt(prompt_token_ids=cur[:-1])],
        SamplingParams(temperature=0, max_tokens=1),
    )[0]
    one_trace = llm.apply_model(_read)[0]
    one_tok = int(out_one.outputs[0].token_ids[0])
    print(f"inc_tok={inc_tok} inc_last={inc_trace[-3:]}", flush=True)
    print(f"one_tok={one_tok} one_last={one_trace[-3:]}", flush=True)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
