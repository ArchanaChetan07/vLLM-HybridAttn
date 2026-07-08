#!/usr/bin/env python3
"""Probe L1 q/k/v history length and peak diff: incremental step14 vs one-shot."""

from __future__ import annotations

import gc
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = "Hello, my name is"
STEP = int(os.environ.get("MINICPM_SALA_MISMATCH_STEP", "14"))


def _reset_hist(model: torch.nn.Module) -> int:
    for layer in model.model.layers:
        reset = getattr(layer.self_attn, "_reset_qkv_history", None)
        if callable(reset):
            reset()
    return 0


def _read_l1(model: torch.nn.Module) -> dict:
    attn = model.model.layers[1].self_attn
    out: dict = {"hist_len": 0}
    if attn._qkv_hist_q is not None:
        out["hist_len"] = int(attn._qkv_hist_q.shape[0])
        out["q"] = attn._qkv_hist_q.detach().float().cpu().clone()
        out["k"] = attn._qkv_hist_k.detach().float().cpu().clone()
    return out


def main() -> int:
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
    llm.generate(
        [TokensPrompt(prompt_token_ids=prompt_ids)],
        SamplingParams(temperature=0, max_tokens=STEP + 1),
    )
    inc = llm.apply_model(_read_l1)[0]
    llm.apply_model(_reset_hist)
    llm.generate(
        [TokensPrompt(prompt_token_ids=cur[:-1])],
        SamplingParams(temperature=0, max_tokens=1),
    )
    one = llm.apply_model(_read_l1)[0]
    print(f"prefix_len={len(cur)-1} step={STEP}", flush=True)
    print(f"incremental hist_len={inc['hist_len']}", flush=True)
    print(f"oneshot hist_len={one['hist_len']}", flush=True)
    if inc.get("q") is not None and one.get("q") is not None:
        qlen = min(inc["q"].shape[0], one["q"].shape[0])
        qp = (inc["q"][:qlen] - one["q"][:qlen]).abs().max().item()
        kp = (inc["k"][:qlen] - one["k"][:qlen]).abs().max().item()
        print(f"L1 q peak diff (first {qlen} tok)={qp:.6g}", flush=True)
        print(f"L1 k peak diff (first {qlen} tok)={kp:.6g}", flush=True)
        qp_last = (inc["q"][-1] - one["q"][-1]).abs().max().item()
        print(f"L1 q peak diff (last tok only)={qp_last:.6g}", flush=True)
    del llm
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
