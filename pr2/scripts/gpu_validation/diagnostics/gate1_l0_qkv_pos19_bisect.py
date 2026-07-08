#!/usr/bin/env python3
from __future__ import annotations
import os
import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
WEIGHTS = os.environ.get("MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA")
PROMPT = "Hello, my name is"
STEP = 14
HF_GREEDY = [2132, 1417, 1523, 7089, 1520, 1606, 5, 1975, 19020, 59324, 59342, 63, 59377, 59320, 16091, 1525]
POS = 19

def _read_dense_hist(model):
    layer = model.model.layers[0]
    out = {}
    for tag in ("q", "k", "v"):
        t = getattr(layer, f"_sala_dense_kv_{tag}", None)
        out[tag] = None if t is None else t.detach().float().cpu().clone()
    return out

def _install_attn_hook(model):
    model._attn_chunks = []
    attn = model.model.layers[0].self_attn
    def _h(_m, _i, o):
        t = o[0] if isinstance(o, tuple) else o
        if isinstance(t, torch.Tensor):
            model._attn_chunks.append(t.detach().float().cpu().clone())
    model._attn_hook = attn.register_forward_hook(_h)
    return 0

def _attn_cat(model):
    c = getattr(model, "_attn_chunks", [])
    return torch.cat(c, dim=0) if c else None

def _peak(a, b, pos):
    n = min(a.shape[0], b.shape[0])
    allp = (a[:n] - b[:n]).abs().amax().item()
    pp = (a[pos] - b[pos]).abs().max().item() if n > pos else float("nan")
    return n, allp, pp

def main():
    from transformers import AutoTokenizer
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    prompt_ids = tok.encode(PROMPT, add_special_tokens=True)
    prefix = prompt_ids + HF_GREEDY[:STEP]
    llm = LLM(model=WEIGHTS, trust_remote_code=True, dtype="bfloat16", max_model_len=4096,
              block_size=256, gpu_memory_utilization=0.45, enforce_eager=True, max_num_seqs=1,
              enable_prefix_caching=False, mamba_cache_mode="none", enable_chunked_prefill=False)
    llm.apply_model(_install_attn_hook)
    llm.generate([TokensPrompt(prompt_token_ids=prompt_ids)], SamplingParams(temperature=0, max_tokens=STEP+1))
    inc_hist = llm.apply_model(_read_dense_hist)[0]
    inc_attn = llm.apply_model(_attn_cat)[0]
    llm.apply_model(lambda m: (setattr(m, "_attn_chunks", []), 0)[1])
    llm.generate([TokensPrompt(prompt_token_ids=prefix)], SamplingParams(temperature=0, max_tokens=1))
    one_hist = llm.apply_model(_read_dense_hist)[0]
    one_attn = llm.apply_model(_attn_cat)[0]
    print(f"prefix_len={len(prefix)} pos={POS}", flush=True)
    for name in ("q", "k", "v"):
        a, b = inc_hist.get(name), one_hist.get(name)
        if a is None or b is None:
            print(f"dense_hist_{name}: missing", flush=True)
            continue
        n, allp, pp = _peak(a, b, POS)
        print(f"dense_hist_{name} n={n} overall_peak={allp:.6g} pos{POS}_peak={pp:.6g}", flush=True)
    if inc_attn is not None and one_attn is not None:
        n, allp, pp = _peak(inc_attn, one_attn, POS)
        print(f"attn_branch n={n} overall_peak={allp:.6g} pos{POS}_peak={pp:.6g}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
