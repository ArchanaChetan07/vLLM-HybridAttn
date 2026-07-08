#!/usr/bin/env python3
"""Same-run L0 vs L1 layer_in peaks at pos 6 (incremental vs oneshot)."""

from __future__ import annotations

import gc
import os

import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = "Hello, my name is"
STEP = 14
HF_GREEDY = [
    2132, 1417, 1523, 7089, 1520, 1606, 5, 1975, 19020, 59324,
    59342, 63, 59377, 59320, 16091, 1525,
]


def _install(model: torch.nn.Module) -> int:
    model._cap = {0: [], 1: []}

    def _mk(i):
        def _pre(_mod, args):
            if len(args) < 2:
                return
            hs = args[1]
            if isinstance(hs, torch.Tensor) and hs.numel():
                model._cap[i].append(hs.detach().float().cpu().clone())

        return _pre

    model._hooks = [
        model.model.layers[0].register_forward_pre_hook(_mk(0)),
        model.model.layers[1].register_forward_pre_hook(_mk(1)),
    ]
    return 0


def _cat(chunks):
    return torch.cat(chunks, dim=0) if chunks else None


def main() -> int:
    from transformers import AutoTokenizer

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    prompt_ids = tok.encode(PROMPT, add_special_tokens=True)
    prefix = prompt_ids + HF_GREEDY[:STEP]

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
    inc = llm.apply_model(lambda m: {k: _cat(v) for k, v in m._cap.items()})[0]

    llm.apply_model(lambda m: setattr(m, "_cap", {0: [], 1: []}) or 0)
    llm.generate(
        [TokensPrompt(prompt_token_ids=prefix)],
        SamplingParams(temperature=0, max_tokens=1),
    )
    one = llm.apply_model(lambda m: {k: _cat(v) for k, v in m._cap.items()})[0]

    for layer in (0, 1):
        a, b = inc[layer], one[layer]
        if a is None or b is None:
            print(f"L{layer}: missing", flush=True)
            continue
        n = min(a.shape[0], b.shape[0])
        a, b = a[:n], b[:n]
        diffs = (a - b).abs().amax(dim=1)
        print(f"L{layer} n={n} overall_peak={diffs.max().item():.6g}", flush=True)
        for pos in range(min(n, 12)):
            print(f"  pos={pos} peak={diffs[pos].item():.6g}", flush=True)
        bad = (diffs > 1e-5).nonzero(as_tuple=False)
        if bad.numel():
            p = int(bad[0].item())
            print(f"  first>{1e-5}: pos={p} peak={diffs[p].item():.6g}", flush=True)
        else:
            print("  all positions within 1e-5", flush=True)

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
