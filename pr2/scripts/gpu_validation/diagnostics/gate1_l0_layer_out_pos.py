#!/usr/bin/env python3
"""Capture L0 layer_out (residual) incremental vs oneshot at Hello seq=21.

layer_in for L0 is embeddings; L1.layer_in is L0.layer_out. W2 showed L0.layer_in
peak=0 but L1.layer_in residual; this isolates whether L0 forward itself drifts.
"""

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
    model._cap = {"in": [], "out": []}

    def _pre(_mod, args):
        if len(args) < 2:
            return
        hs = args[1]
        if isinstance(hs, torch.Tensor) and hs.numel():
            model._cap["in"].append(hs.detach().float().cpu().clone())

    def _post(_mod, _args, out):
        hs = out[0] if isinstance(out, tuple) else out
        if isinstance(hs, torch.Tensor) and hs.numel():
            model._cap["out"].append(hs.detach().float().cpu().clone())

    layer0 = model.model.layers[0]
    model._hooks = [
        layer0.register_forward_pre_hook(_pre),
        layer0.register_forward_hook(_post),
    ]
    return 0


def _reset(model: torch.nn.Module) -> int:
    model._cap = {"in": [], "out": []}
    return 0


def _cat(chunks):
    return torch.cat(chunks, dim=0) if chunks else None


def _report(label: str, a: torch.Tensor, b: torch.Tensor) -> None:
    n = min(a.shape[0], b.shape[0])
    diffs = (a[:n] - b[:n]).abs().amax(dim=1)
    print(f"{label} n={n} overall_peak={diffs.max().item():.6g}", flush=True)
    for pos in range(min(n, 12)):
        print(f"  pos={pos} peak={diffs[pos].item():.6g}", flush=True)
    if n > 12:
        for pos in (n - 3, n - 2, n - 1):
            print(f"  pos={pos} peak={diffs[pos].item():.6g}", flush=True)
    bad = (diffs > 1e-5).nonzero(as_tuple=False)
    if bad.numel():
        p = int(bad[0].item())
        print(f"  first>1e-5: pos={p} peak={diffs[p].item():.6g}", flush=True)
    else:
        print("  all positions within 1e-5", flush=True)


def main() -> int:
    from transformers import AutoTokenizer

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    prompt_ids = tok.encode(PROMPT, add_special_tokens=True)
    prefix = prompt_ids + HF_GREEDY[:STEP]
    print(f"prompt_len={len(prompt_ids)} prefix_len={len(prefix)}", flush=True)

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
    inc = llm.apply_model(
        lambda m: {k: _cat(v) for k, v in m._cap.items()}
    )[0]

    llm.apply_model(_reset)
    llm.generate(
        [TokensPrompt(prompt_token_ids=prefix)],
        SamplingParams(temperature=0, max_tokens=1),
    )
    one = llm.apply_model(
        lambda m: {k: _cat(v) for k, v in m._cap.items()}
    )[0]

    _report("L0_layer_in", inc["in"], one["in"])
    _report("L0_layer_out", inc["out"], one["out"])

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
