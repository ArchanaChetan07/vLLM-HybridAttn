#!/usr/bin/env python3
"""Log slot_mapping vs block_table across decode steps."""

from __future__ import annotations

import os

import torch

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = "Hello, my name is"
_MAX = 0


def _install(model: torch.nn.Module) -> int:
    from vllm.forward_context import get_forward_context

    model._slots: list[dict] = []

    def _pre(_mod, args):
        global _MAX
        if len(args) < 2 or not isinstance(args[1], torch.Tensor):
            return
        if args[1].shape[0] != 1:
            return
        ctx = get_forward_context()
        md = ctx.attn_metadata
        if not isinstance(md, dict):
            return
        key = model.model.layers[0].self_attn.attn.layer_name
        if key not in md:
            return
        meta = md[key]
        row = {
            "seq_lens": meta.seq_lens.tolist(),
            "slot_mapping": meta.slot_mapping.tolist(),
            "block_table": meta.block_table.tolist(),
        }
        if not model._slots or model._slots[-1] != row:
            model._slots.append(row)
        _MAX += 1

    model._hook = model.model.layers[0].register_forward_pre_hook(_pre)
    return 0


def _read(model: torch.nn.Module) -> list:
    return list(getattr(model, "_slots", []))


def main() -> int:
    global _MAX
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids = tok.encode(PROMPT, add_special_tokens=True)
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
        [TokensPrompt(prompt_token_ids=ids)],
        SamplingParams(temperature=0, max_tokens=15),
    )
    rows = llm.apply_model(_read)[0]
    for i, r in enumerate(rows):
        print(f"step={i} {r}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
