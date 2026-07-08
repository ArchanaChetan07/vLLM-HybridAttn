#!/usr/bin/env python3
"""Capture L0 slot_mapping on prefill vs first decode."""

from __future__ import annotations

import os

import torch

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = "Hello, my name is"


def _install(model: torch.nn.Module) -> int:
    from vllm.forward_context import get_forward_context

    model._rows: list[dict] = []

    def _pre(_mod, args):
        if len(args) < 2 or not isinstance(args[1], torch.Tensor):
            return
        h = args[1]
        ctx = get_forward_context()
        md = ctx.attn_metadata
        if not isinstance(md, dict):
            return
        key = model.model.layers[0].self_attn.attn.layer_name
        if key not in md:
            return
        meta = md[key]
        model._rows.append(
            {
                "rows": int(h.shape[0]),
                "seq_lens": meta.seq_lens.tolist(),
                "slot_mapping": meta.slot_mapping.tolist(),
                "block_table": meta.block_table.tolist(),
                "qsl": meta.query_start_loc.tolist(),
            }
        )

    model._hook = model.model.layers[0].register_forward_pre_hook(_pre)
    return 0


def _read(model: torch.nn.Module) -> list:
    return list(getattr(model, "_rows", []))


def main() -> int:
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
        SamplingParams(temperature=0, max_tokens=1),
    )
    for r in llm.apply_model(_read)[0]:
        print(r, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
