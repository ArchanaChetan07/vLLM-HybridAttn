#!/usr/bin/env python3
"""Capture engine decode attention metadata at mismatch step."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import torch

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = os.environ.get("MINICPM_SALA_PROMPT", "Hello, my name is")
TARGET_STEP = int(os.environ.get("MINICPM_SALA_MISMATCH_STEP", "14"))
TRACE = Path(__file__).resolve().parent / "traces" / "decode_meta_latest.json"

_DECODE_COUNT = 0
_TARGET = 0


def _install(model: torch.nn.Module) -> int:
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.models.minicpm_sala import (
        is_lightning_layer,
        is_sparse_layer,
    )

    model._dm: dict = {"steps": []}

    def _pre(_mod, args):
        global _DECODE_COUNT
        if len(args) < 2:
            return
        h = args[1]
        if not isinstance(h, torch.Tensor) or h.shape[0] != 1:
            return
        _DECODE_COUNT += 1
        if _DECODE_COUNT != _TARGET:
            return
        ctx = get_forward_context()
        md = ctx.attn_metadata
        if not isinstance(md, dict):
            return
        snap: dict = {"decode_idx": _DECODE_COUNT}
        for layer in model.model.layers:
            if is_sparse_layer(layer.mixer_type):
                key = layer.self_attn.attn.layer_name
            elif is_lightning_layer(layer.mixer_type):
                key = layer.self_attn.prefix
            else:
                continue
            if key not in md:
                continue
            meta = md[key]
            row = {"layer": layer.mixer_type}
            for attr in (
                "num_prefills",
                "num_decodes",
                "num_prefill_tokens",
                "num_decode_tokens",
                "dense_len",
                "num_actual_tokens",
                "max_query_len",
                "max_seq_len",
            ):
                if hasattr(meta, attr):
                    row[attr] = getattr(meta, attr)
            for attr in (
                "query_start_loc",
                "seq_lens",
                "slot_mapping",
                "state_indices_tensor",
                "block_table",
            ):
                if hasattr(meta, attr):
                    t = getattr(meta, attr)
                    if isinstance(t, torch.Tensor):
                        row[attr] = t.detach().cpu().tolist()
            snap[key] = row
        model._dm["steps"].append(snap)

    model._dm_hook = model.model.layers[0].register_forward_pre_hook(_pre)
    return 0


def _read(model: torch.nn.Module) -> dict:
    return dict(getattr(model, "_dm", {}))


def main() -> int:
    global _TARGET
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    _TARGET = TARGET_STEP
    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids = tok.encode(PROMPT, add_special_tokens=True)

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
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
    out = llm.generate(
        [TokensPrompt(prompt_token_ids=ids)],
        SamplingParams(temperature=0, max_tokens=TARGET_STEP + 1),
    )[0]
    dm = llm.apply_model(_read)[0]
    dm["prompt_len"] = len(ids)
    dm["target_step"] = TARGET_STEP
    dm["tokens"] = list(out.outputs[0].token_ids)
    TRACE.parent.mkdir(parents=True, exist_ok=True)
    TRACE.write_text(json.dumps(dm, indent=2), encoding="utf-8")
    print(json.dumps(dm, indent=2), flush=True)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
