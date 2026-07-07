#!/usr/bin/env python3
"""Gate 1: audit HF checkpoint keys vs vLLM model parameters."""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)


def _checkpoint_keys(weights_dir: str) -> set[str]:
    index_path = Path(weights_dir) / "model.safetensors.index.json"
    keys: set[str] = set()
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map", {})
        for ckpt_name in sorted(set(weight_map.values())):
            with safe_open(str(Path(weights_dir) / ckpt_name), framework="pt") as f:
                keys.update(f.keys())
        return keys
    for shard in sorted(Path(weights_dir).glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            keys.update(f.keys())
    return keys


def _vllm_param_keys() -> set[str]:
    from vllm import LLM

    llm = LLM(
        model=WEIGHTS,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=2048,
        block_size=256,
        gpu_memory_utilization=0.45,
        enforce_eager=True,
    )
    runner = llm.llm_engine.engine_core
    # v1 engine: pull model from executor via config path
    from vllm.model_executor.model_loader import get_model

    model_config = llm.llm_engine.model_config
    model = get_model(model_config=model_config)
    return {n for n, _ in model.named_parameters()}


def main() -> int:
    if not Path(WEIGHTS).is_dir():
        print(f"FAIL: weights not found at {WEIGHTS}", flush=True)
        return 1

    ckpt = _checkpoint_keys(WEIGHTS)
    print(f"checkpoint keys: {len(ckpt)}", flush=True)

    # Load vLLM model parameters without full engine to avoid OOM/time.
    from transformers import AutoConfig

    from vllm.config import ModelConfig, VllmConfig
    from vllm.model_executor.model_loader import get_model_loader

    hf_config = AutoConfig.from_pretrained(WEIGHTS, trust_remote_code=True)
    model_config = ModelConfig(
        model=WEIGHTS,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=2048,
    )
    vllm_config = VllmConfig(model_config=model_config)
    loader = get_model_loader(model_config)
    model = loader.load_model(vllm_config=vllm_config)
    params = {n for n, _ in model.named_parameters()}
    print(f"vllm params: {len(params)}", flush=True)

    def norm(k: str) -> str:
        return k.replace("model.", "", 1) if k.startswith("model.") else k

    ckpt_norm = {norm(k) for k in ckpt}
    missing_in_model = sorted(k for k in ckpt_norm if k not in params)
    extra_in_model = sorted(k for k in params if k not in ckpt_norm)

    print(f"missing_in_vllm ({len(missing_in_model)}):", flush=True)
    for k in missing_in_model[:40]:
        print(f"  {k}", flush=True)
    if len(missing_in_model) > 40:
        print(f"  ... +{len(missing_in_model) - 40} more", flush=True)

    print(f"extra_in_vllm ({len(extra_in_model)}):", flush=True)
    for k in extra_in_model[:40]:
        print(f"  {k}", flush=True)

    # High-signal patterns
    patterns = ("o_gate", "z_proj", "qkv_proj", "lm_head", "embed_tokens", "slope")
    by_pat: dict[str, list[str]] = defaultdict(list)
    for k in missing_in_model:
        for p in patterns:
            if p in k:
                by_pat[p].append(k)
    print("missing by pattern:", flush=True)
    for p, ks in sorted(by_pat.items()):
        print(f"  {p}: {len(ks)}", flush=True)
        for k in ks[:5]:
            print(f"    {k}", flush=True)

    ok = len(missing_in_model) == 0
    print(f"WEIGHT_AUDIT pass={ok}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
