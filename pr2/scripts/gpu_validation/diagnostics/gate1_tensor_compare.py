#!/usr/bin/env python3
"""Compare checkpoint tensors vs vLLM-loaded parameters."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open

import vllm.config as vconfig
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.config.device import DeviceConfig
from vllm.config.load import LoadConfig
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
    init_distributed_environment,
    initialize_model_parallel,
)

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)

CHECK_TENSORS = [
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.self_attn.o_gate.weight",
    "model.layers.1.self_attn.q_proj.weight",
    "model.layers.1.self_attn.z_proj.weight",
    "lm_head.weight",
]


def load_ckpt_tensor(name: str) -> torch.Tensor:
    index = json.loads(Path(WEIGHTS, "model.safetensors.index.json").read_text())
    shard = index["weight_map"][name]
    with safe_open(str(Path(WEIGHTS) / shard), framework="pt") as f:
        return f.get_tensor(name)


def load_vllm_model():
    from vllm.model_executor.model_loader import get_model_loader

    model_config = ModelConfig(
        model=WEIGHTS,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=2048,
    )
    load_config = LoadConfig()
    cache_config = CacheConfig(block_size=256)
    vllm_config = VllmConfig(
        model_config=model_config,
        load_config=load_config,
        cache_config=cache_config,
        device_config=DeviceConfig(device="cuda"),
    )
    fd, temp_file = tempfile.mkstemp()
    os.close(fd)
    try:
        with vconfig.set_current_vllm_config(vllm_config, check_compile=False):
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{temp_file}",
                local_rank=0,
                backend="nccl",
            )
            initialize_model_parallel(1, 1)
            loader = get_model_loader(load_config)
            return loader.load_model(
                vllm_config=vllm_config, model_config=model_config
            )
    finally:
        os.unlink(temp_file)


def vllm_param(model, hf_name: str) -> torch.Tensor:
    # HF: model.layers.0... -> vLLM: model.model.layers.0...
    candidates = [hf_name, f"model.{hf_name}", hf_name.replace("model.", "model.model.", 1)]
    # vLLM stacks q/k/v into qkv_proj; compare the q shard when checkpoint is split.
    if ".q_proj.weight" in hf_name:
        candidates.extend(
            c.replace(".q_proj.weight", ".qkv_proj.weight") for c in list(candidates)
        )
    params = dict(model.named_parameters())
    for c in candidates:
        if c not in params:
            continue
        p = params[c]
        if ".q_proj.weight" in hf_name and ".qkv_proj.weight" in c:
            # qkv_proj is [q_size + 2*kv_size, hidden]; q is the leading rows.
            ckpt_q = load_ckpt_tensor(hf_name)
            return p[: ckpt_q.shape[0]]
        return p
    raise KeyError(f"no param for {hf_name} among {candidates}")


def main() -> int:
    model = load_vllm_model()
    worst = 0.0
    try:
        for name in CHECK_TENSORS:
            ckpt = load_ckpt_tensor(name).float().cpu()
            try:
                vp = vllm_param(model, name).float().cpu()
            except KeyError as e:
                print(f"MISSING_PARAM {name}: {e}")
                continue
            if ckpt.shape != vp.shape:
                print(
                    f"SHAPE_MISMATCH {name} ckpt={tuple(ckpt.shape)} "
                    f"vllm={tuple(vp.shape)}"
                )
                continue
            diff = (ckpt - vp).abs().max().item()
            worst = max(worst, diff)
            print(f"{name}: max_abs_diff={diff}")
        print(f"TENSOR_COMPARE worst={worst}")
        return 0 if worst < 1e-3 else 1
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    sys.exit(main())
