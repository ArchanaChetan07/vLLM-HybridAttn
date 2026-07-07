#!/usr/bin/env python3
"""Bisect HF vs vLLM hidden states for prompt and prompt+first-token sequences."""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import tempfile

import torch

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = "Hello, my name is"


def _patch_hf() -> None:
    script = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "remote",
        "patch_hf_transformers_compat.py",
    )
    script = os.path.normpath(script)
    if os.path.isfile(script):
        subprocess.run([sys.executable, script], check=False)


def hf_trace(ids: torch.Tensor) -> dict[str, torch.Tensor]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    traces: dict[str, torch.Tensor] = {}
    hooks = []

    def hook(name):
        def _fn(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            traces[name] = h[0, -1].detach().float().cpu()

        return _fn

    hooks.append(model.model.layers[0].register_forward_hook(hook("layer0")))
    hooks.append(model.model.layers[1].register_forward_hook(hook("layer1")))
    with torch.no_grad():
        attn = torch.ones_like(ids)
        logits = model(input_ids=ids, attention_mask=attn).logits
        traces["logits"] = logits[0, -1].float().cpu()
        traces["greedy"] = logits[0, -1].argmax().cpu()
    for h in hooks:
        h.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return traces


def vllm_trace(ids: torch.Tensor) -> dict[str, torch.Tensor]:
    import vllm.config as vconfig
    from vllm.config import CacheConfig, ModelConfig, VllmConfig
    from vllm.config.device import DeviceConfig
    from vllm.config.load import LoadConfig
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.model_loader import get_model_loader

    traces: dict[str, torch.Tensor] = {}
    hooks = []

    def hook(name):
        def _fn(_mod, _inp, out):
            h = out if isinstance(out, torch.Tensor) else out[0]
            traces[name] = h[-1].detach().float().cpu()

        return _fn

    model_config = ModelConfig(
        model=WEIGHTS,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
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
            model = get_model_loader(load_config).load_model(
                vllm_config=vllm_config, model_config=model_config
            )
            model.eval().cuda()
            hooks.append(model.model.layers[0].register_forward_hook(hook("layer0")))
            hooks.append(model.model.layers[1].register_forward_hook(hook("layer1")))
            positions = torch.arange(ids.shape[1], device="cuda", dtype=torch.long)
            with torch.no_grad():
                with set_forward_context(None, vllm_config):
                    hidden = model.model(ids, positions)
                    logits = model.compute_logits(hidden)
                    traces["logits"] = logits[0, -1].float().cpu()
                    traces["greedy"] = logits[0, -1].argmax().cpu()
            for h in hooks:
                h.remove()
            destroy_model_parallel()
            destroy_distributed_environment()
    finally:
        import contextlib

        with contextlib.suppress(OSError):
            os.unlink(temp_file)
    gc.collect()
    torch.cuda.empty_cache()
    return traces


def _diff(hf: dict, vv: dict, label: str) -> None:
    print(f"--- {label} ---", flush=True)
    for key in ("layer0", "layer1", "logits"):
        if key not in hf or key not in vv:
            continue
        d = (hf[key] - vv[key]).abs()
        print(
            f"{key} max_abs={d.max().item():.6g} "
            f"hf_greedy={int(hf['greedy'])} v_greedy={int(vv['greedy'])}"
        )
    top_hf = torch.topk(hf["logits"], 5)
    top_v = torch.topk(vv["logits"], 5)
    print(f"hf top5: {list(zip(top_hf.indices.tolist(), top_hf.values.tolist()))}")
    print(f"v  top5: {list(zip(top_v.indices.tolist(), top_v.values.tolist()))}")


def main() -> int:
    _patch_hf()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids1 = tok.encode(PROMPT, return_tensors="pt").to("cuda")
    hf1 = hf_trace(ids1)
    v1 = vllm_trace(ids1)
    _diff(hf1, v1, "prompt only")

    t1 = int(hf1["greedy"])
    ids2 = torch.cat(
        [ids1, torch.tensor([[t1]], device=ids1.device, dtype=ids1.dtype)], dim=1
    )
    hf2 = hf_trace(ids2)
    v2 = vllm_trace(ids2)
    _diff(hf2, v2, f"prompt + t1={t1}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
