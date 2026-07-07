#!/usr/bin/env python3
"""Gate 1: bisect HF vs vLLM hidden states after embed and selected layers."""

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
    script = "/workspace/hybridattn/scripts/remote/patch_hf_transformers_compat.py"
    if os.path.isfile(script):
        subprocess.run([sys.executable, script], check=False)


def hf_hidden_trace() -> dict[str, torch.Tensor]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids = tok.encode(PROMPT, return_tensors="pt").to("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    traces: dict[str, torch.Tensor] = {}

    def hook(name):
        def _fn(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            traces[name] = h[0, -1].detach().float().cpu()

        return _fn

    model.model.layers[0].register_forward_hook(hook("layer0"))
    model.model.layers[1].register_forward_hook(hook("layer1"))
    with torch.no_grad():
        emb = model.model.embed_tokens(ids) * model.config.scale_emb
        traces["embed"] = emb[0, -1].float().cpu()
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
        traces["logits"] = logits[0, -1].float().cpu()
        traces["greedy"] = logits[0, -1].argmax().cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return traces


def vllm_hidden_trace() -> dict[str, torch.Tensor]:
    import vllm.config as vconfig
    from transformers import AutoTokenizer

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
            model.model.layers[0].register_forward_hook(hook("layer0"))
            model.model.layers[1].register_forward_hook(hook("layer1"))

            tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
            ids = tok.encode(PROMPT, return_tensors="pt").to("cuda")
            positions = torch.arange(ids.shape[1], device="cuda", dtype=torch.long)
            with torch.no_grad():
                with set_forward_context(None, vllm_config):
                    emb = model.embed_input_ids(ids)
                    traces["embed"] = emb[0, -1].float().cpu()
                    hidden = model.model(ids, positions)
                    logits = model.compute_logits(hidden)
                    traces["logits"] = logits[0, -1].float().cpu()
                    traces["greedy"] = logits[0, -1].argmax().cpu()
            destroy_model_parallel()
            destroy_distributed_environment()
    finally:
        os.unlink(temp_file)
    return traces


def main() -> int:
    _patch_hf()
    hf = hf_hidden_trace()
    vllm = vllm_hidden_trace()
    for key in ("embed", "layer0", "layer1", "logits"):
        if key in hf and key in vllm:
            diff = (hf[key] - vllm[key]).abs().max().item()
            print(f"{key} max_abs_diff={diff}")
    print(f"HF greedy={int(hf['greedy'])} vLLM greedy={int(vllm['greedy'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
