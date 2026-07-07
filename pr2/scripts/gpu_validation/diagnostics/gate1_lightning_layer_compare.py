#!/usr/bin/env python3
"""Compare HF LightningAttention vs vLLM layer-1 output on real weights."""

from __future__ import annotations

import contextlib
import gc
import math
import os
import subprocess
import sys
import tempfile

import torch
from einops import rearrange

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = "Hello, my name is"


def _patch_hf() -> None:
    script = "/workspace/hybridattn/scripts/remote/patch_hf_transformers_compat.py"
    if os.path.isfile(script):
        subprocess.run([sys.executable, script], check=False)


def hf_hidden_after_layer0() -> torch.Tensor:
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
    with torch.no_grad():
        emb = model.model.embed_tokens(ids) * model.config.scale_emb
        pos = torch.arange(ids.shape[1], device="cuda").unsqueeze(0)
        mask = torch.ones_like(ids)
        h = emb
        out = model.model.layers[0](
            h, attention_mask=mask, position_ids=pos, use_cache=False
        )
        h1 = out[0] if isinstance(out, tuple) else out
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return h1


def hf_layer1_out(hidden: torch.Tensor) -> torch.Tensor:
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
    pos = torch.arange(ids.shape[1], device="cuda").unsqueeze(0)
    mask = torch.ones_like(ids)
    with torch.no_grad():
        out = model.model.layers[1](
            hidden, attention_mask=mask, position_ids=pos, use_cache=False
        )
        y = out[0] if isinstance(out, tuple) else out
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return y


def vllm_layer1_out(hidden: torch.Tensor) -> torch.Tensor:
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
    from vllm.v1.attention.backends.linear_attn import LinearAttentionMetadata

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids = tok.encode(PROMPT, return_tensors="pt").to("cuda")
    seq_len = ids.shape[1]
    positions = torch.arange(seq_len, device="cuda", dtype=torch.long)

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
    out_tensor = torch.zeros_like(hidden)
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
            layer1 = model.model.layers[1]
            attn = layer1.self_attn
            prefix = attn.prefix
            state_shape = attn.get_state_shape()
            attn.kv_cache = (
                torch.zeros(
                    1, *state_shape[0], device="cuda", dtype=attn.get_state_dtype()[0]
                ),
            )
            ln_meta = LinearAttentionMetadata(
                num_prefills=1,
                num_prefill_tokens=seq_len,
                num_decodes=0,
                num_decode_tokens=0,
                query_start_loc=torch.tensor(
                    [0, seq_len], device="cuda", dtype=torch.int32
                ),
                seq_lens=torch.tensor([seq_len], device="cuda", dtype=torch.int32),
                state_indices_tensor=torch.tensor(
                    [0], device="cuda", dtype=torch.int32
                ),
            )
            residual = hidden
            h_norm = layer1.input_layernorm(hidden)
            attn_out = torch.zeros_like(h_norm)
            with set_forward_context(
                attn_metadata={prefix: ln_meta}, vllm_config=vllm_config
            ):
                attn.forward(
                    hidden_states=h_norm,
                    output=attn_out,
                    positions=positions,
                )
            mlp_in = layer1._add_scaled_residual(residual, attn_out)
            residual2 = mlp_in
            h2 = layer1.post_attention_layernorm(mlp_in)
            h2 = layer1.mlp(h2)
            out_tensor = layer1._add_scaled_residual(residual2, h2)
            destroy_model_parallel()
            destroy_distributed_environment()
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp_file)
    return out_tensor


def main() -> int:
    _patch_hf()
    h0 = hf_hidden_after_layer0()
    hf_y = hf_layer1_out(h0)
    v_y = vllm_layer1_out(h0)
    diff = (hf_y.float() - v_y.float()).abs()
    print(f"hidden_in shape={tuple(h0.shape)}")
    print(f"layer1 max_abs_diff={diff.max().item():.6g}")
    print(f"layer1 mean_abs_diff={diff.mean().item():.6g}")
    print(f"HF last-token norm={hf_y[0,-1].float().norm().item():.6g}")
    print(f"vLLM last-token norm={v_y[0,-1].float().norm().item():.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
