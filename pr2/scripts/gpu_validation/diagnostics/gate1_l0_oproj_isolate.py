#!/usr/bin/env python3
"""Isolate o_proj: weight parity + manual matmul on shared gated input."""

from __future__ import annotations

import gc
import os
import sys

import torch
import torch.nn.functional as F

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = os.environ.get("MINICPM_SALA_PROMPT", "Briefly explain gravity:")


def main() -> int:
    sys.path.insert(0, os.path.dirname(__file__))
    from gate1_l0_sparse_bisect import _patch_hf, hf_l0_traces, vllm_l0_traces

    _patch_hf()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids = tok.encode(PROMPT, add_special_tokens=True)
    hf = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    with torch.no_grad():
        t1 = int(
            hf(
                torch.tensor([ids], device="cuda"),
                attention_mask=torch.ones(1, len(ids), device="cuda"),
            )
            .logits[0, -1]
            .argmax()
        )
        hf_sa = hf.model.layers[0].self_attn
        hf_w = hf_sa.o_proj.weight.float()
        hf_b = hf_sa.o_proj.bias.float() if hf_sa.o_proj.bias is not None else None
    del hf
    gc.collect()
    torch.cuda.empty_cache()

    ids2 = ids + [t1]
    print(f"prompt={PROMPT!r} seqlen={len(ids2)}", flush=True)

    hf_t = hf_l0_traces(ids2)
    vv_t = vllm_l0_traces(ids2)

    gated_hf = hf_t["o_proj_in"]
    gated_vv = vv_t["o_proj_in"]
    d_in = (gated_hf - gated_vv).abs().max().item()
    print(f"gated_input_hf_vs_vllm peak={d_in:.6g}", flush=True)

    # vLLM weight from traces path reload
    import tempfile
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
    from vllm.model_executor.model_loader import get_model_loader

    model_config = ModelConfig(
        model=WEIGHTS, trust_remote_code=True, dtype="bfloat16", max_model_len=4096
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        load_config=LoadConfig(),
        cache_config=CacheConfig(block_size=256),
        device_config=DeviceConfig(device="cuda"),
    )
    fd, temp = tempfile.mkstemp()
    os.close(fd)
    try:
        with vconfig.set_current_vllm_config(vllm_config, check_compile=False):
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{temp}",
                local_rank=0,
                backend="nccl",
            )
            initialize_model_parallel(1, 1)
            model = get_model_loader(LoadConfig()).load_model(
                vllm_config=vllm_config, model_config=model_config
            )
            vv_w = model.model.layers[0].self_attn.o_proj.weight.float().cpu()
            vv_b = model.model.layers[0].self_attn.o_proj.bias
            vv_b = vv_b.float().cpu() if vv_b is not None else None
            destroy_model_parallel()
            destroy_distributed_environment()
    finally:
        os.unlink(temp)

    w_diff = (hf_w.cpu() - vv_w).abs().max().item()
    print(f"o_proj_weight_hf_vs_vllm peak={w_diff:.6g}", flush=True)

    gated = gated_vv.to(torch.float32)
    manual_hf_w = F.linear(gated, hf_w.cpu(), hf_b.cpu() if hf_b is not None else None)
    manual_vv_w = F.linear(gated, vv_w, vv_b)
    d_hf_w_on_v_gated = (manual_hf_w - hf_t["o_proj_out"].float()).abs().max().item()
    d_vv_w_on_v_gated = (manual_vv_w - vv_t["o_proj_out"].float()).abs().max().item()
    d_cross = (manual_hf_w - vv_t["o_proj_out"].float()).abs().max().item()
    print(f"manual_hf_weight_on_vllm_gated vs hf_out peak={d_hf_w_on_v_gated:.6g}", flush=True)
    print(f"manual_vllm_weight_on_vllm_gated vs vllm_out peak={d_vv_w_on_v_gated:.6g}", flush=True)
    print(f"manual_hf_weight_on_vllm_gated vs vllm_out peak={d_cross:.6g}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
