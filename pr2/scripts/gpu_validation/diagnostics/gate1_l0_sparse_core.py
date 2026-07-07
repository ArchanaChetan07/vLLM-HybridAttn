#!/usr/bin/env python3
"""sparse_core vs HF + in-model o_proj replay (Briefly seqlen=7)."""

from __future__ import annotations

import contextlib
import gc
import os
import sys
import tempfile

import torch
import torch.nn.functional as F

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = os.environ.get("MINICPM_SALA_PROMPT", "Briefly explain gravity:")


def _pos_peak(a: torch.Tensor, b: torch.Tensor) -> tuple[float, list[float]]:
    d = (a.float() - b.float()).abs()
    if d.dim() == 1:
        diffs = [d.max().item()]
    else:
        diffs = [d[i].max().item() for i in range(d.shape[0])]
    return max(diffs), diffs


def main() -> int:
    sys.path.insert(0, os.path.dirname(__file__))
    from gate1_l0_sparse_bisect import (
        _patch_hf,
        _print_stage,
        hf_l0_traces,
        vllm_l0_traces,
    )

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
    del hf
    gc.collect()
    torch.cuda.empty_cache()

    ids2 = ids + [t1]
    print(f"prompt={PROMPT!r} t1={t1} seqlen={len(ids2)}", flush=True)

    hf_t = hf_l0_traces(ids2)
    vv_t = vllm_l0_traces(ids2)

    for stage in (
        "flash_raw",
        "sparse_core",
        "gated",
        "o_proj_out",
        "attn_branch",
        "layer0",
    ):
        if stage in hf_t and stage in vv_t:
            _print_stage(stage, hf_t[stage], vv_t[stage])

    # In-model o_proj replay on GPU (vLLM weights + gated from traces).
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
    from vllm.model_executor.models.minicpm_sala import _dense_o_proj

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
            sa = model.model.layers[0].self_attn
            g = vv_t["o_proj_in"].cuda().bfloat16()
            with torch.no_grad():
                mod_out = _dense_o_proj(sa.o_proj, g)
                bias = sa.o_proj.bias
                lin_out = F.linear(
                    g.float(),
                    sa.o_proj.weight.float(),
                    bias.float() if bias is not None else None,
                ).to(g.dtype)
            destroy_model_parallel()
            destroy_distributed_environment()
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp)

    d_mod = (mod_out.float().cpu() - vv_t["o_proj_out"].float()).abs().max().item()
    d_lin = (lin_out.float().cpu() - vv_t["o_proj_out"].float()).abs().max().item()
    d_mod_lin = (mod_out.float().cpu() - lin_out.float().cpu()).abs().max().item()
    print(f"gpu_o_proj_module vs trace_o_proj_out peak={d_mod:.6g}", flush=True)
    print(f"gpu_F_linear vs trace_o_proj_out peak={d_lin:.6g}", flush=True)
    print(f"gpu_o_proj_module vs gpu_F_linear peak={d_mod_lin:.6g}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
