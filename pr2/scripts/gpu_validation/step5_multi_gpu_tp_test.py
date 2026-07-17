#!/usr/bin/env python3
"""GPU Step 5: real multi-GPU TP test, under `nccl`.

Extends `validation/stage2_tp2_sharding_test.py` (which used `gloo` on
CPU, confirming the decay-slope sharding ARITHMETIC is correct) with a
real multi-GPU run under vLLM's actual production backend (`nccl`).
This does NOT re-derive new test logic -- it reuses the exact same
verification (each rank's `tp_slope` shard matches an independently
computed reference, and concatenating all ranks' shards exactly
reconstructs the full array) against real GPU hardware and the real
distributed backend.

Requires >= 2 real GPUs on one node. Does NOT require Ampere+ or
infllm_v2 -- this tests Lightning Attention's TP sharding, which is
pure tensor indexing, not kernel dispatch.

Usage: torchrun --nproc_per_node=2 step5_multi_gpu_tp_test.py
(NOT plain `python3` -- this needs torchrun's real multi-process launch,
confirmed against how vLLM's own distributed tests are actually run,
not the mp.spawn approach the earlier CPU-only test used, since that
was sufficient for a single-node CPU gloo test but real multi-GPU nccl
setups are conventionally torchrun-launched.)
"""

import os
import sys

import torch
from transformers import PretrainedConfig

REAL_LIGHTNING_CONFIG = {
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "intermediate_size": 16384,
    "vocab_size": 73448,
    "rms_norm_eps": 1e-06,
    "attention_bias": False,
    "lightning_nh": 32,
    "lightning_nkv": 32,
    "lightning_head_dim": 128,
    "lightning_scale": "1/sqrt(d)",
    "lightning_use_rope": True,
    "qk_norm": True,
    "use_output_norm": True,
    "use_output_gate": True,
    "max_position_embeddings": 524288,
    "rope_theta": 10000.0,
    "rope_scaling": None,
}


def main() -> int:
    assert torch.cuda.is_available(), "This script requires real GPUs."
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size < 2:
        print(
            "WORLD_SIZE < 2 -- this script must be launched with "
            "torchrun --nproc_per_node=<N>, N >= 2. Running under plain "
            "python3 (world_size=1) does not exercise real multi-GPU "
            "sharding -- that's already covered by the CPU gloo TP=2 "
            "test. Aborting rather than silently passing a "
            "non-representative single-process run."
        )
        return 1

    torch.cuda.set_device(local_rank)

    import vllm.config as vconfig
    from vllm.config import CacheConfig, VllmConfig
    from vllm.config.device import DeviceConfig
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    hf_config = PretrainedConfig(**REAL_LIGHTNING_CONFIG)
    cache_config = CacheConfig()
    vllm_config = VllmConfig(
        cache_config=cache_config, device_config=DeviceConfig(device="cuda")
    )

    with vconfig.set_current_vllm_config(vllm_config, check_compile=False):
        # Real nccl backend, real torchrun-provided env vars -- NOT the
        # file://-init-method + gloo pattern the CPU-only test used
        # (that pattern is vLLM's own single-node CPU-test convention;
        # this is the real multi-GPU convention, env://-based, matching
        # how torchrun sets MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE).
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method="env://",
            local_rank=local_rank,
            backend="nccl",
        )
        initialize_model_parallel(tensor_model_parallel_size=world_size)

        from vllm.model_executor.models.minicpm_sala import (
            MiniCPMSALALightningAttention,
            build_lightning_decay_rate,
        )

        if rank == 0:
            print(
                f"world_size={world_size}, real nccl backend, "
                f"{torch.cuda.get_device_name(local_rank)}"
            )

        layer = MiniCPMSALALightningAttention(
            config=hf_config,
            cache_config=cache_config,
            quant_config=None,
            prefix="model.layers.1.self_attn",
        ).to(device=f"cuda:{local_rank}")

        tp_heads = REAL_LIGHTNING_CONFIG["num_attention_heads"] // world_size
        # POSITIVE decay rate -- the single source of truth is
        # build_lightning_decay_rate (the kernels apply exp(-rate*d)
        # internally). An earlier revision of this test expected the
        # NEGATED slope (* -1.0), i.e. the exact sign bug the decay-sign
        # fix removed, and therefore failed against correct layers.
        full_reference = build_lightning_decay_rate(
            REAL_LIGHTNING_CONFIG["num_attention_heads"]
        )
        expected_shard = full_reference[rank * tp_heads : (rank + 1) * tp_heads].to(
            layer.tp_slope.device
        )

        matches = torch.allclose(layer.tp_slope, expected_shard, atol=1e-9)
        print(
            f"rank {rank} (GPU {local_rank}): tp_heads={tp_heads}, "
            f"tp_slope.shape={tuple(layer.tp_slope.shape)}, "
            f"matches expected shard: {matches}"
        )
        assert matches, f"rank {rank}: real multi-GPU sharding MISMATCH"

        # Gather all ranks' shards to rank 0 and verify reconstruction,
        # same check as the CPU test but now across real GPU processes
        # and real nccl collectives.
        gathered = [torch.zeros_like(layer.tp_slope) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, layer.tp_slope)

        if rank == 0:
            reconstructed = torch.cat(gathered)
            full_ref_check = build_lightning_decay_rate(
                REAL_LIGHTNING_CONFIG["num_attention_heads"]
            ).to(reconstructed.device)
            reconstruction_matches = torch.allclose(
                reconstructed, full_ref_check, atol=1e-9
            )
            print(
                f"\nReconstructed full array via real nccl all_gather "
                f"matches from-scratch computation: {reconstruction_matches}"
            )
            assert reconstruction_matches, (
                "real multi-GPU reconstruction mismatch -- sharding bug "
                "under real nccl, not just CPU gloo"
            )
            print(
                "\nPASS: real multi-GPU (nccl) TP sharding confirmed "
                "correct, extending the CPU gloo result to real hardware."
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
