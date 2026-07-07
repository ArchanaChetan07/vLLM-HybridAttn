#!/usr/bin/env python3
"""GPU Step 4: end-to-end sparse attention path test, past `dense_len`.

Requires: real GPU, `infllm_v2` actually installed (confirmed possible
per docs/minicpm_sala_known_limitations.md's real-hardware report --
run `patches/fix_cutlass_submodule.sh` first if not yet built), and
Ampere+ (sm_80+) hardware for the real kernel dispatch inside
`infllmv2_attn_with_kvcache` -- confirmed via real testing that
Turing-class GPUs (T1000, sm_75) fail here with a real, unpatchable
hardware-floor error.

This is the FIRST script in this project to exercise the full chain:
CompressK -> compressed_attention -> infllmv2_attn_with_kvcache with a
real (non-None) topk_idx. Everything before this step has tested these
pieces individually or with the dense (topk_idx=None) fallback only.

Run after GPU steps 1-3 pass.
"""

import os
import sys
import tempfile

import torch


def main() -> int:
    assert torch.cuda.is_available(), "This script requires a real GPU."

    from vllm.v1.attention.backends.minicpm_sala_sparse import (
        INFLLM_V2_AVAILABLE,
        MiniCPMSALASparseAttentionImpl,
    )

    if not INFLLM_V2_AVAILABLE:
        print(
            "INFLLM_V2_AVAILABLE is False -- infllm_v2 is not installed "
            "in this environment. Run patches/fix_cutlass_submodule.sh "
            "against a real clone of github.com/OpenBMB/infllmv2_cuda_impl "
            "first (see docs/minicpm_sala_known_limitations.md for the "
            "confirmed-working procedure), then re-run this script."
        )
        return 1
    print("INFLLM_V2_AVAILABLE: True -- proceeding.")

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    compute_capability = props.major * 10 + props.minor
    print(f"Device: {props.name}, compute capability sm_{compute_capability}")
    if compute_capability < 80:
        print(
            f"\nWARNING: compute capability sm_{compute_capability} < 80. "
            f"Per this project's confirmed real-hardware finding, the "
            f"kernel dispatch below is EXPECTED to fail with "
            f"'Flash attention currently only supported for compute "
            f"capability >= 80' -- this is a real, unpatchable hardware "
            f"floor, not a bug. Proceeding anyway so the exact failure "
            f"is captured for the record, not to suggest it will pass."
        )

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    fd, temp_file = tempfile.mkstemp()
    os.close(fd)
    os.unlink(temp_file)

    try:
        vllm_config = VllmConfig()
        with set_current_vllm_config(vllm_config, check_compile=False):
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{temp_file}",
                local_rank=0,
                backend="nccl",
            )
            initialize_model_parallel(1, 1)

        # Real checkpoint values (Phase 1 report): 2 KV heads, head_dim
        # 128, block_size must be multiple of 256.
        num_heads, num_kv_heads, head_size = 32, 2, 128
        block_size = 256
        dense_len = 8192  # real checkpoint sparse_config.dense_len

        from vllm.v1.attention.backends.minicpm_sala_sparse import (
            MiniCPMSALASparseAttentionMetadata,
            parse_sparse_config,
        )
        from transformers import PretrainedConfig

        sparse_config = parse_sparse_config(
            PretrainedConfig(
                sparse_config={
                    "kernel_size": 32,
                    "kernel_stride": 16,
                    "init_blocks": 1,
                    "block_size": 64,
                    "window_size": 2048,
                    "topk": 64,
                    "dense_len": dense_len,
                }
            )
        )

        impl = MiniCPMSALASparseAttentionImpl(
            num_heads=num_heads,
            head_size=head_size,
            scale=head_size**-0.5,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            sparse_config=sparse_config,
        )
        print(
            "MiniCPMSALASparseAttentionImpl constructed (stateless Impl, no parameters)"
        )

        # A sequence PAST dense_len -- the actual point of this test.
        # Kept as small as correctness testing allows (dense_len + a
        # small margin, not a huge multiple) to keep this script fast;
        # a realistic long-context benchmark run is a separate concern
        # (see docs/minicpm_sala_benchmark_plan.md).
        seq_len = dense_len + 256
        num_blocks_needed = (seq_len + block_size - 1) // block_size + 1

        print(
            f"Constructing a sequence of length {seq_len} "
            f"(dense_len={dense_len} + 256) -- this MUST trigger the "
            f"sparse regime, not silently fall back to dense."
        )

        kv_cache = torch.zeros(
            num_blocks_needed,
            2,
            block_size,
            num_kv_heads,
            head_size,
            device=device,
            dtype=torch.bfloat16,
        )
        query = torch.randn(
            seq_len, num_heads, head_size, device=device, dtype=torch.bfloat16
        )
        key = torch.randn(
            seq_len, num_kv_heads, head_size, device=device, dtype=torch.bfloat16
        )
        value = torch.randn(
            seq_len, num_kv_heads, head_size, device=device, dtype=torch.bfloat16
        )
        output = torch.zeros_like(query)

        block_table = torch.arange(
            num_blocks_needed, device=device, dtype=torch.int32
        ).unsqueeze(0)

        attn_metadata = MiniCPMSALASparseAttentionMetadata(
            query_start_loc=torch.tensor(
                [0, seq_len], device=device, dtype=torch.int32
            ),
            seq_lens=torch.tensor([seq_len], device=device, dtype=torch.int32),
            block_table=block_table,
            dense_len=dense_len,
            page_block_size=block_size,
        )

        print(
            "\nCalling forward() -- this should dispatch to "
            "_forward_sparse() and exercise CompressK -> "
            "compressed_attention -> infllmv2_attn_with_kvcache with "
            "a REAL (non-None) topk_idx for the first time in this "
            "project's history ..."
        )

        result = impl.forward(
            layer=None,  # AttentionLayer protocol unused by this Impl's forward
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
        )
        torch.cuda.synchronize()

        print("\nforward() returned without raising.")
        print(f"output.shape={tuple(result.shape)}, dtype={result.dtype}")
        assert not torch.isnan(result).any(), "NaN in sparse-path output"
        assert not torch.isinf(result).any(), "Inf in sparse-path output"
        assert result.abs().sum().item() > 0, (
            "output is all zeros -- the sparse kernel path likely did "
            "not actually execute (check whether the dense_len dispatch "
            "in forward() correctly routed to _forward_sparse for this "
            "seq_len)"
        )
        print(
            "\nPASS: sparse regime executed end-to-end, output is "
            "finite and non-trivial. This does NOT yet confirm "
            "NUMERICAL correctness against the reference model -- "
            "only that the pipeline runs without error and produces "
            "plausible-looking output. See "
            "tests/models/language/generation/test_minicpm_sala.py "
            "for the actual HF-comparison test, which still needs "
            "real weights to run."
        )
        return 0

    except Exception as e:
        print(f"\nFAILED: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        if "compute capability" in str(e):
            print(
                "\nThis matches the known, confirmed sm_80+ hardware "
                "floor -- expected on Turing-class GPUs, not a new bug. "
                "Re-run on Ampere+ hardware."
            )
        return 1
    finally:
        import contextlib

        with contextlib.suppress(OSError):
            os.unlink(temp_file)


if __name__ == "__main__":
    sys.exit(main())
