#!/usr/bin/env python3
"""GPU Step 6: mixed dense/sparse batch invariance.

A packed varlen batch containing one sequence BELOW `dense_len` (dense
regime) and one AT/ABOVE it (sparse regime) must produce, per sequence,
the same output as running each sequence alone. This exercises
`_forward_mixed`'s per-sequence sub-batch extraction and scatter-back
against the REAL kernels (the CPU unit test
pr2/tests/v1/attention/test_minicpm_sala_forward_mixed.py covers the
same routing with mocks only).

Requires: sm_80+ GPU, infllm_v2 installed, PR2 overlay installed.
Run after steps 0-4 pass.
"""

import os
import sys
import tempfile

import torch


def _make_metadata(meta_cls, seq_lens, block_tables, dense_len, block_size, device):
    starts = [0]
    for n in seq_lens:
        starts.append(starts[-1] + n)
    return meta_cls(
        query_start_loc=torch.tensor(starts, device=device, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, device=device, dtype=torch.int32),
        block_table=block_tables,
        dense_len=dense_len,
        page_block_size=block_size,
    )


def main() -> int:
    assert torch.cuda.is_available(), "This script requires a real GPU."

    from vllm.v1.attention.backends.minicpm_sala_sparse import (
        INFLLM_V2_AVAILABLE,
        MiniCPMSALASparseAttentionImpl,
        MiniCPMSALASparseAttentionMetadata,
        parse_sparse_config,
    )

    if not INFLLM_V2_AVAILABLE:
        print("infllm_v2 not installed -- run scripts/install_infllm_v2.sh first.")
        return 1

    device = torch.device("cuda:0")
    from transformers import PretrainedConfig

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    fd, temp_file = tempfile.mkstemp()
    os.close(fd)
    os.unlink(temp_file)

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

    num_heads, num_kv_heads, head_size = 32, 2, 128
    block_size = 256
    dense_len = 8192

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

    torch.manual_seed(0)
    short_len = 512  # dense regime
    long_len = dense_len + 256  # sparse regime
    seq_lens = [short_len, long_len]
    total = sum(seq_lens)
    blocks_per_seq = [
        (n + block_size - 1) // block_size + 1 for n in seq_lens
    ]
    num_blocks = sum(blocks_per_seq) + 1

    def fresh_cache():
        return torch.zeros(
            num_blocks, 2, block_size, num_kv_heads, head_size,
            device=device, dtype=torch.bfloat16,
        )

    query = torch.randn(total, num_heads, head_size, device=device, dtype=torch.bfloat16)
    key = torch.randn(total, num_kv_heads, head_size, device=device, dtype=torch.bfloat16)
    value = torch.randn(total, num_kv_heads, head_size, device=device, dtype=torch.bfloat16)

    # Disjoint physical blocks per sequence, padded to equal width.
    max_blocks = max(blocks_per_seq)
    bt = torch.zeros(2, max_blocks, device=device, dtype=torch.int32)
    bt[0, : blocks_per_seq[0]] = torch.arange(blocks_per_seq[0], device=device)
    bt[1, : blocks_per_seq[1]] = torch.arange(blocks_per_seq[1], device=device) + blocks_per_seq[0]

    # --- Mixed batch ---
    mixed_meta = _make_metadata(
        MiniCPMSALASparseAttentionMetadata, seq_lens, bt, dense_len, block_size, device
    )
    mixed_out = torch.zeros_like(query)
    impl.forward(None, query, key, value, fresh_cache(), mixed_meta, mixed_out)
    torch.cuda.synchronize()

    # --- Each sequence alone (fresh caches, same physical block ids) ---
    solo_out = torch.zeros_like(query)
    start = 0
    for i, n in enumerate(seq_lens):
        end = start + n
        meta = _make_metadata(
            MiniCPMSALASparseAttentionMetadata,
            [n],
            bt[i : i + 1],
            dense_len,
            block_size,
            device,
        )
        out_i = torch.zeros(n, num_heads, head_size, device=device, dtype=torch.bfloat16)
        impl.forward(
            None, query[start:end], key[start:end], value[start:end],
            fresh_cache(), meta, out_i,
        )
        solo_out[start:end] = out_i
        start = end
    torch.cuda.synchronize()

    max_diff = (mixed_out.float() - solo_out.float()).abs().max().item()
    print(f"max |mixed - solo| = {max_diff:.6f}")
    # bf16 kernels; identical math up to reduction-order noise.
    tol = 2e-2
    if not torch.isfinite(mixed_out.float()).all():
        print("STEP 6 FAIL: non-finite values in mixed-batch output", file=sys.stderr)
        return 1
    if max_diff > tol:
        print(
            f"STEP 6 FAIL: mixed-batch output diverges from per-sequence "
            f"runs (max diff {max_diff} > tol {tol})",
            file=sys.stderr,
        )
        return 1
    print("STEP 6 PASS: mixed dense/sparse batch matches per-sequence runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
