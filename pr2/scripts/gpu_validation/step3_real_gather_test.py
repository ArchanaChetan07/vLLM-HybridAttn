#!/usr/bin/env python3
"""GPU Step 3: real paged-cache gather test.

Flagged since `_gather_full_k_with_new_tokens` was first written as the
single highest-risk function in this project -- the CPU test
(tests/v1/attention/test_minicpm_sala_gather.py) uses a small,
hand-built synthetic cache. This script constructs a REAL
GPU-allocated cache tensor at production scale (same shape
`get_kv_cache_shape` would actually produce) and re-runs the same
correctness check: reconstruct exact token order across non-contiguous
physical blocks and a partial final block.

Run after GPU steps 1 and 2. Does NOT need infllm_v2 -- this only
exercises the gather function, which is pure PyTorch tensor indexing.
"""

import sys

import torch


def main() -> int:
    assert torch.cuda.is_available(), "This script requires a real GPU."
    device = torch.device("cuda:0")

    from vllm.v1.attention.backends.minicpm_sala_sparse import (
        _gather_full_k_with_new_tokens,
    )

    # Production-scale shapes (real checkpoint values, Phase 1 report):
    # num_kv_heads=2, head_size=128, block_size=256 (real
    # infllmv2_attn_with_kvcache constraint: multiple of 256).
    block_size = 256
    num_kv_heads, head_size = 2, 128
    num_physical_blocks = 64

    print(
        f"Allocating real GPU cache tensor: "
        f"({num_physical_blocks}, {block_size}, {num_kv_heads}, {head_size}) "
        f"on {device} ..."
    )
    # NOTE: k_cache dtype here is float32, NOT the real production
    # bfloat16 -- caught as a real bug in an earlier draft of this
    # exact script: bfloat16 cannot exactly represent integers beyond
    # ~256 (its ~8-bit mantissa), so using it to encode literal token-id
    # markers for THIS correctness check caused false collisions (token
    # 256 and 257 both rounding to the same bf16 value) that looked
    # like a gather bug but were actually a test-encoding artifact. The
    # gather function's correctness (index arithmetic) is dtype-
    # independent, so float32 here checks the same real property
    # without the encoding collision -- confirmed by finding this exact
    # failure mode when this script was first written and validated
    # on CPU before being trusted enough to ship.
    k_cache = torch.zeros(
        num_physical_blocks,
        block_size,
        num_kv_heads,
        head_size,
        device=device,
        dtype=torch.float32,
    )

    # Same non-contiguous-block, partial-final-block scenario as the
    # CPU test, but now at real block_size=256 and on real GPU memory,
    # with THREE sequences instead of two (more realistic batch).
    seq_configs = [
        # (physical_blocks, tokens_in_last_block, num_new_tokens)
        ([37, 5, 12], 100, 8),  # 2 full blocks + 100 in 3rd, 8 new
        ([0], 50, 4),  # 1 partial block, 4 new
        ([22, 8], 256, 16),  # 2 full blocks (no partial), 16 new
    ]

    seq_lens_before = []
    query_start_loc = [0]
    block_table_rows = []
    max_blocks = max(len(c[0]) for c in seq_configs)

    token_id_counter = 0
    expected_order = []
    new_key_rows = []

    for physical_blocks, last_block_tokens, num_new in seq_configs:
        n_full_blocks = len(physical_blocks) - 1
        cached_len = n_full_blocks * block_size + last_block_tokens
        seq_lens_before.append(cached_len)

        # Fill each physical block with sequential real token ids so we
        # can verify reconstruction order later.
        for i, pb in enumerate(physical_blocks):
            n_tokens_this_block = (
                block_size if i < len(physical_blocks) - 1 else last_block_tokens
            )
            for local_tok in range(n_tokens_this_block):
                k_cache[pb, local_tok] = float(token_id_counter)
                expected_order.append(float(token_id_counter))
                token_id_counter += 1

        padded_blocks = physical_blocks + [0] * (max_blocks - len(physical_blocks))
        block_table_rows.append(padded_blocks)

        for _ in range(num_new):
            new_key_rows.append([float(token_id_counter)] * head_size)
            expected_order.append(float(token_id_counter))
            token_id_counter += 1

        query_start_loc.append(query_start_loc[-1] + num_new)

    block_table = torch.tensor(block_table_rows, dtype=torch.int32, device=device)
    seq_lens_before_t = torch.tensor(seq_lens_before, dtype=torch.int32, device=device)
    query_start_loc_t = torch.tensor(query_start_loc, dtype=torch.int32, device=device)
    new_key = (
        torch.tensor(new_key_rows, dtype=torch.float32, device=device)
        .unsqueeze(1)
        .expand(-1, num_kv_heads, -1)
        .contiguous()
    )

    n_new = sum(c[2] for c in seq_configs)
    print(
        f"Gathering across {len(seq_configs)} sequences, "
        f"{sum(seq_lens_before)} cached + {n_new} new tokens ..."
    )

    full_k, cu_seqlens = _gather_full_k_with_new_tokens(
        k_cache=k_cache,
        new_key=new_key,
        block_table=block_table,
        seq_lens_before=seq_lens_before_t,
        query_start_loc=query_start_loc_t,
        block_size=block_size,
    )
    torch.cuda.synchronize()

    print(f"cu_seqlens: {cu_seqlens.tolist()}")
    print(f"full_k shape: {tuple(full_k.shape)}")

    actual_order = full_k[:, 0, 0].float().cpu().tolist()
    expected_len = sum(seq_lens_before) + sum(c[2] for c in seq_configs)
    assert len(actual_order) == expected_len, (
        f"length mismatch: expected {expected_len}, got {len(actual_order)}"
    )
    assert actual_order == expected_order, (
        "REAL GPU GATHER MISMATCH -- token order not correctly "
        "reconstructed. This is exactly the failure mode flagged as "
        "the highest risk in this project; investigate "
        "_gather_full_k_with_new_tokens against this exact scenario."
    )
    print(
        "\nPASS: real GPU gather correctly reconstructs token order "
        "across non-contiguous blocks, partial final blocks, and "
        "multiple sequences, at production block_size=256."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
