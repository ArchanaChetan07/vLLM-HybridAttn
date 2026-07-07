# SPDX-License-Identifier: Apache-2.0
"""Unit tests for infllm kvcache varlen packing helpers."""

import torch

from vllm.v1.attention.backends.minicpm_sala_sparse import (
    _pack_varlen_qkv_for_infllm_kvcache,
    _unpack_batched_output_for_varlen,
)


def test_pack_unpack_roundtrip() -> None:
    device = torch.device("cpu")
    query = torch.arange(28, dtype=torch.float32, device=device).view(7, 2, 2)
    key = query + 100
    value = query + 200
    query_start_loc = torch.tensor([0, 3, 5, 7], dtype=torch.int32, device=device)

    q_b, k_b, v_b = _pack_varlen_qkv_for_infllm_kvcache(
        query, key, value, query_start_loc
    )
    # Packed dim-1 is max(q_lens)=3 (longest sequence in the batch).
    assert q_b.shape == (3, 3, 2, 2)
    assert torch.equal(q_b[0, :3], query[:3])
    assert torch.equal(q_b[1, :2], query[3:5])
    assert torch.equal(q_b[2, :2], query[5:7])

    out_b = torch.randn(3, 3, 2, 2, device=device)
    output = torch.zeros(7, 2, 2, device=device)
    _unpack_batched_output_for_varlen(out_b, query_start_loc, output)
    assert torch.equal(output[:3], out_b[0, :3])
    assert torch.equal(output[3:5], out_b[1, :2])
    assert torch.equal(output[5:7], out_b[2, :2])
