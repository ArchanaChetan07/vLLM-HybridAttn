# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for per-sequence sparse/dense dispatch helpers (H2)."""

import torch

from vllm.v1.attention.backends.minicpm_sala_sparse import (
    MiniCPMSALASparseAttentionBackend,
    MiniCPMSALASparseAttentionMetadata,
    _correct_dense_prefill_metadata,
    _packed_num_tokens,
    _select_varlen_sequences,
    sequence_sparse_mask,
)


def _metadata(
    seq_lens: list[int],
    q_tokens_per_seq: list[int],
    page_block_size: int = 256,
    dense_len: int = 8192,
) -> MiniCPMSALASparseAttentionMetadata:
    qsl = [0]
    for n in q_tokens_per_seq:
        qsl.append(qsl[-1] + n)
    num_tokens = qsl[-1]
    return MiniCPMSALASparseAttentionMetadata(
        query_start_loc=torch.tensor(qsl, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        block_table=torch.zeros(len(seq_lens), 4, dtype=torch.int32),
        slot_mapping=torch.zeros(num_tokens, dtype=torch.int64),
        dense_len=dense_len,
        page_block_size=page_block_size,
        num_actual_tokens=num_tokens,
        max_query_len=max(q_tokens_per_seq) if q_tokens_per_seq else 0,
        max_seq_len=max(seq_lens) if seq_lens else 0,
    )


class TestSparseBackendKvCachePolicy:
    def test_dense_path_uses_external_kv_cache_update(self) -> None:
        assert (
            MiniCPMSALASparseAttentionBackend.forward_includes_kv_cache_update
            is False
        )

    def test_dense_eager_prefill_default_enabled(self) -> None:
        from vllm.v1.attention.backends import minicpm_sala_sparse as sparse_mod

        assert sparse_mod._DENSE_EAGER_PREFILL is True


class TestCorrectDensePrefillMetadata:
    def test_clamps_inflated_seq_lens_on_new_token_forward(self) -> None:
        meta = _metadata(seq_lens=[8], q_tokens_per_seq=[7])
        query = torch.zeros(7, 16, 64)
        fixed = _correct_dense_prefill_metadata(meta, query)
        assert fixed.seq_lens.tolist() == [7]
        assert fixed.max_seq_len == 7
        assert fixed.max_query_len == 7

    def test_noop_when_seq_lens_matches_num_new(self) -> None:
        meta = _metadata(seq_lens=[7], q_tokens_per_seq=[7])
        query = torch.zeros(7, 16, 64)
        fixed = _correct_dense_prefill_metadata(meta, query)
        assert fixed is meta

    def test_noop_on_single_token_decode(self) -> None:
        meta = _metadata(seq_lens=[8], q_tokens_per_seq=[1])
        query = torch.zeros(1, 16, 64)
        fixed = _correct_dense_prefill_metadata(meta, query)
        assert fixed is meta

    def test_clamps_with_padded_query_tensor(self) -> None:
        meta = _metadata(seq_lens=[12], q_tokens_per_seq=[6])
        meta.num_actual_tokens = 8
        query = torch.zeros(8, 16, 64)
        fixed = _correct_dense_prefill_metadata(meta, query)
        assert fixed.seq_lens.tolist() == [6]
        assert _packed_num_tokens(fixed) == 6


class TestSequenceSparseMask:
    def test_all_dense(self) -> None:
        seq_lens = torch.tensor([100, 4096, 8191], dtype=torch.int32)
        mask = sequence_sparse_mask(seq_lens, dense_len=8192)
        assert mask.tolist() == [False, False, False]

    def test_all_sparse(self) -> None:
        seq_lens = torch.tensor([8192, 9000, 100000], dtype=torch.int32)
        mask = sequence_sparse_mask(seq_lens, dense_len=8192)
        assert mask.tolist() == [True, True, True]

    def test_mixed_batch(self) -> None:
        seq_lens = torch.tensor([4096, 8192, 8193, 0], dtype=torch.int32)
        mask = sequence_sparse_mask(seq_lens, dense_len=8192)
        assert mask.tolist() == [False, True, True, False]

    def test_boundary_exact_dense_len(self) -> None:
        seq_lens = torch.tensor([8191, 8192], dtype=torch.int32)
        mask = sequence_sparse_mask(seq_lens, dense_len=8192)
        assert mask.tolist() == [False, True]


class TestSelectVarlenSequences:
    def test_subset_preserves_token_order(self) -> None:
        meta = _metadata(seq_lens=[100, 9000], q_tokens_per_seq=[2, 3])
        total = 5
        query = torch.arange(total).view(total, 1, 1).expand(total, 1, 2).float()
        key = query.clone()
        value = query.clone()

        sub_q, sub_k, sub_v, sub_meta, ranges = _select_varlen_sequences(
            [1], query, key, value, meta
        )
        assert sub_q.shape[0] == 3
        assert ranges == [(2, 5)]
        assert sub_meta.seq_lens.tolist() == [9000]
        assert sub_meta.query_start_loc.tolist() == [0, 3]
        assert sub_q[:, 0, 0].tolist() == [2.0, 3.0, 4.0]

    def test_empty_new_sequence_q_tokens(self) -> None:
        meta = _metadata(seq_lens=[0], q_tokens_per_seq=[0])
        query = torch.zeros(0, 1, 2)
        key = query.clone()
        value = query.clone()
        sub_q, _, _, sub_meta, ranges = _select_varlen_sequences(
            [0], query, key, value, meta
        )
        assert sub_q.shape[0] == 0
        assert ranges == [(0, 0)]
        assert sub_meta.seq_lens.tolist() == [0]
