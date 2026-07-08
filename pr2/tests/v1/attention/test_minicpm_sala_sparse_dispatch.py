# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for per-sequence sparse/dense dispatch helpers (H2)."""

import torch

from vllm.v1.attention.backends.minicpm_sala_sparse import (
    MiniCPMSALASparseAttentionBackend,
    MiniCPMSALASparseAttentionMetadata,
    _append_dense_kv_history,
    _correct_dense_prefill_metadata,
    _dense_kv_history_prefix,
    _DENSE_HISTORY_DECODE_MAX_SEQ,
    _packed_num_tokens,
    _reset_dense_kv_history,
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

    def test_dense_history_decode_max_seq_default_64(self) -> None:
        assert _DENSE_HISTORY_DECODE_MAX_SEQ == 64


class TestDenseKvHistory:
    def test_prefix_matches_n_before_after_prefill_append(self) -> None:
        layer = torch.nn.Module()
        _reset_dense_kv_history(layer)
        q = torch.randn(6, 2, 4)
        k = torch.randn(6, 1, 4)
        v = torch.randn(6, 1, 4)
        _append_dense_kv_history(layer, q, k, v, 6)
        prefix = _dense_kv_history_prefix(layer, 6)
        assert prefix is not None
        hist_q, hist_k, hist_v = prefix
        assert hist_q.shape[0] == 6
        assert torch.equal(hist_q, q)
        assert torch.equal(hist_k, k)
        assert torch.equal(hist_v, v)

    def test_append_extends_history_on_decode_step(self) -> None:
        layer = torch.nn.Module()
        _reset_dense_kv_history(layer)
        _append_dense_kv_history(
            layer,
            torch.randn(6, 2, 4),
            torch.randn(6, 1, 4),
            torch.randn(6, 1, 4),
            6,
        )
        _append_dense_kv_history(
            layer,
            torch.randn(1, 2, 4),
            torch.randn(1, 1, 4),
            torch.randn(1, 1, 4),
            1,
        )
        prefix = _dense_kv_history_prefix(layer, 7)
        assert prefix is not None
        assert prefix[0].shape[0] == 7

    def test_prefix_none_when_length_mismatch(self) -> None:
        layer = torch.nn.Module()
        _reset_dense_kv_history(layer)
        _append_dense_kv_history(
            layer,
            torch.randn(6, 2, 4),
            torch.randn(6, 1, 4),
            torch.randn(6, 1, 4),
            6,
        )
        assert _dense_kv_history_prefix(layer, 5) is None
        assert _dense_kv_history_prefix(layer, 7) is None


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

    def test_decode_block_table_follows_slot_mapping(self) -> None:
        from vllm.v1.attention.backends.minicpm_sala_sparse import (
            _correct_dense_decode_block_table,
        )

        meta = _metadata(seq_lens=[20], q_tokens_per_seq=[1])
        meta.slot_mapping = torch.tensor([2067], dtype=torch.int64)
        meta.block_table = torch.tensor([[1, 0]], dtype=torch.int32)
        fixed = _correct_dense_decode_block_table(meta)
        assert fixed.block_table[0, 0].item() == 8

    def test_sparse_decode_kv_slot_replay_steps_10_through_15(self) -> None:
        """CPU replay from decode_kv_slot_capture_latest.json (ISSUE-03)."""
        from vllm.v1.attention.backends.minicpm_sala_sparse import (
            _correct_dense_decode_block_table,
            _gather_cached_tokens_for_decode,
            _gather_full_k_with_new_tokens,
        )

        page = 256
        traces = [
            (10, 16, 2063),
            (11, 17, 2064),
            (12, 18, 2065),
            (13, 19, 2066),
            (14, 20, 2067),
            (15, 21, 2068),
        ]
        k_cache = torch.zeros(10, page, 1, 2)
        for phys in (1, 8):
            for off in range(page):
                k_cache[phys, off, 0, 0] = float(phys * 1000 + off)

        for _step, seq_len, slot in traces:
            n_before = seq_len - 1
            meta = _metadata(seq_lens=[seq_len], q_tokens_per_seq=[1], page_block_size=page)
            meta.slot_mapping = torch.tensor([slot], dtype=torch.int64)
            meta.block_table = torch.tensor([[1, 0]], dtype=torch.int32)

            pos = seq_len - 1
            bt_head = int(meta.block_table[0, 0].item())
            delta_before = bt_head * page + (pos % page) - slot
            assert delta_before == -1792

            fixed = _correct_dense_decode_block_table(meta)
            bt_fixed = int(fixed.block_table[0, 0].item())
            assert bt_fixed == slot // page
            delta_after = bt_fixed * page + (pos % page) - slot
            assert delta_after == 0

            new_key = torch.tensor([[float(slot), float(slot)]]).view(1, 1, 2)
            full_k, _cu = _gather_full_k_with_new_tokens(
                k_cache=k_cache,
                new_key=new_key,
                block_table=fixed.block_table,
                seq_lens_before=torch.tensor([n_before], dtype=torch.int32),
                query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
                block_size=page,
            )
            expected_tail = float(8 * 1000 + (n_before - 1))
            assert full_k[-2, 0, 0].item() == expected_tail
            assert full_k[-1, 0, 0].item() == float(slot)

            gathered = _gather_cached_tokens_for_decode(
                k_cache,
                n_before,
                torch.tensor([slot], dtype=torch.int64),
                page,
            )
            assert gathered.shape[0] == n_before
            assert gathered[-1, 0, 0].item() == expected_tail

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
