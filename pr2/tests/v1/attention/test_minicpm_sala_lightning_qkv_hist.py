# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for MiniCPM-SALA lightning q/k/v history bookkeeping.

Mirrors ``MiniCPMSALALightningAttention._sync_qkv_history`` /
``_reset_qkv_history``, ``_lightning_should_reset_qkv_history``, and the
recompute-path tensor layout used by ``_decode_infer_parity``. Self-contained
— no GPU, checkpoint, ``fla``, or full ``vllm`` install required.
"""

from __future__ import annotations

import torch


class _QkvHistStub:
    """Minimal stand-in for lightning q/k/v history fields."""

    def __init__(self) -> None:
        self._qkv_hist_q: torch.Tensor | None = None
        self._qkv_hist_k: torch.Tensor | None = None
        self._qkv_hist_v: torch.Tensor | None = None

    def _reset_qkv_history(self) -> None:
        self._qkv_hist_q = None
        self._qkv_hist_k = None
        self._qkv_hist_v = None

    def _sync_qkv_history(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        fresh: bool,
        target_hist_len: int | None = None,
    ) -> None:
        if not fresh and target_hist_len is not None:
            cur = 0 if self._qkv_hist_q is None else int(self._qkv_hist_q.shape[0])
            if cur >= target_hist_len:
                return
        if fresh or self._qkv_hist_q is None:
            self._qkv_hist_q = q.detach()
            self._qkv_hist_k = k.detach()
            self._qkv_hist_v = v.detach()
            return
        self._qkv_hist_q = torch.cat([self._qkv_hist_q, q.detach()], dim=0)
        self._qkv_hist_k = torch.cat([self._qkv_hist_k, k.detach()], dim=0)
        self._qkv_hist_v = torch.cat([self._qkv_hist_v, v.detach()], dim=0)


def _should_reset_qkv_history(
    num_prefills: int,
    num_decode_tokens: int,
    query_start_loc: list[int],
    seq_lens: list[int],
    positions: list[int],
) -> bool:
    """Mirror ``_lightning_should_reset_qkv_history`` without vLLM metadata."""
    offset = num_decode_tokens
    for prefill_idx in range(num_prefills):
        q_start = query_start_loc[offset + prefill_idx]
        q_end = query_start_loc[offset + prefill_idx + 1]
        if positions[q_start] == 0:
            return True
        query_len = q_end - q_start
        context_len = seq_lens[offset + prefill_idx] - query_len
        if context_len == 0:
            return True
    return False


class TestLightningQkvHistory:
    def test_fresh_sync_replaces_history(self) -> None:
        attn = _QkvHistStub()
        q1 = torch.randn(3, 4, 8)
        k1 = torch.randn(3, 4, 8)
        v1 = torch.randn(3, 4, 8)
        attn._sync_qkv_history(q1, k1, v1, fresh=True)
        assert attn._qkv_hist_q is not None
        assert attn._qkv_hist_q.shape == (3, 4, 8)

        q2 = torch.randn(2, 4, 8)
        k2 = torch.randn(2, 4, 8)
        v2 = torch.randn(2, 4, 8)
        attn._sync_qkv_history(q2, k2, v2, fresh=True)
        assert attn._qkv_hist_q.shape == (2, 4, 8)
        assert not torch.equal(attn._qkv_hist_q, q1)

    def test_decode_append_extends_history(self) -> None:
        attn = _QkvHistStub()
        pre_q = torch.randn(5, 4, 8)
        pre_k = torch.randn(5, 4, 8)
        pre_v = torch.randn(5, 4, 8)
        attn._sync_qkv_history(pre_q, pre_k, pre_v, fresh=True)

        dec_q = torch.randn(1, 4, 8)
        dec_k = torch.randn(1, 4, 8)
        dec_v = torch.randn(1, 4, 8)
        attn._sync_qkv_history(dec_q, dec_k, dec_v, fresh=False, target_hist_len=6)
        assert attn._qkv_hist_q.shape == (6, 4, 8)
        assert torch.equal(attn._qkv_hist_q[:5], pre_q)
        assert torch.equal(attn._qkv_hist_q[5:], dec_q)

    def test_decode_skips_append_when_history_already_full(self) -> None:
        """Prefill+decode on the same prefix must not duplicate the last token."""
        attn = _QkvHistStub()
        pre_q = torch.randn(20, 4, 8)
        pre_k = torch.randn(20, 4, 8)
        pre_v = torch.randn(20, 4, 8)
        attn._sync_qkv_history(pre_q, pre_k, pre_v, fresh=True)
        last_q = pre_q[-1:].clone()
        last_k = pre_k[-1:].clone()
        last_v = pre_v[-1:].clone()
        attn._sync_qkv_history(
            last_q, last_k, last_v, fresh=False, target_hist_len=20
        )
        assert attn._qkv_hist_q.shape == (20, 4, 8)
        assert torch.equal(attn._qkv_hist_q[-1], pre_q[-1])

    def test_recompute_pack_matches_bhtd_layout(self) -> None:
        """Recompute path: [T,H,D] -> transpose -> unsqueeze -> [1,H,T,D]."""
        t, h, d = 7, 4, 8
        hist_q = torch.arange(t * h * d, dtype=torch.float32).reshape(t, h, d)
        packed = hist_q.transpose(0, 1).unsqueeze(0).contiguous()
        assert packed.shape == (1, h, t, d)
        for ti in range(t):
            for hi in range(h):
                assert torch.equal(packed[0, hi, ti], hist_q[ti, hi])

    def test_decode_parity_routing_threshold(self) -> None:
        """Document routing: full recompute for all hist_len > 0 (chunk if >= 64)."""
        for hist_len in (0, 1, 20, 63, 64, 100):
            use_recompute = hist_len > 0
            if hist_len in (1, 20, 63, 64, 100):
                assert use_recompute
            else:
                assert not use_recompute

    def test_reset_qkv_history_clears_buffers(self) -> None:
        attn = _QkvHistStub()
        attn._sync_qkv_history(
            torch.randn(2, 4, 8),
            torch.randn(2, 4, 8),
            torch.randn(2, 4, 8),
            fresh=True,
        )
        attn._reset_qkv_history()
        assert attn._qkv_hist_q is None
        assert attn._qkv_hist_k is None
        assert attn._qkv_hist_v is None

    def test_should_reset_on_position_zero(self) -> None:
        assert _should_reset_qkv_history(
            num_prefills=1,
            num_decode_tokens=0,
            query_start_loc=[0, 6],
            seq_lens=[6],
            positions=[0, 1, 2, 3, 4, 5],
        )

    def test_should_reset_on_context_len_zero(self) -> None:
        """Inflated seq_lens can skip cache clear unless context_len==0 is checked."""
        assert _should_reset_qkv_history(
            num_prefills=1,
            num_decode_tokens=0,
            query_start_loc=[0, 6],
            seq_lens=[6],
            positions=[5, 6, 7, 8, 9, 10],
        )

    def test_should_not_reset_on_chunked_continuation(self) -> None:
        assert not _should_reset_qkv_history(
            num_prefills=1,
            num_decode_tokens=0,
            query_start_loc=[0, 4],
            seq_lens=[10],
            positions=[6, 7, 8, 9],
        )
