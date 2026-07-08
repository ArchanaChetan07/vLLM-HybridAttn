# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for dense-layer K/V history bookkeeping (Blocker 2).

Mirrors ``_append_dense_kv_history``, ``_dense_kv_history_prefix``, and
``_reset_dense_kv_history`` without a full vLLM install.
"""

from __future__ import annotations

import torch


class _FakeLayer:
    pass


def _reset_dense_kv_history(layer: _FakeLayer) -> None:
    layer._sala_dense_kv_q = None
    layer._sala_dense_kv_k = None
    layer._sala_dense_kv_v = None


def _append_dense_kv_history(
    layer: _FakeLayer,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    n: int,
) -> None:
    q = query[:n].detach().float()
    k = key[:n].detach().float()
    v = value[:n].detach().float()
    hist_q = getattr(layer, "_sala_dense_kv_q", None)
    if hist_q is None:
        layer._sala_dense_kv_q = q
        layer._sala_dense_kv_k = k
        layer._sala_dense_kv_v = v
        return
    layer._sala_dense_kv_q = torch.cat([hist_q, q], dim=0)
    layer._sala_dense_kv_k = torch.cat([layer._sala_dense_kv_k, k], dim=0)
    layer._sala_dense_kv_v = torch.cat([layer._sala_dense_kv_v, v], dim=0)


def _dense_kv_history_prefix(
    layer: _FakeLayer,
    n_before: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    hist_q = getattr(layer, "_sala_dense_kv_q", None)
    hist_k = getattr(layer, "_sala_dense_kv_k", None)
    if (
        hist_q is None
        or hist_k is None
        or int(hist_q.shape[0]) != n_before
        or int(hist_k.shape[0]) != n_before
    ):
        return None
    return hist_q, hist_k, layer._sala_dense_kv_v


def test_dense_kv_history_append_and_prefix() -> None:
    layer = _FakeLayer()
    q1 = torch.randn(3, 4, 8)
    k1 = torch.randn(3, 2, 8)
    v1 = torch.randn(3, 2, 8)
    _append_dense_kv_history(layer, q1, k1, v1, 3)
    hist = _dense_kv_history_prefix(layer, 3)
    assert hist is not None
    hq, hk, hv = hist
    assert hq.dtype == torch.float32
    assert hk.dtype == torch.float32
    assert hv.dtype == torch.float32
    assert torch.equal(hq, q1.float())
    assert torch.equal(hk, k1.float())
    assert torch.equal(hv, v1.float())

    q2 = torch.randn(1, 4, 8)
    k2 = torch.randn(1, 2, 8)
    v2 = torch.randn(1, 2, 8)
    _append_dense_kv_history(layer, q2, k2, v2, 1)
    hist4 = _dense_kv_history_prefix(layer, 4)
    assert hist4 is not None
    _, hk4, hv4 = hist4
    assert hk4.shape == (4, 2, 8)
    assert torch.equal(hk4[:3], k1.float())
    assert torch.equal(hk4[3:], k2.float())
    assert torch.equal(hv4[:3], v1.float())
    assert torch.equal(hv4[3:], v2.float())


def test_dense_kv_history_prefix_miss_on_length() -> None:
    layer = _FakeLayer()
    q = torch.randn(2, 4, 8)
    k = torch.randn(2, 2, 8)
    v = torch.randn(2, 2, 8)
    _append_dense_kv_history(layer, q, k, v, 2)
    assert _dense_kv_history_prefix(layer, 1) is None


def test_reset_dense_kv_history() -> None:
    layer = _FakeLayer()
    _append_dense_kv_history(
        layer,
        torch.randn(1, 4, 8),
        torch.randn(1, 2, 8),
        torch.randn(1, 2, 8),
        1,
    )
    _reset_dense_kv_history(layer)
    assert _dense_kv_history_prefix(layer, 1) is None
