#!/usr/bin/env python3
"""Synthetic qkv GEMM vs split isolation (no full model load)."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

torch.manual_seed(0)
HIDDEN = 4096
Q_SIZE = 32 * 128
KV_SIZE = 2 * 128
SEQ = 7
DTYPE = torch.bfloat16


def _peak(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def main() -> int:
    x = torch.randn(SEQ, HIDDEN, dtype=DTYPE)
    w_q = torch.randn(Q_SIZE, HIDDEN, dtype=DTYPE)
    w_k = torch.randn(KV_SIZE, HIDDEN, dtype=DTYPE)
    w_v = torch.randn(KV_SIZE, HIDDEN, dtype=DTYPE)
    w_fused = torch.cat([w_q, w_k, w_v], dim=0)

    hf_q = F.linear(x, w_q)
    hf_k = F.linear(x, w_k)
    hf_v = F.linear(x, w_v)

    fused = F.linear(x, w_fused)
    fq, fk, fv = fused.split([Q_SIZE, KV_SIZE, KV_SIZE], dim=-1)

    tq, tk, tv = F.linear(x, w_q), F.linear(x, w_k), F.linear(x, w_v)

    print(f"synthetic seqlen={SEQ} hidden={HIDDEN}", flush=True)
    print(f"fused split q={_peak(fq, hf_q):.6g} k={_peak(fk, hf_k):.6g} v={_peak(fv, hf_v):.6g}", flush=True)
    print(f"three matmul q={_peak(tq, hf_q):.6g} k={_peak(tk, hf_k):.6g} v={_peak(tv, hf_v):.6g}", flush=True)
    print("H1 fused GEMM: IN if fused k/v > three k/v", flush=True)
    print("H2 split: OUT if fused manual slice peak=0", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
