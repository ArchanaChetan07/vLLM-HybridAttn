#!/usr/bin/env python3
"""Rewrite minicpm_sala_sparse_wiring.py as UTF-8 (fixes Windows UTF-16 corruption)."""

from __future__ import annotations

from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "vllm/model_executor/models/minicpm_sala_sparse_wiring.py"
DST = Path(__file__).resolve().parents[2] / "vllm/model_executor/models/minicpm_sala_sparse_wiring.utf8.py"

raw = SRC.read_bytes()
if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
    text = raw.decode("utf-16")
elif b"\x00" in raw[:80]:
    text = raw.decode("utf-16-le")
else:
    text = raw.decode("utf-8")

DST.write_bytes(text.encode("utf-8"))
SRC.write_bytes(text.encode("utf-8"))
print("rewrote", SRC, "first bytes", SRC.read_bytes()[:4])
