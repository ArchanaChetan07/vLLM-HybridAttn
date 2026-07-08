#!/usr/bin/env python3
"""CPU gate (ISSUE-03): replay decode_kv_slot_capture metadata at steps 10-12.

Confirms block_table correction aligns read physical page with slot_mapping
writes (expected_vs_actual_slot_delta -> 0) without GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

TRACE_JSON = (
    Path(__file__).resolve().parent / "traces" / "decode_kv_slot_capture_latest.json"
)
BLOCK_SIZE = 256
TARGET_STEPS = (10, 11, 12)


def _read_phys_from_slot(slot: int, n_before: int, page: int) -> int:
    return (slot - n_before) // page


def _slot_delta(block_table_head: int, slot: int, pos: int, page: int) -> int:
    return block_table_head * page + (pos % page) - slot


def main() -> int:
    if not TRACE_JSON.is_file():
        print(f"FAIL: missing trace {TRACE_JSON}", file=sys.stderr)
        return 1

    trace = json.loads(TRACE_JSON.read_text(encoding="utf-8"))
    captures = trace.get("captures", {})
    failures: list[str] = []

    # Import correction + gather when vLLM overlay is on PYTHONPATH (A100/CI).
    try:
        from vllm.v1.attention.backends.minicpm_sala_sparse import (
            MiniCPMSALASparseAttentionMetadata,
            _correct_dense_decode_block_table,
            _gather_full_k_with_new_tokens,
        )
    except ImportError:
        _correct_dense_decode_block_table = None
        _gather_full_k_with_new_tokens = None
        MiniCPMSALASparseAttentionMetadata = None

    print("ISSUE-03 CPU replay (steps 10-12)", flush=True)
    for step in TARGET_STEPS:
        cap = captures.get(str(step))
        if cap is None:
            failures.append(f"step {step}: missing capture")
            continue

        seq_len = int(cap["seq_lens"][0])
        slot = int(cap["slot_mapping"][0])
        bt_head = int(cap["block_table"][0][0])
        pos = seq_len - 1
        n_before = pos

        delta_before = _slot_delta(bt_head, slot, pos, BLOCK_SIZE)
        phys = _read_phys_from_slot(slot, n_before, BLOCK_SIZE)
        delta_after = _slot_delta(phys, slot, pos, BLOCK_SIZE)

        row = {
            "step": step,
            "seq_len": seq_len,
            "slot": slot,
            "block_table_head_before": bt_head,
            "read_phys_after_fix": phys,
            "delta_before": delta_before,
            "delta_after": delta_after,
        }
        print(json.dumps(row), flush=True)

        if delta_before != -1792:
            failures.append(f"step {step}: expected delta_before=-1792, got {delta_before}")
        if delta_after != 0:
            failures.append(f"step {step}: delta_after={delta_after}, expected 0")
        if phys != slot // BLOCK_SIZE:
            failures.append(
                f"step {step}: phys={phys} != slot//page={slot // BLOCK_SIZE}"
            )

        if (
            _correct_dense_decode_block_table is not None
            and MiniCPMSALASparseAttentionMetadata is not None
        ):
            meta = MiniCPMSALASparseAttentionMetadata(
                query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
                seq_lens=torch.tensor([seq_len], dtype=torch.int32),
                block_table=torch.tensor([cap["block_table"][0]], dtype=torch.int32),
                slot_mapping=torch.tensor([slot], dtype=torch.int64),
                dense_len=int(cap.get("dense_len", 8192)),
                page_block_size=BLOCK_SIZE,
                num_actual_tokens=1,
                max_query_len=1,
                max_seq_len=seq_len,
            )
            fixed = _correct_dense_decode_block_table(meta)
            if int(fixed.block_table[0, 0].item()) != phys:
                failures.append(
                    f"step {step}: _correct_dense_decode_block_table -> "
                    f"{int(fixed.block_table[0, 0].item())}, expected {phys}"
                )

            if _gather_full_k_with_new_tokens is not None:
                k_cache = torch.zeros(10, BLOCK_SIZE, 1, 2)
                for p in (1, 8):
                    for off in range(BLOCK_SIZE):
                        k_cache[p, off, 0, 0] = float(p * 1000 + off)
                new_key = torch.tensor([[float(slot), float(slot)]]).view(1, 1, 2)
                full_k, _ = _gather_full_k_with_new_tokens(
                    k_cache=k_cache,
                    new_key=new_key,
                    block_table=fixed.block_table,
                    seq_lens_before=torch.tensor([n_before], dtype=torch.int32),
                    query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
                    block_size=BLOCK_SIZE,
                )
                expected_tail = float(8 * 1000 + (slot % BLOCK_SIZE))
                if full_k[-2, 0, 0].item() != expected_tail:
                    failures.append(
                        f"step {step}: gather tail {full_k[-2, 0, 0].item()} "
                        f"!= {expected_tail}"
                    )

    if failures:
        print("FAIL:", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)
        return 1

    print("PASS: all steps delta_after=0 (read page matches slot_mapping write)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
