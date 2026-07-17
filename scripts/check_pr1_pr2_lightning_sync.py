#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gate PR1/PR2 lightning-logic drift.

The repo intentionally carries two copies of minicpm_sala.py:

  vllm/model_executor/models/minicpm_sala.py       (PR1: model only)
  pr2/vllm/model_executor/models/minicpm_sala.py   (PR2: + sparse wiring)

The ONLY allowed differences are the PR2 sparse-wiring deltas (header
comment, the ``create_sparse_attention_if_available`` import, and the
sparse-attention branch inside ``MiniCPMSALADenseAttention.__init__``).
Everything lightning-related must stay byte-identical: the two copies
drifting apart is exactly how the zero-RoPE / kernel-mismatch bugs slipped
in before (see docs/VALIDATION_REPORT.md history).

Pure stdlib (ast + difflib): runs anywhere, no vLLM install needed.

Usage: python3 scripts/check_pr1_pr2_lightning_sync.py
Exit 0 on sync, 1 on drift.
"""

import ast
import difflib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PR1 = REPO_ROOT / "vllm/model_executor/models/minicpm_sala.py"
PR2 = REPO_ROOT / "pr2/vllm/model_executor/models/minicpm_sala.py"

# Top-level definitions that must be byte-identical across both copies.
SYNC_DEFS = [
    "validate_mixer_schedule",
    "is_sparse_layer",
    "is_lightning_layer",
    "build_alibi_slopes",
    "build_lightning_decay_rate",
    "_rotate_half",
    "_build_rope_inv_freq",
    "_apply_hf_rotary_bhtd",
    "_minicpm_sala_lightning_forward_prefix",
    "MiniCPMSALAMLP",
    "MiniCPMSALADenseAttention",  # presence checked; content delta allowed
    "MiniCPMSALALightningAttention",
    "MiniCPMSALADecoderLayer",
    "MiniCPMSALAModel",
    "MiniCPMSALAForCausalLM",
]


def extract_defs(path: Path) -> dict[str, str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[node.name] = ast.get_source_segment(source, node) or ""
    return out


def main() -> int:
    pr1_defs = extract_defs(PR1)
    pr2_defs = extract_defs(PR2)
    failures = []
    for name in SYNC_DEFS:
        a = pr1_defs.get(name)
        b = pr2_defs.get(name)
        if a is None or b is None:
            failures.append(f"{name}: missing in {'PR1' if a is None else 'PR2'} copy")
            continue
        if name == "MiniCPMSALADenseAttention":
            continue  # allowed to differ (sparse wiring branch)
        if a != b:
            diff = "\n".join(
                difflib.unified_diff(
                    a.splitlines(), b.splitlines(), "PR1", "PR2", lineterm="", n=2
                )
            )
            failures.append(f"{name}: DRIFT\n{diff}")
    if failures:
        print("PR1/PR2 lightning sync check FAILED:\n", file=sys.stderr)
        for f in failures:
            print(f, file=sys.stderr)
            print(file=sys.stderr)
        return 1
    print(f"PR1/PR2 sync OK: {len(SYNC_DEFS)} definitions identical "
          "(MiniCPMSALADenseAttention wiring delta allowed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
