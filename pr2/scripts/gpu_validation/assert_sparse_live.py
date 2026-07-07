#!/usr/bin/env python3
"""Step 0 gate: sparse path must be live before any validation is trusted."""
from __future__ import annotations

import os
import sys


def _site_packages_wiring_path() -> str:
    import vllm

    return os.path.join(
        os.path.dirname(vllm.__file__),
        "model_executor",
        "models",
        "minicpm_sala_sparse_wiring.py",
    )


def _check_wiring_file_encoding(path: str) -> str | None:
    """Return wiring source text, or None after printing a FAIL message."""
    if not os.path.isfile(path):
        print(f"FAIL: sparse wiring not installed at {path}")
        print("      Run: bash scripts/install_pr2_overlay.sh")
        return None

    raw = open(path, "rb").read()
    if b"\x00" in raw:
        print(f"FAIL: {path} contains null bytes (UTF-16 corruption)")
        print("      Re-copy UTF-8 sources: bash scripts/install_pr2_overlay.sh")
        return None

    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as e:
        print(f"FAIL: {path} is not valid UTF-8: {e}")
        return None


def _check_fail_loud_wiring(source: str) -> bool:
    import ast

    tree = ast.parse(source)
    target: ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "create_sparse_attention_if_available":
            target = node
            break

    if target is None:
        print("FAIL: create_sparse_attention_if_available not found in sparse_wiring")
        return False

    func_src = ast.get_source_segment(source, target)
    if func_src is None:
        print("FAIL: could not extract create_sparse_attention_if_available source")
        return False

    if "if not INFLLM_V2_AVAILABLE" in func_src and "return None" in func_src:
        print(
            "FAIL: create_sparse_attention_if_available silently returns None "
            "when infllm_v2 missing"
        )
        return False
    if "raise RuntimeError" not in func_src and "raise ImportError" not in func_src:
        print(
            "FAIL: create_sparse_attention_if_available does not fail loud "
            "when infllm_v2 missing"
        )
        return False
    return True


def main() -> int:
    try:
        from vllm.v1.attention.backends.minicpm_sala_sparse import (
            INFLLM_V2_AVAILABLE,
            infllmv2_attn_varlen_func,
        )
    except ImportError as e:
        print(f"FAIL: cannot import minicpm_sala_sparse backend: {e}")
        return 1

    if not INFLLM_V2_AVAILABLE:
        print("FAIL: INFLLM_V2_AVAILABLE is False — infllm_v2 not installed")
        return 1

    try:
        import infllm_v2  # noqa: F401
    except ImportError as e:
        print(f"FAIL: infllm_v2 import failed: {e}")
        return 1

    if infllmv2_attn_varlen_func is None:
        print("FAIL: infllmv2_attn_varlen_func is None")
        return 1

    wiring_path = _site_packages_wiring_path()
    wiring_source = _check_wiring_file_encoding(wiring_path)
    if wiring_source is None:
        return 1

    if not _check_fail_loud_wiring(wiring_source):
        return 1

    # Import must succeed after encoding check (proves overlay is executable Python).
    try:
        from vllm.model_executor.models.minicpm_sala_sparse_wiring import (  # noqa: F401
            create_sparse_attention_if_available,
        )
    except SyntaxError as e:
        print(f"FAIL: cannot import sparse_wiring after encoding check: {e}")
        return 1

    print(
        "PASS: sparse path LIVE (INFLLM_V2_AVAILABLE=True, infllm_v2 importable, "
        "fail-loud wiring, UTF-8 overlay)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
