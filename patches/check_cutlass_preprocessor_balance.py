#!/usr/bin/env python3
"""Structural checker for #if/#else/#endif balance in a CUTLASS header.

Used to verify fix_cutlass_submodule.sh before attempting pip install -e .
Does not invoke the CUDA compiler — catches the exact class of bug that
produced '#else after #else' from inert (non-directive) lines.
"""

from __future__ import annotations

import sys
from pathlib import Path

IF_FAMILY = ("#if ", "#ifdef ", "#ifndef ", "#elif ", "#else", "#endif")


def check(path: Path) -> list[str]:
    errors: list[str] = []
    stack: list[tuple[int, str]] = []

    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.lstrip()
        if not stripped.startswith("#"):
            # Flag lines that look like broken preprocessor directives (missing '#').
            for token in ("if __CUDACC", "if CUDACC", "if ((CUDACC", "else", "endif"):
                if stripped.startswith(token) and not stripped.startswith("#"):
                    errors.append(
                        f"{lineno}: line looks like a preprocessor directive "
                        f"but is missing leading '#': {stripped!r}"
                    )
            continue

        if stripped.startswith("#if ") or stripped.startswith("#ifdef ") or stripped.startswith(
            "#ifndef "
        ):
            stack.append((lineno, stripped.split()[0]))
        elif stripped.startswith("#elif "):
            if not stack:
                errors.append(f"{lineno}: #elif without matching #if")
        elif stripped.startswith("#else"):
            if not stack:
                errors.append(f"{lineno}: #else without matching #if")
            elif any(e.startswith(f"{lineno}:") and "#else after #else" in e for e in errors):
                pass
            # Detect consecutive #else at same nesting level (simplified).
        elif stripped.startswith("#endif"):
            if not stack:
                errors.append(f"{lineno}: #endif without matching #if")
            else:
                stack.pop()

    # Second pass: detect #else after #else by simulating stack with branch flags.
    stack2: list[dict[str, bool]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.lstrip()
        if not stripped.startswith("#"):
            continue
        if stripped.startswith(("#if ", "#ifdef ", "#ifndef ")):
            stack2.append({"saw_else": False})
        elif stripped.startswith("#elif "):
            if stack2 and stack2[-1]["saw_else"]:
                errors.append(f"{lineno}: #elif after #else in same block")
        elif stripped.startswith("#else"):
            if not stack2:
                errors.append(f"{lineno}: #else without matching #if")
            elif stack2[-1]["saw_else"]:
                errors.append(f"{lineno}: #else after #else")
            else:
                stack2[-1]["saw_else"] = True
        elif stripped.startswith("#endif"):
            if not stack2:
                errors.append(f"{lineno}: #endif without matching #if")
            else:
                stack2.pop()

    if stack:
        for open_line, kind in stack:
            errors.append(f"{open_line}: unclosed {kind} (no matching #endif)")

    if stack2:
        errors.append(f"unclosed preprocessor block(s): {len(stack2)} remaining")

    return errors


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path/to/cuda_host_adapter.hpp>", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"FAIL: not a file: {path}", file=sys.stderr)
        return 1

    errors = check(path)
    if errors:
        print("FAIL: preprocessor structure errors:")
        for err in errors:
            print(f"  {err}")
        return 1

    print(f"PASS: {path} preprocessor structure OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
