#!/usr/bin/env bash
# Apply the verified 2-line fix to the pinned CUTLASS submodule commit
# (zhangyan-didu/cutlass @ 424c5a03220f58a56ddd754e0e2d4eabdf01c802).
#
# Bug: two lines in cuda_host_adapter.hpp are missing the leading '#'
# and use CUDACC_VER_MAJOR/MINOR instead of __CUDACC_VER_MAJOR__/__MINOR__,
# so they are inert text that desyncs every #else/#endif after them.
#
# Run from inside infllmv2_cuda_impl/ after git submodule update --init.

set -euo pipefail

HEADER="csrc/cutlass/include/cutlass/cuda_host_adapter.hpp"
if [[ ! -f "${HEADER}" ]]; then
  echo "ERROR: ${HEADER} not found — run from infllmv2_cuda_impl root" >&2
  exit 1
fi

python3 - "${HEADER}" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
broken = (
    "if ((CUDACC_VER_MAJOR > 12) || "
    "(CUDACC_VER_MAJOR == 12 && CUDACC_VER_MINOR >= 5))"
)
fixed = (
    "#if ((__CUDACC_VER_MAJOR__ > 12) || "
    "(__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 5))"
)
count = text.count(broken)
if count:
    path.write_text(text.replace(broken, fixed), encoding="utf-8")
    print(f"Patched {count} line(s): restored # and __CUDACC_VER_*__ macros")
elif text.count(fixed) >= 2:
    print(f"Already patched ({text.count(fixed)} correct #if lines present)")
else:
    raise SystemExit(
        f"ERROR: expected broken or fixed lines not found in {path}"
    )
PY

echo "Done. Re-run check_cutlass_preprocessor_balance.py on ${HEADER} to confirm."
