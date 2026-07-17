#!/usr/bin/env bash
# Build and install the infllm_v2 CUDA package (OpenBMB/infllmv2_cuda_impl)
# for the MiniCPM-SALA sparse path. Requires an sm_80+ GPU toolchain.
#
# Usage: scripts/install_infllm_v2.sh [BUILD_DIR]
#   BUILD_DIR defaults to /tmp/infllmv2_cuda_impl
#
# Applies patches/fix_cutlass_submodule.sh (the verified 2-line CUTLASS
# header fix) before building -- without it the build fails on CUDA >= 12.5
# with desynced #if/#endif errors in cuda_host_adapter.hpp.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-/tmp/infllmv2_cuda_impl}"

if python3 -c "import infllm_v2" 2>/dev/null; then
  echo "infllm_v2 already importable -- nothing to do."
  exit 0
fi

if [[ ! -d "${BUILD_DIR}/.git" ]]; then
  git clone https://github.com/OpenBMB/infllmv2_cuda_impl.git "${BUILD_DIR}"
fi
cd "${BUILD_DIR}"
git submodule update --init --recursive

bash "${REPO_ROOT}/patches/fix_cutlass_submodule.sh"
python3 "${REPO_ROOT}/patches/check_cutlass_preprocessor_balance.py" \
  csrc/cutlass/include/cutlass/cuda_host_adapter.hpp || true

# sm_80 (A100). Override for other Ampere+ parts, e.g. "8.9" for RTX 4090.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
echo "Building infllm_v2 for TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} ..."
pip install --no-build-isolation -v .

# Verify from OUTSIDE the source tree: importing with cwd inside
# infllmv2_cuda_impl/ picks up the source package (which has no compiled
# C extension) and fails with a bogus circular-import error.
cd /
python3 - <<'PY'
from infllm_v2 import (
    infllmv2_attn_stage1,
    infllmv2_attn_with_kvcache,
    max_pooling_1d_varlen,
)
print("infllm_v2 import OK:", infllmv2_attn_stage1.__name__,
      infllmv2_attn_with_kvcache.__name__, max_pooling_1d_varlen.__name__)
PY
echo "infllm_v2 installed."
