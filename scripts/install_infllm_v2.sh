#!/usr/bin/env bash
# Build and install infllm_v2 (OpenBMB/infllmv2_cuda_impl) with CUTLASS patch.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCHES="${REPO_ROOT}/patches"
WORKDIR="${INFLLM_BUILD_DIR:-/tmp/infllmv2_cuda_impl}"

export PIP_ROOT_USER_ACTION=ignore
pip install -q packaging setuptools wheel psutil numpy

rm -rf "${WORKDIR}"
git clone --depth 1 https://github.com/OpenBMB/infllmv2_cuda_impl.git "${WORKDIR}"
cd "${WORKDIR}"
git submodule update --init --recursive
bash "${PATCHES}/fix_cutlass_submodule.sh"
python3 "${PATCHES}/check_cutlass_preprocessor_balance.py" \
  csrc/cutlass/include/cutlass/cuda_host_adapter.hpp
pip install --no-build-isolation -e .
python3 -c "import infllm_v2; print('infllm_v2 import OK')"
