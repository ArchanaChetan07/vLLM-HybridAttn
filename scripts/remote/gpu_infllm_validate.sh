#!/usr/bin/env bash
# Optional GPU host runner: overlay PR2, build infllm_v2, run full validation.
# Usage (from repo root):
#   REPO_ROOT=$PWD bash scripts/remote/gpu_infllm_validate.sh
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PR2="${REPO_ROOT}/pr2"
LOG="${GPU_VALIDATION_LOG:-/tmp/gpu_validation_full.log}"

export PIP_ROOT_USER_ACTION=ignore
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${REPO_ROOT}/models}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/pip_cache}"
export TMPDIR="${TMPDIR:-/tmp}"

exec > >(tee "${LOG}") 2>&1

echo "=== Repo: ${REPO_ROOT} ==="
echo "=== Overlay PR2 ==="
bash "${REPO_ROOT}/scripts/install_pr2_overlay.sh"

echo "=== Build infllm_v2 ==="
bash "${REPO_ROOT}/scripts/install_infllm_v2.sh"

echo "=== GPU validation suite ==="
bash "${PR2}/scripts/gpu_validation/run_all_gpu_validation.sh"

echo "=== DONE log at ${LOG} ==="
