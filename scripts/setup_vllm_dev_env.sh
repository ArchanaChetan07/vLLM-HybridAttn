#!/usr/bin/env bash
# Local vLLM dev env: pr2 overlay wins over vllm_ref for imports + pytest harness.
# Usage (Linux/WSL/A100):
#   source scripts/setup_vllm_dev_env.sh
#   python -m pytest tests/models/language/generation/test_minicpm_sala_long_context.py --collect-only
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -d "${REPO_ROOT}/vllm_ref" ]]; then
  export VLLM_REF_ROOT="${REPO_ROOT}/vllm_ref"
elif [[ -d "${REPO_ROOT}/../vllm_ref" ]]; then
  export VLLM_REF_ROOT="$(cd "${REPO_ROOT}/../vllm_ref" && pwd)"
else
  echo "WARN: vllm_ref not found; GPU harness tests will skip (set VLLM_REF_ROOT manually)"
fi

# pr2 must precede vllm_ref so overlay modules override site-packages / vllm_ref.
export PYTHONPATH="${REPO_ROOT}/pr2${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${VLLM_REF_ROOT:-}" ]]; then
  export PYTHONPATH="${PYTHONPATH}:${VLLM_REF_ROOT}"
fi

echo "PYTHONPATH=${PYTHONPATH}"
echo "VLLM_REF_ROOT=${VLLM_REF_ROOT:-<unset>}"
