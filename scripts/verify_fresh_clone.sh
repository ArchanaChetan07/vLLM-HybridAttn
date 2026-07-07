#!/usr/bin/env bash
# Fresh-clone verification: install vLLM, overlay PR2, run CPU tests + ruff.
# Run from repo root inside Linux (Docker or WSL). No manual site-packages edits.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PIP_ROOT_USER_ACTION=ignore

echo "=== git status (must be clean or only expected untracked) ==="
if command -v git >/dev/null 2>&1; then
  git status --short | head -20
else
  echo "SKIP: git not installed (optional in minimal containers)"
fi

echo "=== install vLLM 0.24.0 ==="
pip install -q "vllm==0.24.0" tblib pytest einops ruff

echo "=== overlay PR2 ==="
bash "${REPO_ROOT}/scripts/install_pr2_overlay.sh"

echo "=== ruff ==="
ruff check "${REPO_ROOT}/vllm/model_executor/models/minicpm_sala.py" \
  "${REPO_ROOT}/pr2/vllm"
ruff format --check "${REPO_ROOT}/vllm/model_executor/models/minicpm_sala.py" \
  "${REPO_ROOT}/pr2/vllm"

echo "=== pytest (CPU unit tests) ==="
rm -rf /tmp/minicpm_verify
mkdir -p /tmp/minicpm_verify/v1/core /tmp/minicpm_verify/v1/attention \
         /tmp/minicpm_verify/models/language/generation
# CPU-only: exclude GPU/HF harness tests (tests.models.registry, hf_runner).
for f in \
  test_minicpm_sala_schedule.py \
  test_minicpm_sala_decay_sign.py \
  test_minicpm_sala_mamba_helpers.py \
  test_minicpm_sala_fused_residual.py; do
  cp "${REPO_ROOT}/tests/models/language/generation/${f}" \
     /tmp/minicpm_verify/models/language/generation/
done
cp "${REPO_ROOT}/pr2/tests/v1/core/test_minicpm_sala_"*.py \
   /tmp/minicpm_verify/v1/core/
cp "${REPO_ROOT}/pr2/tests/v1/attention/test_minicpm_sala_"*.py \
   /tmp/minicpm_verify/v1/attention/
cd /tmp
python3 -m pytest --noconftest --rootdir=/tmp/minicpm_verify \
  /tmp/minicpm_verify/models/language/generation/ \
  /tmp/minicpm_verify/v1/core/ \
  /tmp/minicpm_verify/v1/attention/ \
  -v --tb=short -q

echo "=== PR1 must not import sparse wiring ==="
if grep -E 'from \.minicpm_sala_sparse_wiring|import minicpm_sala_sparse' \
    "${REPO_ROOT}/vllm/model_executor/models/minicpm_sala.py"; then
  echo "FAIL: sparse imports in PR1 minicpm_sala.py"
  exit 1
fi
echo "PR1 sparse imports: none"

echo "=== PASS: fresh-clone CPU verification ==="
