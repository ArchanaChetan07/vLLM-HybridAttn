#!/usr/bin/env bash
# Fresh-clone CPU verification: overlay the PR2 stack onto an installed
# vLLM and run every CPU-runnable gate. No GPU, no weights, no network
# beyond pip. This is the command docs/VALIDATION_REPORT.md points at.
#
# Usage: scripts/verify_fresh_clone.sh
# Expects: vllm==0.24.0 already installed (pip install "vllm==0.24.0"),
# plus pytest + einops (installed here if missing).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
log() { echo "[$(date +%H:%M:%S)] $*"; }
FAILURES=0

log "=== 0. PR1/PR2 lightning drift gate (pure stdlib) ==="
python3 scripts/check_pr1_pr2_lightning_sync.py || FAILURES=$((FAILURES + 1))

log "=== 1. Ensure test deps ==="
pip install -q pytest einops tblib 2>&1 | tail -1

log "=== 2. Install PR2 overlay into site-packages ==="
bash scripts/install_pr2_overlay.sh || FAILURES=$((FAILURES + 1))

log "=== 3. PR1 CPU unit tests ==="
PR1_TESTS=$(mktemp -d)
mkdir -p "${PR1_TESTS}/models/language/generation"
for f in tests/models/language/generation/test_minicpm_sala_*.py; do
  case "$(basename "${f}")" in
    test_minicpm_sala.py|test_minicpm_sala_long_context.py) continue ;;
  esac
  cp "${f}" "${PR1_TESTS}/models/language/generation/"
done
python3 -m pytest --noconftest --rootdir="${PR1_TESTS}" \
  "${PR1_TESTS}/models/language/generation/" -q || FAILURES=$((FAILURES + 1))

log "=== 4. PR2 CPU unit tests ==="
python3 -m pytest --noconftest --rootdir=pr2/tests pr2/tests/ -q \
  || FAILURES=$((FAILURES + 1))

log "=== Summary ==="
if [[ ${FAILURES} -ne 0 ]]; then
  echo "verify_fresh_clone: ${FAILURES} gate(s) FAILED"
  exit 1
fi
echo "verify_fresh_clone: all CPU gates PASS"
echo "Next (GPU host): bash scripts/install_infllm_v2.sh && \\"
echo "  bash pr2/scripts/gpu_validation/run_all_gpu_validation.sh"
