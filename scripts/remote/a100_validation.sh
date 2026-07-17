#!/usr/bin/env bash
# One-shot A100 validation host runner -- the reproduction command
# docs/VALIDATION_REPORT.md points at. Runs the CPU gates, builds
# infllm_v2, then the gated GPU suite (Steps 0-6) and HF parity (Step B).
#
# Usage:
#   export MINICPM_SALA_WEIGHTS=/path/to/openbmb/MiniCPM-SALA  # or hub id
#   bash scripts/remote/a100_validation.sh
#
# Logs land in ${LOG_DIR:-/tmp/phase2_logs}.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
LOG_DIR="${LOG_DIR:-/tmp/phase2_logs}"
mkdir -p "${LOG_DIR}"
log() { echo "[$(date +%H:%M:%S)] $*"; }
FAILURES=0

log "=== CPU gates (fresh clone) ==="
bash scripts/verify_fresh_clone.sh 2>&1 | tee "${LOG_DIR}/cpu_gates.log" \
  || FAILURES=$((FAILURES + 1))

log "=== Build infllm_v2 (sm_80) ==="
bash scripts/install_infllm_v2.sh 2>&1 | tee "${LOG_DIR}/infllm_build.log" \
  || FAILURES=$((FAILURES + 1))

log "=== Gated GPU suite (Steps 0-6) ==="
bash pr2/scripts/gpu_validation/run_all_gpu_validation.sh 2>&1 \
  | tee "${LOG_DIR}/gated_run.log" || FAILURES=$((FAILURES + 1))

if [[ -n "${MINICPM_SALA_WEIGHTS:-}" ]]; then
  log "=== Step B: HF vs vLLM parity (short prompts) ==="
  python3 pr2/scripts/gpu_validation/run_parity_sequential.py 2>&1 \
    | tee "${LOG_DIR}/step_b_parity.log" || FAILURES=$((FAILURES + 1))

  log "=== Step B-long: parity in the >=8192 sparse regime ==="
  MINICPM_SALA_LONG=1 python3 pr2/scripts/gpu_validation/run_parity_sequential.py \
    2>&1 | tee "${LOG_DIR}/step_b_parity_long.log" || FAILURES=$((FAILURES + 1))
else
  log "MINICPM_SALA_WEIGHTS not set -- skipping parity (Step B)."
fi

log "=== Overall ==="
if [[ ${FAILURES} -ne 0 ]]; then
  echo "a100_validation: ${FAILURES} stage(s) FAILED -- see ${LOG_DIR}/"
  exit 1
fi
echo "a100_validation: ALL stages PASS -- logs in ${LOG_DIR}/"
echo "Update docs/VALIDATION_REPORT.md with these results and log paths."
