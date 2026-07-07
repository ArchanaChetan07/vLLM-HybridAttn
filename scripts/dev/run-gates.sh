#!/usr/bin/env bash
# Layered quality gates. Usage: run-gates.sh [pr1|full|quick]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

GATE="${1:-quick}"

run_quick() {
  log "Gate: quick (static)"
  python3 -m py_compile \
    "${REPO_ROOT}/vllm/model_executor/models/minicpm_sala.py" \
    "${REPO_ROOT}/pr2/vllm/v1/attention/backends/minicpm_sala_sparse.py" \
    "${REPO_ROOT}/pr2/vllm/v1/core/minicpm_sala_kv_cache_spec.py" \
    "${REPO_ROOT}/pr2/vllm/model_executor/models/minicpm_sala.py"
  ruff check "${REPO_ROOT}/vllm/model_executor/models/minicpm_sala.py" "${REPO_ROOT}/pr2/vllm"
  ruff format --check "${REPO_ROOT}/vllm/model_executor/models/minicpm_sala.py" "${REPO_ROOT}/pr2/vllm"
}

run_pr1() {
  log "Gate: PR1 (docker)"
  bash "${REPO_ROOT}/docker_run_pr1.sh"
}

run_full() {
  log "Gate: full stack (docker integration)"
  bash "${REPO_ROOT}/docker_run_integration.sh"
}

case "${GATE}" in
  quick) run_quick ;;
  pr1) run_pr1 ;;
  full) run_full ;;
  *)
    echo "Usage: $0 [quick|pr1|full]" >&2
    exit 2
    ;;
esac

log "Gate '${GATE}' passed"
