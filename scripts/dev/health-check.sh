#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/checks.sh"
EXIT=0
note_fail() { warn "$*"; EXIT=1; }
log "=== repository health check ==="
check_os || true
check_disk || true
check_git || true
check_python || true
check_gpu || true
check_pr1_boundary || note_fail "PR1 boundary check failed"
check_shell_scripts_lf || true
check_cuda_torch_vllm || true
if have_cmd ruff; then
  log "Running ruff check (scoped) ..."
  if ruff check "${REPO_ROOT}/vllm/model_executor/models/minicpm_sala.py" "${REPO_ROOT}/pr2/vllm" 2>/dev/null; then
    log "ruff: OK"
  else
    note_fail "ruff check failed"
  fi
else
  warn "ruff not installed — run make init"
fi
if [[ -f "${REPO_ROOT}/.dev/init.stamp" ]]; then
  log "Init stamp present: $(cat "${REPO_ROOT}/.dev/init.stamp")"
else
  warn "Dev environment not initialized — run: make init"
fi
log "=== health check done (exit ${EXIT}) ==="
exit "${EXIT}"