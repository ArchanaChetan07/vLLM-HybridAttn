#!/usr/bin/env bash
# One-command Cursor <-> Vast.ai workflow orchestrator.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
VAST_HOST="${VAST_SSH_HOST:-vast4090}"
REMOTE_REPO="${VAST_REMOTE_REPO:-/workspace/hybridattn}"
MODE="${1:-local}"
print_cursor_instructions() {
  log "=== Cursor connection ==="
  log "1. Remote-SSH: Connect to Host -> ${VAST_HOST}"
  log "2. Open folder: ${REMOTE_REPO}"
}
run_remote_pipeline() {
  cd "${REPO_ROOT}"
  log "=== connect_cursor.sh --remote ==="
  bash "${SCRIPT_DIR}/sync-repo.sh" || warn "sync-repo skipped"
  bash "${SCRIPT_DIR}/bootstrap-vast.sh" || warn "bootstrap skipped"
  bash "${SCRIPT_DIR}/health_check.sh" || bash "${SCRIPT_DIR}/health-check.sh" || warn "health check issues"
  log "=== Remote pipeline complete ==="
}
run_local_preflight() {
  log "=== connect_cursor.sh (local preflight) ==="
  ssh -o BatchMode=yes -o ConnectTimeout=20 "${VAST_HOST}" "echo SSH_OK && test -d '${REMOTE_REPO}'" || fail "SSH or repo path failed"
  ssh "${VAST_HOST}" "cd '${REMOTE_REPO}' && bash scripts/dev/connect_cursor.sh --remote" || warn "Remote pipeline incomplete"
  print_cursor_instructions
  log "=== PASS: SSH preflight ==="
}
case "${MODE}" in
  --remote|remote) run_remote_pipeline ;;
  *) run_local_preflight ;;
esac