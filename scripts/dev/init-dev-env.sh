#!/usr/bin/env bash
# Idempotent developer environment initialization for local and Cursor Remote SSH.
# Safe to run repeatedly; skips work when config fingerprint is unchanged.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
# shellcheck source=lib/checks.sh
source "${SCRIPT_DIR}/lib/checks.sh"

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

main() {
  if [[ "${FORCE}" -eq 0 ]] && should_skip_init; then
    log "Environment already initialized (stamp: $(cat "${DEV_MARKER}")). Use --force to re-run."
    exit 0
  fi

  exec > >(tee -a "${DEV_LOG}") 2>&1
  log "=== vLLM-HybridAttn dev init ==="
  log "Repo: ${REPO_ROOT}"

  if is_remote_ssh; then
    log "Session: Remote SSH ($(echo "${SSH_CONNECTION:-unknown}" | awk '{print $1" -> "$3}'))"
  else
    log "Session: local"
  fi
  if is_cursor_remote; then
    log "Editor: Cursor/VS Code remote workspace detected"
  fi

  check_os
  check_disk
  check_git
  check_python
  check_gpu
  check_pr1_boundary
  check_shell_scripts_lf

  setup_venv
  install_dev_deps
  check_cuda_torch_vllm
  install_git_hooks
  write_env_summary

  mark_init_complete
  log "=== init complete ==="
  log "Next: make health   |   make gate-pr1   |   make gate-full"
}

setup_venv() {
  local venv="${REPO_ROOT}/.venv"
  if [[ ! -d "${venv}" ]]; then
    log "Creating .venv ..."
    python3 -m venv "${venv}"
  else
    log ".venv exists"
  fi
  # shellcheck disable=SC1091
  source "${venv}/bin/activate"
  python -m pip install -q --upgrade pip wheel
}

install_dev_deps() {
  local req="${REPO_ROOT}/requirements-dev.txt"
  if [[ ! -f "${req}" ]]; then
    warn "requirements-dev.txt missing — skipping pip install"
    return 0
  fi
  log "Installing dev dependencies (idempotent pip) ..."
  pip install -q -r "${req}"
}

install_git_hooks() {
  if [[ -x "${SCRIPT_DIR}/install-git-hooks.sh" ]]; then
    bash "${SCRIPT_DIR}/install-git-hooks.sh" || warn "git hooks install skipped"
  fi
}

write_env_summary() {
  cat >"${DEV_STATE_DIR}/environment.json" <<EOF
{
  "repo_root": "${REPO_ROOT}",
  "remote_ssh": $(is_remote_ssh && echo true || echo false),
  "cursor_remote": $(is_cursor_remote && echo true || echo false),
  "python": "$(python3 --version 2>&1 | tr -d '\n')",
  "initialized_at": "$(date -Iseconds)"
}
EOF
  log "Wrote ${DEV_STATE_DIR}/environment.json"
}

main "$@"
