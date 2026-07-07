#!/usr/bin/env bash
# Shared helpers for repository dev automation (source, do not execute).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DEV_STATE_DIR="${REPO_ROOT}/.dev"
DEV_MARKER="${DEV_STATE_DIR}/init.stamp"
DEV_LOG="${DEV_STATE_DIR}/init.log"

mkdir -p "${DEV_STATE_DIR}"

log() { printf '[dev-init %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '[dev-init %s] WARN: %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
fail() { printf '[dev-init %s] ERROR: %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }

is_remote_ssh() {
  [[ -n "${SSH_CONNECTION:-}" || -n "${SSH_CLIENT:-}" || -n "${SSH_TTY:-}" ]]
}

is_cursor_remote() {
  [[ -n "${CURSOR_AGENT:-}" || -n "${VSCODE_GIT_IPC_HANDLE:-}" || "${TERM_PROGRAM:-}" == "vscode" ]]
}

have_cmd() { command -v "$1" >/dev/null 2>&1; }

config_fingerprint() {
  # Re-run init when these inputs change.
  cat "${REPO_ROOT}/requirements-dev.txt" \
      "${REPO_ROOT}/scripts/dev/init-dev-env.sh" 2>/dev/null \
    | sha256sum | awk '{print $1}'
}

should_skip_init() {
  [[ -f "${DEV_MARKER}" ]] || return 1
  [[ "$(cat "${DEV_MARKER}")" == "$(config_fingerprint)" ]]
}

mark_init_complete() {
  config_fingerprint >"${DEV_MARKER}"
}
