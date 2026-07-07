#!/usr/bin/env bash
# Install optional git hooks (non-destructive; skips if already installed).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HOOKS_SRC="${SCRIPT_DIR}/hooks"
HOOKS_DST="${REPO_ROOT}/.git/hooks"

if [[ ! -d "${REPO_ROOT}/.git" ]]; then
  echo "Not a git repository — skipping hook install"
  exit 0
fi

install_hook() {
  local name="$1"
  if [[ -f "${HOOKS_DST}/${name}" && ! -f "${HOOKS_DST}/${name}.hybridattn.bak" ]]; then
    cp "${HOOKS_DST}/${name}" "${HOOKS_DST}/${name}.hybridattn.bak"
  fi
  cp "${HOOKS_SRC}/${name}" "${HOOKS_DST}/${name}"
  chmod +x "${HOOKS_DST}/${name}"
  echo "Installed hook: ${name}"
}

install_hook pre-commit
install_hook pre-push
