#!/usr/bin/env bash
# Environment validation checks (source from init-dev-env.sh).
set -euo pipefail

check_os() {
  log "OS: $(uname -srmo 2>/dev/null || uname -a)"
}

check_disk() {
  local avail_gb
  avail_gb="$(df -BG "${REPO_ROOT}" 2>/dev/null | awk 'NR==2 {gsub(/G/,"",$4); print $4}' || echo 0)"
  log "Disk free at repo root: ${avail_gb} GB"
  if [[ "${avail_gb}" =~ ^[0-9]+$ ]] && (( avail_gb < 10 )); then
    warn "Less than 10 GB free — model weights and Docker layers may fail."
  fi
}

check_git() {
  if ! have_cmd git; then
    warn "git not found"
    return 0
  fi
  log "Git: $(git --version)"
  log "Branch: $(git -C "${REPO_ROOT}" branch --show-current 2>/dev/null || echo unknown)"
  if ! git -C "${REPO_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    warn "Not inside a git work tree"
    return 0
  fi
  if [[ -z "$(git -C "${REPO_ROOT}" config user.email 2>/dev/null || true)" ]]; then
    warn "git user.email not set (commits will fail until configured)"
  fi
  if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain 2>/dev/null)" ]]; then
    warn "Working tree has uncommitted changes"
  fi
}

check_python() {
  if ! have_cmd python3; then
    warn "python3 not found — install Python 3.10+"
    return 0
  fi
  log "Python: $(python3 --version 2>&1)"
}

check_gpu() {
  if have_cmd nvidia-smi; then
    log "GPU: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1)"
    return 0
  fi
  if is_remote_ssh; then
    warn "nvidia-smi not found on remote host — GPU validation steps will be skipped"
  else
    log "GPU: not detected (CPU-only session)"
  fi
}

check_cuda_torch_vllm() {
  if ! have_cmd python3; then
    return 0
  fi
  python3 - <<'PY' || warn "CUDA/PyTorch/vLLM check failed (venv may not be ready yet)"
import sys
try:
    import torch
except ImportError:
    print("torch: not installed")
    sys.exit(0)
print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda device: {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"compute capability: sm_{cap[0]}{cap[1]}")
try:
    import vllm
    print(f"vllm: {vllm.__version__}")
except ImportError:
    print("vllm: not installed")
PY
}

check_pr1_boundary() {
  local model="${REPO_ROOT}/vllm/model_executor/models/minicpm_sala.py"
  if [[ -f "${model}" ]]; then
    if grep -qE '^from .*minicpm_sala_sparse|^import .*minicpm_sala_sparse' "${model}"; then
      warn "PR1 boundary violation: sparse imports in minicpm_sala.py"
    else
      log "PR1 boundary: no sparse imports in root minicpm_sala.py"
    fi
  fi
}

check_shell_scripts_lf() {
  local bad=0
  while IFS= read -r f; do
    if grep -q $'\r' "$f" 2>/dev/null; then
      warn "CRLF in shell script: ${f#${REPO_ROOT}/}"
      bad=1
    fi
  done < <(find "${REPO_ROOT}" -name '*.sh' -not -path '*/.git/*' 2>/dev/null)
  if [[ "${bad}" -eq 0 ]]; then
    log "Shell scripts: LF line endings OK"
  fi
}
