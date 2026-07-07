#!/usr/bin/env bash
# Environment validation checks (source from init-dev-env.sh / health_check.sh).
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
  if have_cmd df; then
    log "Disk /tmp: $(df -h /tmp 2>/dev/null | awk 'NR==2 {print $4 " free of " $2}')"
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
  local upstream
  upstream="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref '@{u}' 2>/dev/null || true)"
  if [[ -n "${upstream}" ]]; then
    git -C "${REPO_ROOT}" fetch origin --quiet 2>/dev/null || true
    local behind ahead
    behind="$(git -C "${REPO_ROOT}" rev-list --count HEAD.."${upstream}" 2>/dev/null || echo 0)"
    ahead="$(git -C "${REPO_ROOT}" rev-list --count "${upstream}"..HEAD 2>/dev/null || echo 0)"
    log "Git sync: behind=${behind} ahead=${ahead} (tracking ${upstream})"
    if [[ "${behind}" != "0" ]]; then
      warn "Repository is ${behind} commit(s) behind ${upstream} — run scripts/dev/sync-repo.sh"
    fi
  fi
}

check_ssh() {
  if [[ -n "${SSH_CONNECTION:-}" ]]; then
    log "SSH session: ${SSH_CONNECTION}"
    return 0
  fi
  if is_remote_ssh; then
    log "SSH client detected (${SSH_CLIENT:-unknown})"
    return 0
  fi
  log "SSH: local session (not over SSH)"
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

check_weights() {
  local weights="${MINICPM_SALA_WEIGHTS:-${HF_HOME:-${REPO_ROOT}/models}/MiniCPM-SALA}"
  if [[ -d "${weights}" ]] && [[ -f "${weights}/config.json" ]]; then
    log "Weights: found at ${weights}"
    return 0
  fi
  warn "Weights not found at ${weights} (Steps B/C parity will be skipped)"
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
    import triton
    print(f"triton: {triton.__version__}")
except ImportError:
    print("triton: not installed")
try:
    import vllm
    print(f"vllm: {vllm.__version__}")
except ImportError:
    print("vllm: not installed")
try:
    import infllm_v2  # noqa: F401
    print("infllm_v2: import OK")
except ImportError:
    print("infllm_v2: not installed")
PY
}

check_pr1_pr2_lightning_sync() {
  local pr1="${REPO_ROOT}/vllm/model_executor/models/minicpm_sala.py"
  local pr2="${REPO_ROOT}/pr2/vllm/model_executor/models/minicpm_sala.py"
  if [[ ! -f "${pr1}" || ! -f "${pr2}" ]]; then
    warn "Lightning sync check skipped: missing PR1 or PR2 minicpm_sala.py"
    return
  fi
  local markers=(
    _minicpm_sala_lightning_forward_prefix
    prefix_fn=_minicpm_sala_lightning_forward_prefix
    self.tp_slope.float()
  )
  local bad=0
  for m in "${markers[@]}"; do
    if grep -qF "${m}" "${pr2}" && ! grep -qF "${m}" "${pr1}"; then
      warn "PR1/PR2 lightning drift: PR2 has ${m} but PR1 does not"
      bad=1
    fi
  done
  if [[ "${bad}" -eq 0 ]]; then
    log "PR1/PR2 lightning shared logic: in sync"
  fi
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
  check_pr1_pr2_lightning_sync
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

check_overlay_scripts() {
  for f in install_pr2_overlay.sh install_infllm_v2.sh verify_fresh_clone.sh; do
    if [[ -x "${REPO_ROOT}/scripts/${f}" ]] || [[ -f "${REPO_ROOT}/scripts/${f}" ]]; then
      log "Script present: scripts/${f}"
    else
      warn "Missing: scripts/${f} (fresh-clone install will fail)"
    fi
  done
}

check_repository() {
  log "Repository root: ${REPO_ROOT}"
  if [[ "${REPO_ROOT}" == "/workspace/hybridattn" ]]; then
    log "Repository path: Vast.ai standard (/workspace/hybridattn)"
  fi
  check_overlay_scripts
}
