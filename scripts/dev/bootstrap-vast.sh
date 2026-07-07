#!/usr/bin/env bash
# Bootstrap GPU environment on Vast.ai (overlay + infllm_v2 if missing).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
export PIP_ROOT_USER_ACTION=ignore
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${REPO_ROOT}/models}"
need_vllm=0; need_overlay=0; need_infllm=0
python3 -c "import vllm" >/dev/null 2>&1 || need_vllm=1
if python3 -c "import vllm" >/dev/null 2>&1; then
  SITE="$(pip show vllm 2>/dev/null | awk '/^Location:/ {print $2}')"
  [[ -f "${SITE}/vllm/v1/attention/backends/minicpm_sala_sparse.py" ]] || need_overlay=1
else need_overlay=1; fi
python3 -c "import infllm_v2" >/dev/null 2>&1 || need_infllm=1
if [[ "${need_vllm}" -eq 0 && "${need_overlay}" -eq 0 && "${need_infllm}" -eq 0 ]]; then
  log "Bootstrap: vllm, PR2 overlay, and infllm_v2 already present"; exit 0
fi
log "=== bootstrap-vast.sh ==="
if [[ "${need_vllm}" -eq 1 ]]; then
  log "Installing vllm==0.24.0 ..."
  pip install -q "vllm==0.24.0" tblib pytest einops ruff packaging setuptools wheel psutil numpy
fi
if [[ "${need_overlay}" -eq 1 ]]; then bash "${REPO_ROOT}/scripts/install_pr2_overlay.sh"; fi
if [[ "${need_infllm}" -eq 1 ]]; then
  if have_cmd nvidia-smi; then bash "${REPO_ROOT}/scripts/install_infllm_v2.sh"
  else warn "Skipping infllm_v2 build — no GPU"; fi
fi
log "Bootstrap complete"