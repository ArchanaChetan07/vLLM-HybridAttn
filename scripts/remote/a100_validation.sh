#!/usr/bin/env bash
# A100 final validation — Phase A through F
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/ArchanaChetan07/vLLM-HybridAttn.git}"
BRANCH="${BRANCH:-feature/minicpm-sala-sparse}"
WORKDIR="${WORKDIR:-/workspace/hybridattn}"
LOG_DIR="${LOG_DIR:-/tmp/phase2_logs}"
WEIGHTS="${MINICPM_SALA_WEIGHTS:-/workspace/models/openbmb/MiniCPM-SALA}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== PHASE A: PRE-FLIGHT ==="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
CAP=$(python3 - <<'PY'
import torch
print(torch.cuda.get_device_capability())
PY
)
log "device_capability=$CAP"
[[ "$CAP" == "(8, 0)" ]] || { log "FAIL: expected sm_80 (8,0), got $CAP"; exit 1; }

NVCC_LINE=$(nvcc --version 2>/dev/null | tail -1 || true)
log "nvcc: $NVCC_LINE"
echo "$NVCC_LINE" | grep -qE 'release 12\.' || { log "FAIL: need CUDA 12.x nvcc"; exit 1; }

df -BG / /workspace 2>/dev/null | head -5
AVAIL=$(df -BG /workspace 2>/dev/null | awk 'NR==2 {gsub(/G/,"",$4); print $4}')
log "workspace_free_gb=${AVAIL:-unknown}"
if [[ "${AVAIL:-0}" =~ ^[0-9]+$ ]] && (( AVAIL < 60 )); then
  log "WARN: less than 60GB free on /workspace"
fi

log "=== install vLLM 0.24.0 ==="
export PIP_ROOT_USER_ACTION=ignore
pip install -q "vllm==0.24.0" tblib pytest einops huggingface_hub transformers accelerate 2>&1 | tail -3
python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
assert torch.version.cuda and torch.version.cuda.startswith("12"), torch.version.cuda
PY

log "=== clone / update repo ==="
if [[ -d "$WORKDIR/.git" ]]; then
  git -C "$WORKDIR" fetch origin
  git -C "$WORKDIR" checkout "$BRANCH"
  git -C "$WORKDIR" pull --ff-only origin "$BRANCH"
else
  git clone --branch "$BRANCH" --depth 50 "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"
log "HEAD=$(git rev-parse --short HEAD) $(git log -1 --oneline)"

log "=== build infllm_v2 for sm_80 ==="
export CUDA_ARCH_LIST="8.0"
export FLASH_ATTN_CUDA_ARCHS="80"
export MAX_JOBS="$(nproc)"
bash scripts/install_infllm_v2.sh 2>&1 | tail -20

log "=== PR2 overlay ==="
bash scripts/install_pr2_overlay.sh

log "=== infllm_v2 import check ==="
python3 - <<'PY'
from infllm_v2 import infllmv2_attn_with_kvcache, infllmv2_attn_stage1, max_pooling_1d_varlen
for name, fn in [
    ("infllmv2_attn_with_kvcache", infllmv2_attn_with_kvcache),
    ("infllmv2_attn_stage1", infllmv2_attn_stage1),
    ("max_pooling_1d_varlen", max_pooling_1d_varlen),
]:
    print(name, "OK", callable(fn))
from vllm.v1.attention.backends.minicpm_sala_sparse import INFLLM_V2_AVAILABLE
print("INFLLM_V2_AVAILABLE", INFLLM_V2_AVAILABLE)
assert INFLLM_V2_AVAILABLE
PY

mkdir -p "$LOG_DIR"

log "=== PHASE B: Step 0 sparse LIVE ==="
python3 pr2/scripts/gpu_validation/assert_sparse_live.py | tee "$LOG_DIR/step0_sparse_live.log"
STEP0=${PIPESTATUS[0]}
[[ "$STEP0" -eq 0 ]] || { log "INVALID RUN: Step 0 failed"; exit 2; }

log "=== PHASE C: gated GPU validation ==="
bash pr2/scripts/gpu_validation/run_all_gpu_validation.sh 2>&1 | tee "$LOG_DIR/gated_run.log"
GATED_EXIT=${PIPESTATUS[0]}

log "gated_exit=$GATED_EXIT"
grep -E '^(  PASS|  FAIL|  SKIPPED)' "$LOG_DIR/gated_run.log" || true

log "=== PHASE D: weights check ==="
if [[ -d "$WEIGHTS" ]] && ls "$WEIGHTS"/*.safetensors "$WEIGHTS"/model*.safetensors 2>/dev/null | head -1 >/dev/null; then
  export MINICPM_SALA_WEIGHTS="$WEIGHTS"
  log "Running Step B parity with weights at $WEIGHTS"
  python3 pr2/scripts/gpu_validation/run_parity_sequential.py 2>&1 | tee "$LOG_DIR/step_b_parity.log"
  PARITY_EXIT=${PIPESTATUS[0]}
else
  log "Downloading MiniCPM-SALA weights (~19GB)..."
  mkdir -p "$(dirname "$WEIGHTS")"
  huggingface-cli download openbmb/MiniCPM-SALA --local-dir "$WEIGHTS" 2>&1 | tail -10
  export MINICPM_SALA_WEIGHTS="$WEIGHTS"
  python3 pr2/scripts/gpu_validation/run_parity_sequential.py 2>&1 | tee "$LOG_DIR/step_b_parity.log"
  PARITY_EXIT=${PIPESTATUS[0]}
fi

log "=== SUMMARY ==="
echo "step0=$STEP0 gated_exit=$GATED_EXIT parity_exit=${PARITY_EXIT:-skipped}"
echo "logs in $LOG_DIR"
