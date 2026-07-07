#!/usr/bin/env bash
set -euo pipefail
cd /workspace/hybridattn
export MINICPM_SALA_WEIGHTS="${MINICPM_SALA_WEIGHTS:-/workspace/models/openbmb/MiniCPM-SALA}"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "gpu_free_mib=${FREE}"
[ "${FREE}" -ge 60000 ] || { echo "FAIL: dirty GPU"; exit 1; }
mkdir -p pr2/scripts/gpu_validation/diagnostics/traces
git pull origin feature/minicpm-sala-sparse
bash scripts/install_pr2_overlay.sh
MINICPM_SALA_PROMPT='Briefly explain gravity:' \
  python3 pr2/scripts/gpu_validation/diagnostics/gate1_dense_flash_compare.py 2>&1 | tee \
  pr2/scripts/gpu_validation/diagnostics/traces/dense_flash_compare_run.log
