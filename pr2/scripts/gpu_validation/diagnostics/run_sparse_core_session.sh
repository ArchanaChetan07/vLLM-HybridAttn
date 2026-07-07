#!/usr/bin/env bash
# Focused sparse_core + o_proj replay session (Briefly, clean GPU).
set -euo pipefail
cd /workspace/hybridattn
export MINICPM_SALA_WEIGHTS="${MINICPM_SALA_WEIGHTS:-/workspace/models/openbmb/MiniCPM-SALA}"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "gpu_free_mib=${FREE}"
if [ "${FREE}" -lt 60000 ]; then
  echo "FAIL: GPU not clean"; exit 1
fi

git pull origin feature/minicpm-sala-sparse
bash scripts/install_pr2_overlay.sh

echo "=== sparse_core bisect (fixed head_dim) ==="
MINICPM_SALA_PROMPT='Briefly explain gravity:' \
  python3 pr2/scripts/gpu_validation/diagnostics/gate1_l0_sparse_core.py 2>&1 \
  | grep -E 'prompt=|peak=|gpu_'

echo "=== full bisect ==="
MINICPM_SALA_PROMPT='Briefly explain gravity:' \
  python3 pr2/scripts/gpu_validation/diagnostics/gate1_l0_sparse_bisect.py 2>&1 \
  | grep -E 'prompt=|peak=|first_stage|all_stages'
