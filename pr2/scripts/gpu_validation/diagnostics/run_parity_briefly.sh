#!/usr/bin/env bash
set -euo pipefail
cd /workspace/hybridattn
export MINICPM_SALA_WEIGHTS="${MINICPM_SALA_WEIGHTS:-/workspace/models/openbmb/MiniCPM-SALA}"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
pkill -9 -f 'VLLM::EngineCore|EngineCore' 2>/dev/null || true
sleep 2
git pull origin feature/minicpm-sala-sparse
bash scripts/install_pr2_overlay.sh
mkdir -p pr2/scripts/gpu_validation/diagnostics/traces

echo "=== gate1_engine_manual_prefill_diff ==="
MINICPM_SALA_PROMPT='Briefly explain gravity:' \
  python3 pr2/scripts/gpu_validation/diagnostics/gate1_engine_manual_prefill_diff.py 2>&1 \
  | tee pr2/scripts/gpu_validation/diagnostics/traces/engine_manual_prefill_diff.log

echo "=== gate1_briefly_only_token2 ==="
python3 pr2/scripts/gpu_validation/diagnostics/gate1_briefly_only_token2.py 2>&1 \
  | tee pr2/scripts/gpu_validation/diagnostics/traces/gate1_briefly_only_token2.log
