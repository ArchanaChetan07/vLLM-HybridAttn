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

echo "=== Engine norm lmhead + chunked A/B ==="
MINICPM_SALA_PROMPT='Briefly explain gravity:' MINICPM_SALA_ENGINE_AB=1 \
  python3 pr2/scripts/gpu_validation/diagnostics/gate1_engine_norm_lmhead.py 2>&1 \
  | tee pr2/scripts/gpu_validation/diagnostics/traces/engine_norm_lmhead.log

echo "=== Engine vs manual logits A/B (retry) ==="
MINICPM_SALA_PROMPT='Briefly explain gravity:' MINICPM_SALA_ENGINE_AB=1 \
  python3 pr2/scripts/gpu_validation/diagnostics/gate1_engine_vs_manual_logits.py 2>&1 \
  | tee pr2/scripts/gpu_validation/diagnostics/traces/engine_vs_manual_ab.log
