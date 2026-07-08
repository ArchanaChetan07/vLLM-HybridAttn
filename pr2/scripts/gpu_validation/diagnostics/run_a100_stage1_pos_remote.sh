#!/usr/bin/env bash
set -euo pipefail

cd /workspace/hybridattn
git fetch origin feature/minicpm-sala-sparse
git reset --hard origin/feature/minicpm-sala-sparse
echo "HEAD=$(git rev-parse HEAD)"

bash scripts/install_pr2_overlay.sh

pkill -9 -f 'EngineCore|VLLM::' 2>/dev/null || true
sleep 2
nvidia-smi --query-gpu=memory.used --format=csv,noheader || true

python3 pr2/scripts/gpu_validation/assert_sparse_live.py

TS="$(date +%Y%m%d_%H%M%S)"
TR="pr2/scripts/gpu_validation/diagnostics/traces"
mkdir -p "$TR"

export MINICPM_SALA_WEIGHTS=/workspace/models/openbmb/MiniCPM-SALA
export MINICPM_SALA_PROMPT="Hello, my name is"
export MINICPM_SALA_MISMATCH_STEP=14
export MINICPM_SALA_POS0="${MINICPM_SALA_POS0:-6}"
export MINICPM_SALA_POS1="${MINICPM_SALA_POS1:-19}"
export MINICPM_SALA_DEBUG_GLA=1
export DEBUG_RUN_ID="stage1-pos-$TS"
export DEBUG_LOG_PATH="$TR/c1c2_split_pos_step14_${TS}.ndjson"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

python3 pr2/scripts/gpu_validation/diagnostics/gate1_recompute_c1_c2_split.py 2>&1 | tee "$TR/c1c2_split_pos_step14_${TS}.log"
python3 pr2/scripts/gpu_validation/diagnostics/gate1_hello_token14_parity.py 2>&1 | tee "$TR/hello_token14_stage1_${TS}.log" || true

echo "DONE_TS=$TS"
