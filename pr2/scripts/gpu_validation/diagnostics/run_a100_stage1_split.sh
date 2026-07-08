#!/usr/bin/env bash
set -euo pipefail
cd /workspace/hybridattn
pkill -9 -f 'EngineCore|VLLM::' 2>/dev/null || true
sleep 2
git fetch origin feature/minicpm-sala-sparse
git reset --hard origin/feature/minicpm-sala-sparse
echo "HEAD=$(git rev-parse HEAD)"
bash scripts/install_pr2_overlay.sh
export MINICPM_SALA_WEIGHTS=/workspace/models/openbmb/MiniCPM-SALA
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export MINICPM_SALA_DEBUG_GLA=1
export DEBUG_RUN_ID=stage1-c1c2-split
export DEBUG_LOG_PATH=/workspace/hybridattn/debug-212a6e.log
export MINICPM_SALA_MISMATCH_STEP=14
rm -f "$DEBUG_LOG_PATH"
TRACES=pr2/scripts/gpu_validation/diagnostics/traces
mkdir -p "$TRACES"
echo "=== assert_sparse_live ==="
python3 pr2/scripts/gpu_validation/assert_sparse_live.py 2>&1 | tee "$TRACES/assert_sparse_live_stage1.log"
echo "=== gate1_recompute_c1_c2_split ==="
python3 pr2/scripts/gpu_validation/diagnostics/gate1_recompute_c1_c2_split.py 2>&1 | tee "$TRACES/recompute_c1c2_split.log" || true
echo "=== gate1_hello_token14_parity ==="
python3 pr2/scripts/gpu_validation/diagnostics/gate1_hello_token14_parity.py 2>&1 | tee "$TRACES/hello_token14_stage1.log" || true
echo "=== STAGE1_DONE ==="
