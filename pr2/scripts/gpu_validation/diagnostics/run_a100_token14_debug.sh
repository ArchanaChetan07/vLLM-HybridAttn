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
export DEBUG_RUN_ID=post-fix-cf66727
export DEBUG_LOG_PATH=/workspace/hybridattn/debug-212a6e.log
rm -f "$DEBUG_LOG_PATH"
TRACES=pr2/scripts/gpu_validation/diagnostics/traces
mkdir -p "$TRACES"
python3 pr2/scripts/gpu_validation/diagnostics/gate1_hello_token14_parity.py 2>&1 | tee "$TRACES/hello_token14_cf66727.log"
echo "=== L1 decode branch layer1 ==="
grep '"layer_idx": 1' "$DEBUG_LOG_PATH" | grep decode || true
echo "=== L1 step14 signals ==="
grep '"layer_idx": 1' "$DEBUG_LOG_PATH" | grep '"hist_len": 14' || true
echo "=== debug log lines ==="
wc -l "$DEBUG_LOG_PATH" || true
echo "=== CAPTURE_DONE ==="
