#!/usr/bin/env bash
set -euo pipefail
cd /workspace/hybridattn
export MINICPM_SALA_WEIGHTS="${MINICPM_SALA_WEIGHTS:-/workspace/models/openbmb/MiniCPM-SALA}"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "gpu_free_mib=${FREE}"
[ "${FREE}" -ge 60000 ] || { echo "FAIL: dirty GPU"; exit 1; }
pkill -9 -f 'VLLM::EngineCore|vllm.entrypoints|EngineCore' 2>/dev/null || true
sleep 2
git pull origin feature/minicpm-sala-sparse
bash scripts/install_pr2_overlay.sh
mkdir -p pr2/scripts/gpu_validation/diagnostics/traces

echo "=== qkv isolation (HF-only) ==="
MINICPM_SALA_DEVICE=cuda python3 pr2/scripts/gpu_validation/diagnostics/gate1_qkv_isolate_hf_only.py \
  2>&1 | tee pr2/scripts/gpu_validation/diagnostics/traces/qkv_isolate_run.log

echo "=== layer-0 bisect Hello ==="
MINICPM_SALA_PROMPT='Hello, my name is' \
  python3 pr2/scripts/gpu_validation/diagnostics/gate1_l0_sparse_bisect.py 2>&1 \
  | tee pr2/scripts/gpu_validation/diagnostics/traces/l0_bisect_hello_postfix.log

echo "=== layer-0 bisect Briefly ==="
MINICPM_SALA_PROMPT='Briefly explain gravity:' \
  python3 pr2/scripts/gpu_validation/diagnostics/gate1_l0_sparse_bisect.py 2>&1 \
  | tee pr2/scripts/gpu_validation/diagnostics/traces/l0_bisect_briefly_postfix.log

echo "=== token-2 probes ==="
python3 pr2/scripts/gpu_validation/diagnostics/gate1_two_token_logits.py 2>&1 \
  | tee pr2/scripts/gpu_validation/diagnostics/traces/token2_postfix.log

echo "=== check_logprobs_close harness ==="
python3 pr2/scripts/gpu_validation/run_parity_sequential.py 2>&1 \
  | tee pr2/scripts/gpu_validation/diagnostics/traces/parity_postfix.log

echo "=== sparse LIVE ==="
python3 pr2/scripts/gpu_validation/assert_sparse_live.py 2>&1 \
  | tee pr2/scripts/gpu_validation/diagnostics/traces/sparse_live_postfix.log

echo "DONE Stage-4 verification"
