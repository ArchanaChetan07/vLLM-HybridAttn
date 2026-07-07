#!/usr/bin/env bash
set -euo pipefail
cd /workspace/hybridattn
export MINICPM_SALA_WEIGHTS="${MINICPM_SALA_WEIGHTS:-/workspace/models/openbmb/MiniCPM-SALA}"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "gpu_free_mib=${FREE}"
pkill -9 -f 'VLLM::EngineCore|EngineCore' 2>/dev/null || true
sleep 2
git pull origin feature/minicpm-sala-sparse
bash scripts/install_pr2_overlay.sh
mkdir -p pr2/scripts/gpu_validation/diagnostics/traces

echo "=== HF vs vLLM greedy t1 (Briefly prompt-only) ==="
python3 pr2/scripts/gpu_validation/diagnostics/gate1_briefly_only_token2.py 2>&1 \
  | tee pr2/scripts/gpu_validation/diagnostics/traces/briefly_t1_latest.log

echo "=== 32-layer stack bisect (prompt-only) ==="
MINICPM_SALA_PROMPT='Briefly explain gravity:' MINICPM_SALA_MODE=prompt \
  python3 pr2/scripts/gpu_validation/diagnostics/gate1_stack_bisect.py 2>&1 \
  | tee pr2/scripts/gpu_validation/diagnostics/traces/stack_bisect_briefly_prompt.log

echo "=== 32-layer stack bisect (Hello prompt-only sanity) ==="
MINICPM_SALA_PROMPT='Hello, my name is' MINICPM_SALA_MODE=prompt \
  python3 pr2/scripts/gpu_validation/diagnostics/gate1_stack_bisect.py 2>&1 \
  | tee pr2/scripts/gpu_validation/diagnostics/traces/stack_bisect_hello_prompt.log
