#!/usr/bin/env bash
# One A100 session: confirm o_proj, test fp32 fix, token-2, parity harness.
set -euo pipefail

cd /workspace/hybridattn
export MINICPM_SALA_WEIGHTS="${MINICPM_SALA_WEIGHTS:-/workspace/models/openbmb/MiniCPM-SALA}"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

echo "=== GPU clean check ==="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv || true
pkill -9 -f 'VLLM::EngineCore' 2>/dev/null || true
pkill -9 -f gate1_l0 2>/dev/null || true
sleep 3
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "gpu_free_mib=${FREE}"
if [ "${FREE}" -lt 60000 ]; then
  echo "FAIL: GPU not clean enough (need ~60GiB free); aborting."
  exit 1
fi

git pull origin feature/minicpm-sala-sparse
bash scripts/install_pr2_overlay.sh

echo "=== Stage 1: o_proj confirm (bf16 RowParallel) ==="
unset MINICPM_SALA_FP32_O_PROJ
MINICPM_SALA_PROMPT='Briefly explain gravity:' \
  python3 pr2/scripts/gpu_validation/diagnostics/gate1_l0_oproj_confirm.py

echo "=== Stage 2: fp32 o_proj bisect ==="
export MINICPM_SALA_FP32_O_PROJ=1
MINICPM_SALA_PROMPT='Briefly explain gravity:' \
  python3 pr2/scripts/gpu_validation/diagnostics/gate1_l0_oproj_confirm.py

echo "=== Token-2 probes ==="
for P in 'Hello, my name is' 'Briefly explain gravity:'; do
  MINICPM_SALA_PROMPT="$P" python3 pr2/scripts/gpu_validation/diagnostics/gate1_prefill_plus_one.py \
    2>&1 | grep -E 'prompt_len|HF t2|vLLM t2|match='
done

echo "=== check_logprobs_close harness ==="
python3 pr2/scripts/gpu_validation/run_parity_sequential.py 2>&1 | tail -25
