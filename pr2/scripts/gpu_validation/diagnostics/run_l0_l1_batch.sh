#!/usr/bin/env bash
# Run L0/L1 isolation probes for Hello and Briefly on A100.
set -eu
set -o pipefail
cd /workspace/hybridattn
git pull origin feature/minicpm-sala-sparse
bash scripts/install_pr2_overlay.sh >/dev/null 2>&1
export MINICPM_SALA_WEIGHTS=/workspace/models/openbmb/MiniCPM-SALA
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

run_probe() {
  local prompt="$1"
  echo "========== PROMPT: $prompt =========="
  export MINICPM_SALA_PROMPT="$prompt"
  python3 pr2/scripts/gpu_validation/diagnostics/gate1_layer0_engine_probe.py 2>&1 \
    | grep -E "^(prompt|layer0|FAIL)" || true
  python3 pr2/scripts/gpu_validation/diagnostics/gate1_l0_l1_isolation.py 2>&1 \
    | grep -E "^(prompt|layer0|l1_attn|l1_full|FAIL)" || true
  MINICPM_SALA_MODE=prompt_plus_t1 python3 pr2/scripts/gpu_validation/diagnostics/gate1_lightning_internals.py 2>&1 \
    | grep -E "^(mode|prompt|q_|gla|_forward)" || true
  python3 pr2/scripts/gpu_validation/diagnostics/gate1_prefill_plus_one.py 2>&1 \
    | grep -E "^(prompt|HF |vLLM)" || true
  python3 pr2/scripts/gpu_validation/diagnostics/gate1_two_token_logits.py 2>&1 \
    | grep -E "^(===|Hello|France|Briefly)" || true
}

run_probe "Hello, my name is"
run_probe "Briefly explain gravity:"
