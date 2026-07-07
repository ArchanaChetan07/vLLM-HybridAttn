#!/usr/bin/env bash
set -eu
export MINICPM_SALA_WEIGHTS="${MINICPM_SALA_WEIGHTS:-/workspace/models/openbmb/MiniCPM-SALA}"
SCRIPT="/workspace/hybridattn/pr2/scripts/gpu_validation/diagnostics/gate1_prefill_plus_one.py"
for P in "Hello, my name is" "The capital of France is" "Briefly explain gravity:"; do
  echo "===== $P ====="
  MINICPM_SALA_PROMPT="$P" python3 "$SCRIPT" 2>&1 | grep -E 'prompt_len|HF |vLLM prefill|match='
done
