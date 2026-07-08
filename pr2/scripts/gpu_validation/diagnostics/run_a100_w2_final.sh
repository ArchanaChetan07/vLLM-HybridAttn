#!/usr/bin/env bash
# A100 W2 one-shot: fp32 dense flash unification (530fbc5+) → token14 closure.
# Prereq: /workspace/hybridattn on feature/minicpm-sala-sparse @ 530fbc5+,
#         overlay installed, weights at MINICPM_SALA_WEIGHTS.
set -euo pipefail

REPO=/workspace/hybridattn
cd "${REPO}"

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export MINICPM_SALA_WEIGHTS="${MINICPM_SALA_WEIGHTS:-/workspace/models/openbmb/MiniCPM-SALA}"
export MINICPM_SALA_PROMPT="${MINICPM_SALA_PROMPT:-Hello, my name is}"
export MINICPM_SALA_MISMATCH_STEP=14
export MINICPM_SALA_LOG_DENSE_PATH=1

TRACES="${REPO}/pr2/scripts/gpu_validation/diagnostics/traces"
mkdir -p "${TRACES}"

echo "=== W2 final @ $(git -C "${REPO}" rev-parse --short HEAD 2>/dev/null || echo unknown) ===" \
  | tee "${TRACES}/w2_final_start.txt"
date -u | tee -a "${TRACES}/w2_final_start.txt"

echo "=== clean GPU ==="
pkill -9 -f 'EngineCore|VLLM::' 2>/dev/null || true
sleep 2
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader || true

echo "=== overlay sanity ==="
python3 -c "
import vllm.v1.attention.backends.minicpm_sala_sparse as m
print('sparse_file', m.__file__)
print('hist_max', m._DENSE_HISTORY_DECODE_MAX_SEQ)
assert m._DENSE_HISTORY_DECODE_MAX_SEQ == 64
assert hasattr(m, '_flash_dense_varlen_causal')
print('OVERLAY_OK')
"

echo "=== 1/4 assert_sparse_live ==="
python3 pr2/scripts/gpu_validation/assert_sparse_live.py \
  2>&1 | tee "${TRACES}/assert_sparse_live_w2_final.log"

echo "=== 2/4 per-position v parity ==="
python3 pr2/scripts/gpu_validation/diagnostics/gate1_per_position_v_parity.py \
  2>&1 | tee "${TRACES}/per_position_v_parity_w2_final.log"

echo "=== 3/4 token14 parity ==="
python3 pr2/scripts/gpu_validation/diagnostics/gate1_hello_token14_parity.py \
  2>&1 | tee "${TRACES}/hello_token14_w2_final.log"

echo "=== 4/4 lightning state compare (step 14) ==="
python3 pr2/scripts/gpu_validation/diagnostics/gate1_lightning_state_compare.py \
  2>&1 | tee "${TRACES}/lightning_state_w2_final.log"

date -u | tee "${TRACES}/w2_final_end.txt"
echo "=== W2 FINAL DONE — summary ==="
grep -E 'token14:|LIVE|PASS|FAIL|GREEN|RED|MISMATCH|overall_peak|first>|append=|hist_ok|delta_after|step.?14' \
  "${TRACES}"/assert_sparse_live_w2_final.log \
  "${TRACES}"/per_position_v_parity_w2_final.log \
  "${TRACES}"/hello_token14_w2_final.log \
  "${TRACES}"/lightning_state_w2_final.log 2>/dev/null | head -n 80
