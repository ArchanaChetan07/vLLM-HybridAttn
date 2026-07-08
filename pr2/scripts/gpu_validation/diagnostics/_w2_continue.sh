#!/usr/bin/env bash
# Continue A100 W2 after history_hit proven via EngineCore dense-path logs.
set -uo pipefail
cd /workspace/hybridattn
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export MINICPM_SALA_WEIGHTS="${MINICPM_SALA_WEIGHTS:-/workspace/models/openbmb/MiniCPM-SALA}"
export MINICPM_SALA_PROMPT="${MINICPM_SALA_PROMPT:-Hello, my name is}"
export MINICPM_SALA_MISMATCH_STEP=14
export MINICPM_SALA_LOG_DENSE_PATH=1
TRACES=pr2/scripts/gpu_validation/diagnostics/traces
mkdir -p "$TRACES"
date -u | tee -a "$TRACES/w2_start.txt"
echo "=== HISTORY NOTE ===" | tee -a "$TRACES/w2_master.log"
echo "EngineCore dense-path showed history_hit on decode; client-side monkeypatch is false miss." | tee -a "$TRACES/w2_master.log"

pkill -9 -f EngineCore 2>/dev/null || true
pkill -9 -f 'VLLM::' 2>/dev/null || true
sleep 2
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader || true

run() {
  local label="$1"
  local cmd="$2"
  local log="$3"
  echo "=== ${label} ===" | tee -a "$TRACES/w2_master.log"
  set +e
  bash -lc "$cmd" 2>&1 | tee "$log" | tee -a "$TRACES/w2_master.log"
  local ec=${PIPESTATUS[0]}
  set -e
  echo "EXIT_${label}=${ec}" | tee -a "$TRACES/w2_master.log"
  return 0
}

run "L0_L1_layer_in" \
  "python3 pr2/scripts/gpu_validation/diagnostics/gate1_l0_l1_layer_in_pos.py" \
  "$TRACES/l0_l1_layer_in_pos_w2.log"

run "per_position_v_parity" \
  "python3 pr2/scripts/gpu_validation/diagnostics/gate1_per_position_v_parity.py" \
  "$TRACES/per_position_v_parity_w2.log"

run "briefly_token2" \
  "python3 pr2/scripts/gpu_validation/diagnostics/gate1_briefly_only_token2.py" \
  "$TRACES/briefly_token2_w2.log"

run "two_token_logits" \
  "python3 pr2/scripts/gpu_validation/diagnostics/gate1_two_token_logits.py" \
  "$TRACES/two_token_logits_w2.log"

run "dense_kv_replay" \
  "python3 pr2/scripts/gpu_validation/diagnostics/gate1_decode_kv_slot_cpu_replay.py" \
  "$TRACES/dense_kv_replay_w2.log"

run "token14" \
  "python3 pr2/scripts/gpu_validation/diagnostics/gate1_hello_token14_parity.py" \
  "$TRACES/hello_token14_w2.log"

run "lightning_state" \
  "python3 pr2/scripts/gpu_validation/diagnostics/gate1_lightning_state_compare.py" \
  "$TRACES/lightning_state_w2.log"

run "parity_sequential" \
  "python3 pr2/scripts/gpu_validation/run_parity_sequential.py" \
  "$TRACES/run_parity_sequential_w2.log"

run "logprobs_hybrid" \
  "python3 -m pytest tests/models/language/generation/test_minicpm_sala.py -m hybrid_model -q" \
  "$TRACES/check_logprobs_close_w2.log"

date -u | tee "$TRACES/w2_end.txt"
echo "=== W2 CONTINUE DONE ===" | tee -a "$TRACES/w2_master.log"
grep -E 'token14:|PASS|FAIL|first>|GREEN|RED|peak=|1420|7670|APPEND|hist_|L0 |L1 ' \
  "$TRACES"/l0_l1_layer_in_pos_w2.log \
  "$TRACES"/per_position_v_parity_w2.log \
  "$TRACES"/briefly_token2_w2.log \
  "$TRACES"/hello_token14_w2.log \
  "$TRACES"/two_token_logits_w2.log 2>/dev/null | head -n 120 || true
