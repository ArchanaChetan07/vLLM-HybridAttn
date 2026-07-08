#!/usr/bin/env bash
# A100 W2 verification for short-seq decode-hidden continuity fix (49d3b0c+).
set -euo pipefail
cd /workspace/hybridattn
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export MINICPM_SALA_WEIGHTS="${MINICPM_SALA_WEIGHTS:-/workspace/models/openbmb/MiniCPM-SALA}"
export MINICPM_SALA_PROMPT="${MINICPM_SALA_PROMPT:-Hello, my name is}"
export MINICPM_SALA_MISMATCH_STEP=14
export MINICPM_SALA_LOG_DENSE_PATH=1
TRACES=pr2/scripts/gpu_validation/diagnostics/traces
mkdir -p "$TRACES"
date -u | tee "$TRACES/w2_start.txt"
echo "=== clean GPU ==="
pkill -9 -f 'EngineCore|VLLM::' 2>/dev/null || true
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader || true

echo "=== overlay sanity ==="
python3 -c "import vllm.v1.attention.backends.minicpm_sala_sparse as m; print('sparse_file', m.__file__); print('hist_max', m._DENSE_HISTORY_DECODE_MAX_SEQ); assert m._DENSE_HISTORY_DECODE_MAX_SEQ == 64; print('OVERLAY_OK')"

echo "=== assert_sparse_live ==="
python3 pr2/scripts/gpu_validation/assert_sparse_live.py 2>&1 | tee "$TRACES/assert_sparse_live_w2.log"

echo "=== dense history hit probe ==="
python3 pr2/scripts/gpu_validation/diagnostics/gate1_dense_history_hit.py 2>&1 | tee "$TRACES/dense_history_hit_w2.log"

echo "=== L0/L1 layer_in pos probe ==="
python3 pr2/scripts/gpu_validation/diagnostics/gate1_l0_l1_layer_in_pos.py 2>&1 | tee "$TRACES/l0_l1_layer_in_pos_w2.log"

echo "=== per-position v parity ==="
python3 pr2/scripts/gpu_validation/diagnostics/gate1_per_position_v_parity.py 2>&1 | tee "$TRACES/per_position_v_parity_w2.log"

echo "=== no-regression briefly token2 ==="
python3 pr2/scripts/gpu_validation/diagnostics/gate1_briefly_only_token2.py 2>&1 | tee "$TRACES/briefly_token2_w2.log"

echo "=== no-regression two-token logits ==="
python3 pr2/scripts/gpu_validation/diagnostics/gate1_two_token_logits.py 2>&1 | tee "$TRACES/two_token_logits_w2.log"

echo "=== dense KV CPU replay ==="
python3 pr2/scripts/gpu_validation/diagnostics/gate1_decode_kv_slot_cpu_replay.py 2>&1 | tee "$TRACES/dense_kv_replay_w2.log" || true

echo "=== token14 parity ==="
python3 pr2/scripts/gpu_validation/diagnostics/gate1_hello_token14_parity.py 2>&1 | tee "$TRACES/hello_token14_w2.log"

echo "=== lightning state compare ==="
python3 pr2/scripts/gpu_validation/diagnostics/gate1_lightning_state_compare.py 2>&1 | tee "$TRACES/lightning_state_w2.log" || true

echo "=== run_parity_sequential ==="
python3 pr2/scripts/gpu_validation/run_parity_sequential.py 2>&1 | tee "$TRACES/run_parity_sequential_w2.log" || true

echo "=== hybrid_model logprobs ==="
python3 -m pytest tests/models/language/generation/test_minicpm_sala.py -m hybrid_model -q 2>&1 | tee "$TRACES/check_logprobs_close_w2.log" || true

date -u | tee "$TRACES/w2_end.txt"
echo "=== W2 DONE ==="
grep -E 'token14:|LIVE|PASS|FAIL|append=|hist_ok=|hist_miss=|first>|GREEN|RED|MISMATCH|peak=|OVERLAY' \
  "$TRACES"/hello_token14_w2.log \
  "$TRACES"/dense_history_hit_w2.log \
  "$TRACES"/l0_l1_layer_in_pos_w2.log \
  "$TRACES"/per_position_v_parity_w2.log \
  "$TRACES"/briefly_token2_w2.log \
  "$TRACES"/assert_sparse_live_w2.log \
  "$TRACES"/w2_master.log 2>/dev/null | head -n 100
