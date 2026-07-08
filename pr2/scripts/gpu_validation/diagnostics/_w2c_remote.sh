#!/usr/bin/env bash
# Focused A100 verify after fp32 dense-history restore (a280419).
set -uo pipefail
cd /workspace/hybridattn
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export MINICPM_SALA_WEIGHTS=/workspace/models/openbmb/MiniCPM-SALA
export MINICPM_SALA_LOG_DENSE_PATH=1
TRACES=pr2/scripts/gpu_validation/diagnostics/traces
mkdir -p "$TRACES"
date -u | tee "$TRACES/w2c_start.txt"
pkill -9 -f EngineCore 2>/dev/null || true
sleep 2

run() {
  local label="$1"; local cmd="$2"; local log="$3"
  echo "=== ${label} ===" | tee -a "$TRACES/w2c_master.log"
  set +e
  bash -lc "$cmd" >"$log" 2>&1
  local ec=$?
  set -e
  echo "EXIT_${label}=${ec}" | tee -a "$TRACES/w2c_master.log"
  grep -E 'token14:|PASS|FAIL|first>|overall_peak|L0_|t1=|GREEN|RED|LIVE|history_hit|1420|7670|delta_after' "$log" | head -n 40 | tee -a "$TRACES/w2c_master.log" || true
}

: > "$TRACES/w2c_master.log"
run assert_sparse "python3 pr2/scripts/gpu_validation/assert_sparse_live.py" "$TRACES/assert_sparse_live_w2c.log"
run l0_out "python3 pr2/scripts/gpu_validation/diagnostics/gate1_l0_layer_out_pos.py" "$TRACES/l0_layer_out_pos_w2c.log"
run briefly "python3 pr2/scripts/gpu_validation/diagnostics/gate1_briefly_only_token2.py" "$TRACES/briefly_token2_w2c.log"
run denskv "python3 pr2/scripts/gpu_validation/diagnostics/gate1_decode_kv_slot_cpu_replay.py" "$TRACES/dense_kv_replay_w2c.log"
run token14 "python3 pr2/scripts/gpu_validation/diagnostics/gate1_hello_token14_parity.py" "$TRACES/hello_token14_w2c.log"
date -u | tee "$TRACES/w2c_end.txt"
echo "=== W2C DONE ===" | tee -a "$TRACES/w2c_master.log"
