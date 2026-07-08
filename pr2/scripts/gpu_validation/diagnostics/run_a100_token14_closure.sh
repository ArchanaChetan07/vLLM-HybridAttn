#!/usr/bin/env bash
# Single A100 verification pass for lightning GLA idx14 fix (a5b9db4) + long-context.
# Prereq: remote at 2b1dd20+, overlay installed, weights at MINICPM_SALA_WEIGHTS.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

export MINICPM_SALA_WEIGHTS="${MINICPM_SALA_WEIGHTS:-/workspace/models/openbmb/MiniCPM-SALA}"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export MINICPM_SALA_PROMPT="${MINICPM_SALA_PROMPT:-Hello, my name is}"
export MINICPM_SALA_MISMATCH_STEP=14

TRACES="${REPO_ROOT}/pr2/scripts/gpu_validation/diagnostics/traces"
mkdir -p "${TRACES}"

echo "=== clean GPU ==="
nvidia-smi || true
pkill -9 -f 'EngineCore|VLLM::' 2>/dev/null || true

echo "=== assert_sparse_live (must be LIVE) ==="
python3 pr2/scripts/gpu_validation/assert_sparse_live.py 2>&1 | tee "${TRACES}/assert_sparse_live_post_fix.log"

echo "=== PRIMARY: token14 parity (0-indexed idx14) ==="
python3 pr2/scripts/gpu_validation/diagnostics/gate1_hello_token14_parity.py 2>&1 | tee "${TRACES}/hello_token14_post_fix.log"

echo "=== L1 GLA state incremental vs one-shot ==="
python3 pr2/scripts/gpu_validation/diagnostics/gate1_lightning_state_compare.py 2>&1 | tee "${TRACES}/lightning_state_step14_post_fix.log"

echo "=== incremental vs oneshot per step ==="
python3 pr2/scripts/gpu_validation/diagnostics/gate1_decode_incremental_vs_oneshot.py 2>&1 | tee "${TRACES}/incremental_vs_oneshot_post_fix.log"

echo "=== no-regression: prefill ==="
python3 pr2/scripts/gpu_validation/diagnostics/gate1_briefly_only_token2.py 2>&1 | tee "${TRACES}/briefly_token2_post_fix.log"
python3 pr2/scripts/gpu_validation/diagnostics/gate1_two_token_logits.py 2>&1 | tee "${TRACES}/two_token_logits_post_fix.log"

echo "=== dense KV replay ==="
python3 pr2/scripts/gpu_validation/diagnostics/gate1_decode_kv_slot_cpu_replay.py 2>&1 | tee "${TRACES}/dense_kv_replay_post_fix.log"

echo "=== end-to-end parity ==="
python3 pr2/scripts/gpu_validation/run_parity_sequential.py 2>&1 | tee "${TRACES}/run_parity_sequential_post_fix.log"

echo "=== PR1 hybrid logprobs (if harness available) ==="
if python3 -m pytest tests/models/language/generation/test_minicpm_sala.py -m hybrid_model --collect-only -q 2>/dev/null; then
  python3 -m pytest tests/models/language/generation/test_minicpm_sala.py -m hybrid_model -q 2>&1 | tee "${TRACES}/check_logprobs_close_short_post_fix.log" || true
  python3 -m pytest tests/models/language/generation/test_minicpm_sala_long_context.py -m hybrid_model -q 2>&1 | tee "${TRACES}/long_context_post_fix.log" || true
else
  echo "SKIP: pytest harness not collectable (install vllm_ref + pr2 overlay)"
fi

echo "=== DONE — review ${TRACES}/*_post_fix.log ==="
