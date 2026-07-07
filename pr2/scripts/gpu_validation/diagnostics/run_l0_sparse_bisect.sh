#!/usr/bin/env bash
set -eu
set -o pipefail
cd /workspace/hybridattn
git pull origin feature/minicpm-sala-sparse
bash scripts/install_pr2_overlay.sh >/dev/null 2>&1
export MINICPM_SALA_WEIGHTS=/workspace/models/openbmb/MiniCPM-SALA

run() {
  echo "========== $* =========="
  "$@" 2>&1 | grep -E '^(prompt|embed|norm|q |k |attn|layer0|first|all_)' || true
}

run python3 pr2/scripts/gpu_validation/diagnostics/gate1_l0_sparse_bisect.py
run env MINICPM_SALA_PROMPT='Briefly explain gravity:' python3 pr2/scripts/gpu_validation/diagnostics/gate1_l0_sparse_bisect.py
run python3 pr2/scripts/gpu_validation/diagnostics/gate1_layer0_engine_probe.py
run env MINICPM_SALA_PROMPT='Briefly explain gravity:' python3 pr2/scripts/gpu_validation/diagnostics/gate1_layer0_engine_probe.py
