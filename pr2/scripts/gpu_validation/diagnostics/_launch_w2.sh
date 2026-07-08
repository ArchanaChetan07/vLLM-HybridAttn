#!/usr/bin/env bash
set -euo pipefail
cd /workspace/hybridattn
pkill -9 -f EngineCore 2>/dev/null || true
pkill -9 -f 'VLLM::' 2>/dev/null || true
# Do not kill ourselves: only sibling gate1 scripts outside our tree of control
sleep 2
nohup bash pr2/scripts/gpu_validation/diagnostics/_w2_remote.sh \
  > pr2/scripts/gpu_validation/diagnostics/traces/w2_master.log 2>&1 < /dev/null &
echo "LAUNCH_PID=$!"
sleep 1
pgrep -af '_w2_remote' || true
