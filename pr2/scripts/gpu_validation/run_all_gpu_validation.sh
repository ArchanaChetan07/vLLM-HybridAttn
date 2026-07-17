#!/usr/bin/env bash
# Master GPU validation runner -- runs every GPU validation step in
# order, stops at the first hard failure (except the known,
# non-blocking sm_80+ wall in step 2/4, which is reported but doesn't
# abort the run since later steps don't depend on it succeeding), and
# produces one consolidated pass/fail summary.
#
# Usage: bash run_all_gpu_validation.sh
# For step 5 (multi-GPU), set MULTI_GPU_NPROC to your GPU count first,
# e.g.: MULTI_GPU_NPROC=2 bash run_all_gpu_validation.sh

set -uo pipefail  # NOT -e: we want to continue past the known sm_80+
                   # failure and still run later independent steps.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS=()

run_step() {
    local name="$1"
    local cmd="$2"
    echo ""
    echo "=================================================================="
    echo "  $name"
    echo "=================================================================="
    if eval "$cmd"; then
        RESULTS+=("PASS: $name")
    else
        local exit_code=$?
        RESULTS+=("FAIL (exit $exit_code): $name")
    fi
}

echo "Real GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")' 2>/dev/null || echo 'torch not available')"

run_step "Step 0: Assert sparse backend LIVE (not dense fallback)" \
    "python3 $SCRIPT_DIR/assert_sparse_live.py"
# NOTE: Step 0 failing means every later "sparse" result would silently
# exercise the dense fallback. Later steps still run so the full picture
# is captured, but treat their results as dense-only until Step 0 is green.

run_step "Step 1: Diagnostic (imports, platform, backend resolution)" \
    "python3 $SCRIPT_DIR/step1_diagnostic.py"

run_step "Step 2: Lightning Attention kernel dispatch (real, single layer)" \
    "python3 $SCRIPT_DIR/step2_kernel_dispatch.py"
# NOTE: Step 2 is EXPECTED to fail on sub-Ampere hardware (confirmed
# real sm_80+ floor, see docs/minicpm_sala_known_limitations.md) -- a
# failure here matching "compute capability >= 80" is not a regression,
# it's the known hardware constraint. Steps 3+ do not depend on step 2
# passing, so the run continues regardless.

run_step "Step 3: Real paged-cache gather test (production block_size)" \
    "python3 $SCRIPT_DIR/step3_real_gather_test.py"

run_step "Step 4: End-to-end sparse path past dense_len" \
    "python3 $SCRIPT_DIR/step4_sparse_e2e_test.py"
# NOTE: also expected to hit the sm_80+ floor on sub-Ampere hardware,
# for the same underlying reason as step 2 (the sparse regime's kernel
# call goes through the same compute-capability-gated path).

if [ -n "${MULTI_GPU_NPROC:-}" ] && [ "$MULTI_GPU_NPROC" -ge 2 ]; then
    run_step "Step 5: Real multi-GPU TP sharding (nccl, $MULTI_GPU_NPROC GPUs)" \
        "torchrun --nproc_per_node=$MULTI_GPU_NPROC $SCRIPT_DIR/step5_multi_gpu_tp_test.py"
else
    echo ""
    echo "Skipping Step 5 (multi-GPU TP): set MULTI_GPU_NPROC>=2 to run it."
    RESULTS+=("SKIPPED: Step 5 (set MULTI_GPU_NPROC>=2 to enable)")
fi

run_step "Step 6: Mixed dense/sparse batch invariance" \
    "python3 $SCRIPT_DIR/step6_mixed_batch_invariance.py"

if [ -n "${MINICPM_SALA_WEIGHTS:-}" ]; then
    run_step "Step B: HF vs vLLM parity (sequential, short prompts)" \
        "python3 $SCRIPT_DIR/run_parity_sequential.py"
else
    echo ""
    echo "Skipping Step B (parity): set MINICPM_SALA_WEIGHTS to the"
    echo "openbmb/MiniCPM-SALA weights path (or hub id) to run it."
    RESULTS+=("SKIPPED: Step B parity (set MINICPM_SALA_WEIGHTS to enable)")
fi

echo ""
echo "=================================================================="
echo "  SUMMARY"
echo "=================================================================="
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
echo ""
echo "Full baseline: run docker_run_integration.sh (66 tests + ruff)"
echo ""
echo "Update docs/minicpm_sala_known_limitations.md with these real"
echo "results before treating any of them as confirmed -- same pattern"
echo "as every prior real-hardware session in that document's history."
