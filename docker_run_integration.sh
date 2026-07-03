#!/usr/bin/env bash
# Full integration run: PR1 + PR2 overlay, unit tests + ruff + infllm_v2 + GPU validation.
set -uo pipefail
export PIP_ROOT_USER_ACTION=ignore
export DEBIAN_FRONTEND=noninteractive

PKG=/deliverable/minicpm_sala_stage1_pr
PR2="${PKG}/pr2"
PATCHES="${PKG}/patches"
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== Install Python + vLLM ==="
apt-get update -qq
apt-get install -y -qq python3 python3-pip git ninja-build > /dev/null
pip install -q "vllm==0.24.0" tblib pytest einops 2>&1 | tail -3

SITE="$(pip show vllm | awk '/^Location:/ {print $2}')"
VLLM_SITE="${SITE}/vllm"
log "Overlay PR2 merged stack into ${VLLM_SITE}"
cp "${PR2}/vllm/model_executor/models/minicpm_sala.py" "${VLLM_SITE}/model_executor/models/"
cp "${PR2}/vllm/model_executor/models/minicpm_sala_sparse_wiring.py" \
   "${VLLM_SITE}/model_executor/models/"
cp "${PR2}/vllm/v1/core/minicpm_sala_kv_cache_spec.py" "${VLLM_SITE}/v1/core/"
cp "${PR2}/vllm/v1/attention/backends/minicpm_sala_sparse.py" \
   "${VLLM_SITE}/v1/attention/backends/"
REG="${VLLM_SITE}/model_executor/models/registry.py"
if ! grep -q MiniCPMSALAForCausalLM "${REG}"; then
  sed -i '/MiniCPM3ForCausalLM/a\    "MiniCPMSALAForCausalLM": ("minicpm_sala", "MiniCPMSALAForCausalLM"),' "${REG}"
fi

log "=== Unit tests + ruff ==="
pip install -q ruff 2>&1 | tail -1
ruff check "${PKG}/vllm/model_executor/models/minicpm_sala.py" \
  "${PR2}/vllm" && ruff format --check "${PKG}/vllm/model_executor/models/minicpm_sala.py" \
  "${PR2}/vllm"
RUFF_EXIT=$?

rm -rf /tmp/minicpm_tests
mkdir -p /tmp/minicpm_tests/v1/core /tmp/minicpm_tests/v1/attention \
         /tmp/minicpm_tests/models/language/generation
cp "${PKG}/tests/models/language/generation/test_minicpm_sala_"*.py \
   /tmp/minicpm_tests/models/language/generation/
cp "${PR2}/tests/v1/core/test_minicpm_sala_"*.py /tmp/minicpm_tests/v1/core/
cp "${PR2}/tests/v1/attention/test_minicpm_sala_"*.py /tmp/minicpm_tests/v1/attention/
cd /tmp
python3 -m pytest --noconftest --rootdir=/tmp/minicpm_tests \
  /tmp/minicpm_tests/models/language/generation/ \
  /tmp/minicpm_tests/v1/core/ \
  /tmp/minicpm_tests/v1/attention/ \
  -v --tb=short -q 2>&1 | tail -12
BASELINE=$?

log "=== Install infllm_v2 ==="
pip install -q packaging setuptools wheel psutil numpy 2>&1 | tail -1
cd /tmp
rm -rf infllmv2_cuda_impl
git clone --depth 1 https://github.com/OpenBMB/infllmv2_cuda_impl.git 2>&1 | tail -1
cd infllmv2_cuda_impl
git submodule update --init --recursive 2>&1 | tail -2
bash "${PATCHES}/fix_cutlass_submodule.sh"
python3 "${PATCHES}/check_cutlass_preprocessor_balance.py" \
  csrc/cutlass/include/cutlass/cuda_host_adapter.hpp
CUTLASS_CHECK=$?
if [[ "${CUTLASS_CHECK}" -eq 0 ]]; then
  pip install -e . 2>&1 | tail -5
  python3 -c "from vllm.v1.attention.backends.minicpm_sala_sparse import INFLLM_V2_AVAILABLE; print('INFLLM_V2_AVAILABLE', INFLLM_V2_AVAILABLE)"
fi

log "=== GPU validation suite ==="
cd /tmp
bash "${PR2}/scripts/gpu_validation/run_all_gpu_validation.sh" 2>&1 | tee /tmp/gpu_validation.out | tail -40
GPU_EXIT=$?

log "=== Summary ==="
echo "ruff_exit=${RUFF_EXIT} baseline_exit=${BASELINE} cutlass_check=${CUTLASS_CHECK} gpu_exit=${GPU_EXIT} gpu_validation_log=/tmp/gpu_validation.out"
exit $(( RUFF_EXIT != 0 ? RUFF_EXIT : (BASELINE != 0 ? BASELINE : (CUTLASS_CHECK != 0 ? CUTLASS_CHECK : GPU_EXIT)) ))
