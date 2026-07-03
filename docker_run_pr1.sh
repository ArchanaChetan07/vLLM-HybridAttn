#!/usr/bin/env bash
# PR1-only gate: model imports and PR1 tests pass with zero PR2 files present.
set -uo pipefail
export PIP_ROOT_USER_ACTION=ignore
export DEBIAN_FRONTEND=noninteractive

PKG=/deliverable/minicpm_sala_stage1_pr
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== Install Python + vLLM ==="
apt-get update -qq
apt-get install -y -qq python3 python3-pip git > /dev/null
pip install -q "vllm==0.24.0" tblib pytest einops 2>&1 | tail -3

SITE="$(pip show vllm | awk '/^Location:/ {print $2}')"
VLLM_SITE="${SITE}/vllm"
log "Overlay PR1 model only into ${VLLM_SITE}"
cp "${PKG}/vllm/model_executor/models/minicpm_sala.py" "${VLLM_SITE}/model_executor/models/"
REG="${VLLM_SITE}/model_executor/models/registry.py"
if ! grep -q MiniCPMSALAForCausalLM "${REG}"; then
  sed -i '/MiniCPM3ForCausalLM/a\    "MiniCPMSALAForCausalLM": ("minicpm_sala", "MiniCPMSALAForCausalLM"),' "${REG}"
fi

log "=== Verify PR2 modules absent ==="
for f in \
  "${VLLM_SITE}/v1/attention/backends/minicpm_sala_sparse.py" \
  "${VLLM_SITE}/v1/core/minicpm_sala_kv_cache_spec.py" \
  "${VLLM_SITE}/model_executor/models/minicpm_sala_sparse_wiring.py"; do
  if [[ -f "${f}" ]]; then
    echo "FAIL: PR2 file still present: ${f}"
    exit 1
  fi
done
echo "PR2 files absent: OK"

log "=== Import PR1 model ==="
python3 -c "
from vllm.model_executor.models.minicpm_sala import MiniCPMSALAForCausalLM
print('PR1-ONLY IMPORT: OK', MiniCPMSALAForCausalLM)
"

log "=== Ruff (PR1 model only) ==="
pip install -q ruff 2>&1 | tail -1
ruff check "${PKG}/vllm/model_executor/models/minicpm_sala.py"
ruff format --check "${PKG}/vllm/model_executor/models/minicpm_sala.py"
RUFF_EXIT=$?

log "=== PR1 unit tests ==="
rm -rf /tmp/minicpm_pr1_tests
mkdir -p /tmp/minicpm_pr1_tests/models/language/generation
cp "${PKG}/tests/models/language/generation/test_minicpm_sala_"*.py \
   /tmp/minicpm_pr1_tests/models/language/generation/
cd /tmp
python3 -m pytest --noconftest --rootdir=/tmp/minicpm_pr1_tests \
  /tmp/minicpm_pr1_tests/models/language/generation/ \
  -v --tb=short -q 2>&1 | tail -15
PR1_EXIT=$?

log "=== Summary ==="
echo "ruff_exit=${RUFF_EXIT} pr1_tests_exit=${PR1_EXIT}"
exit $(( RUFF_EXIT != 0 ? RUFF_EXIT : PR1_EXIT ))
