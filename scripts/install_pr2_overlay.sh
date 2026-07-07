#!/usr/bin/env bash
# Overlay PR2 MiniCPM-SALA files into an installed vLLM site-packages tree.
# Usage: scripts/install_pr2_overlay.sh [VLLM_SITE]
# If VLLM_SITE is omitted, resolves from `pip show vllm`.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PR2="${REPO_ROOT}/pr2"

if [[ $# -ge 1 ]]; then
  VLLM_SITE="$1"
else
  SITE="$(pip show vllm | awk '/^Location:/ {print $2}')"
  VLLM_SITE="${SITE}/vllm"
fi

if [[ ! -d "${VLLM_SITE}" ]]; then
  echo "ERROR: vLLM site-packages not found at ${VLLM_SITE}" >&2
  exit 1
fi

echo "Overlaying PR2 into ${VLLM_SITE}"
cp "${PR2}/vllm/model_executor/models/minicpm_sala.py" \
   "${VLLM_SITE}/model_executor/models/"
cp "${PR2}/vllm/model_executor/models/minicpm_sala_sparse_wiring.py" \
   "${VLLM_SITE}/model_executor/models/"
cp "${PR2}/vllm/v1/core/minicpm_sala_kv_cache_spec.py" \
   "${VLLM_SITE}/v1/core/"
cp "${PR2}/vllm/v1/attention/backends/minicpm_sala_sparse.py" \
   "${VLLM_SITE}/v1/attention/backends/"

REG="${VLLM_SITE}/model_executor/models/registry.py"
if ! grep -q MiniCPMSALAForCausalLM "${REG}"; then
  sed -i '/MiniCPM3ForCausalLM/a\    "MiniCPMSALAForCausalLM": ("minicpm_sala", "MiniCPMSALAForCausalLM"),' \
    "${REG}"
fi

KVREG="${VLLM_SITE}/v1/kv_cache_spec_registry.py"
if [[ -f "${KVREG}" ]] && ! grep -q minicpm_sala_kv_cache_spec "${KVREG}" 2>/dev/null; then
  if ! grep -q "import vllm.v1.core.minicpm_sala_kv_cache_spec" "${KVREG}"; then
    echo "NOTE: ensure minicpm_sala_kv_cache_spec is imported for @register_kv_cache_spec"
  fi
fi

echo "PR2 overlay complete"
