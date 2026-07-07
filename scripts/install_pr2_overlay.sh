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

verify_utf8_py() {
  python3 - "$1" <<'PY'
import sys
path = sys.argv[1]
raw = open(path, "rb").read()
if b"\x00" in raw:
    print(f"ERROR: {path} contains null bytes (UTF-16?) — refusing overlay", file=sys.stderr)
    sys.exit(1)
try:
    raw.decode("utf-8")
except UnicodeDecodeError as e:
    print(f"ERROR: {path} is not valid UTF-8: {e}", file=sys.stderr)
    sys.exit(1)
PY
}

OVERLAY_FILES=(
  "${PR2}/vllm/model_executor/models/minicpm_sala.py"
  "${PR2}/vllm/model_executor/models/minicpm_sala_sparse_wiring.py"
  "${PR2}/vllm/v1/core/minicpm_sala_kv_cache_spec.py"
  "${PR2}/vllm/v1/attention/backends/minicpm_sala_sparse.py"
)

for src in "${OVERLAY_FILES[@]}"; do
  if [[ ! -f "${src}" ]]; then
    echo "ERROR: overlay source missing: ${src}" >&2
    exit 1
  fi
  verify_utf8_py "${src}"
done

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
