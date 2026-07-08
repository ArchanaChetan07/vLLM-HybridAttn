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
    print(f"ERROR: {path} contains null bytes (UTF-16?) - refusing overlay", file=sys.stderr)
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
  "${PR2}/vllm/model_executor/models/minicpm_sala_parity.py"
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
cp "${PR2}/vllm/model_executor/models/minicpm_sala_parity.py" \
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

if ! grep -q minicpm_sala_parity "${REG}"; then
  cat >> "${REG}" <<'EOF'

# MiniCPM-SALA: native RMSNorm parity under enforce_eager.
try:
    import vllm.model_executor.models.minicpm_sala_parity as _  # noqa: F401
except ImportError:
    pass
EOF
fi

python3 - <<'PY'
import inspect
import sys

import vllm.model_executor.models.minicpm_sala_parity as _parity  # noqa: F401
from vllm.config import VllmConfig

if not getattr(VllmConfig, "_minicpm_sala_parity_patched", False):
    print("ERROR: minicpm_sala_parity did not install VllmConfig hook", file=sys.stderr)
    sys.exit(1)

from vllm.model_executor.models import minicpm_sala as m

src = inspect.getsource(m._minicpm_sala_lightning_forward_prefix)
if "chunk_simple_gla" not in src:
    print("ERROR: overlaid minicpm_sala lacks fla lightning prefill", file=sys.stderr)
    sys.exit(1)
fwd = inspect.getsource(m.MiniCPMSALALightningAttention._forward)
if "torch.zeros_like(q)" not in fwd:
    print("ERROR: overlaid minicpm_sala lacks HF-effective RoPE policy", file=sys.stderr)
    sys.exit(1)
print("Overlay parity kernels: OK")
PY

echo "PR2 overlay complete"
