#!/usr/bin/env bash
set -euo pipefail
VLLM_SITE=$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')
cp /workspace/hybridattn/pr2/vllm/model_executor/models/minicpm_sala.py "${VLLM_SITE}/model_executor/models/"
cp /workspace/hybridattn/pr2/vllm/model_executor/models/minicpm_sala_parity.py "${VLLM_SITE}/model_executor/models/"
REG="${VLLM_SITE}/model_executor/models/registry.py"
if ! grep -q minicpm_sala_parity "${REG}"; then
  cat >> "${REG}" <<'EOF'

# MiniCPM-SALA: native RMSNorm parity under enforce_eager.
try:
    import vllm.model_executor.models.minicpm_sala_parity as _  # noqa: F401
except ImportError:
    pass
EOF
fi
python3 <<'PY'
import vllm.model_executor.models.minicpm_sala_parity  # noqa: F401
from vllm.config import VllmConfig
print("patched", getattr(VllmConfig, "_minicpm_sala_parity_patched", False))
PY
