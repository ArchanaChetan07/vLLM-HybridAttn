#!/usr/bin/env bash
# Create a venv for the HF-reference side of parity (Step B).
#
# The reference modeling_minicpm_sala.py targets transformers==4.56 (its
# config's transformers_version) and imports 4.x-era private APIs
# (is_torch_fx_available, _prepare_4d_causal_attention_mask, ...) that
# transformers 5.x removed. Recent vLLM needs 5.x. Solution: parity runs
# the HF phase in a subprocess (MINICPM_SALA_HF_PYTHON) inside this venv,
# which sees the system torch + the main env's fla/infllm_v2/flash_attn
# shim, but pins its own transformers 4.56.
#
# Usage: bash scripts/setup_hf_reference_env.sh [ENV_DIR]
#   then: export MINICPM_SALA_HF_PYTHON=${ENV_DIR:-/workspace/hfenv}/bin/python
set -euo pipefail

ENV_DIR="${1:-/workspace/hfenv}"
MAIN_SITE="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

python3 -m venv --system-site-packages "${ENV_DIR}"
VENV_SITE="$("${ENV_DIR}/bin/python" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"
# Make the main env's extras (fla, infllm_v2, flash_attn shim) visible.
echo "${MAIN_SITE}" > "${VENV_SITE}/zz_main_env.pth"

"${ENV_DIR}/bin/pip" install -q "transformers==4.56.*" "tokenizers" "accelerate" 2>&1 | tail -2

# fla >= 2026 removed the `head_first` kwarg (raises if passed); the
# reference modeling file passes `head_first=False` explicitly, which is
# exactly the layout new fla mandates -- so tolerating (and dropping)
# head_first=False is semantically exact. Runtime monkeypatching does not
# stick (fla lazy-loads its op modules and rebinds names on
# materialization; a system sitecustomize also shadows venv ones), so
# patch the installed fla source in place, guarded and idempotent.
"${ENV_DIR}/bin/python" - <<'PY'
import fla.ops.simple_gla.chunk as chunk_mod
from pathlib import Path

p = Path(chunk_mod.__file__)
s = p.read_text()
old = """    if 'head_first' in kwargs:
        raise DeprecationWarning("""
new = """    if kwargs.pop('head_first', False):
        # Compat (vLLM-HybridAttn parity host): head_first=False is exactly
        # the mandatory [B, T, H, ...] layout, so tolerate it; only the
        # removed head-first layout is an error.
        raise DeprecationWarning("""
if old in s:
    p.write_text(s.replace(old, new, 1))
    print("fla chunk_simple_gla head_first guard patched:", p)
elif "kwargs.pop('head_first', False)" in s:
    print("fla already patched:", p)
else:
    print("fla accepts head_first natively or layout changed -- no patch needed")
PY

"${ENV_DIR}/bin/python" - <<'PY'
import transformers, torch
import flash_attn, fla, infllm_v2  # noqa: F401
from fla.ops.simple_gla import chunk_simple_gla  # noqa: F401
print("hfenv OK: transformers", transformers.__version__, "| torch", torch.__version__)
PY
echo "HF reference env at ${ENV_DIR} -- export MINICPM_SALA_HF_PYTHON=${ENV_DIR}/bin/python"
