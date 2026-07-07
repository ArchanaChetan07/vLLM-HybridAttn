from pathlib import Path

OLD = "from transformers.utils.import_utils import is_torch_fx_available"
NEW = """try:
    from transformers.utils.import_utils import is_torch_fx_available
except ImportError:
    def is_torch_fx_available():
        return False"""

base = Path("/workspace/.hf_home/modules/transformers_modules/MiniCPM_hyphen_SALA")
for p in base.glob("*/modeling_minicpm_sala.py"):
    text = p.read_text(encoding="utf-8")
    if OLD in text:
        p.write_text(text.replace(OLD, NEW), encoding="utf-8")
        print("patched", p)

weights_model = Path("/workspace/models/openbmb/MiniCPM-SALA/modeling_minicpm_sala.py")
if weights_model.is_file():
    text = weights_model.read_text(encoding="utf-8")
    if OLD in text:
        weights_model.write_text(text.replace(OLD, NEW), encoding="utf-8")
        print("patched", weights_model)
