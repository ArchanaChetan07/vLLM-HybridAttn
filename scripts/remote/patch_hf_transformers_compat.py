from pathlib import Path

OLD_IMPORT = "from transformers.utils.import_utils import is_torch_fx_available"
NEW_IMPORT = (
    "from transformers.utils.import_utils import is_torch_available as "
    "is_torch_fx_available"
)

OLD_PRETRAINED = "    _supports_flash_attn_2 = True\n    _supports_sdpa = True"
NEW_PRETRAINED = (
    "    _supports_flash_attn = True\n"
    "    _supports_flash_attn_2 = True\n"
    "    _supports_sdpa = True"
)

OLD_CLASS = "class MiniCPMSALAForCausalLM(MiniCPMSALAPreTrainedModel):\n    _tied_weights_keys"
NEW_CLASS = (
    "class MiniCPMSALAForCausalLM(MiniCPMSALAPreTrainedModel):\n"
    "    _supports_flash_attn = True\n"
    "    _tied_weights_keys"
)

paths = list(
    Path("/workspace/.hf_home/modules/transformers_modules/MiniCPM_hyphen_SALA").glob(
        "*/modeling_minicpm_sala.py"
    )
)
paths.append(Path("/workspace/models/openbmb/MiniCPM-SALA/modeling_minicpm_sala.py"))

seen: set[str] = set()
for p in paths:
    if not p.is_file():
        continue
    key = str(p.resolve())
    if key in seen:
        continue
    seen.add(key)
    text = p.read_text(encoding="utf-8")
    changed = False
    if OLD_IMPORT in text and NEW_IMPORT not in text:
        text = text.replace(OLD_IMPORT, NEW_IMPORT)
        changed = True
        print("patched import", p)
    if OLD_PRETRAINED in text and "_supports_flash_attn = True" not in text:
        text = text.replace(OLD_PRETRAINED, NEW_PRETRAINED)
        changed = True
        print("patched pretrained", p)
    if OLD_CLASS in text and "_supports_flash_attn = True\n    _tied_weights_keys" not in text:
        text = text.replace(OLD_CLASS, NEW_CLASS)
        changed = True
        print("patched class", p)
    if "head_first=False," in text:
        text = text.replace("                head_first=False,\n", "")
        changed = True
        print("patched fla head_first", p)
    if changed:
        p.write_text(text, encoding="utf-8")
    else:
        print("skip", p)
