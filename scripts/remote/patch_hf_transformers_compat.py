#!/usr/bin/env python3
"""Patch cached HF MiniCPM-SALA modeling for transformers/fla compatibility."""

from __future__ import annotations

import os
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


def _candidate_paths() -> list[Path]:
    weights = os.environ.get(
        "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
    )
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    paths: list[Path] = []
    paths.append(Path(weights) / "modeling_minicpm_sala.py")
    paths.extend(
        hf_home.glob("modules/transformers_modules/MiniCPM_hyphen_SALA/*/modeling_minicpm_sala.py")
    )
    paths.extend(hf_home.glob("hub/models--openbmb--MiniCPM-SALA/snapshots/*/modeling_minicpm_sala.py"))
    return paths


def _patch_file(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    changed = False
    if OLD_IMPORT in text and NEW_IMPORT not in text:
        text = text.replace(OLD_IMPORT, NEW_IMPORT)
        changed = True
        print(f"patched import {path}")
    if OLD_PRETRAINED in text and "_supports_flash_attn = True" not in text:
        text = text.replace(OLD_PRETRAINED, NEW_PRETRAINED)
        changed = True
        print(f"patched pretrained {path}")
    if (
        OLD_CLASS in text
        and "_supports_flash_attn = True\n    _tied_weights_keys" not in text
    ):
        text = text.replace(OLD_CLASS, NEW_CLASS)
        changed = True
        print(f"patched class {path}")
    if "head_first=False," in text:
        text = text.replace("                head_first=False,\n", "")
        changed = True
        print(f"patched fla head_first {path}")
    if changed:
        path.write_text(text, encoding="utf-8")
    else:
        print("skip", path)
    return changed


def main() -> int:
    seen: set[str] = set()
    any_changed = False
    for path in _candidate_paths():
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        any_changed = _patch_file(path) or any_changed
    if not seen:
        print("WARN: no modeling_minicpm_sala.py candidates found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
