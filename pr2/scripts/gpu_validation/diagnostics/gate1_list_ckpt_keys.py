#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

from safetensors import safe_open

WEIGHTS = os.environ.get("MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA")
PATTERNS = ("o_gate", "z_proj", "qkv_proj", "q_proj", "lm_head", "embed", "slope", "decay")


def main() -> int:
    index = json.loads(Path(WEIGHTS, "model.safetensors.index.json").read_text())
    keys: set[str] = set()
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(str(Path(WEIGHTS) / shard), framework="pt") as f:
            keys.update(f.keys())
    print(f"total keys {len(keys)}")
    for pat in PATTERNS:
        hits = sorted(k for k in keys if pat in k)
        print(f"\n== {pat} ({len(hits)}) ==")
        for k in hits[:8]:
            print(k)
        if len(hits) > 8:
            print(f"... +{len(hits)-8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
