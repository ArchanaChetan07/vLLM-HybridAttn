#!/usr/bin/env python3
import json
import os
from pathlib import Path

WEIGHTS = os.environ.get("MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA")
c = json.loads(Path(WEIGHTS, "config.json").read_text())
mixers = c["mixer_types"]
print("num_layers", len(mixers))
print("attn_use_output_gate", c.get("attn_use_output_gate"))
print("use_output_gate", c.get("use_output_gate"))
sparse_idx = [i for i, m in enumerate(mixers) if m == "minicpm4"]
light_idx = [i for i, m in enumerate(mixers) if "lightning" in m]
print("sparse layers", len(sparse_idx), sparse_idx[:10], "...")
print("lightning layers", len(light_idx), light_idx[:10], "...")
index = json.loads(Path(WEIGHTS, "model.safetensors.index.json").read_text())
ogate_layers = sorted(
    int(k.split(".")[2])
    for k in index["weight_map"]
    if ".self_attn.o_gate." in k
)
print("o_gate in ckpt layers", ogate_layers)
print("sparse without o_gate", [i for i in sparse_idx if i not in ogate_layers])
