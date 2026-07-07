#!/usr/bin/env python3
"""Dump HF layer-0 self_attn forward source and o_proj shapes."""
import inspect
import os
import torch
from transformers import AutoModelForCausalLM

W = os.environ.get("MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA")
m = AutoModelForCausalLM.from_pretrained(
    W, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda"
)
sa = m.model.layers[0].self_attn
print("type", type(sa))
print("o_proj", sa.o_proj)
print("o_proj.weight.shape", sa.o_proj.weight.shape)
print("has o_gate", hasattr(sa, "o_gate"))
if hasattr(sa, "o_gate"):
    print("o_gate.weight.shape", sa.o_gate.weight.shape)
print("--- forward ---")
print(inspect.getsource(sa.forward))
