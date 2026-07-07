#!/usr/bin/env python3
"""Dump HF _flash_attention_forward_dense source."""
import inspect
import os
from transformers import AutoModelForCausalLM

W = os.environ.get("MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA")
m = AutoModelForCausalLM.from_pretrained(
    W, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cpu"
)
sa = m.model.layers[0].self_attn
print("head_dim", sa.head_dim, "num_heads", sa.num_heads, "num_kv", sa.num_key_value_heads)
print("scale", getattr(sa, "scale", None))
print("--- _flash_attention_forward_dense ---")
print(inspect.getsource(sa._flash_attention_forward_dense))
