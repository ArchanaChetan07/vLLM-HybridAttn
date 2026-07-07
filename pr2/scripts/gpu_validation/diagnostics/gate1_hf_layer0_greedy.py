#!/usr/bin/env python3
"""Compare HF hidden state after layer 0 only (isolates sparse GQA path)."""

from __future__ import annotations

import gc
import os
import subprocess
import sys

import torch

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = "Hello, my name is"


def main() -> int:
    script = "/workspace/hybridattn/scripts/remote/patch_hf_transformers_compat.py"
    if os.path.isfile(script):
        subprocess.run([sys.executable, script], check=False)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids = tok.encode(PROMPT, return_tensors="pt").to("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    with torch.no_grad():
        emb = model.model.embed_tokens(ids) * model.config.scale_emb
        pos = torch.arange(ids.shape[1], device="cuda").unsqueeze(0)
        mask = torch.ones_like(ids)
        h = emb
        for i, layer in enumerate(model.model.layers):
            out = layer(
                h,
                attention_mask=mask,
                position_ids=pos,
                use_cache=False,
            )
            h = out[0] if isinstance(out, tuple) else out
            if i == 0:
                h0 = h[0, -1].float().cpu()
                logits0 = model.lm_head(
                    h / (model.config.hidden_size / model.config.dim_model_base)
                )
                g0 = int(logits0[0, -1].argmax())
                break
        full = model(input_ids=ids, attention_mask=mask).logits
        g_full = int(full[0, -1].argmax())
    print("embed last token norm", emb[0, -1].float().norm().item())
    print("after layer0 greedy", g0)
    print("full model greedy", g_full)
    print("layer0 hidden max_abs", h0.abs().max().item())
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
