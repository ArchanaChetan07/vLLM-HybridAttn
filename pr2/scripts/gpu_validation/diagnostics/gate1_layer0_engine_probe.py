#!/usr/bin/env python3
"""Compare HF vs vLLM engine layer-0 output at last position (seqlen=7)."""

from __future__ import annotations

import gc
import os
import sys

import torch

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = os.environ.get("MINICPM_SALA_PROMPT", "Hello, my name is")


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.model_executor.models.minicpm_sala import MiniCPMSALADecoderLayer

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids = tok.encode(PROMPT, add_special_tokens=True)
    hf = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    with torch.no_grad():
        t1 = int(
            hf(
                torch.tensor([ids], device="cuda"),
                attention_mask=torch.ones(1, len(ids), device="cuda"),
            )
            .logits[0, -1]
            .argmax()
        )
        ids2 = ids + [t1]
        pos = torch.arange(len(ids2), device="cuda").unsqueeze(0)
        mask = torch.ones(1, len(ids2), device="cuda")
        emb = hf.model.embed_tokens(torch.tensor([ids2], device="cuda")) * hf.config.scale_emb
        h0 = hf.model.layers[0](
            emb, attention_mask=mask, position_ids=pos, use_cache=False
        )[0][0, -1].float().cpu()
    del hf
    gc.collect()
    torch.cuda.empty_cache()

    captured: dict[str, torch.Tensor] = {}
    orig = MiniCPMSALADecoderLayer.forward

    def traced_forward(self, positions, hidden_states):
        out = orig(self, positions, hidden_states)
        if "layers.0." in getattr(self.self_attn, "prefix", ""):
            captured["layer0"] = out[-1].detach().float().cpu()
        return out

    MiniCPMSALADecoderLayer.forward = traced_forward
    try:
        llm = LLM(
            model=WEIGHTS,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=4096,
            block_size=256,
            gpu_memory_utilization=0.5,
            enforce_eager=True,
            max_num_seqs=1,
            enable_prefix_caching=False,
            mamba_cache_mode="none",
        )
        llm.generate(
            [TokensPrompt(prompt_token_ids=ids2)],
            SamplingParams(temperature=0, max_tokens=1),
        )
    finally:
        MiniCPMSALADecoderLayer.forward = orig

    if "layer0" not in captured:
        print("FAIL: layer0 hook did not fire", flush=True)
        return 1
    v0 = captured["layer0"]
    diff = (h0 - v0).abs().max().item()
    print(f"prompt={PROMPT!r} t1={t1} seqlen={len(ids2)}", flush=True)
    print(f"layer0_engine_last max_abs_diff={diff:.6g}", flush=True)
    return 0 if diff < 1e-3 else 1


if __name__ == "__main__":
    sys.exit(main())
