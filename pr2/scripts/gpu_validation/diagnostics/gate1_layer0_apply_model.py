#!/usr/bin/env python3
"""Check vLLM engine layer-0 last hidden via apply_model hook."""

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

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids = tok.encode(PROMPT, add_special_tokens=True)
    hf = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    traces: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        t1 = int(
            hf(
                torch.tensor([ids], device="cuda"),
                attention_mask=torch.ones(1, len(ids), device="cuda"),
            )
            .logits[0, -1]
            .argmax()
        )
        for label, seq in [("len6", ids), ("len7", ids + [t1])]:
            pos = torch.arange(len(seq), device="cuda").unsqueeze(0)
            emb = hf.model.embed_tokens(torch.tensor([seq], device="cuda")) * hf.config.scale_emb
            h = hf.model.layers[0](
                emb,
                attention_mask=torch.ones(1, len(seq), device="cuda"),
                position_ids=pos,
                use_cache=False,
            )[0][0]
            traces[f"hf_{label}_last"] = h[-1].float().cpu()
            traces[f"hf_{label}_pos5"] = h[5].float().cpu()
    del hf
    gc.collect()
    torch.cuda.empty_cache()

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

    def _install_hook(model: torch.nn.Module) -> dict[str, float]:
        layer0 = model.model.layers[0]
        box: dict[str, torch.Tensor] = {}

        def hook(_mod, _inp, out):
            box["last"] = out[-1].detach().float().cpu()
            if out.shape[0] > 5:
                box["pos5"] = out[5].detach().float().cpu()

        handle = layer0.register_forward_hook(hook)
        try:
            for label, seq in [("len6", ids), ("len7", ids + [t1])]:
                box.clear()
                llm.generate(
                    [TokensPrompt(prompt_token_ids=seq)],
                    SamplingParams(temperature=0, max_tokens=1),
                )
                if "last" in box:
                    traces[f"v_{label}_last"] = box["last"]
                if "pos5" in box:
                    traces[f"v_{label}_pos5"] = box["pos5"]
        finally:
            handle.remove()
        return {}

    llm.apply_model(_install_hook)

    print(f"prompt={PROMPT!r} t1={t1}", flush=True)
    for label in ("len6", "len7"):
        d_last = (traces[f"hf_{label}_last"] - traces[f"v_{label}_last"]).abs().max()
        d_p5 = (traces[f"hf_{label}_pos5"] - traces[f"v_{label}_pos5"]).abs().max()
        print(
            f"{label}: layer0_last_diff={d_last:.6g} pos5_diff={d_p5:.6g}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
