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


def _install_l0_hook(model: torch.nn.Module) -> int:
    model._l0_capture = None

    def hook(_mod, _inp, out):
        h = out if isinstance(out, torch.Tensor) else out
        model._l0_capture = h[-1].detach().float().cpu()

    model._l0_hook = model.model.layers[0].register_forward_hook(hook)
    return 0


def _read_l0_capture(model: torch.nn.Module) -> torch.Tensor | None:
    cap = getattr(model, "_l0_capture", None)
    return cap.clone() if cap is not None else None


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
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
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
    llm.apply_model(_install_l0_hook)
    llm.generate(
        [TokensPrompt(prompt_token_ids=ids2)],
        SamplingParams(temperature=0, max_tokens=1),
    )
    caps = llm.apply_model(_read_l0_capture)
    if caps and caps[0] is not None:
        captured["layer0"] = caps[0]
    del llm
    gc.collect()
    torch.cuda.empty_cache()

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
