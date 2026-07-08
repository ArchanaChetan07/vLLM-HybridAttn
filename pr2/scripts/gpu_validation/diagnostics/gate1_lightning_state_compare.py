#!/usr/bin/env python3
"""Compare lightning GLA kv_cache state: incremental step14 vs one-shot prefix."""

from __future__ import annotations

import gc
import os

import torch

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = "Hello, my name is"
STEP = 14
LIGHTNING_LAYERS = (1, 6, 9, 31)


def _install(model: torch.nn.Module) -> int:
    model._lightning_snap: dict[int, torch.Tensor] = {}

    def _capture(layer_idx: int):
        def fn(mod, _inp, _out):
            cache = getattr(mod, "kv_cache", None)
            if cache is None or not cache:
                return
            model._lightning_snap[layer_idx] = cache[0].detach().float().cpu().clone()

        return fn

    model._lightning_hooks = []
    for i in LIGHTNING_LAYERS:
        layer = model.model.layers[i]
        if hasattr(layer.self_attn, "kv_cache"):
            model._lightning_hooks.append(
                layer.self_attn.register_forward_hook(_capture(i))
            )
    return 0


def _read(model: torch.nn.Module) -> dict[int, torch.Tensor]:
    return dict(getattr(model, "_lightning_snap", {}))


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    prompt_ids = tok.encode(PROMPT, add_special_tokens=True)
    hf = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    cur = prompt_ids[:]
    for _ in range(STEP + 1):
        with torch.no_grad():
            nxt = int(hf(torch.tensor([cur], device="cuda")).logits[0, -1].argmax())
        cur.append(nxt)
    del hf
    gc.collect()
    torch.cuda.empty_cache()

    llm = LLM(
        model=WEIGHTS,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
        block_size=256,
        gpu_memory_utilization=0.45,
        enforce_eager=True,
        max_num_seqs=1,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        enable_chunked_prefill=False,
    )
    llm.apply_model(_install)
    llm.generate(
        [TokensPrompt(prompt_token_ids=prompt_ids)],
        SamplingParams(temperature=0, max_tokens=STEP + 1),
    )
    inc = llm.apply_model(_read)[0]
    llm.apply_model(lambda m: setattr(m, "_lightning_snap", {}) or 0)
    llm.generate(
        [TokensPrompt(prompt_token_ids=cur[:-1])],
        SamplingParams(temperature=0, max_tokens=1),
    )
    one = llm.apply_model(_read)[0]
    print(f"prefix_len={len(cur)-1} hf_next={cur[-1]}", flush=True)
    for layer_idx in LIGHTNING_LAYERS:
        if layer_idx in inc and layer_idx in one:
            p = (inc[layer_idx] - one[layer_idx]).abs().max().item()
            print(f"L{layer_idx} state peak={p:.6g}", flush=True)
    del llm
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
