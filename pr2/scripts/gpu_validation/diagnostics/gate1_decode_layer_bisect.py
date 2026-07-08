#!/usr/bin/env python3
"""Per-layer hidden peak: incremental step14 vs one-shot prefix."""

from __future__ import annotations

import gc
import os

import torch

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = "Hello, my name is"
STEP = 14
LAYERS = (0, 6, 9, 31)


def _install(model: torch.nn.Module) -> int:
    model._snap: dict[str, torch.Tensor] = {}

    def _post(idx: int):
        def fn(_mod, _inp, out):
            h = out if isinstance(out, torch.Tensor) else out
            if isinstance(h, torch.Tensor) and h.shape[0] == 1:
                model._snap[f"layer{idx}"] = h[-1].detach().float().cpu()

        return fn

    def _norm(_mod, _inp, out):
        h = out if isinstance(out, torch.Tensor) else out
        if isinstance(h, torch.Tensor) and h.shape[0] == 1:
            model._snap["norm"] = h[-1].detach().float().cpu()

    model._hooks = [
        model.model.layers[i].register_forward_hook(_post(i)) for i in LAYERS
    ]
    model._hooks.append(model.model.norm.register_forward_hook(_norm))
    return 0


def _read(model: torch.nn.Module) -> dict:
    return dict(getattr(model, "_snap", {}))


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
    llm.apply_model(lambda m: setattr(m, "_snap", {}) or 0)
    llm.generate(
        [TokensPrompt(prompt_token_ids=cur[:-1])],
        SamplingParams(temperature=0, max_tokens=1),
    )
    one = llm.apply_model(_read)[0]
    for key in sorted(set(inc) | set(one)):
        if key in inc and key in one:
            p = (inc[key].float() - one[key].float()).abs().max().item()
            print(f"{key} peak={p:.6g}", flush=True)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
