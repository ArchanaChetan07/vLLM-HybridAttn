#!/usr/bin/env python3
"""32-layer stack bisect: HF vs vLLM engine hidden states at last position.

Records max_abs_diff at every decoder layer output to locate where
~bf16 rounding noise becomes token-flipping magnitude.

Usage:
  export MINICPM_SALA_WEIGHTS=/path/to/MiniCPM-SALA
  python3 gate1_stack_bisect.py
  MINICPM_SALA_PROMPT="Briefly explain gravity:" python3 gate1_stack_bisect.py
  MINICPM_SALA_MODE=prompt_plus_t1 python3 gate1_stack_bisect.py
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys

import torch

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = os.environ.get("MINICPM_SALA_PROMPT", "Hello, my name is")
MODE = os.environ.get(
    "MINICPM_SALA_MODE", "prompt"
)  # "prompt" or "prompt_plus_t1"


def _patch_hf() -> None:
    script = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "remote",
        "patch_hf_transformers_compat.py",
    )
    script = os.path.normpath(script)
    if os.path.isfile(script):
        subprocess.run([sys.executable, script], check=False)


def _hf_layer_last_hiddens(ids: list[int]) -> dict[int, torch.Tensor]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    captured: dict[int, torch.Tensor] = {}
    hooks = []

    def _hook(idx: int):
        def fn(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[idx] = h[0, -1].detach().float().cpu()

        return fn

    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(_hook(i)))

    ids_t = torch.tensor([ids], device="cuda")
    attn = torch.ones_like(ids_t)
    with torch.no_grad():
        model(input_ids=ids_t, attention_mask=attn)
    for h in hooks:
        h.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return captured


def _vllm_layer_last_hiddens(ids: list[int]) -> dict[int, torch.Tensor]:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.model_executor.models.minicpm_sala import MiniCPMSALADecoderLayer

    captured: dict[int, torch.Tensor] = {}
    orig = MiniCPMSALADecoderLayer.forward

    def traced_forward(self, positions, hidden_states):
        out = orig(self, positions, hidden_states)
        prefix = getattr(self.self_attn, "prefix", "")
        if "layers." in prefix:
            idx = int(prefix.split("layers.")[1].split(".")[0])
            captured[idx] = out[-1].detach().float().cpu()
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
            [TokensPrompt(prompt_token_ids=ids)],
            SamplingParams(temperature=0, max_tokens=1),
        )
    finally:
        MiniCPMSALADecoderLayer.forward = orig
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return captured


def _greedy_token(ids: list[int]) -> int:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    with torch.no_grad():
        t = int(
            model(
                torch.tensor([ids], device="cuda"),
                attention_mask=torch.ones(1, len(ids), device="cuda"),
            )
            .logits[0, -1]
            .argmax()
            .item()
        )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return t


def main() -> int:
    _patch_hf()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids = tok.encode(PROMPT, add_special_tokens=True)
    if MODE == "prompt_plus_t1":
        t1 = _greedy_token(ids)
        ids = ids + [t1]
        print(f"mode=prompt_plus_t1 t1={t1} seqlen={len(ids)}", flush=True)
    else:
        print(f"mode=prompt seqlen={len(ids)}", flush=True)

    print(f"prompt={PROMPT!r}", flush=True)
    hf = _hf_layer_last_hiddens(ids)
    vv = _vllm_layer_last_hiddens(ids)

    print(f"{'layer':>5} {'max_abs':>12} {'mean_abs':>12} {'hf_norm':>12} {'v_norm':>12}", flush=True)
    first_flip_layer: int | None = None
    for i in range(32):
        if i not in hf or i not in vv:
            print(f"{i:5d} MISSING", flush=True)
            continue
        d = (hf[i] - vv[i]).abs()
        mx = d.max().item()
        mn = d.mean().item()
        print(
            f"{i:5d} {mx:12.6g} {mn:12.6g} "
            f"{hf[i].norm().item():12.6g} {vv[i].norm().item():12.6g}",
            flush=True,
        )
        if first_flip_layer is None and mx > 0.05:
            first_flip_layer = i

    if first_flip_layer is not None:
        print(f"first_layer_above_0.05={first_flip_layer}", flush=True)
    else:
        max_layer = max(
            range(32),
            key=lambda i: (hf[i] - vv[i]).abs().max().item()
            if i in hf and i in vv
            else 0.0,
        )
        mx = (hf[max_layer] - vv[max_layer]).abs().max().item()
        print(f"max_diff_layer={max_layer} max_abs={mx:.6g}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
