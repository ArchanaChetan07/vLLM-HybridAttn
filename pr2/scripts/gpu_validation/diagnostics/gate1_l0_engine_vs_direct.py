#!/usr/bin/env python3
"""Compare vLLM engine vs direct-load layer-0 prefill (isolates scheduler bug)."""

from __future__ import annotations

import gc
import os
import sys

import torch

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = os.environ.get("MINICPM_SALA_PROMPT", "Hello, my name is")


def _engine_l0(ids: list[int]) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    expected = len(ids)
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
        enable_chunked_prefill=False,
    )

    def _install(model: torch.nn.Module) -> int:
        model._l0_capture = None
        model._l0_input = None
        model._attn_branch = None

        def hook(_mod, _inp, out):
            h = out if isinstance(out, torch.Tensor) else out
            if h.shape[0] == expected:
                model._l0_capture = h.detach().float().cpu()

        def pre_hook(_mod, args):
            if len(args) >= 2:
                hs = args[1]
                if hs.shape[0] == expected:
                    model._l0_input = hs.detach().float().cpu()

        model._l0_hook = model.model.layers[0].register_forward_hook(hook)
        model._l0_pre = model.model.layers[0].register_forward_pre_hook(pre_hook)

        def attn_hook(_mod, _inp, out):
            if out.shape[0] == expected:
                model._attn_branch = out.detach().float().cpu()

        model._attn_hook = model.model.layers[0].self_attn.register_forward_hook(
            attn_hook
        )
        return 0

    def _read(model: torch.nn.Module) -> dict[str, torch.Tensor | None]:
        cap = getattr(model, "_l0_capture", None)
        inp = getattr(model, "_l0_input", None)
        return {
            "out": cap.clone() if cap is not None else None,
            "inp": inp.clone() if inp is not None else None,
            "attn": (
                getattr(model, "_attn_branch", None).clone()
                if getattr(model, "_attn_branch", None) is not None
                else None
            ),
        }

    llm.apply_model(_install)
    llm.generate(
        [TokensPrompt(prompt_token_ids=ids)],
        SamplingParams(temperature=0, max_tokens=1),
    )
    caps = llm.apply_model(_read)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    if caps and caps[0] is not None:
        return caps[0].get("out"), caps[0].get("inp"), caps[0].get("attn")
    return None, None, None


def _direct_l0(ids: list[int]) -> torch.Tensor:
    from gate1_l0_sparse_bisect import vllm_l0_traces

    return vllm_l0_traces(ids)["layer0"]


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from gate1_l0_sparse_bisect import vllm_l0_traces

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
    del hf
    gc.collect()
    torch.cuda.empty_cache()

    ids2 = ids + [t1]
    print(f"prompt={PROMPT!r} t1={t1} seqlen={len(ids2)}", flush=True)

    direct = _direct_l0(ids2)
    traces = vllm_l0_traces(ids2)
    direct_emb = traces["embed"]
    direct_attn = traces["attn_branch"]
    engine, engine_in, engine_attn = _engine_l0(ids2)
    if engine_in is not None:
        din = (direct_emb - engine_in).abs().max().item()
        print(f"engine_vs_direct_input peak={din:.6g}", flush=True)
    if engine_attn is not None:
        da = (direct_attn - engine_attn).abs().max().item()
        print(f"engine_vs_direct_attn peak={da:.6g}", flush=True)
    if engine is None:
        print("FAIL: engine prefill capture missing", flush=True)
        return 1
    if engine.shape[0] != len(ids2):
        print(
            f"FAIL: engine seqlen={engine.shape[0]} expected={len(ids2)}",
            flush=True,
        )
        return 1

    diff = (direct - engine).abs()
    print(f"engine_vs_direct peak={diff.max().item():.6g}", flush=True)
    for i in range(diff.shape[0]):
        print(f"pos{i} engine_vs_direct={diff[i].max().item():.6g}", flush=True)
    return 0


if __name__ == "__main__":
    # Allow import from same directory when run as script.
    sys.path.insert(0, os.path.dirname(__file__))
    sys.exit(main())
