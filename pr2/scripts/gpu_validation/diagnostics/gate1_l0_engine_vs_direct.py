#!/usr/bin/env python3
"""Compare vLLM engine vs manual layer-0 prefill inside the same worker."""

from __future__ import annotations

import gc
import os
import sys

import torch

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = os.environ.get("MINICPM_SALA_PROMPT", "Hello, my name is")


def _run_in_one_worker(
    ids: list[int],
) -> dict[str, torch.Tensor | None]:
    from vllm import LLM, SamplingParams
    from vllm.config import get_current_vllm_config
    from vllm.inputs import TokensPrompt

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    expected = len(ids)
    out: dict[str, torch.Tensor | None] = {
        "manual_layer0": None,
        "manual_attn": None,
        "engine_layer0": None,
        "engine_attn": None,
        "engine_input": None,
    }

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

    def _manual(model: torch.nn.Module) -> int:
        from gate1_l0_sparse_bisect import manual_l0_from_model

        vllm_config = get_current_vllm_config()
        traces = manual_l0_from_model(model, vllm_config, ids)
        out["manual_layer0"] = traces["layer0"]
        out["manual_attn"] = traces["attn_branch"]
        return 0

    def _install(model: torch.nn.Module) -> int:
        def hook(_mod, _inp, h_out):
            h = h_out if isinstance(h_out, torch.Tensor) else h_out
            if h.shape[0] == expected:
                out["engine_layer0"] = h.detach().float().cpu()

        def pre_hook(_mod, args):
            if len(args) >= 2 and args[1].shape[0] == expected:
                out["engine_input"] = args[1].detach().float().cpu()

        def attn_hook(_mod, _inp, h_out):
            if h_out.shape[0] == expected:
                out["engine_attn"] = h_out.detach().float().cpu()

        model._eng_l0_hook = model.model.layers[0].register_forward_hook(hook)
        model._eng_l0_pre = model.model.layers[0].register_forward_pre_hook(pre_hook)
        model._eng_attn_hook = model.model.layers[0].self_attn.register_forward_hook(
            attn_hook
        )
        return 0

    llm.apply_model(_manual)
    llm.apply_model(_install)
    llm.generate(
        [TokensPrompt(prompt_token_ids=ids)],
        SamplingParams(temperature=0, max_tokens=1),
    )
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, os.path.dirname(__file__))
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

    parent = vllm_l0_traces(ids2)
    worker = _run_in_one_worker(ids2)

    for key in ("manual_layer0", "manual_attn", "engine_layer0", "engine_attn"):
        if worker[key] is None:
            print(f"FAIL: missing worker {key}", flush=True)
            return 1

    d_parent_worker = (parent["layer0"] - worker["manual_layer0"]).abs().max().item()
    print(f"parent_vs_worker_manual peak={d_parent_worker:.6g}", flush=True)

    d_mw_attn = (worker["manual_attn"] - worker["engine_attn"]).abs().max().item()
    d_mw_l0 = (worker["manual_layer0"] - worker["engine_layer0"]).abs().max().item()
    print(f"worker_manual_vs_engine_attn peak={d_mw_attn:.6g}", flush=True)
    print(f"worker_manual_vs_engine_layer0 peak={d_mw_l0:.6g}", flush=True)

    d_hf = (parent["layer0"] - worker["engine_layer0"]).abs().max().item()
    print(f"hf_ref_vs_engine_layer0 peak={d_hf:.6g}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
