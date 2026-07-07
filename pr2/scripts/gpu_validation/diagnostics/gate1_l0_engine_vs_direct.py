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


def _engine_l0(ids: list[int]) -> tuple[torch.Tensor | None, torch.Tensor | None, list[dict]]:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.v1.attention.backends.minicpm_sala_sparse import _num_new_tokens_per_seq

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    expected = len(ids)
    meta_log: list[dict] = []
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
        model._dense_meta_log = meta_log

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

        sparse_attn = model.model.layers[0].self_attn.attn
        impl = getattr(sparse_attn, "impl", None)
        if impl is None:
            impl = getattr(sparse_attn, "attn_impl", None)
        if impl is not None:
            orig_dense = impl._forward_dense

            def _patched_dense(
                self, layer, query, key, value, kv_cache, attn_metadata, output
            ):
                num_new = _num_new_tokens_per_seq(attn_metadata)
                seq_lens_before = attn_metadata.seq_lens - num_new
                model._dense_meta_log.append(
                    {
                        "seq_lens": attn_metadata.seq_lens.tolist(),
                        "num_new": num_new.tolist(),
                        "seq_lens_before": seq_lens_before.tolist(),
                        "num_actual_tokens": attn_metadata.num_actual_tokens,
                        "eager": bool((seq_lens_before == 0).all().item()),
                        "q_tokens": int(query.shape[0]),
                    }
                )
                return orig_dense(
                    self, layer, query, key, value, kv_cache, attn_metadata, output
                )

            impl._forward_dense = _patched_dense.__get__(impl, type(impl))
        return 0

    def _read(model: torch.nn.Module) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        cap = getattr(model, "_l0_capture", None)
        inp = getattr(model, "_l0_input", None)
        return (
            cap.clone() if cap is not None else None,
            inp.clone() if inp is not None else None,
        )

    llm.apply_model(_install)
    llm.generate(
        [TokensPrompt(prompt_token_ids=ids)],
        SamplingParams(temperature=0, max_tokens=1),
    )
    caps = llm.apply_model(_read)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    if caps:
        return caps[0][0], caps[0][1], meta_log
    return None, None, meta_log


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
    direct_emb = vllm_l0_traces(ids2)["embed"]
    engine, engine_in, meta_log = _engine_l0(ids2)
    for i, m in enumerate(meta_log):
        print(f"dense_call{i}={m}", flush=True)
    if engine_in is not None:
        din = (direct_emb - engine_in).abs().max().item()
        print(f"engine_vs_direct_input peak={din:.6g}", flush=True)
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
