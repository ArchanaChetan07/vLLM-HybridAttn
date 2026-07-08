#!/usr/bin/env python3
"""Bisect decode-phase HF vs vLLM drift at first greedy mismatch.

Runs HF greedy to find the mismatch step, then compares HF full-forward
logits/hiddens vs vLLM engine generate at the same prefix. Captures decode
attention metadata on the mismatch forward.

Usage:
  MINICPM_SALA_PROMPT='Hello, my name is' python3 gate1_decode_first_mismatch.py
  MINICPM_SALA_MISMATCH_STEP=14 python3 gate1_decode_first_mismatch.py
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = os.environ.get("MINICPM_SALA_PROMPT", "Hello, my name is")
MAX_STEPS = int(os.environ.get("MINICPM_SALA_MAX_DECODE_STEPS", "20"))
FORCED_STEP = os.environ.get("MINICPM_SALA_MISMATCH_STEP")
CAPTURE_LAYERS = tuple(
    int(x)
    for x in os.environ.get("MINICPM_SALA_DECODE_CAPTURE_LAYERS", "0,6,9,31").split(",")
    if x.strip()
)
TRACE_DIR = Path(__file__).resolve().parent / "traces"

# Pickle-safe worker globals
_WORKER_PREFIX: list[int] = []
_WORKER_STEP: int = 0


def _patch_hf() -> None:
    script = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "remote",
        "patch_hf_transformers_compat.py",
    )
    script = os.path.normpath(script)
    if os.path.isfile(script):
        subprocess.run([sys.executable, script], check=False)


def _hf_greedy_ids(prompt: str, max_steps: int) -> tuple[list[int], list[int], int]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids = tok.encode(prompt, add_special_tokens=True)
    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    gen: list[int] = []
    cur = ids[:]
    with torch.no_grad():
        for _ in range(max_steps):
            out = model(
                torch.tensor([cur], device="cuda"),
                attention_mask=torch.ones(1, len(cur), device="cuda"),
            )
            nxt = int(out.logits[0, -1].argmax().item())
            gen.append(nxt)
            cur.append(nxt)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return ids, gen, len(ids)


def _meta_dict(meta: Any) -> dict[str, Any]:
    if meta is None:
        return {"kind": "none"}
    from vllm.v1.attention.backends.linear_attn import LinearAttentionMetadata
    from vllm.v1.attention.backends.minicpm_sala_sparse import (
        MiniCPMSALASparseAttentionMetadata,
    )

    if isinstance(meta, MiniCPMSALASparseAttentionMetadata):
        return {
            "kind": "sparse",
            "seq_lens": meta.seq_lens.detach().cpu().tolist(),
            "query_start_loc": meta.query_start_loc.detach().cpu().tolist(),
            "num_actual_tokens": int(meta.num_actual_tokens),
            "max_query_len": int(meta.max_query_len),
            "max_seq_len": int(meta.max_seq_len),
            "dense_len": int(meta.dense_len),
        }
    if isinstance(meta, LinearAttentionMetadata):
        return {
            "kind": "linear",
            "num_prefills": int(meta.num_prefills),
            "num_decodes": int(meta.num_decodes),
            "num_prefill_tokens": int(meta.num_prefill_tokens),
            "num_decode_tokens": int(meta.num_decode_tokens),
            "seq_lens": meta.seq_lens.detach().cpu().tolist(),
            "query_start_loc": meta.query_start_loc.detach().cpu().tolist(),
            "state_indices": meta.state_indices_tensor.detach().cpu().tolist(),
        }
    return {"kind": "unknown", "type": type(meta).__name__}


def _hf_logits_and_hiddens(ids: list[int]) -> tuple[int, dict[str, torch.Tensor]]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    pos = torch.arange(len(ids), device="cuda").unsqueeze(0)
    mask = torch.ones(1, len(ids), device="cuda")
    hiddens: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        h = model.model.embed_tokens(torch.tensor([ids], device="cuda")) * model.config.scale_emb
        for i, layer in enumerate(model.model.layers):
            h = layer(h, attention_mask=mask, position_ids=pos, use_cache=False)[0]
            if i in CAPTURE_LAYERS:
                hiddens[f"layer{i}"] = h[0, -1].detach().float().cpu()
        h = model.model.norm(h)
        hiddens["norm"] = h[0, -1].detach().float().cpu()
        greedy = int(model(torch.tensor([ids], device="cuda")).logits[0, -1].argmax().item())
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return greedy, hiddens


def _install_decode_capture(model: torch.nn.Module) -> int:
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.models.minicpm_sala import (
        is_lightning_layer,
        is_sparse_layer,
    )

    prefix_len = len(_WORKER_PREFIX)
    step = _WORKER_STEP
    model._dec: dict[str, Any] = {
        "prefix_len": prefix_len,
        "step": step,
        "decode_meta": {},
        "hiddens": {},
        "logits_rows": [],
    }

    def _pre(layer_idx: int):
        def fn(_mod, args):
            if len(args) < 2:
                return
            h = args[1]
            if not isinstance(h, torch.Tensor):
                return
            # decode forward carries one token; prefill on continuation uses full prefix
            if h.shape[0] != 1:
                return
            ctx = get_forward_context()
            md = ctx.attn_metadata
            if not isinstance(md, dict):
                return
            layer = model.model.layers[layer_idx]
            if is_sparse_layer(layer.mixer_type):
                key = layer.self_attn.attn.layer_name
            elif is_lightning_layer(layer.mixer_type):
                key = layer.self_attn.prefix
            else:
                key = None
            if key and key in md:
                tag = f"layer{layer_idx}"
                if tag not in model._dec["decode_meta"]:
                    model._dec["decode_meta"][tag] = _meta_dict(md[key])

        return fn

    def _post(layer_idx: int):
        def fn(_mod, _inp, h_out):
            h = h_out if isinstance(h_out, torch.Tensor) else h_out
            if isinstance(h, torch.Tensor) and h.shape[0] == 1 and layer_idx in CAPTURE_LAYERS:
                model._dec["hiddens"][f"layer{layer_idx}"] = h[-1].detach().float().cpu()

        return fn

    model._dec_hooks = []
    for idx in CAPTURE_LAYERS:
        model._dec_hooks.append(
            model.model.layers[idx].register_forward_pre_hook(_pre(idx))
        )
        model._dec_hooks.append(
            model.model.layers[idx].register_forward_hook(_post(idx))
        )

    def _norm_hook(_mod, _inp, h_out):
        h = h_out if isinstance(h_out, torch.Tensor) else h_out
        if isinstance(h, torch.Tensor) and h.shape[0] == 1:
            model._dec["hiddens"]["norm"] = h[-1].detach().float().cpu()

    model._dec_norm = model.model.norm.register_forward_hook(_norm_hook)
    orig = model.compute_logits

    def _logits_wrap(hidden_states: torch.Tensor):
        logits = orig(hidden_states)
        if logits is not None:
            for i in range(logits.shape[0]):
                model._dec["logits_rows"].append(
                    {
                        "rows": int(hidden_states.shape[0]),
                        "argmax": int(logits[i].argmax().item()),
                    }
                )
        return logits

    model.compute_logits = _logits_wrap
    return 0


def _read_decode_capture(model: torch.nn.Module) -> dict[str, Any]:
    dec = dict(getattr(model, "_dec", {}))
    return dec


def _vllm_greedy_at_prefix(prefix_ids: list[int], gen_steps: int) -> tuple[list[int], dict]:
    global _WORKER_PREFIX, _WORKER_STEP
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    llm = LLM(
        model=WEIGHTS,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=max(len(prefix_ids) + gen_steps + 64, 4096),
        block_size=256,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        max_num_seqs=1,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        enable_chunked_prefill=False,
    )
    out = llm.generate(
        [TokensPrompt(prompt_token_ids=prefix_ids)],
        SamplingParams(temperature=0, max_tokens=gen_steps),
    )[0]
    v_ids = list(out.outputs[0].token_ids)

    # Re-run one more token with hooks on the mismatch decode step
    full_prefix = prefix_ids + v_ids[: max(0, gen_steps - 1)]
    _WORKER_PREFIX = full_prefix
    _WORKER_STEP = len(v_ids[: max(0, gen_steps - 1)])
    llm2 = LLM(
        model=WEIGHTS,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=max(len(full_prefix) + 4, 4096),
        block_size=256,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        max_num_seqs=1,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        enable_chunked_prefill=False,
    )
    llm2.apply_model(_install_decode_capture)
    gen2 = llm2.generate(
        [TokensPrompt(prompt_token_ids=full_prefix)],
        SamplingParams(temperature=0, max_tokens=1),
    )[0]
    capture = llm2.apply_model(_read_decode_capture)[0]
    capture["engine_token"] = int(gen2.outputs[0].token_ids[0])
    del llm, llm2
    gc.collect()
    torch.cuda.empty_cache()
    return v_ids, capture


def _peak(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def main() -> int:
    _patch_hf()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    prompt_ids, hf_gen, prompt_len = _hf_greedy_ids(PROMPT, MAX_STEPS)

    # vLLM full greedy for mismatch index
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    llm = LLM(
        model=WEIGHTS,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=max(prompt_len + MAX_STEPS + 64, 4096),
        block_size=256,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        max_num_seqs=1,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
    )
    v_out = llm.generate(
        [TokensPrompt(prompt_token_ids=prompt_ids)],
        SamplingParams(temperature=0, max_tokens=MAX_STEPS),
    )[0]
    v_gen = list(v_out.outputs[0].token_ids)
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    mismatch = None
    limit = min(len(hf_gen), len(v_gen))
    for i in range(limit):
        if hf_gen[i] != v_gen[i]:
            mismatch = i
            break
    if FORCED_STEP is not None:
        mismatch = int(FORCED_STEP)

    print(f"prompt={PROMPT!r} prompt_len={prompt_len}", flush=True)
    print(f"hf_gen[:16]={hf_gen[:16]}", flush=True)
    print(f"v_gen[:16]={v_gen[:16]}", flush=True)
    if mismatch is None:
        print("no mismatch within max_steps", flush=True)
        return 0

    prefix = prompt_ids + hf_gen[:mismatch]
    hf_next = hf_gen[mismatch]
    v_next = v_gen[mismatch] if mismatch < len(v_gen) else -1
    print(
        f"mismatch_step={mismatch} total_seq_before={len(prefix)} "
        f"hf_next={hf_next} vllm_next={v_next}",
        flush=True,
    )

    hf_greedy, hf_h = _hf_logits_and_hiddens(prefix)
    v_gen2, capture = _vllm_greedy_at_prefix(prefix, 1)
    v_greedy = int(v_gen2[0]) if v_gen2 else -1

    print(f"hf_greedy_at_prefix={hf_greedy} vllm_greedy_at_prefix={v_greedy}", flush=True)
    print(f"capture_engine_token={capture.get('engine_token')}", flush=True)
    print("decode_metadata:", flush=True)
    for k, v in sorted(capture.get("decode_meta", {}).items()):
        print(f"  {k}: {v}", flush=True)
    print("hidden_peaks hf_vs_engine:", flush=True)
    peaks: dict[str, float] = {}
    for key in sorted(set(hf_h) | set(capture.get("hiddens", {}))):
        if key in hf_h and key in capture.get("hiddens", {}):
            p = _peak(hf_h[key], capture["hiddens"][key])
            peaks[key] = p
            print(f"  {key}: {p:.6g}", flush=True)

    worst = max(peaks.values()) if peaks else 0.0
    worst_layer = max(peaks, key=peaks.get) if peaks else "none"
    result = {
        "prompt": PROMPT,
        "prompt_len": prompt_len,
        "mismatch_step": mismatch,
        "total_seq_before": len(prefix),
        "hf_next": hf_next,
        "vllm_next": v_next,
        "hf_greedy_at_prefix": hf_greedy,
        "vllm_greedy_at_prefix": v_greedy,
        "decode_meta": capture.get("decode_meta", {}),
        "hidden_peaks": peaks,
        "worst_layer": worst_layer,
        "worst_peak": worst,
        "logits_rows": capture.get("logits_rows", []),
    }
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TRACE_DIR / "decode_first_mismatch_latest.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"worst_layer={worst_layer} worst_peak={worst:.6g}", flush=True)
    print(f"trace={out_path}", flush=True)
    return 0 if hf_greedy == v_greedy else 1


if __name__ == "__main__":
    raise SystemExit(main())
