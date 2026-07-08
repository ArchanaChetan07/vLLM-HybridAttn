#!/usr/bin/env python3
"""Engine prefill norm vs decode logits: manual lm_head on hooked hiddens (in worker).

Usage:
  MINICPM_SALA_PROMPT='Briefly explain gravity:' python3 gate1_engine_norm_lmhead.py
  MINICPM_SALA_ENGINE_AB=1 python3 gate1_engine_norm_lmhead.py
"""

from __future__ import annotations

import contextlib
import gc
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = os.environ.get("MINICPM_SALA_PROMPT", "Briefly explain gravity:")
ENGINE_AB = os.environ.get("MINICPM_SALA_ENGINE_AB", "0") == "1"
CHUNKED = os.environ.get("MINICPM_SALA_CHUNKED_PREFILL", "default")


def _patch_hf() -> None:
    script = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "remote",
        "patch_hf_transformers_compat.py",
    )
    script = os.path.normpath(script)
    if os.path.isfile(script):
        subprocess.run([sys.executable, script], check=False)


def _llm_kwargs() -> dict:
    kw = dict(
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
    if CHUNKED != "default":
        kw["enable_chunked_prefill"] = CHUNKED.lower() in ("1", "true", "yes")
    return kw


def _hf_greedy(ids: list[int]) -> int:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="flash_attention_2",
    ).eval()
    with torch.no_grad():
        g = int(
            model(
                torch.tensor([ids], device="cuda"),
                attention_mask=torch.ones(1, len(ids), device="cuda"),
            ).logits[0, -1].argmax()
        )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return g


def _manual_greedy_in_worker(model, ids: list[int], seq_len: int) -> tuple[int, torch.Tensor, torch.Tensor]:
    import vllm.config as vconfig
    from vllm.config import CacheConfig, ModelConfig, VllmConfig
    from vllm.config.device import DeviceConfig
    from vllm.config.load import LoadConfig
    from vllm.forward_context import set_forward_context

    sys.path.insert(0, os.path.dirname(__file__))
    from gate1_cascade_inject import _setup_attn_context

    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model=WEIGHTS, trust_remote_code=True, dtype="bfloat16", max_model_len=4096
        ),
        load_config=LoadConfig(),
        cache_config=CacheConfig(block_size=256),
        device_config=DeviceConfig(device="cuda"),
    )
    positions = torch.arange(seq_len, device="cuda", dtype=torch.long)
    with vconfig.set_current_vllm_config(vllm_config, check_compile=False):
        attn_metadata, slot_mapping = _setup_attn_context(model, seq_len, vllm_config)
        with torch.no_grad():
            ids_t = torch.tensor(ids, device="cuda")
            with set_forward_context(
                attn_metadata=attn_metadata,
                vllm_config=vllm_config,
                num_tokens=seq_len,
                slot_mapping=slot_mapping,
            ):
                h = model.model.get_input_embeddings(ids_t)
                for layer in model.model.layers:
                    h = layer(positions, h)
                l31 = h[-1].detach().float().cpu()
                h = model.model.norm(h)
                norm_last = h[-1].detach().float().cpu()
                logits = model._orig_compute_logits(h)
                greedy = int(logits[-1].float().argmax().item())
    return greedy, norm_last, l31


def _greedy_on_post_norm(model, hidden_vec: torch.Tensor) -> int:
    """Hidden already passed through model.norm (compute_logits input)."""
    with torch.no_grad():
        h = hidden_vec.to(device="cuda", dtype=torch.bfloat16).unsqueeze(0)
        logits = model._orig_compute_logits(h)
        return int(logits[0].float().argmax().item())


def _greedy_on_l31(model, l31_vec: torch.Tensor) -> int:
    with torch.no_grad():
        h = l31_vec.to(device="cuda", dtype=torch.bfloat16).unsqueeze(0)
        h = model.model.norm(h)
        logits = model._orig_compute_logits(h)
        return int(logits[0].float().argmax().item())


def _engine_run(ids: list[int]) -> dict:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    seq_len = len(ids)
    llm = LLM(**_llm_kwargs())
    llm._diag_ids = ids
    llm._diag_seq_len = seq_len
    llm._diag_engine_token = None

    def _install(model: torch.nn.Module) -> int:
        model._cap: dict = {}
        model._orig_compute_logits = model.compute_logits
        model._diag_ids = ids
        model._diag_seq_len = seq_len

        def _layer_hook(idx: int):
            def fn(_mod, _inp, h_out):
                h = h_out if isinstance(h_out, torch.Tensor) else h_out
                if h.shape[0] == seq_len:
                    model._cap["l31_prefill_last"] = h[-1].detach().float().cpu()
                    model._cap[f"l{idx}_prefill_shape"] = tuple(h.shape)
                elif h.shape[0] == 1 and idx == 31:
                    model._cap["l31_decode"] = h[0].detach().float().cpu()

            return fn

        model._hooks = [
            layer.register_forward_hook(_layer_hook(i))
            for i, layer in enumerate(model.model.layers)
        ]

        def _norm_hook(_mod, _inp, h_out):
            h = h_out if isinstance(h_out, torch.Tensor) else h_out
            if h.shape[0] == seq_len:
                model._cap["norm_prefill_last"] = h[-1].detach().float().cpu()
            elif h.shape[0] == 1:
                model._cap["norm_decode"] = h[0].detach().float().cpu()

        model._norm_hook = model.model.norm.register_forward_hook(_norm_hook)

        def _logits_wrap(hidden_states: torch.Tensor):
            shape = tuple(hidden_states.shape)
            hs = hidden_states.detach().float().cpu()
            for i in range(hs.shape[0]):
                model._cap.setdefault("logits_in", []).append((i, shape, hs[i].clone()))
            logits = model._orig_compute_logits(hidden_states)
            if logits is not None:
                for i in range(logits.shape[0]):
                    model._cap.setdefault("logits_greedy", []).append(
                        (i, shape, int(logits[i].argmax()))
                    )
            return logits

        model.compute_logits = _logits_wrap
        return 0

    def _analyze(model: torch.nn.Module) -> dict:
        cap = getattr(model, "_cap", {})
        out: dict = {}
        try:
            manual_g, manual_norm, manual_l31 = _manual_greedy_in_worker(
                model, ids, seq_len
            )
        except Exception as e:
            manual_g, manual_norm, manual_l31 = -1, None, None
            out["manual_error"] = str(e)

        out["manual_greedy"] = manual_g
        if manual_norm is not None:
            out["manual_norm_greedy"] = _greedy_on_post_norm(model, manual_norm)
        if manual_l31 is not None:
            out["manual_l31_greedy"] = _greedy_on_l31(model, manual_l31)

        if "norm_prefill_last" in cap:
            out["engine_prefill_norm_greedy"] = _greedy_on_post_norm(
                model, cap["norm_prefill_last"]
            )
        if "l31_prefill_last" in cap:
            out["engine_prefill_l31_greedy"] = _greedy_on_l31(
                model, cap["l31_prefill_last"]
            )
        if "norm_decode" in cap:
            out["engine_decode_norm_greedy"] = _greedy_on_post_norm(
                model, cap["norm_decode"]
            )
        if "l31_decode" in cap:
            out["engine_decode_l31_greedy"] = _greedy_on_l31(
                model, cap["l31_decode"]
            )

        for i, (row, shape, vec) in enumerate(cap.get("logits_in", [])):
            out[f"logits_in_{i}_shape"] = shape
            out[f"logits_in_{i}_greedy"] = _greedy_on_post_norm(model, vec)

        out["logits_greedy"] = cap.get("logits_greedy", [])
        if manual_norm is not None and "norm_prefill_last" in cap:
            out["manual_vs_engine_prefill_norm_peak"] = (
                manual_norm - cap["norm_prefill_last"]
            ).abs().max().item()
        if "norm_prefill_last" in cap and "norm_decode" in cap:
            out["prefill_vs_decode_norm_peak"] = (
                cap["norm_prefill_last"] - cap["norm_decode"]
            ).abs().max().item()
        return out

    llm.apply_model(_install)
    gen = llm.generate(
        [TokensPrompt(prompt_token_ids=ids)],
        SamplingParams(temperature=0, max_tokens=1, logprobs=3),
    )[0]
    engine_token = int(gen.outputs[0].token_ids[0])
    analysis = llm.apply_model(_analyze)[0]
    analysis["engine_token"] = engine_token
    analysis["hf_greedy"] = None  # filled by caller
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return analysis


def main() -> int:
    _patch_hf()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids = tok.encode(PROMPT, add_special_tokens=True)
    hf_greedy = _hf_greedy(ids)
    print(f"prompt={PROMPT!r} seqlen={len(ids)} hf_greedy={hf_greedy}", flush=True)

    global CHUNKED
    results = []
    if ENGINE_AB:
        for name, chunked in (
            ("chunked_default", "default"),
            ("chunked_off", "false"),
            ("chunked_on", "true"),
        ):
            CHUNKED = chunked
            r = _engine_run(ids)
            r["label"] = name
            r["hf_greedy"] = hf_greedy
            results.append(r)
            print(f"\n=== {name} ===", flush=True)
            for k, v in sorted(r.items()):
                print(f"{k}={v}", flush=True)
        print(f"\nab_summary={json.dumps(results, default=str)}", flush=True)
    else:
        r = _engine_run(ids)
        r["label"] = "default"
        r["hf_greedy"] = hf_greedy
        results.append(r)
        print(f"\n=== default ===", flush=True)
        for k, v in sorted(r.items()):
            print(f"{k}={v}", flush=True)

    trace_dir = Path(__file__).parent / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / "engine_norm_lmhead_latest.json").write_text(
        json.dumps(results, indent=2, default=str) + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
