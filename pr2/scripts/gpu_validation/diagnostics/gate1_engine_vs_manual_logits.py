#!/usr/bin/env python3
"""Engine vs manual full-stack logits on prompt-only prefill.

Compares HF greedy, engine LLM.generate(), hooked engine logits, and manual
forward inside the same worker (cascade-inject metadata).

Usage:
  MINICPM_SALA_PROMPT='Briefly explain gravity:' python3 gate1_engine_vs_manual_logits.py
  MINICPM_SALA_ENGINE_AB=1 python3 gate1_engine_vs_manual_logits.py
  MINICPM_SALA_CHUNKED_PREFILL=false python3 gate1_engine_vs_manual_logits.py
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
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


def _hf_greedy_top5(ids: list[int]) -> tuple[int, list[tuple[int, float]]]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    with torch.no_grad():
        logits = model(
            torch.tensor([ids], device="cuda"),
            attention_mask=torch.ones(1, len(ids), device="cuda"),
        ).logits[0, -1].float()
    top = torch.topk(logits, 5)
    greedy = int(logits.argmax())
    top5 = [(int(i), float(v)) for i, v in zip(top.indices, top.values)]
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return greedy, top5


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


def _build_vllm_config():
    from vllm.config import CacheConfig, ModelConfig, VllmConfig
    from vllm.config.device import DeviceConfig
    from vllm.config.load import LoadConfig

    model_config = ModelConfig(
        model=WEIGHTS, trust_remote_code=True, dtype="bfloat16", max_model_len=4096
    )
    return VllmConfig(
        model_config=model_config,
        load_config=LoadConfig(),
        cache_config=CacheConfig(block_size=256),
        device_config=DeviceConfig(device="cuda"),
    )


def _manual_forward_greedy(
    model, ids: list[int], seq_len: int, vllm_config
) -> tuple[int, torch.Tensor, torch.Tensor]:
    import vllm.config as vconfig
    from vllm.forward_context import set_forward_context

    sys.path.insert(0, os.path.dirname(__file__))
    from gate1_cascade_inject import _setup_attn_context

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
    return greedy, l31, norm_last


def _engine_probe(ids: list[int]) -> dict:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    seq_len = len(ids)
    llm = LLM(**_llm_kwargs())

    def _install(model: torch.nn.Module) -> int:
        model._probe: dict = {"layers": {}, "shapes": []}
        model._orig_compute_logits = model.compute_logits

        def _layer_hook(idx: int):
            def fn(_mod, _inp, h_out):
                h = h_out if isinstance(h_out, torch.Tensor) else h_out
                model._probe["shapes"].append(("layer", idx, tuple(h.shape)))
                if h.shape[0] == seq_len:
                    model._probe["layers"][idx] = h[-1].detach().float().cpu()

            return fn

        model._probe_hooks = [
            layer.register_forward_hook(_layer_hook(i))
            for i, layer in enumerate(model.model.layers)
        ]

        def _norm_hook(_mod, _inp, h_out):
            h = h_out if isinstance(h_out, torch.Tensor) else h_out
            model._probe["shapes"].append(("norm", -1, tuple(h.shape)))
            if h.shape[0] == seq_len:
                model._probe["norm_last"] = h[-1].detach().float().cpu()

        model._norm_hook = model.model.norm.register_forward_hook(_norm_hook)

        def _logits_wrap(hidden_states: torch.Tensor):
            model._probe["shapes"].append(
                ("logits_in", -1, tuple(hidden_states.shape))
            )
            logits = model._orig_compute_logits(hidden_states)
            if logits is not None and logits.shape[0] >= 1:
                row = logits[-1].float().cpu()
                model._probe["logits_last"] = row
                model._probe["logits_greedy"] = int(row.argmax())
            return logits

        model.compute_logits = _logits_wrap
        return 0

    def _manual(model: torch.nn.Module) -> int:
        g, l31, norm_last = _manual_forward_greedy(
            model, ids, seq_len, _build_vllm_config()
        )
        model._probe["manual_greedy"] = g
        model._probe["manual_l31"] = l31
        model._probe["manual_norm"] = norm_last
        return 0

    def _read(model: torch.nn.Module) -> dict:
        return dict(getattr(model, "_probe", {}))

    llm.apply_model(_install)
    gen = llm.generate(
        [TokensPrompt(prompt_token_ids=ids)],
        SamplingParams(temperature=0, max_tokens=1, logprobs=5),
    )[0]
    engine_token = int(gen.outputs[0].token_ids[0])
    lps = gen.outputs[0].logprobs[0]
    engine_top5 = sorted(
        ((int(k), float(v.logprob)) for k, v in lps.items()),
        key=lambda x: -x[1],
    )[:5]

    llm.apply_model(_manual)
    probe = llm.apply_model(_read)[0]

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    out: dict = {
        "engine_token": engine_token,
        "engine_top5_logprob": engine_top5,
        "engine_logits_greedy": probe.get("logits_greedy"),
        "manual_greedy": probe.get("manual_greedy"),
        "shapes": probe.get("shapes", []),
        "engine_layer_count": len(probe.get("layers", {})),
        "engine_l31": probe.get("layers", {}).get(31),
        "manual_l31": probe.get("manual_l31"),
        "engine_norm": probe.get("norm_last"),
        "manual_norm": probe.get("manual_norm"),
    }
    if out["engine_l31"] is not None and out["manual_l31"] is not None:
        out["l31_peak"] = (out["engine_l31"] - out["manual_l31"]).abs().max().item()
    if out["engine_norm"] is not None and out["manual_norm"] is not None:
        out["norm_peak"] = (out["engine_norm"] - out["manual_norm"]).abs().max().item()
    return out


def _print_result(label: str, hf: int, result: dict) -> None:
    eng = result.get("engine_token")
    manual = result.get("manual_greedy")
    eng_logits = result.get("engine_logits_greedy")
    print(f"\n=== {label} ===", flush=True)
    print(f"hf_greedy={hf}", flush=True)
    print(f"engine_generate_token={eng} match_hf={eng == hf}", flush=True)
    print(
        f"engine_logits_greedy={eng_logits} match_hf={eng_logits == hf}",
        flush=True,
    )
    print(f"manual_greedy={manual} match_hf={manual == hf}", flush=True)
    print(f"engine_top5_logprob={result.get('engine_top5_logprob')}", flush=True)
    if "l31_peak" in result:
        print(f"engine_vs_manual_l31_peak={result['l31_peak']:.6g}", flush=True)
    if "norm_peak" in result:
        print(f"engine_vs_manual_norm_peak={result['norm_peak']:.6g}", flush=True)
    print(f"engine_prefill_layer_captures={result.get('engine_layer_count')}", flush=True)
    uniq = sorted(set(tuple(s) for s in result.get("shapes", [])))
    print(f"forward_shapes={uniq[:24]}", flush=True)


def main() -> int:
    _patch_hf()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids = tok.encode(PROMPT, add_special_tokens=True)
    hf_greedy, hf_top5 = _hf_greedy_top5(ids)
    print(f"prompt={PROMPT!r} seqlen={len(ids)}", flush=True)
    print(f"hf_greedy={hf_greedy} hf_top5={hf_top5}", flush=True)

    global CHUNKED
    if ENGINE_AB:
        summary = []
        for name, chunked in (
            ("chunked_default", "default"),
            ("chunked_off", "false"),
            ("chunked_on", "true"),
        ):
            CHUNKED = chunked
            r = _engine_probe(ids)
            _print_result(name, hf_greedy, r)
            summary.append(
                {
                    "case": name,
                    "engine_token": r.get("engine_token"),
                    "engine_logits_greedy": r.get("engine_logits_greedy"),
                    "manual_greedy": r.get("manual_greedy"),
                    "match_hf": r.get("engine_token") == hf_greedy,
                }
            )
        print(f"\nab_summary={json.dumps(summary)}", flush=True)
        return 0

    result = _engine_probe(ids)
    _print_result("default_engine", hf_greedy, result)

    trace_dir = Path(__file__).parent / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "prompt": PROMPT,
        "seqlen": len(ids),
        "hf_greedy": hf_greedy,
        "hf_top5": hf_top5,
        **{
            k: v
            for k, v in result.items()
            if k
            not in ("engine_l31", "manual_l31", "engine_norm", "manual_norm", "shapes")
        },
    }
    (trace_dir / "engine_vs_manual_logits_latest.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n"
    )
    return 0 if result.get("engine_token") == hf_greedy else 1


if __name__ == "__main__":
    sys.exit(main())
