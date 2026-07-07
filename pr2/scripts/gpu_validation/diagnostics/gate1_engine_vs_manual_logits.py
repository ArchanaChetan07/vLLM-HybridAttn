#!/usr/bin/env python3
"""Engine vs manual full-stack logits on prompt-only prefill.

Hooks engine prefill during LLM.generate(); compares to standalone manual
forward (gate1_cascade_inject path, known HF-parity).

Usage:
  MINICPM_SALA_PROMPT='Briefly explain gravity:' python3 gate1_engine_vs_manual_logits.py
  MINICPM_SALA_ENGINE_AB=1 python3 gate1_engine_vs_manual_logits.py
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


def _standalone_manual_greedy(ids: list[int]) -> int:
    import vllm.config as vconfig
    from vllm.config import CacheConfig, ModelConfig, VllmConfig
    from vllm.config.device import DeviceConfig
    from vllm.config.load import LoadConfig
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.model_loader import get_model_loader

    sys.path.insert(0, os.path.dirname(__file__))
    from gate1_cascade_inject import _setup_attn_context, _vllm_pure_baseline

    seq_len = len(ids)
    positions = torch.arange(seq_len, device="cuda", dtype=torch.long)
    model_config = ModelConfig(
        model=WEIGHTS, trust_remote_code=True, dtype="bfloat16", max_model_len=4096
    )
    load_config = LoadConfig()
    vllm_config = VllmConfig(
        model_config=model_config,
        load_config=load_config,
        cache_config=CacheConfig(block_size=256),
        device_config=DeviceConfig(device="cuda"),
    )
    fd, temp = tempfile.mkstemp()
    os.close(fd)
    greedy = -1
    try:
        with vconfig.set_current_vllm_config(vllm_config, check_compile=False):
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{temp}",
                local_rank=0,
                backend="nccl",
            )
            initialize_model_parallel(1, 1)
            model = get_model_loader(load_config).load_model(
                vllm_config=vllm_config, model_config=model_config
            )
            model.eval().cuda()
            attn_metadata, slot_mapping = _setup_attn_context(
                model, seq_len, vllm_config
            )
            greedy = _vllm_pure_baseline(
                model, ids, positions, seq_len, vllm_config, attn_metadata, slot_mapping
            )
            destroy_model_parallel()
            destroy_distributed_environment()
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp)
    gc.collect()
    torch.cuda.empty_cache()
    return greedy


def _engine_capture(ids: list[int]) -> dict:
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
            if logits is not None:
                for row_idx in range(logits.shape[0]):
                    row = logits[row_idx].float().cpu()
                    model._probe.setdefault("logits_rows", []).append(
                        (row_idx, int(row.argmax()), tuple(hidden_states.shape))
                    )
                if logits.shape[0] >= 1:
                    model._probe["logits_greedy_last"] = int(
                        logits[-1].float().argmax()
                    )
            return logits

        model.compute_logits = _logits_wrap
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
    probe = llm.apply_model(_read)[0]
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "engine_token": engine_token,
        "engine_top5_logprob": engine_top5,
        "engine_logits_greedy_last": probe.get("logits_greedy_last"),
        "engine_layer_count": len(probe.get("layers", {})),
        "engine_l31": probe.get("layers", {}).get(31),
        "engine_norm": probe.get("norm_last"),
        "logits_rows": probe.get("logits_rows", []),
        "shapes": probe.get("shapes", []),
    }


def _print_result(label: str, hf: int, manual: int, result: dict) -> None:
    eng = result.get("engine_token")
    eng_logits = result.get("engine_logits_greedy_last")
    print(f"\n=== {label} ===", flush=True)
    print(f"hf_greedy={hf}", flush=True)
    print(f"manual_standalone={manual} match_hf={manual == hf}", flush=True)
    print(f"engine_generate_token={eng} match_hf={eng == hf}", flush=True)
    print(
        f"engine_logits_greedy_last={eng_logits} match_hf={eng_logits == hf}",
        flush=True,
    )
    print(f"engine_top5_logprob={result.get('engine_top5_logprob')}", flush=True)
    print(f"engine_prefill_layer_captures={result.get('engine_layer_count')}", flush=True)
    print(f"logits_rows={result.get('logits_rows')}", flush=True)
    uniq = sorted(set(tuple(s) for s in result.get("shapes", [])))
    print(f"forward_shapes={uniq[:30]}", flush=True)


def main() -> int:
    _patch_hf()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids = tok.encode(PROMPT, add_special_tokens=True)
    hf_greedy, hf_top5 = _hf_greedy_top5(ids)
    manual_greedy = _standalone_manual_greedy(ids)
    print(f"prompt={PROMPT!r} seqlen={len(ids)}", flush=True)
    print(f"hf_greedy={hf_greedy} hf_top5={hf_top5}", flush=True)
    print(f"manual_standalone={manual_greedy} match_hf={manual_greedy == hf_greedy}", flush=True)

    global CHUNKED
    if ENGINE_AB:
        summary = []
        for name, chunked in (
            ("chunked_default", "default"),
            ("chunked_off", "false"),
            ("chunked_on", "true"),
        ):
            CHUNKED = chunked
            r = _engine_capture(ids)
            _print_result(name, hf_greedy, manual_greedy, r)
            summary.append(
                {
                    "case": name,
                    "engine_token": r.get("engine_token"),
                    "engine_logits_greedy_last": r.get("engine_logits_greedy_last"),
                    "match_hf": r.get("engine_token") == hf_greedy,
                }
            )
        print(f"\nab_summary={json.dumps(summary)}", flush=True)
        return 0

    result = _engine_capture(ids)
    _print_result("default_engine", hf_greedy, manual_greedy, result)

    trace_dir = Path(__file__).parent / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "prompt": PROMPT,
        "seqlen": len(ids),
        "hf_greedy": hf_greedy,
        "hf_top5": hf_top5,
        "manual_standalone": manual_greedy,
        **{k: v for k, v in result.items() if k not in ("engine_l31", "engine_norm")},
    }
    (trace_dir / "engine_vs_manual_logits_latest.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n"
    )
    return 0 if result.get("engine_token") == hf_greedy else 1


if __name__ == "__main__":
    sys.exit(main())
