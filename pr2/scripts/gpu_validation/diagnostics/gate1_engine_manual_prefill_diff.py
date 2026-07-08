#!/usr/bin/env python3
"""Engine vs manual prefill hidden diff at last position (prefill-only hooks).

Only records layer outputs when batch dim == prompt seqlen (prefill pass).
Decode pass (batch=1) stored separately to detect overwrite bugs.

Usage:
  MINICPM_SALA_PROMPT='Briefly explain gravity:' python3 gate1_engine_manual_prefill_diff.py
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


def _patch_hf() -> None:
    script = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "remote",
        "patch_hf_transformers_compat.py",
    )
    script = os.path.normpath(script)
    if os.path.isfile(script):
        subprocess.run([sys.executable, script], check=False)


def _manual_prefill_capture(ids: list[int]) -> dict[str, torch.Tensor | int]:
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
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.model_loader import get_model_loader

    sys.path.insert(0, os.path.dirname(__file__))
    from gate1_cascade_inject import _setup_attn_context

    seq_len = len(ids)
    positions = torch.arange(seq_len, device="cuda", dtype=torch.long)
    cap: dict[str, torch.Tensor | int] = {}
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
    try:
        with vconfig.set_current_vllm_config(vllm_config, check_compile=False):
            init_distributed_environment(
                world_size=1, rank=0,
                distributed_init_method=f"file://{temp}",
                local_rank=0, backend="nccl",
            )
            initialize_model_parallel(1, 1)
            model = get_model_loader(load_config).load_model(
                vllm_config=vllm_config, model_config=model_config
            )
            model.eval().cuda()
            attn_metadata, slot_mapping = _setup_attn_context(
                model, seq_len, vllm_config
            )
            with torch.no_grad():
                ids_t = torch.tensor(ids, device="cuda")
                with set_forward_context(
                    attn_metadata=attn_metadata,
                    vllm_config=vllm_config,
                    num_tokens=seq_len,
                    slot_mapping=slot_mapping,
                ):
                    h = model.model.get_input_embeddings(ids_t)
                    cap["embed"] = h[-1].detach().float().cpu()
                    for i, layer in enumerate(model.model.layers):
                        h = layer(positions, h)
                        cap[f"layer{i}"] = h[-1].detach().float().cpu()
                    h = model.model.norm(h)
                    cap["norm"] = h[-1].detach().float().cpu()
                    logits = model.compute_logits(h)
                    cap["greedy"] = int(logits[-1].float().argmax().item())
            destroy_model_parallel()
            destroy_distributed_environment()
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp)
    gc.collect()
    torch.cuda.empty_cache()
    return cap


def _engine_prefill_capture(ids: list[int]) -> dict:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    seq_len = len(ids)
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
        model._cap: dict = {"prefill": {}, "decode": {}}

        def _l0_pre(_mod, args):
            if len(args) >= 2:
                h = args[1]
                if isinstance(h, torch.Tensor):
                    if h.shape[0] == seq_len:
                        model._cap["prefill"]["embed"] = h[-1].detach().float().cpu()
                    elif h.shape[0] == 1:
                        model._cap["decode"]["embed"] = h[0].detach().float().cpu()

        model._l0_pre = model.model.layers[0].register_forward_pre_hook(_l0_pre)

        def _layer_hook(idx: int):
            def fn(_mod, _inp, h_out):
                h = h_out if isinstance(h_out, torch.Tensor) else h_out
                if h.shape[0] == seq_len:
                    model._cap["prefill"][f"layer{idx}"] = h[-1].detach().float().cpu()
                elif h.shape[0] == 1:
                    model._cap["decode"][f"layer{idx}"] = h[0].detach().float().cpu()

            return fn

        model._hooks = [
            layer.register_forward_hook(_layer_hook(i))
            for i, layer in enumerate(model.model.layers)
        ]

        def _norm_hook(_mod, _inp, h_out):
            h = h_out if isinstance(h_out, torch.Tensor) else h_out
            if h.shape[0] == seq_len:
                model._cap["prefill"]["norm"] = h[-1].detach().float().cpu()
            elif h.shape[0] == 1:
                model._cap["decode"]["norm"] = h[0].detach().float().cpu()

        model._norm_hook = model.model.norm.register_forward_hook(_norm_hook)

        orig_logits = model.compute_logits

        def _logits_wrap(hidden_states: torch.Tensor):
            sh = tuple(hidden_states.shape)
            logits = orig_logits(hidden_states)
            if logits is not None:
                bucket = "prefill" if hidden_states.shape[0] == seq_len else "decode"
                for i in range(logits.shape[0]):
                    model._cap.setdefault(f"{bucket}_logits", []).append(
                        (i, sh, int(logits[i].argmax()))
                    )
            return logits

        model.compute_logits = _logits_wrap
        return 0

    def _read(model: torch.nn.Module) -> dict:
        cap = getattr(model, "_cap", {})
        # Return CPU tensors (pickle-safe)
        return {
            "prefill": dict(cap.get("prefill", {})),
            "decode": dict(cap.get("decode", {})),
            "prefill_logits": cap.get("prefill_logits", []),
            "decode_logits": cap.get("decode_logits", []),
        }

    llm.apply_model(_install)
    gen = llm.generate(
        [TokensPrompt(prompt_token_ids=ids)],
        SamplingParams(temperature=0, max_tokens=1),
    )[0]
    cap = llm.apply_model(_read)[0]
    cap["engine_token"] = int(gen.outputs[0].token_ids[0])
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return cap


def _peak(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def _print_table(manual: dict, engine: dict) -> dict:
    summary: dict = {"layers": {}, "first_above_0.01": None, "first_above_0.05": None}
    keys = ["embed"] + [f"layer{i}" for i in range(32)] + ["norm"]
    print(f"{'stage':>8} {'peak':>12} {'manual_greedy':>14} {'engine_greedy':>14}", flush=True)
    for key in keys:
        if key not in manual or key not in engine.get("prefill", {}):
            print(f"{key:>8} MISSING manual={key in manual} engine={key in engine.get('prefill', {})}", flush=True)
            continue
        m = manual[key]
        e = engine["prefill"][key]
        if not isinstance(m, torch.Tensor):
            continue
        pk = _peak(m, e)
        summary["layers"][key] = pk
        mg = manual.get("greedy") if key == "norm" else ""
        eg = ""
        if key == "norm":
            for row in engine.get("prefill_logits", []):
                eg = row[2]
            for row in engine.get("decode_logits", []):
                eg = row[2]
        print(f"{key:>8} {pk:12.6g}", flush=True)
        if summary["first_above_0.01"] is None and pk > 0.01:
            summary["first_above_0.01"] = key
        if summary["first_above_0.05"] is None and pk > 0.05:
            summary["first_above_0.05"] = key

    if "norm" in manual and "norm" in engine.get("prefill", {}):
        summary["norm_peak"] = _peak(manual["norm"], engine["prefill"]["norm"])
    if "norm" in engine.get("prefill", {}) and "norm" in engine.get("decode", {}):
        summary["engine_prefill_vs_decode_norm_peak"] = _peak(
            engine["prefill"]["norm"], engine["decode"]["norm"]
        )
    summary["manual_greedy"] = manual.get("greedy")
    summary["engine_token"] = engine.get("engine_token")
    summary["prefill_logits"] = engine.get("prefill_logits", [])
    summary["decode_logits"] = engine.get("decode_logits", [])
    return summary


def main() -> int:
    _patch_hf()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids = tok.encode(PROMPT, add_special_tokens=True)
    print(f"prompt={PROMPT!r} seqlen={len(ids)}", flush=True)

    print("=== manual prefill capture ===", flush=True)
    manual = _manual_prefill_capture(ids)
    print(f"manual_greedy={manual.get('greedy')}", flush=True)

    print("=== engine prefill capture (prefill-only hooks) ===", flush=True)
    engine = _engine_prefill_capture(ids)
    print(f"engine_token={engine.get('engine_token')}", flush=True)
    print(f"prefill_layers={sorted(engine['prefill'].keys())}", flush=True)
    print(f"decode_layers={sorted(engine['decode'].keys())}", flush=True)
    print(f"prefill_logits={engine.get('prefill_logits')}", flush=True)
    print(f"decode_logits={engine.get('decode_logits')}", flush=True)

    print("=== engine vs manual (prefill last position) ===", flush=True)
    summary = _print_table(manual, engine)
    print(f"first_above_0.01={summary.get('first_above_0.01')}", flush=True)
    print(f"first_above_0.05={summary.get('first_above_0.05')}", flush=True)
    print(f"engine_prefill_vs_decode_norm_peak={summary.get('engine_prefill_vs_decode_norm_peak')}", flush=True)

    trace_dir = Path(__file__).parent / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "prompt": PROMPT,
        "seqlen": len(ids),
        **{k: v for k, v in summary.items() if k != "layers"},
        "layer_peaks": summary.get("layers", {}),
    }
    (trace_dir / "engine_manual_prefill_diff_latest.json").write_text(
        json.dumps(out, indent=2, default=str) + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
