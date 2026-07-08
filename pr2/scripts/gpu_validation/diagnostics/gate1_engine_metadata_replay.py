#!/usr/bin/env python3
"""In-worker full-stack replay using engine-captured attention metadata.

Runs LLM.generate() on the Briefly prompt, captures forward-context metadata
during the 6-token prefill, then replays embed->layers->norm->greedy in the
same worker with that metadata (not ideal _setup_attn_context).

Usage:
  MINICPM_SALA_PROMPT='Briefly explain gravity:' \\
    python3 gate1_engine_metadata_replay.py
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
from typing import Any

import torch

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = os.environ.get("MINICPM_SALA_PROMPT", "Briefly explain gravity:")
CAPTURE_LAYERS = tuple(
    int(x)
    for x in os.environ.get("MINICPM_SALA_META_CAPTURE_LAYERS", "0,6,9").split(",")
    if x.strip() != ""
)

# Pickle-safe module globals set before apply_model (not closures over ids).
_CAPTURE_SEQ_LEN: int = 0
_CAPTURE_IDS: list[int] = []


def _patch_hf() -> None:
    script = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "remote",
        "patch_hf_transformers_compat.py",
    )
    script = os.path.normpath(script)
    if os.path.isfile(script):
        subprocess.run([sys.executable, script], check=False)


def _tensor_to_list(t: torch.Tensor | None) -> list | None:
    if t is None:
        return None
    return t.detach().cpu().tolist()


def _meta_to_dict(meta: Any) -> dict[str, Any]:
    if meta is None:
        return {"kind": "none"}
    from vllm.v1.attention.backends.linear_attn import LinearAttentionMetadata
    from vllm.v1.attention.backends.minicpm_sala_sparse import (
        MiniCPMSALASparseAttentionMetadata,
    )

    if isinstance(meta, MiniCPMSALASparseAttentionMetadata):
        return {
            "kind": "sparse",
            "query_start_loc": _tensor_to_list(meta.query_start_loc),
            "seq_lens": _tensor_to_list(meta.seq_lens),
            "block_table": _tensor_to_list(meta.block_table),
            "slot_mapping": _tensor_to_list(meta.slot_mapping),
            "dense_len": int(meta.dense_len),
            "page_block_size": int(meta.page_block_size),
            "num_actual_tokens": int(meta.num_actual_tokens),
            "max_query_len": int(meta.max_query_len),
            "max_seq_len": int(meta.max_seq_len),
        }
    if isinstance(meta, LinearAttentionMetadata):
        return {
            "kind": "linear",
            "num_prefills": int(meta.num_prefills),
            "num_prefill_tokens": int(meta.num_prefill_tokens),
            "num_decodes": int(meta.num_decodes),
            "num_decode_tokens": int(meta.num_decode_tokens),
            "query_start_loc": _tensor_to_list(meta.query_start_loc),
            "seq_lens": _tensor_to_list(meta.seq_lens),
            "state_indices_tensor": _tensor_to_list(meta.state_indices_tensor),
        }
    out: dict[str, Any] = {"kind": "unknown", "type": type(meta).__name__}
    for key in (
        "num_prefills",
        "num_prefill_tokens",
        "num_decodes",
        "num_decode_tokens",
        "dense_len",
        "page_block_size",
        "num_actual_tokens",
        "max_query_len",
        "max_seq_len",
    ):
        if hasattr(meta, key):
            out[key] = getattr(meta, key)
    for key in (
        "query_start_loc",
        "seq_lens",
        "slot_mapping",
        "state_indices_tensor",
        "block_table",
    ):
        if hasattr(meta, key):
            val = getattr(meta, key)
            if isinstance(val, torch.Tensor):
                out[key] = _tensor_to_list(val)
    return out


def _meta_from_dict(data: dict[str, Any], device: torch.device) -> Any:
    kind = data.get("kind")
    if kind == "none":
        return None
    if kind == "sparse":
        from vllm.v1.attention.backends.minicpm_sala_sparse import (
            MiniCPMSALASparseAttentionMetadata,
        )

        return MiniCPMSALASparseAttentionMetadata(
            query_start_loc=torch.tensor(
                data["query_start_loc"], device=device, dtype=torch.int32
            ),
            seq_lens=torch.tensor(data["seq_lens"], device=device, dtype=torch.int32),
            block_table=torch.tensor(
                data["block_table"], device=device, dtype=torch.int32
            ),
            slot_mapping=torch.tensor(
                data["slot_mapping"], device=device, dtype=torch.int64
            ),
            dense_len=int(data["dense_len"]),
            page_block_size=int(data["page_block_size"]),
            num_actual_tokens=int(data["num_actual_tokens"]),
            max_query_len=int(data["max_query_len"]),
            max_seq_len=int(data["max_seq_len"]),
        )
    if kind == "linear":
        from vllm.v1.attention.backends.linear_attn import LinearAttentionMetadata

        return LinearAttentionMetadata(
            num_prefills=int(data["num_prefills"]),
            num_prefill_tokens=int(data["num_prefill_tokens"]),
            num_decodes=int(data["num_decodes"]),
            num_decode_tokens=int(data["num_decode_tokens"]),
            query_start_loc=torch.tensor(
                data["query_start_loc"], device=device, dtype=torch.int32
            ),
            seq_lens=torch.tensor(data["seq_lens"], device=device, dtype=torch.int32),
            state_indices_tensor=torch.tensor(
                data["state_indices_tensor"], device=device, dtype=torch.int32
            ),
        )
    raise ValueError(f"unsupported metadata kind: {kind!r}")


def _slot_mapping_from_dict(
    slot_map: dict[str, list], device: torch.device
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, vals in slot_map.items():
        if vals is None:
            continue
        dtype = torch.int64 if key else torch.int64
        out[key] = torch.tensor(vals, device=device, dtype=dtype)
    return out


def _install_capture(model: torch.nn.Module) -> int:
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.models.minicpm_sala import (
        is_lightning_layer,
        is_sparse_layer,
    )

    seq_len = _CAPTURE_SEQ_LEN
    model._mr: dict[str, Any] = {
        "seq_len": seq_len,
        "ids": list(_CAPTURE_IDS),
        "engine_hiddens": {},
        "meta_by_layer": {},
    }

    def _capture_ctx(layer_idx: int):
        def pre(_mod, args):
            if len(args) < 2:
                return
            h = args[1]
            if not isinstance(h, torch.Tensor) or h.shape[0] != seq_len:
                return
            ctx = get_forward_context()
            md = ctx.attn_metadata
            if not isinstance(md, dict):
                return
            if "attn_metadata_json" not in model._mr:
                model._mr["attn_metadata_json"] = {
                    k: _meta_to_dict(v) for k, v in md.items()
                }
                model._mr["slot_mapping_json"] = {
                    k: _tensor_to_list(v) for k, v in ctx.slot_mapping.items()
                }
                model._mr["no_compile_layers"] = ctx.no_compile_layers
            layer = model.model.layers[layer_idx]
            if is_sparse_layer(layer.mixer_type):
                key = layer.self_attn.attn.layer_name
            elif is_lightning_layer(layer.mixer_type):
                key = layer.self_attn.prefix
            else:
                key = None
            if key and key in md and f"layer{layer_idx}" not in model._mr["meta_by_layer"]:
                model._mr["meta_by_layer"][f"layer{layer_idx}"] = _meta_to_dict(
                    md[key]
                )

        return pre

    def _layer_post(idx: int):
        def fn(_mod, _inp, h_out):
            h = h_out if isinstance(h_out, torch.Tensor) else h_out
            if isinstance(h, torch.Tensor) and h.shape[0] == seq_len:
                model._mr["engine_hiddens"][f"layer{idx}"] = (
                    h[-1].detach().float().cpu()
                )

        return fn

    model._mr_hooks = []
    model._mr_hooks.append(
        model.model.layers[0].register_forward_pre_hook(_capture_ctx(0))
    )
    for idx in CAPTURE_LAYERS:
        if idx != 0:
            model._mr_hooks.append(
                model.model.layers[idx].register_forward_pre_hook(_capture_ctx(idx))
            )
        model._mr_hooks.append(
            model.model.layers[idx].register_forward_hook(_layer_post(idx))
        )

    def _norm_hook(_mod, _inp, h_out):
        h = h_out if isinstance(h_out, torch.Tensor) else h_out
        if isinstance(h, torch.Tensor) and h.shape[0] == seq_len:
            model._mr["engine_hiddens"]["norm"] = h[-1].detach().float().cpu()

    model._mr_norm_hook = model.model.norm.register_forward_hook(_norm_hook)

    orig_logits = model.compute_logits

    def _logits_wrap(hidden_states: torch.Tensor):
        logits = orig_logits(hidden_states)
        if logits is not None:
            bucket = "prefill" if hidden_states.shape[0] == seq_len else "decode"
            for i in range(logits.shape[0]):
                model._mr.setdefault(f"{bucket}_logits", []).append(
                    (i, tuple(hidden_states.shape), int(logits[i].argmax()))
                )
        return logits

    model.compute_logits = _logits_wrap
    return 0


def _replay_with_ideal_metadata(model: torch.nn.Module) -> int:
    """Replay in worker using ideal _setup_attn_context metadata (bisect)."""
    import vllm.config as vconfig
    from vllm.config import CacheConfig, DeviceConfig, LoadConfig, ModelConfig, VllmConfig
    from vllm.forward_context import ForwardContext, override_forward_context

    sys.path.insert(0, os.path.dirname(__file__))
    from gate1_cascade_inject import _setup_attn_context

    mr = model._mr
    seq_len = mr["seq_len"]
    ids = mr["ids"]
    device = torch.device("cuda")
    positions = torch.arange(seq_len, device=device, dtype=torch.long)
    ids_t = torch.tensor(ids, device=device)

    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model=WEIGHTS, trust_remote_code=True, dtype="bfloat16", max_model_len=4096
        ),
        load_config=LoadConfig(),
        cache_config=CacheConfig(block_size=256),
        device_config=DeviceConfig(device="cuda"),
    )
    with vconfig.set_current_vllm_config(vllm_config, check_compile=False):
        attn_metadata, slot_mapping = _setup_attn_context(model, seq_len, vllm_config)
    fc = ForwardContext(
        no_compile_layers=mr.get("no_compile_layers"),
        attn_metadata=attn_metadata,
        slot_mapping=slot_mapping,
    )
    with torch.no_grad():
        with override_forward_context(fc):
            h = model.model.get_input_embeddings(ids_t)
            for layer in model.model.layers:
                h = layer(positions, h)
            h = model.model.norm(h)
            logits = model.compute_logits(h)
            mr["ideal_replay_greedy"] = int(logits[-1].float().argmax().item())
    return 0


def _replay_with_engine_metadata(model: torch.nn.Module) -> int:
    from vllm.forward_context import ForwardContext, override_forward_context

    mr = model._mr
    seq_len = mr["seq_len"]
    ids = mr["ids"]
    device = torch.device("cuda")
    positions = torch.arange(seq_len, device=device, dtype=torch.long)
    ids_t = torch.tensor(ids, device=device)

    attn_metadata = {
        k: _meta_from_dict(v, device) for k, v in mr["attn_metadata_json"].items()
    }
    slot_mapping = _slot_mapping_from_dict(mr["slot_mapping_json"], device)
    ncl = mr.get("no_compile_layers")
    fc = ForwardContext(
        no_compile_layers=ncl,
        attn_metadata=attn_metadata,
        slot_mapping=slot_mapping,
    )

    replay_hiddens: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        with override_forward_context(fc):
            h = model.model.get_input_embeddings(ids_t)
            replay_hiddens["embed"] = h[-1].detach().float().cpu()
            for i, layer in enumerate(model.model.layers):
                h = layer(positions, h)
                if i in CAPTURE_LAYERS or i == 31:
                    replay_hiddens[f"layer{i}"] = h[-1].detach().float().cpu()
            h = model.model.norm(h)
            replay_hiddens["norm"] = h[-1].detach().float().cpu()
            logits = model.compute_logits(h)
            mr["replay_greedy"] = int(logits[-1].float().argmax().item())
    mr["replay_hiddens"] = replay_hiddens
    return 0


def _read_capture(model: torch.nn.Module) -> dict[str, Any]:
    mr = dict(getattr(model, "_mr", {}))
    # Drop unpicklable references before returning to parent.
    mr.pop("no_compile_layers", None)
    return mr


def _run_in_worker(ids: list[int]) -> dict[str, Any]:
    global _CAPTURE_SEQ_LEN, _CAPTURE_IDS
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    _CAPTURE_SEQ_LEN = len(ids)
    _CAPTURE_IDS = list(ids)

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

    llm.apply_model(_install_capture)
    gen = llm.generate(
        [TokensPrompt(prompt_token_ids=ids)],
        SamplingParams(temperature=0, max_tokens=1),
    )[0]
    llm.apply_model(_replay_with_engine_metadata)
    llm.apply_model(_replay_with_ideal_metadata)
    mr = llm.apply_model(_read_capture)[0]
    mr["engine_greedy"] = int(gen.outputs[0].token_ids[0])
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return mr


def _ideal_manual_greedy(ids: list[int]) -> tuple[int, dict[str, torch.Tensor]]:
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
    hiddens: dict[str, torch.Tensor] = {}
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
            with torch.no_grad():
                ids_t = torch.tensor(ids, device="cuda")
                with set_forward_context(
                    attn_metadata=attn_metadata,
                    vllm_config=vllm_config,
                    num_tokens=seq_len,
                    slot_mapping=slot_mapping,
                ):
                    h = model.model.get_input_embeddings(ids_t)
                    for i, layer in enumerate(model.model.layers):
                        h = layer(positions, h)
                        if i in CAPTURE_LAYERS or i == 31:
                            hiddens[f"layer{i}"] = h[-1].detach().float().cpu()
                    h = model.model.norm(h)
                    hiddens["norm"] = h[-1].detach().float().cpu()
                    logits = model.compute_logits(h)
                    greedy = int(logits[-1].float().argmax().item())
            destroy_model_parallel()
            destroy_distributed_environment()
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp)
    gc.collect()
    torch.cuda.empty_cache()
    return greedy, hiddens


def _peak(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def _layer_peaks(
    engine: dict[str, torch.Tensor],
    replay: dict[str, torch.Tensor],
    ideal: dict[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    keys = sorted(set(engine) | set(replay) | set(ideal))
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        row: dict[str, float] = {}
        if key in engine and key in replay:
            row["engine_vs_replay"] = _peak(engine[key], replay[key])
        if key in engine and key in ideal:
            row["engine_vs_ideal"] = _peak(engine[key], ideal[key])
        if key in replay and key in ideal:
            row["replay_vs_ideal"] = _peak(replay[key], ideal[key])
        if row:
            out[key] = row
    return out


def _conclude(engine_greedy: int, replay_greedy: int, ideal_greedy: int) -> str:
    if replay_greedy == engine_greedy and replay_greedy != ideal_greedy:
        return (
            "metadata_kv_side_effects: replay with engine metadata matches engine "
            f"({engine_greedy}) not ideal manual ({ideal_greedy})"
        )
    if replay_greedy == ideal_greedy and engine_greedy != ideal_greedy:
        return (
            "logits_indexing_bug: replay matches ideal manual "
            f"({ideal_greedy}) but engine generate token is {engine_greedy}"
        )
    if replay_greedy == ideal_greedy == engine_greedy:
        return "all_match"
    return (
        f"inconclusive: engine={engine_greedy} replay={replay_greedy} "
        f"ideal={ideal_greedy}"
    )


def main() -> int:
    _patch_hf()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids = tok.encode(PROMPT, add_special_tokens=True)
    print(f"prompt={PROMPT!r} seqlen={len(ids)}", flush=True)

    print("=== ideal manual prefill (parent, _setup_attn_context) ===", flush=True)
    ideal_greedy, ideal_hiddens = _ideal_manual_greedy(ids)
    print(f"ideal_manual_greedy={ideal_greedy}", flush=True)

    print("=== engine capture + metadata replay (same worker) ===", flush=True)
    worker = _run_in_worker(ids)
    engine_greedy = worker.get("engine_greedy")
    replay_greedy = worker.get("replay_greedy")
    print(f"engine_greedy={engine_greedy}", flush=True)
    print(f"replay_greedy={replay_greedy}", flush=True)
    print(f"ideal_replay_greedy={worker.get('ideal_replay_greedy')}", flush=True)
    print(f"prefill_logits={worker.get('prefill_logits', [])}", flush=True)
    print(f"decode_logits={worker.get('decode_logits', [])}", flush=True)

    engine_h = worker.get("engine_hiddens", {})
    replay_h = worker.get("replay_hiddens", {})
    peaks = _layer_peaks(engine_h, replay_h, ideal_hiddens)
    print("=== per-layer last-position peaks ===", flush=True)
    for key in sorted(peaks):
        print(f"{key:>8} {peaks[key]}", flush=True)

    conclusion = _conclude(
        int(engine_greedy or -1),
        int(replay_greedy or -1),
        int(ideal_greedy),
    )
    print(f"conclusion={conclusion}", flush=True)

    trace_dir = Path(__file__).parent / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "prompt": PROMPT,
        "seqlen": len(ids),
        "engine_greedy": engine_greedy,
        "replay_greedy": replay_greedy,
        "ideal_manual_greedy": ideal_greedy,
        "prefill_logits": worker.get("prefill_logits", []),
        "decode_logits": worker.get("decode_logits", []),
        "meta_by_layer": worker.get("meta_by_layer", {}),
        "layer_peaks": peaks,
        "conclusion": conclusion,
    }
    trace_path = trace_dir / "engine_metadata_replay_latest.json"
    trace_path.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(f"trace={trace_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
