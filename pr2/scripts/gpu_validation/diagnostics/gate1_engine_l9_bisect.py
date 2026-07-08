#!/usr/bin/env python3
"""Engine prefill bisect at sparse L0/L9 + lightning L6 metadata audit.

Captures engine attention metadata and dense-path decisions during the 6-token
Briefly prefill, then compares L9 output when replaying with engine vs manual
sparse metadata on the engine L8 hidden.

Usage:
  MINICPM_SALA_PROMPT='Briefly explain gravity:' python3 gate1_engine_l9_bisect.py
"""

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import torch

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = os.environ.get("MINICPM_SALA_PROMPT", "Briefly explain gravity:")
LAYERS = (0, 6, 9)


def _meta_dict(meta) -> dict:
    if meta is None:
        return {}
    out: dict = {}
    for key in (
        "num_prefills",
        "num_prefill_tokens",
        "num_decodes",
        "num_decode_tokens",
        "dense_len",
        "num_actual_tokens",
        "max_query_len",
        "max_seq_len",
    ):
        if hasattr(meta, key):
            val = getattr(meta, key)
            out[key] = val if isinstance(val, (int, float)) else None
    for key in ("query_start_loc", "seq_lens", "slot_mapping", "state_indices_tensor"):
        if hasattr(meta, key):
            t = getattr(meta, key)
            if isinstance(t, torch.Tensor):
                out[key] = t.detach().cpu().tolist()
    return out


def _run_engine(ids: list[int]) -> dict:
    from vllm import LLM, SamplingParams
    from vllm.forward_context import get_forward_context
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
        model._diag: dict = {"meta": {}, "dense_path": {}, "hiddens": {}}

        import vllm.v1.attention.backends.minicpm_sala_sparse as sparse_mod

        orig_dense = sparse_mod.MiniCPMSALASparseAttentionImpl._forward_dense

        def _dense_wrap(self, layer, query, key, value, kv_cache, attn_metadata, output):
            for idx in (0, 9):
                attn = model.model.layers[idx].self_attn.attn
                if (
                    attn_metadata is not None
                    and getattr(layer, "layer_name", None) == attn.layer_name
                ):
                    packed = int(attn_metadata.query_start_loc[-1].item())
                    num_new = int(
                        (
                            attn_metadata.query_start_loc[1:]
                            - attn_metadata.query_start_loc[:-1]
                        )
                        .sum()
                        .item()
                    )
                    model._diag["dense_path"][f"layer{idx}"] = {
                        "q_rows": int(query.shape[0]),
                        "packed": packed,
                        "num_new": num_new,
                        "num_actual": int(attn_metadata.num_actual_tokens),
                        "seq_lens": attn_metadata.seq_lens.tolist(),
                    }
            return orig_dense(self, layer, query, key, value, kv_cache, attn_metadata, output)

        sparse_mod.MiniCPMSALASparseAttentionImpl._forward_dense = _dense_wrap
        model._diag["_restore_dense"] = orig_dense

        def _layer_pre(idx: int):
            def fn(_mod, args):
                if len(args) < 2:
                    return
                h = args[1]
                if not isinstance(h, torch.Tensor) or h.shape[0] != seq_len:
                    return
                if idx == 8:
                    model._diag["hiddens"]["l8_in"] = h[-1].detach().float().cpu()
                ctx = get_forward_context()
                md = ctx.attn_metadata
                if not isinstance(md, dict):
                    return
                layer = model.model.layers[idx]
                if idx in (0, 9):
                    key = layer.self_attn.attn.layer_name
                else:
                    key = layer.self_attn.prefix
                if key in md:
                    model._diag["meta"][f"layer{idx}"] = _meta_dict(md[key])

            return fn

        def _layer_post(idx: int):
            def fn(_mod, _inp, h_out):
                h = h_out if isinstance(h_out, torch.Tensor) else h_out
                if isinstance(h, torch.Tensor) and h.shape[0] == seq_len:
                    model._diag["hiddens"][f"layer{idx}"] = h[-1].detach().float().cpu()

            return fn

        model._hooks = []
        for idx in LAYERS:
            model._hooks.append(
                model.model.layers[idx].register_forward_pre_hook(_layer_pre(idx))
            )
            model._hooks.append(
                model.model.layers[idx].register_forward_hook(_layer_post(idx))
            )
        model._hooks.append(
            model.model.layers[8].register_forward_hook(
                lambda _m, _i, h: model._diag["hiddens"].update(
                    {
                        "l8_out": (
                            h[-1].detach().float().cpu()
                            if isinstance(h, torch.Tensor) and h.shape[0] == seq_len
                            else model._diag["hiddens"].get("l8_out")
                        )
                    }
                )
            )
        )
        return 0

    def _read(model: torch.nn.Module) -> dict:
        diag = dict(getattr(model, "_diag", {}))
        import vllm.v1.attention.backends.minicpm_sala_sparse as sparse_mod

        if "_restore_dense" in diag:
            sparse_mod.MiniCPMSALASparseAttentionImpl._forward_dense = diag["_restore_dense"]
        return diag

    llm.apply_model(_install)
    gen = llm.generate(
        [TokensPrompt(prompt_token_ids=ids)],
        SamplingParams(temperature=0, max_tokens=1),
    )[0]
    diag = llm.apply_model(_read)[0]
    diag["engine_token"] = int(gen.outputs[0].token_ids[0])
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return diag


def _replay_l9(model, h8: torch.Tensor, meta, positions, vllm_config) -> torch.Tensor:
    from vllm.forward_context import set_forward_context

    layer9 = model.model.layers[9]
    prefix = layer9.self_attn.attn.layer_name
    h = h8.to(device="cuda", dtype=torch.bfloat16).unsqueeze(0).expand(
        positions.shape[0], -1
    )
    with torch.no_grad():
        with set_forward_context(
            attn_metadata={prefix: meta},
            vllm_config=vllm_config,
            num_tokens=positions.shape[0],
            slot_mapping={prefix: meta.slot_mapping},
        ):
            out = layer9(positions, h)
    return out[-1].detach().float().cpu()


def _manual_replay(ids: list[int], engine_diag: dict) -> dict:
    import contextlib
    import tempfile

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
    from gate1_cascade_inject import _make_sparse_prefill_metadata, _setup_attn_context

    seq_len = len(ids)
    positions = torch.arange(seq_len, device="cuda", dtype=torch.long)
    model_config = ModelConfig(
        model=WEIGHTS, trust_remote_code=True, dtype="bfloat16", max_model_len=4096
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        load_config=LoadConfig(),
        cache_config=CacheConfig(block_size=256),
        device_config=DeviceConfig(device="cuda"),
    )
    fd, temp = tempfile.mkstemp()
    os.close(fd)
    result: dict = {}
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
            model = get_model_loader(LoadConfig()).load_model(
                vllm_config=vllm_config, model_config=model_config
            )
            model.eval().cuda()
            manual_meta_map, _ = _setup_attn_context(model, seq_len, vllm_config)
            manual_l9_meta = manual_meta_map[
                model.model.layers[9].self_attn.attn.layer_name
            ]
            h8 = engine_diag["hiddens"].get("l8_out")
            if h8 is None:
                return result
            eng_l9 = engine_diag["hiddens"].get("layer9")
            man_l9 = _replay_l9(model, h8, manual_l9_meta, positions, vllm_config)
            result["manual_l9_from_engine_l8"] = man_l9
            if eng_l9 is not None:
                result["engine_vs_manual_l9_peak"] = (eng_l9 - man_l9).abs().max().item()
            destroy_model_parallel()
            destroy_distributed_environment()
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp)
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids = tok.encode(PROMPT, add_special_tokens=True)
    print(f"prompt={PROMPT!r} seqlen={len(ids)}", flush=True)

    engine = _run_engine(ids)
    print(f"engine_token={engine.get('engine_token')}", flush=True)
    print(f"dense_path={json.dumps(engine.get('dense_path', {}), indent=2)}", flush=True)
    print(f"meta_layer0={json.dumps(engine.get('meta', {}).get('layer0', {}), indent=2)}", flush=True)
    print(f"meta_layer6={json.dumps(engine.get('meta', {}).get('layer6', {}), indent=2)}", flush=True)
    print(f"meta_layer9={json.dumps(engine.get('meta', {}).get('layer9', {}), indent=2)}", flush=True)

    replay = _manual_replay(ids, engine)
    print(f"engine_vs_manual_l9_peak={replay.get('engine_vs_manual_l9_peak')}", flush=True)

    trace_dir = Path(__file__).parent / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "prompt": PROMPT,
        "seqlen": len(ids),
        "engine_token": engine.get("engine_token"),
        "dense_path": engine.get("dense_path", {}),
        "meta": engine.get("meta", {}),
        **replay,
    }
    (trace_dir / "engine_l9_bisect_latest.json").write_text(
        json.dumps(out, indent=2, default=str) + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
