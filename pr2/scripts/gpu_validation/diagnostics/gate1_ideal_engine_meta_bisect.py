#!/usr/bin/env python3
"""Bisect: ideal manual prefill with engine-captured metadata overlays."""

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
TRACE = Path(__file__).parent / "traces" / "engine_metadata_replay_latest.json"


def _patch_hf() -> None:
    script = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "remote",
        "patch_hf_transformers_compat.py",
    )
    script = os.path.normpath(script)
    if os.path.isfile(script):
        subprocess.run([sys.executable, script], check=False)


def _greedy_with_meta_overrides(
    ids: list[int],
    *,
    sparse_override: dict | None,
    linear_override: dict | None,
    swap_block_table_only: bool = False,
) -> int:
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
    from gate1_engine_metadata_replay import _meta_from_dict

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
            device = torch.device("cuda")
            if sparse_override is not None:
                for key, meta in list(attn_metadata.items()):
                    if not hasattr(meta, "block_table"):
                        continue
                    if swap_block_table_only:
                        eng = _meta_from_dict(sparse_override, device)
                        attn_metadata[key] = type(meta)(
                            query_start_loc=meta.query_start_loc,
                            seq_lens=meta.seq_lens,
                            block_table=eng.block_table,
                            slot_mapping=eng.slot_mapping,
                            dense_len=meta.dense_len,
                            page_block_size=eng.page_block_size,
                            num_actual_tokens=meta.num_actual_tokens,
                            max_query_len=meta.max_query_len,
                            max_seq_len=meta.max_seq_len,
                        )
                    else:
                        attn_metadata[key] = _meta_from_dict(sparse_override, device)
                    slot_mapping[key] = attn_metadata[key].slot_mapping
            if linear_override is not None:
                for key, meta in list(attn_metadata.items()):
                    if not hasattr(meta, "state_indices_tensor"):
                        continue
                    attn_metadata[key] = _meta_from_dict(linear_override, device)
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
                    h = model.model.norm(h)
                    logits = model.compute_logits(h)
                    greedy = int(logits[-1].float().argmax().item())
            destroy_model_parallel()
            destroy_distributed_environment()
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp)
    gc.collect()
    torch.cuda.empty_cache()
    return greedy


def main() -> int:
    _patch_hf()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids = tok.encode(PROMPT, add_special_tokens=True)
    trace = json.loads(TRACE.read_text())
    l0 = trace["meta_by_layer"]["layer0"]
    l6 = trace["meta_by_layer"]["layer6"]

    baseline = _greedy_with_meta_overrides(ids, sparse_override=None, linear_override=None)
    full_sparse = _greedy_with_meta_overrides(
        ids, sparse_override=l0, linear_override=None
    )
    swap_only = _greedy_with_meta_overrides(
        ids,
        sparse_override=l0,
        linear_override=None,
        swap_block_table_only=True,
    )

    print(f"ideal_baseline={baseline}", flush=True)
    print(f"ideal+engine_sparse_meta={full_sparse}", flush=True)
    print(f"ideal+swap_block_table_only={swap_only}", flush=True)
    print(f"(skip linear-only: ideal kv_cache size=1 cannot index engine slot 4)", flush=True)
    print(f"engine_greedy={trace.get('engine_greedy')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
