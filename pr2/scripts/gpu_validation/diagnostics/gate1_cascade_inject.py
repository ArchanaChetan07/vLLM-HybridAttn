#!/usr/bin/env python3
"""Cascade injection: HF hidden after layer K -> vLLM layers K+1..31 -> greedy t1.

Finds the smallest K where injecting HF upstream state makes vLLM greedy match HF.

Usage:
  MINICPM_SALA_PROMPT='Briefly explain gravity:' python3 gate1_cascade_inject.py
  MINICPM_SALA_CASCADE_STEP=4 python3 gate1_cascade_inject.py   # test -1,3,7,...
  MINICPM_SALA_CASCADE_LAYERS=0,7,8,9,31 python3 gate1_cascade_inject.py
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
STEP = int(os.environ.get("MINICPM_SALA_CASCADE_STEP", "1"))
LAYERS_ENV = os.environ.get("MINICPM_SALA_CASCADE_LAYERS", "")


def _patch_hf() -> None:
    script = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "remote",
        "patch_hf_transformers_compat.py",
    )
    script = os.path.normpath(script)
    if os.path.isfile(script):
        subprocess.run([sys.executable, script], check=False)


def _make_sparse_prefill_metadata(
    seq_len: int, block_size: int, dense_len: int, device: torch.device
):
    from vllm.v1.attention.backends.minicpm_sala_sparse import (
        MiniCPMSALASparseAttentionMetadata,
    )

    num_blocks = max(1, (seq_len + block_size - 1) // block_size)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).unsqueeze(
        0
    )
    slot_mapping = torch.arange(seq_len, device=device, dtype=torch.int64)
    return MiniCPMSALASparseAttentionMetadata(
        query_start_loc=torch.tensor([0, seq_len], device=device, dtype=torch.int32),
        seq_lens=torch.tensor([seq_len], device=device, dtype=torch.int32),
        block_table=block_table,
        slot_mapping=slot_mapping,
        dense_len=dense_len,
        page_block_size=block_size,
        num_actual_tokens=seq_len,
        max_query_len=seq_len,
        max_seq_len=seq_len,
    )


def _bind_sparse_kv_cache(attn, block_size: int, seq_len: int) -> None:
    from vllm.v1.attention.backends.minicpm_sala_sparse import (
        MiniCPMSALASparseAttentionBackend,
    )

    num_blocks = max(1, (seq_len + block_size - 1) // block_size)
    shape = MiniCPMSALASparseAttentionBackend.get_kv_cache_shape(
        num_blocks, block_size, attn.num_kv_heads, attn.head_size
    )
    attn.kv_cache = torch.zeros(shape, device="cuda", dtype=torch.bfloat16)


def _hf_hiddens_and_ref(ids: list[int]) -> tuple[dict[int, torch.Tensor], int]:
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
    hiddens: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        h = model.model.embed_tokens(torch.tensor([ids], device="cuda")) * model.config.scale_emb
        hiddens[-1] = h[0].detach().float().cpu()
        for i, layer in enumerate(model.model.layers):
            out = layer(h, attention_mask=mask, position_ids=pos, use_cache=False)
            h = out[0] if isinstance(out, tuple) else out
            hiddens[i] = h[0].detach().float().cpu()
        ref_t1 = int(
            model(
                torch.tensor([ids], device="cuda"),
                attention_mask=mask,
            )
            .logits[0, -1]
            .argmax()
        )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return hiddens, ref_t1


def _setup_attn_context(model, seq_len: int, vllm_config) -> tuple[dict, dict]:
    from vllm.model_executor.models.minicpm_sala import (
        is_lightning_layer,
        is_sparse_layer,
    )
    from vllm.v1.attention.backends.linear_attn import LinearAttentionMetadata
    from vllm.v1.attention.backends.minicpm_sala_sparse import parse_sparse_config

    block_size = vllm_config.cache_config.block_size
    dense_len = parse_sparse_config(vllm_config.model_config.hf_config).dense_len
    device = torch.device("cuda")
    attn_metadata: dict = {}
    slot_mapping: dict = {}

    for layer in model.model.layers:
        mixer = layer.mixer_type
        if is_sparse_layer(mixer):
            sparse_attn = layer.self_attn.attn
            _bind_sparse_kv_cache(sparse_attn, block_size, seq_len)
            meta = _make_sparse_prefill_metadata(
                seq_len, block_size, dense_len, device
            )
            attn_metadata[sparse_attn.layer_name] = meta
            slot_mapping[sparse_attn.layer_name] = meta.slot_mapping
        elif is_lightning_layer(mixer):
            attn = layer.self_attn
            attn.kv_cache = (
                torch.zeros(
                    1,
                    *attn.get_state_shape()[0],
                    device=device,
                    dtype=attn.get_state_dtype()[0],
                ),
            )
            meta = LinearAttentionMetadata(
                num_prefills=1,
                num_prefill_tokens=seq_len,
                num_decodes=0,
                num_decode_tokens=0,
                query_start_loc=torch.tensor(
                    [0, seq_len], device=device, dtype=torch.int32
                ),
                seq_lens=torch.tensor([seq_len], device=device, dtype=torch.int32),
                state_indices_tensor=torch.tensor([0], device=device, dtype=torch.int32),
            )
            attn_metadata[attn.prefix] = meta
    return attn_metadata, slot_mapping


def _vllm_greedy_from_inject(
    model,
    inject_after: int,
    hidden_cpu: torch.Tensor,
    positions: torch.Tensor,
    seq_len: int,
    vllm_config,
    attn_metadata: dict,
    slot_mapping: dict,
) -> int:
    from vllm.forward_context import set_forward_context

    start = inject_after + 1
    h = hidden_cpu.to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        with set_forward_context(
            attn_metadata=attn_metadata,
            vllm_config=vllm_config,
            num_tokens=seq_len,
            slot_mapping=slot_mapping,
        ):
            for i in range(start, len(model.model.layers)):
                h = model.model.layers[i](positions, h)
            h = model.model.norm(h)
            logits = model.compute_logits(h)
    return int(logits[-1].float().argmax().item())


def _vllm_pure_baseline(
    model,
    ids: list[int],
    positions: torch.Tensor,
    seq_len: int,
    vllm_config,
    attn_metadata: dict,
    slot_mapping: dict,
) -> int:
    from vllm.forward_context import set_forward_context

    ids_t = torch.tensor(ids, device="cuda")
    with torch.no_grad():
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
    return int(logits[-1].float().argmax().item())


def _cutoff_layers(num_layers: int) -> list[int]:
    if LAYERS_ENV.strip():
        return sorted(int(x) for x in LAYERS_ENV.split(",") if x.strip() != "")
    layers = [-1] + list(range(0, num_layers, max(1, STEP)))
    if (num_layers - 1) not in layers:
        layers.append(num_layers - 1)
    return sorted(set(layers))


def main() -> int:
    _patch_hf()
    from transformers import AutoTokenizer

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

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids = tok.encode(PROMPT, add_special_tokens=True)
    seq_len = len(ids)
    positions = torch.arange(seq_len, device="cuda", dtype=torch.long)

    print(f"prompt={PROMPT!r} seqlen={seq_len}", flush=True)
    hiddens, hf_ref = _hf_hiddens_and_ref(ids)
    print(f"hf_ref_t1={hf_ref}", flush=True)

    model_config = ModelConfig(
        model=WEIGHTS, trust_remote_code=True, dtype="bfloat16", max_model_len=4096
    )
    load_config = LoadConfig()
    cache_config = CacheConfig(block_size=256)
    vllm_config = VllmConfig(
        model_config=model_config,
        load_config=load_config,
        cache_config=cache_config,
        device_config=DeviceConfig(device="cuda"),
    )
    fd, temp = tempfile.mkstemp()
    os.close(fd)
    results: list[dict] = []
    first_match = None
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

            pure = _vllm_pure_baseline(
                model, ids, positions, seq_len, vllm_config, attn_metadata, slot_mapping
            )
            print(f"vllm_pure_t1={pure} match_hf={pure == hf_ref}", flush=True)

            num_layers = len(model.model.layers)
            for k in _cutoff_layers(num_layers):
                greedy = _vllm_greedy_from_inject(
                    model,
                    k,
                    hiddens[k],
                    positions,
                    seq_len,
                    vllm_config,
                    attn_metadata,
                    slot_mapping,
                )
                match = greedy == hf_ref
                row = {
                    "inject_after_layer": k,
                    "vllm_start_layer": k + 1,
                    "greedy": greedy,
                    "match_hf": match,
                }
                results.append(row)
                tag = "MATCH" if match else "miss"
                print(
                    f"inject_after={k:2d} start={k+1:2d} greedy={greedy} {tag}",
                    flush=True,
                )
                if match and first_match is None:
                    first_match = k

            destroy_model_parallel()
            destroy_distributed_environment()
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp)

    gc.collect()
    torch.cuda.empty_cache()

    print(f"first_inject_after_match={first_match}", flush=True)
    trace_dir = Path(__file__).parent / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    out = trace_dir / "cascade_inject_latest.json"
    payload = {
        "prompt": PROMPT,
        "seqlen": seq_len,
        "hf_ref_t1": hf_ref,
        "vllm_pure_t1": pure,
        "first_inject_after_match": first_match,
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"trace={out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
