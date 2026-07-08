#!/usr/bin/env python3
"""Bisect: ideal prefill inside LLM worker before vs after generate."""

from __future__ import annotations

import gc
import os
import sys

import torch

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = "Briefly explain gravity:"


_LABEL: str = "before_generate"


def _ideal_greedy_in_worker(model: torch.nn.Module) -> int:
    return _ideal_greedy_in_worker_labeled(model, _LABEL)


def _ideal_greedy_in_worker_labeled(model: torch.nn.Module, label: str) -> int:
    import vllm.config as vconfig
    from vllm.config import CacheConfig, DeviceConfig, LoadConfig, ModelConfig, VllmConfig
    from vllm.forward_context import set_forward_context

    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from gate1_cascade_inject import _setup_attn_context

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids = tok.encode(PROMPT, add_special_tokens=True)
    seq_len = len(ids)
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
    from vllm.forward_context import set_forward_context

    with torch.no_grad():
        with vconfig.set_current_vllm_config(vllm_config, check_compile=False):
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
    if not hasattr(model, "_bisect"):
        model._bisect = {}
    model._bisect[label] = greedy
    return 0


def _read_bisect(model: torch.nn.Module) -> dict:
    return dict(getattr(model, "_bisect", {}))


def _run_before(model: torch.nn.Module) -> int:
    global _LABEL
    _LABEL = "before_generate"
    return _ideal_greedy_in_worker(model)


def _run_after(model: torch.nn.Module) -> int:
    global _LABEL
    _LABEL = "after_generate"
    return _ideal_greedy_in_worker(model)


def main() -> int:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids = tok.encode(PROMPT, add_special_tokens=True)

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
    llm.apply_model(_run_before)
    before = llm.apply_model(_read_bisect)[0].get("before_generate")

    gen = llm.generate(
        [TokensPrompt(prompt_token_ids=ids)],
        SamplingParams(temperature=0, max_tokens=1),
    )[0]
    engine = int(gen.outputs[0].token_ids[0])

    llm.apply_model(_run_after)
    after = llm.apply_model(_read_bisect)[0].get("after_generate")

    print(f"ideal_in_worker_before_generate={before}", flush=True)
    print(f"engine_generate={engine}", flush=True)
    print(f"ideal_in_worker_after_generate={after}", flush=True)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
