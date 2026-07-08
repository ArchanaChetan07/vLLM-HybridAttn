#!/usr/bin/env python3
"""Gate 1: bisect HF vs vLLM hidden states (prefill or decode mismatch step).

Modes (``MINICPM_SALA_BISECT_MODE``):
  * ``prefill`` (default): last-token hidden after prompt prefill only.
  * ``decode``: HF greedy prefix through ``MINICPM_SALA_MISMATCH_STEP`` (default 14),
    then compare HF vs vLLM one-shot hidden at lightning layers.

A100 decode bisect (token-14 lightning drift)::

  export MINICPM_SALA_WEIGHTS=/workspace/models/openbmb/MiniCPM-SALA
  export MINICPM_SALA_PROMPT="Hello, my name is"
  export MINICPM_SALA_MISMATCH_STEP=14
  export MINICPM_SALA_BISECT_MODE=decode
  cd /workspace/hybridattn/pr2
  python scripts/gpu_validation/diagnostics/gate1_layer_bisect.py 2>&1 | tee \\
    scripts/gpu_validation/diagnostics/traces/layer_bisect_decode_step14.log

Prefill sanity (original gate)::

  export MINICPM_SALA_BISECT_MODE=prefill
  python scripts/gpu_validation/diagnostics/gate1_layer_bisect.py
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import tempfile

import torch

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = os.environ.get("MINICPM_SALA_PROMPT", "Hello, my name is")
MODE = os.environ.get("MINICPM_SALA_BISECT_MODE", "prefill").lower()
MISMATCH_STEP = int(os.environ.get("MINICPM_SALA_MISMATCH_STEP", "14"))
LAYERS = tuple(
    int(x) for x in os.environ.get("MINICPM_SALA_BISECT_LAYERS", "0,1,6,9,31").split(",")
)


def _patch_hf() -> None:
    script = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "remote",
        "patch_hf_transformers_compat.py",
    )
    script = os.path.normpath(script)
    if os.path.isfile(script):
        subprocess.run([sys.executable, script], check=False)
    alt = "/workspace/hybridattn/scripts/remote/patch_hf_transformers_compat.py"
    if os.path.isfile(alt):
        subprocess.run([sys.executable, alt], check=False)


def _hf_greedy_prefix(prompt_ids: list[int], steps: int) -> list[int]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    cur = prompt_ids[:]
    with torch.no_grad():
        for _ in range(steps):
            nxt = int(
                model(
                    torch.tensor([cur], device="cuda"),
                    attention_mask=torch.ones(1, len(cur), device="cuda"),
                )
                .logits[0, -1]
                .argmax()
                .item()
            )
            cur.append(nxt)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return cur


def hf_hidden_trace(token_ids: list[int]) -> dict[str, torch.Tensor]:
    from transformers import AutoModelForCausalLM

    ids = torch.tensor([token_ids], device="cuda")
    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    traces: dict[str, torch.Tensor] = {}

    def hook(name):
        def _fn(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            traces[name] = h[0, -1].detach().float().cpu()

        return _fn

    for layer_idx in LAYERS:
        model.model.layers[layer_idx].register_forward_hook(hook(f"layer{layer_idx}"))
    with torch.no_grad():
        emb = model.model.embed_tokens(ids) * model.config.scale_emb
        traces["embed"] = emb[0, -1].float().cpu()
        logits = model(
            input_ids=ids, attention_mask=torch.ones_like(ids)
        ).logits
        traces["logits"] = logits[0, -1].float().cpu()
        traces["greedy"] = logits[0, -1].argmax().cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return traces


def vllm_hidden_trace_prefill() -> dict[str, torch.Tensor]:
    import vllm.config as vconfig
    from transformers import AutoTokenizer

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

    traces: dict[str, torch.Tensor] = {}

    def hook(name):
        def _fn(_mod, _inp, out):
            h = out if isinstance(out, torch.Tensor) else out[0]
            traces[name] = h[-1].detach().float().cpu()

        return _fn

    model_config = ModelConfig(
        model=WEIGHTS,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
    )
    load_config = LoadConfig()
    cache_config = CacheConfig(block_size=256)
    vllm_config = VllmConfig(
        model_config=model_config,
        load_config=load_config,
        cache_config=cache_config,
        device_config=DeviceConfig(device="cuda"),
    )
    fd, temp_file = tempfile.mkstemp()
    os.close(fd)
    try:
        with vconfig.set_current_vllm_config(vllm_config, check_compile=False):
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{temp_file}",
                local_rank=0,
                backend="nccl",
            )
            initialize_model_parallel(1, 1)
            model = get_model_loader(load_config).load_model(
                vllm_config=vllm_config, model_config=model_config
            )
            model.eval().cuda()
            for layer_idx in LAYERS:
                model.model.layers[layer_idx].register_forward_hook(
                    hook(f"layer{layer_idx}")
                )

            tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
            ids = tok.encode(PROMPT, return_tensors="pt").to("cuda")
            positions = torch.arange(ids.shape[1], device="cuda", dtype=torch.long)
            with torch.no_grad():
                with set_forward_context(None, vllm_config):
                    emb = model.embed_input_ids(ids)
                    traces["embed"] = emb[0, -1].float().cpu()
                    hidden = model.model(ids, positions)
                    logits = model.compute_logits(hidden)
                    traces["logits"] = logits[0, -1].float().cpu()
                    traces["greedy"] = logits[0, -1].argmax().cpu()
            destroy_model_parallel()
            destroy_distributed_environment()
    finally:
        os.unlink(temp_file)
    return traces


def vllm_hidden_trace_decode(prefix_ids: list[int]) -> dict[str, torch.Tensor]:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    traces: dict[str, torch.Tensor] = {}

    def _install(model: torch.nn.Module) -> int:
        model._snap: dict[str, torch.Tensor] = {}

        def _post(idx: int):
            def fn(_mod, _inp, out):
                h = out if isinstance(out, torch.Tensor) else out
                if isinstance(h, torch.Tensor) and h.shape[0] >= 1:
                    model._snap[f"layer{idx}"] = h[-1].detach().float().cpu()

            return fn

        model._hooks = [
            model.model.layers[i].register_forward_hook(_post(i)) for i in LAYERS
        ]
        return 0

    def _read(model: torch.nn.Module) -> dict[str, torch.Tensor]:
        return dict(getattr(model, "_snap", {}))

    llm = LLM(
        model=WEIGHTS,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=max(len(prefix_ids) + 8, 4096),
        block_size=256,
        gpu_memory_utilization=0.45,
        enforce_eager=True,
        max_num_seqs=1,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        enable_chunked_prefill=False,
    )
    llm.apply_model(_install)
    llm.generate(
        [TokensPrompt(prompt_token_ids=prefix_ids)],
        SamplingParams(temperature=0, max_tokens=1),
    )
    traces = llm.apply_model(_read)[0]
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return traces


def main() -> int:
    _patch_hf()
    if MODE == "decode":
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
        prompt_ids = tok.encode(PROMPT, add_special_tokens=True)
        prefix = _hf_greedy_prefix(prompt_ids, MISMATCH_STEP)
        print(
            f"mode=decode prefix_len={len(prefix)} mismatch_step={MISMATCH_STEP}",
            flush=True,
        )
        hf = hf_hidden_trace(prefix)
        vllm = vllm_hidden_trace_decode(prefix)
    else:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
        prompt_ids = tok.encode(PROMPT, add_special_tokens=True)
        hf = hf_hidden_trace(prompt_ids)
        vllm = vllm_hidden_trace_prefill()

    keys = sorted(set(hf) | set(vllm))
    for key in keys:
        if key in hf and key in vllm:
            diff = (hf[key] - vllm[key]).abs().max().item()
            print(f"{key} max_abs_diff={diff:.6g}", flush=True)
    if "greedy" in hf:
        print(f"HF greedy={int(hf['greedy'])}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
