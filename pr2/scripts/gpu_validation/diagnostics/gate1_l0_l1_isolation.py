#!/usr/bin/env python3
"""Isolate layer-0 vs layer-1 divergence: engine L0, HF L0, L1 with each input.

Usage:
  MINICPM_SALA_PROMPT='Hello, my name is' python3 gate1_l0_l1_isolation.py
  MINICPM_SALA_PROMPT='Briefly explain gravity:' python3 gate1_l0_l1_isolation.py
"""

from __future__ import annotations

import contextlib
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


def _install_l0_hook(model: torch.nn.Module) -> int:
    model._l0_capture = None

    def hook(_mod, _inp, out):
        h = out if isinstance(out, torch.Tensor) else out
        model._l0_capture = h[-1].detach().float().cpu()

    model._l0_hook = model.model.layers[0].register_forward_hook(hook)
    return 0


def _read_l0_capture(model: torch.nn.Module) -> torch.Tensor | None:
    cap = getattr(model, "_l0_capture", None)
    return cap.clone() if cap is not None else None


def _patch_hf() -> None:
    script = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "remote",
        "patch_hf_transformers_compat.py",
    )
    script = os.path.normpath(script)
    if os.path.isfile(script):
        subprocess.run([sys.executable, script], check=False)


def _engine_l0_last(ids: list[int]) -> torch.Tensor | None:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
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
    )
    llm.apply_model(_install_l0_hook)
    llm.generate(
        [TokensPrompt(prompt_token_ids=ids)],
        SamplingParams(temperature=0, max_tokens=1),
    )
    caps = llm.apply_model(_read_l0_capture)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return caps[0] if caps else None


def _hf_l0_last(ids: list[int]) -> torch.Tensor:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    pos = torch.arange(len(ids), device="cuda").unsqueeze(0)
    with torch.no_grad():
        emb = model.model.embed_tokens(torch.tensor([ids], device="cuda")) * model.config.scale_emb
        h0 = model.model.layers[0](
            emb,
            attention_mask=torch.ones(1, len(ids), device="cuda"),
            position_ids=pos,
            use_cache=False,
        )[0][0, -1].float().cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return h0


def _l1_last_with_h0_seq(
    h0_seq: torch.Tensor,
    seq_len: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Run vLLM layer-1 given full layer-0 hidden sequence [seq_len, hidden]."""
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
    from vllm.v1.attention.backends.linear_attn import LinearAttentionMetadata

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
    positions = torch.arange(seq_len, device="cuda", dtype=torch.long)
    out_last = None
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
            vm = get_model_loader(load_config).load_model(
                vllm_config=vllm_config, model_config=model_config
            )
            vm.eval().cuda()
            layer1 = vm.model.layers[1]
            attn = layer1.self_attn
            attn.kv_cache = (
                torch.zeros(
                    1,
                    *attn.get_state_shape()[0],
                    device="cuda",
                    dtype=attn.get_state_dtype()[0],
                ),
            )
            meta = LinearAttentionMetadata(
                num_prefills=1,
                num_prefill_tokens=seq_len,
                num_decodes=0,
                num_decode_tokens=0,
                query_start_loc=torch.tensor(
                    [0, seq_len], device="cuda", dtype=torch.int32
                ),
                seq_lens=torch.tensor([seq_len], device="cuda", dtype=torch.int32),
                state_indices_tensor=torch.tensor([0], device="cuda", dtype=torch.int32),
            )
            h0 = h0_seq.to(device="cuda", dtype=dtype)
            with torch.no_grad():
                x = layer1.input_layernorm(h0)
                attn_out = torch.zeros_like(x)
                with set_forward_context(
                    attn_metadata={attn.prefix: meta}, vllm_config=vllm_config
                ):
                    attn.forward(hidden_states=x, output=attn_out, positions=positions)
                out_last = attn_out[-1].float().cpu()
            destroy_model_parallel()
            destroy_distributed_environment()
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp)
    gc.collect()
    torch.cuda.empty_cache()
    return out_last


def _hf_l1_attn_last(ids: list[int]) -> torch.Tensor:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    pos = torch.arange(len(ids), device="cuda").unsqueeze(0)
    with torch.no_grad():
        emb = model.model.embed_tokens(torch.tensor([ids], device="cuda")) * model.config.scale_emb
        h0 = model.model.layers[0](
            emb,
            attention_mask=torch.ones(1, len(ids), device="cuda"),
            position_ids=pos,
            use_cache=False,
        )[0]
        x = model.model.layers[1].input_layernorm(h0)
        attn_out, _, _ = model.model.layers[1].self_attn(
            x,
            attention_mask=torch.ones(1, len(ids), device="cuda"),
            position_ids=pos,
            use_cache=False,
        )
        out_last = attn_out[0, -1].float().cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out_last


def _l1_full_with_h0_seq(
    h0_seq: torch.Tensor,
    seq_len: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Run full vLLM layer-1 given layer-0 hidden sequence [seq_len, hidden]."""
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
    from vllm.v1.attention.backends.linear_attn import LinearAttentionMetadata

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
    positions = torch.arange(seq_len, device="cuda", dtype=torch.long)
    out_last = None
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
            vm = get_model_loader(load_config).load_model(
                vllm_config=vllm_config, model_config=model_config
            )
            vm.eval().cuda()
            layer1 = vm.model.layers[1]
            attn = layer1.self_attn
            attn.kv_cache = (
                torch.zeros(
                    1,
                    *attn.get_state_shape()[0],
                    device="cuda",
                    dtype=attn.get_state_dtype()[0],
                ),
            )
            meta = LinearAttentionMetadata(
                num_prefills=1,
                num_prefill_tokens=seq_len,
                num_decodes=0,
                num_decode_tokens=0,
                query_start_loc=torch.tensor(
                    [0, seq_len], device="cuda", dtype=torch.int32
                ),
                seq_lens=torch.tensor([seq_len], device="cuda", dtype=torch.int32),
                state_indices_tensor=torch.tensor([0], device="cuda", dtype=torch.int32),
            )
            h0 = h0_seq.to(device="cuda", dtype=dtype)
            with torch.no_grad():
                with set_forward_context(
                    attn_metadata={attn.prefix: meta}, vllm_config=vllm_config
                ):
                    out = layer1(positions, h0)
                out_last = out[-1].float().cpu()
            destroy_model_parallel()
            destroy_distributed_environment()
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp)
    gc.collect()
    torch.cuda.empty_cache()
    return out_last


def _hf_l1_last(ids: list[int]) -> torch.Tensor:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    pos = torch.arange(len(ids), device="cuda").unsqueeze(0)
    with torch.no_grad():
        emb = model.model.embed_tokens(torch.tensor([ids], device="cuda")) * model.config.scale_emb
        h0 = model.model.layers[0](
            emb,
            attention_mask=torch.ones(1, len(ids), device="cuda"),
            position_ids=pos,
            use_cache=False,
        )[0]
        out1 = model.model.layers[1](
            h0,
            attention_mask=torch.ones(1, len(ids), device="cuda"),
            position_ids=pos,
            use_cache=False,
        )[0][0, -1].float().cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out1


def _hf_full_h0(ids: list[int]) -> torch.Tensor:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    pos = torch.arange(len(ids), device="cuda").unsqueeze(0)
    with torch.no_grad():
        emb = model.model.embed_tokens(torch.tensor([ids], device="cuda")) * model.config.scale_emb
        h0 = model.model.layers[0](
            emb,
            attention_mask=torch.ones(1, len(ids), device="cuda"),
            position_ids=pos,
            use_cache=False,
        )[0][0].float().cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return h0


def main() -> int:
    _patch_hf()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids = tok.encode(PROMPT, add_special_tokens=True)
    hf = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    with torch.no_grad():
        t1 = int(
            hf(
                torch.tensor([ids], device="cuda"),
                attention_mask=torch.ones(1, len(ids), device="cuda"),
            )
            .logits[0, -1]
            .argmax()
        )
    del hf
    gc.collect()
    torch.cuda.empty_cache()

    ids2 = ids + [t1]
    seq_len = len(ids2)
    print(f"prompt={PROMPT!r} t1={t1} seqlen={seq_len}", flush=True)

    hf_l0 = _hf_l0_last(ids2)
    eng_l0 = _engine_l0_last(ids2)
    if eng_l0 is None:
        print("FAIL: engine layer0 capture empty", flush=True)
        return 1
    l0_diff = (hf_l0 - eng_l0).abs().max().item()
    print(f"layer0_last max_abs_diff={l0_diff:.6g}", flush=True)

    hf_h0_seq = _hf_full_h0(ids2)
    dtype = torch.bfloat16
    hf_l1_attn = _hf_l1_attn_last(ids2)
    v_l1_hf_h0 = _l1_last_with_h0_seq(hf_h0_seq.to(dtype), seq_len, dtype)

    hybrid = hf_h0_seq.clone()
    hybrid[-1] = eng_l0
    v_l1_hybrid = _l1_last_with_h0_seq(hybrid.to(dtype), seq_len, dtype)

    d_hf_h0 = (hf_l1_attn - v_l1_hf_h0).abs().max().item()
    d_hybrid = (hf_l1_attn - v_l1_hybrid).abs().max().item()
    print(f"l1_attn vLLM(HF_h0_seq) max_abs_diff={d_hf_h0:.6g}", flush=True)
    print(f"l1_attn vLLM(HF_seq+engine_last) max_abs_diff={d_hybrid:.6g}", flush=True)

    hf_l1_full = _hf_l1_last(ids2)
    v_l1_full_hf = _l1_full_with_h0_seq(hf_h0_seq.to(dtype), seq_len, dtype)
    v_l1_full_hybrid = _l1_full_with_h0_seq(hybrid.to(dtype), seq_len, dtype)
    d_full_hf = (hf_l1_full - v_l1_full_hf).abs().max().item()
    d_full_hybrid = (hf_l1_full - v_l1_full_hybrid).abs().max().item()
    print(f"l1_full vLLM(HF_h0_seq) max_abs_diff={d_full_hf:.6g}", flush=True)
    print(f"l1_full vLLM(HF_seq+engine_last) max_abs_diff={d_full_hybrid:.6g}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
