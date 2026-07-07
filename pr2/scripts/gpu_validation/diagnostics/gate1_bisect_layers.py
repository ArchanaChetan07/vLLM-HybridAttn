#!/usr/bin/env python3
"""Bisect HF vs vLLM: embed, layer0, layer1 attn-only, full layer1."""

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
PROMPT = "Hello, my name is"


def _patch_hf() -> None:
    script = "/workspace/hybridattn/scripts/remote/patch_hf_transformers_compat.py"
    if os.path.isfile(script):
        subprocess.run([sys.executable, script], check=False)


def _diff(a: torch.Tensor, b: torch.Tensor, label: str) -> None:
    d = (a.float() - b.float()).abs()
    print(f"{label} max_abs={d.max().item():.6g} mean_abs={d.mean().item():.6g}")


def hf_trace() -> dict[str, torch.Tensor]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids = tok.encode(PROMPT, return_tensors="pt").to("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    traces: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        emb = model.model.embed_tokens(ids) * model.config.scale_emb
        traces["embed"] = emb[0, -1].float().cpu()
        pos = torch.arange(ids.shape[1], device="cuda").unsqueeze(0)
        mask = torch.ones_like(ids)
        h = emb
        out0 = model.model.layers[0](
            h, attention_mask=mask, position_ids=pos, use_cache=False
        )
        h0 = out0[0] if isinstance(out0, tuple) else out0
        traces["layer0"] = h0[0, -1].float().cpu()
        # layer1 attn branch only
        l1 = model.model.layers[1]
        res = h0
        x = l1.input_layernorm(h0)
        attn_out, _, _ = l1.self_attn(
            x, attention_mask=mask, position_ids=pos, use_cache=False
        )
        traces["l1_attn"] = attn_out[0, -1].float().cpu()
        out1 = l1(h0, attention_mask=mask, position_ids=pos, use_cache=False)
        h1 = out1[0] if isinstance(out1, tuple) else out1
        traces["layer1"] = h1[0, -1].float().cpu()
        logits = model.lm_head(
            model.model.norm(h1)
            / (model.config.hidden_size / model.config.dim_model_base)
        )
        traces["greedy"] = logits[0, -1].argmax().cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return traces


def vllm_trace() -> dict[str, torch.Tensor]:
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
    from vllm.v1.attention.backends.linear_attn import LinearAttentionMetadata

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids = tok.encode(PROMPT, return_tensors="pt").to("cuda")
    seq_len = ids.shape[1]
    positions = torch.arange(seq_len, device="cuda", dtype=torch.long)

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
    traces: dict[str, torch.Tensor] = {}
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

            # layer0 is sparse — use full model forward for embed+layer0 via ids
            # Build minimal sparse metadata is hard; run layers[0] via engine-style
            # hidden [T, H] layout.
            with torch.no_grad():
                emb = model.model.get_input_embeddings(ids.squeeze(0))
                traces["embed"] = emb[-1].float().cpu()

            layer0 = model.model.layers[0]
            with torch.no_grad():
                h_in = emb
                h_out = layer0(positions, h_in)
                traces["layer0"] = h_out[-1].float().cpu()

            layer1 = model.model.layers[1]
            attn = layer1.self_attn
            prefix = attn.prefix
            state_shape = attn.get_state_shape()
            attn.kv_cache = (
                torch.zeros(
                    1, *state_shape[0], device="cuda", dtype=attn.get_state_dtype()[0]
                ),
            )
            ln_meta = LinearAttentionMetadata(
                num_prefills=1,
                num_prefill_tokens=seq_len,
                num_decodes=0,
                num_decode_tokens=0,
                query_start_loc=torch.tensor(
                    [0, seq_len], device="cuda", dtype=torch.int32
                ),
                seq_lens=torch.tensor([seq_len], device="cuda", dtype=torch.int32),
                state_indices_tensor=torch.tensor(
                    [0], device="cuda", dtype=torch.int32
                ),
            )
            with torch.no_grad():
                res = h_out
                x = layer1.input_layernorm(h_out)
                attn_out = torch.zeros_like(x)
                with set_forward_context(
                    attn_metadata={prefix: ln_meta}, vllm_config=vllm_config
                ):
                    attn.forward(
                        hidden_states=x,
                        output=attn_out,
                        positions=positions,
                    )
                traces["l1_attn"] = attn_out[-1].float().cpu()
                h1 = layer1._add_scaled_residual(res, attn_out)
                res2 = h1
                h2 = layer1.post_attention_layernorm(h1)
                h2 = layer1.mlp(h2)
                h1full = layer1._add_scaled_residual(res2, h2)
                traces["layer1"] = h1full[-1].float().cpu()
                logits = model.compute_logits(model.model.norm(h1full))
                traces["greedy"] = logits[-1].argmax().cpu()

            destroy_model_parallel()
            destroy_distributed_environment()
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp_file)
    gc.collect()
    torch.cuda.empty_cache()
    return traces


def main() -> int:
    _patch_hf()
    hf = hf_trace()
    v = vllm_trace()
    for key in ("embed", "layer0", "l1_attn", "layer1"):
        _diff(hf[key], v[key], key)
    print(f"HF greedy={int(hf['greedy'])} vLLM greedy={int(v['greedy'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
