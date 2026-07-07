#!/usr/bin/env python3
"""Compare HF vs vLLM lightning layer internals on the production path.

vLLM mirrors ``MiniCPMSALALightningAttention._forward``: qk_norm, identity
RoPE (``zeros_like`` q/k), then ``fused_recurrent_simple_gla`` for seqlen<64.
"""

from __future__ import annotations

import contextlib
import gc
import os
import subprocess
import sys
import tempfile

import torch
from einops import rearrange

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = os.environ.get("MINICPM_SALA_PROMPT", "Hello, my name is")
MODE = os.environ.get("MINICPM_SALA_MODE", "prompt")


def _patch_hf() -> None:
    script = "/workspace/hybridattn/scripts/remote/patch_hf_transformers_compat.py"
    if os.path.isfile(script):
        subprocess.run([sys.executable, script], check=False)


def hf_internals(x: torch.Tensor, pos: torch.Tensor) -> dict[str, torch.Tensor]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    layer = model.model.layers[1].self_attn
    modeling = sys.modules[type(layer).__module__]
    out: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        q = layer.q_proj(x)
        k = layer.k_proj(x)
        v = layer.v_proj(x)
        q = rearrange(q, "b t (h d) -> b h t d", d=layer.head_dim)
        k = rearrange(k, "b t (h d) -> b h t d", d=layer.head_dim)
        v = rearrange(v, "b t (h d) -> b h t d", d=layer.head_dim)
        if layer.qk_norm:
            q = layer.q_norm(q)
            k = layer.k_norm(k)
        out["q_norm"] = q[0, :, -1, :8].float().cpu()
        if layer.use_rope:
            kv_seq_len = pos.max().item() + 1
            cos, sin = layer.rotary_emb(v.to(torch.float32), seq_len=kv_seq_len)
            q, k = modeling.apply_rotary_pos_emb(q, k, cos, sin, pos)
        out["q_rope"] = q[0, :, -1, :8].float().cpu()
        slopes = modeling._build_slope_tensor(layer.num_attention_heads).to(
            "cuda", dtype=torch.float32
        ) * (-1.0)
        qf = rearrange(q, "b h t d -> b t h d").to(torch.float32)
        kf = rearrange(k, "b h t d -> b t h d").to(torch.float32)
        vf = rearrange(v, "b h t d -> b t h d").to(torch.float32)
        seqlen = qf.shape[1]
        from fla.ops.simple_gla import chunk_simple_gla, fused_recurrent_simple_gla

        gla_fn = (
            fused_recurrent_simple_gla if seqlen < 64 else chunk_simple_gla
        )
        o, _ = gla_fn(
            q=qf, k=kf, v=vf, g_gamma=slopes, scale=layer.scale, initial_state=None
        )
        out["gla"] = o[0, -1, 0, :8].float().cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out


def vllm_internals(x: torch.Tensor, positions: torch.Tensor) -> dict[str, torch.Tensor]:
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
    from vllm.model_executor.models.minicpm_sala import (
        _minicpm_sala_lightning_forward_prefix,
    )
    from vllm.v1.attention.backends.linear_attn import LinearAttentionMetadata

    out: dict[str, torch.Tensor] = {}
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
    fd, temp_file = tempfile.mkstemp()
    os.close(fd)
    seq_len = x.shape[0]
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
            vm = get_model_loader(load_config).load_model(
                vllm_config=vllm_config, model_config=model_config
            )
            attn = vm.model.layers[1].self_attn
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
            attn_out = torch.zeros_like(x)
            with torch.no_grad():
                with set_forward_context(
                    attn_metadata={attn.prefix: meta}, vllm_config=vllm_config
                ):
                    attn._forward(
                        hidden_states=x,
                        output=attn_out,
                        positions=positions,
                    )
                out["gla"] = attn_out[-1, :8].float().cpu()

                qkv, _ = attn.qkv_proj(x)
                h, d = attn.tp_heads, attn.head_dim
                q, k, v = qkv.split([h * d, h * d, h * d], dim=-1)
                q = q.view(-1, h, d)
                k = k.view(-1, h, d)
                v = v.view(-1, h, d)
                if attn.qk_norm:
                    q = attn.q_norm(q)
                    k = attn.k_norm(k)
                out["q_norm"] = q[-1, :, :8].float().cpu()
                q = torch.zeros_like(q)
                k = torch.zeros_like(k)
                out["q_rope"] = q[-1, :, :8].float().cpu()

                q4 = q.transpose(0, 1).unsqueeze(0)
                k4 = k.transpose(0, 1).unsqueeze(0)
                v4 = v.transpose(0, 1).unsqueeze(0)
                kv = attn.kv_cache[0]
                gla_flat = _minicpm_sala_lightning_forward_prefix(
                    q4,
                    k4,
                    v4,
                    kv,
                    attn.tp_slope,
                    attn.block_size,
                    scale=attn.scale,
                )
                out["gla_kernel"] = gla_flat[-1, :8].float().cpu()
            destroy_model_parallel()
            destroy_distributed_environment()
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp_file)
    return out


def main() -> int:
    _patch_hf()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids = tok.encode(PROMPT, add_special_tokens=True)
    if MODE == "prompt_plus_t1":
        m = AutoModelForCausalLM.from_pretrained(
            WEIGHTS,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="flash_attention_2",
        ).eval()
        with torch.no_grad():
            t1 = int(
                m(
                    torch.tensor([ids], device="cuda"),
                    attention_mask=torch.ones(1, len(ids), device="cuda"),
                )
                .logits[0, -1]
                .argmax()
            )
        ids = ids + [t1]
        del m
        gc.collect()
        torch.cuda.empty_cache()
        print(f"mode=prompt_plus_t1 t1={t1} seqlen={len(ids)}", flush=True)
    ids_t = torch.tensor([ids], device="cuda")
    pos = torch.arange(len(ids), device="cuda").unsqueeze(0)
    positions = torch.arange(len(ids), device="cuda", dtype=torch.long)
    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    with torch.no_grad():
        h = model.model.embed_tokens(ids_t) * model.config.scale_emb
        mask = torch.ones_like(ids_t)
        h0 = model.model.layers[0](
            h, attention_mask=mask, position_ids=pos, use_cache=False
        )[0]
        x = model.model.layers[1].input_layernorm(h0)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    x_v = x.squeeze(0) if x.dim() == 3 else x
    hf = hf_internals(x, pos)
    vv = vllm_internals(x_v, positions)
    print(f"prompt={PROMPT!r}", flush=True)
    for key in ("q_norm", "q_rope", "gla", "gla_kernel"):
        if key not in hf and key == "gla_kernel":
            d = (vv["gla"] - vv["gla_kernel"]).abs().max().item()
            print(f"_forward vs kernel gla max_abs_diff={d:.6g}")
            continue
        if key not in hf:
            continue
        d = (hf[key] - vv[key]).abs().max().item()
        print(f"{key} max_abs_diff={d:.6g}")
    if "gla_kernel" in vv:
        d = (vv["gla"] - vv["gla_kernel"]).abs().max().item()
        print(f"_forward vs kernel gla max_abs_diff={d:.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
