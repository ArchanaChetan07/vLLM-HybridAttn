#!/usr/bin/env python3
"""Compare HF vs vLLM lightning layer internals (q/k after RoPE, raw GLA out)."""

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
PROMPT = "Hello, my name is"


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
        o, _ = layer.attn_fn(
            q=qf, k=kf, v=vf, decay=slopes, scale=layer.scale, initial_state=None
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
    from vllm.model_executor.model_loader import get_model_loader

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
            with torch.no_grad():
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
                from vllm.model_executor.models.minicpm_sala import (
                    _apply_hf_rotary_bhtd,
                )

                q, k = _apply_hf_rotary_bhtd(q, k, positions, attn.rope_inv_freq)
                out["q_rope"] = q[-1, :, :8].float().cpu()
                from fla.ops.simple_gla import fused_recurrent_simple_gla

                g_gamma = (-attn.tp_slope.to(torch.float32)).reshape(h)
                qf = q.transpose(0, 1).unsqueeze(0).to(torch.float32)
                kf = k.transpose(0, 1).unsqueeze(0).to(torch.float32)
                vf = v.transpose(0, 1).unsqueeze(0).to(torch.float32)
                o, _ = fused_recurrent_simple_gla(
                    q=qf,
                    k=kf,
                    v=vf,
                    g_gamma=g_gamma,
                    scale=attn.scale,
                    initial_state=None,
                    output_final_state=True,
                )
                out["gla"] = o[0, -1, 0, :8].float().cpu()
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
    ids = tok.encode(PROMPT, return_tensors="pt").to("cuda")
    pos = torch.arange(ids.shape[1], device="cuda").unsqueeze(0)
    positions = torch.arange(ids.shape[1], device="cuda", dtype=torch.long)
    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    with torch.no_grad():
        h = model.model.embed_tokens(ids) * model.config.scale_emb
        mask = torch.ones_like(ids)
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
    for key in ("q_norm", "q_rope", "gla"):
        d = (hf[key] - vv[key]).abs().max().item()
        print(f"{key} max_abs_diff={d:.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
