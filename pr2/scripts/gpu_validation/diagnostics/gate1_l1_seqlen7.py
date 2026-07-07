#!/usr/bin/env python3
"""Layer-1 attn compare at last position for prompt+t1 (7-token prefill)."""

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


def main() -> int:
    script = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "remote",
        "patch_hf_transformers_compat.py",
    )
    if os.path.isfile(script):
        subprocess.run([sys.executable, os.path.normpath(script)], check=False)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids = tok.encode(PROMPT, add_special_tokens=True)
    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    with torch.no_grad():
        t1 = int(
            model(
                torch.tensor([ids], device="cuda"),
                attention_mask=torch.ones(1, len(ids), device="cuda"),
            )
            .logits[0, -1]
            .argmax()
        )
    ids2 = ids + [t1]
    seq_len = len(ids2)
    ids_t = torch.tensor([ids2], device="cuda")
    pos = torch.arange(seq_len, device="cuda").unsqueeze(0)
    mask = torch.ones_like(ids_t)

    with torch.no_grad():
        h = model.model.embed_tokens(ids_t) * model.config.scale_emb
        out0 = model.model.layers[0](
            h, attention_mask=mask, position_ids=pos, use_cache=False
        )[0]
        h0 = out0[0]
        x = model.model.layers[1].input_layernorm(h0.unsqueeze(0))
        attn_out, _, _ = model.model.layers[1].self_attn(
            x, attention_mask=mask, position_ids=pos, use_cache=False
        )
        hf_l0 = h0[-1].float().cpu()
        hf_l1 = attn_out[0, -1].float().cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()

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
    vllm_config = VllmConfig(
        model_config=model_config,
        load_config=LoadConfig(),
        cache_config=CacheConfig(block_size=256),
        device_config=DeviceConfig(device="cuda"),
    )
    fd, temp = tempfile.mkstemp()
    os.close(fd)
    positions = torch.arange(seq_len, device="cuda", dtype=torch.long)
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
            vm = get_model_loader(LoadConfig()).load_model(
                vllm_config=vllm_config, model_config=model_config
            )
            vm.eval().cuda()
            layer0 = vm.model.layers[0]
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
                state_indices_tensor=torch.tensor(
                    [0], device="cuda", dtype=torch.int32
                ),
            )
            with torch.no_grad():
                emb = vm.model.get_input_embeddings(torch.tensor(ids2, device="cuda"))
                x = layer1.input_layernorm(h0.to(vm.model.embed_tokens.weight.dtype))
                out = torch.zeros_like(x)
                with set_forward_context(
                    attn_metadata={attn.prefix: meta}, vllm_config=vllm_config
                ):
                    attn.forward(hidden_states=x, output=out, positions=positions)
            l1d = (hf_l1 - out[-1].float().cpu()).abs().max().item()
            print(f"prompt={PROMPT!r} t1={t1} seqlen={seq_len}")
            print(f"hf_layer0_last norm={hf_l0.norm().item():.4g} (vLLM layer0 skipped)")
            print(f"l1_attn_last max_abs_diff={l1d:.6g}")
            destroy_model_parallel()
            destroy_distributed_environment()
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
