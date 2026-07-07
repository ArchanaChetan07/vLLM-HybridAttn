#!/usr/bin/env python3
"""Step 2: run this on your T1000 AFTER step 1 passes. Exercises the REAL
`linear_attention_prefill_and_mix` kernel dispatch (Triton, needs actual
GPU -- this is the piece that could NOT be tested in the CPU-only
sandbox this port was developed in), using real `LinearAttentionMetadata`
fields confirmed by introspection (not guessed):

    num_prefills: int, num_prefill_tokens: int, num_decodes: int,
    num_decode_tokens: int, query_start_loc: Tensor, seq_lens: Tensor,
    state_indices_tensor: Tensor

Single-layer, single short sequence, small enough to comfortably fit a
T1000's 4-8GB VRAM (this is ONE attention layer's weights, ~84M params
~168MB at bf16 -- not the full 9B model, which will NOT fit on this
card; see the note in the previous message).
"""

import os
import tempfile

import torch
from transformers import PretrainedConfig

import vllm.config as vconfig
from vllm.config import CacheConfig, VllmConfig
from vllm.config.device import DeviceConfig
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.forward_context import set_forward_context

REAL_LIGHTNING_CONFIG = {
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "intermediate_size": 16384,
    "vocab_size": 73448,
    "rms_norm_eps": 1e-06,
    "attention_bias": False,
    "lightning_nh": 32,
    "lightning_nkv": 32,
    "lightning_head_dim": 128,
    "lightning_scale": "1/sqrt(d)",
    "lightning_use_rope": True,
    "qk_norm": True,
    "use_output_norm": True,
    "use_output_gate": True,
    "max_position_embeddings": 524288,
    "rope_theta": 10000.0,
    "rope_scaling": None,
}


def main() -> int:
    assert torch.cuda.is_available(), "This script requires a real GPU."
    device = torch.device("cuda:0")

    hf_config = PretrainedConfig(**REAL_LIGHTNING_CONFIG)
    cache_config = CacheConfig()
    vllm_config = VllmConfig(
        cache_config=cache_config, device_config=DeviceConfig(device="cuda")
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
                backend="nccl",  # real GPU -- use vLLM's real default, not gloo
            )
            initialize_model_parallel(1, 1)

            from vllm.model_executor.models.minicpm_sala import (
                MiniCPMSALALightningAttention,
            )
            from vllm.v1.attention.backends.linear_attn import (
                LinearAttentionMetadata,
            )

            print("Constructing lightning layer on GPU ...")
            layer = MiniCPMSALALightningAttention(
                config=hf_config,
                cache_config=cache_config,
                quant_config=None,
                prefix="model.layers.1.self_attn",
            ).to(device=device, dtype=torch.bfloat16)
            n_params = sum(p.numel() for p in layer.parameters())
            print(
                f"Real parameters: {n_params:,} (~{n_params * 2 / 1e6:.1f} MB at bf16)"
            )

            state_shape = layer.get_state_shape()
            print(f"KV (recurrent-state) cache shape: {state_shape}")
            state_dtype = layer.get_state_dtype()[0]
            print(f"KV recurrent state dtype: {state_dtype} (must be fp32 for lightning kernels)")
            # One cache slot; dtype must match get_state_dtype() — bf16 state causes
            # Triton dtype mismatches; fp32 activations are NOT required (bf16 q/k/v OK).
            layer.kv_cache = (
                torch.zeros(
                    1, *state_shape[0], device=device, dtype=state_dtype
                ),
            )

            # A single short prefill sequence, 8 tokens, one request.
            seq_len = 8
            hidden_states = torch.randn(
                seq_len,
                REAL_LIGHTNING_CONFIG["hidden_size"],
                device=device,
                dtype=torch.bfloat16,
            )
            output = torch.zeros_like(hidden_states)
            positions = torch.arange(seq_len, device=device)

            attn_metadata = LinearAttentionMetadata(
                num_prefills=1,
                num_prefill_tokens=seq_len,
                num_decodes=0,
                num_decode_tokens=0,
                query_start_loc=torch.tensor(
                    [0, seq_len], device=device, dtype=torch.int32
                ),
                seq_lens=torch.tensor([seq_len], device=device, dtype=torch.int32),
                state_indices_tensor=torch.tensor(
                    [0], device=device, dtype=torch.int32
                ),
            )
            # Layer dispatches via get_forward_context().attn_metadata[prefix]
            # -- a dict keyed by layer prefix, matching
            # BailingMoELinearAttention/MiniMaxText01LinearAttention's own
            # _forward dispatch convention.
            metadata_dict = {layer.prefix: attn_metadata}

            print(f"Running REAL kernel dispatch (prefill, seq_len={seq_len}) ...")
            with set_forward_context(
                attn_metadata=metadata_dict, vllm_config=vllm_config
            ):
                layer.forward(
                    hidden_states=hidden_states,
                    output=output,
                    positions=positions,
                )
            torch.cuda.synchronize()

            print("KERNEL DISPATCH OK")
            print(f"output.shape={tuple(output.shape)}, dtype={output.dtype}")
            assert not torch.isnan(output).any(), "NaN in output"
            assert not torch.isinf(output).any(), "Inf in output"
            assert output.abs().sum().item() > 0, (
                "output is all zeros -- kernel likely didn't actually run "
                "(check the attn_metadata=None fallback wasn't silently "
                "taken)"
            )
            print("PASS: real kernel executed, output is finite and non-trivial")

            # Second call: decode step (1 new token, continuing the same
            # sequence) -- exercises linear_attention_decode /
            # linear_decode_forward_triton, the OTHER kernel path, not
            # yet tested at all (prefill != decode code path).
            print("\nRunning REAL kernel dispatch (decode, 1 new token) ...")
            decode_hidden = torch.randn(
                1,
                REAL_LIGHTNING_CONFIG["hidden_size"],
                device=device,
                dtype=torch.bfloat16,
            )
            decode_output = torch.zeros_like(decode_hidden)
            decode_positions = torch.tensor([seq_len], device=device)
            decode_metadata = LinearAttentionMetadata(
                num_prefills=0,
                num_prefill_tokens=0,
                num_decodes=1,
                num_decode_tokens=1,
                query_start_loc=torch.tensor([0, 1], device=device, dtype=torch.int32),
                seq_lens=torch.tensor([seq_len + 1], device=device, dtype=torch.int32),
                state_indices_tensor=torch.tensor(
                    [0], device=device, dtype=torch.int32
                ),
            )
            with set_forward_context(
                attn_metadata={layer.prefix: decode_metadata},
                vllm_config=vllm_config,
            ):
                layer.forward(
                    hidden_states=decode_hidden,
                    output=decode_output,
                    positions=decode_positions,
                )
            torch.cuda.synchronize()
            print("DECODE KERNEL DISPATCH OK")
            print(f"decode_output.shape={tuple(decode_output.shape)}")
            assert not torch.isnan(decode_output).any()
            assert not torch.isinf(decode_output).any()
            print("PASS: decode-path kernel executed, output is finite")

    finally:
        import contextlib

        with contextlib.suppress(OSError):
            os.unlink(temp_file)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
