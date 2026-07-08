# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from
# https://huggingface.co/openbmb/MiniCPM-SALA/blob/main/modeling_minicpm_sala.py
# (OpenBMB, Apache-2.0)
#
# PR2 merged model: dense GQA fallback plus optional InfLLM-V2 sparse backend
# when infllm_v2 is installed (see minicpm_sala_sparse_wiring.py).
"""Inference-only MiniCPM-SALA model compatible with HuggingFace weights.

Pinned reference: vllm-project/vllm @ 8cfeb84dba41a0c56570334757d921abd71e5288
(main, 2026-07-01). API surface referenced here (Attention, LinearAttention,
MambaBase, RMSNorm, get_rope, AutoWeightsLoader, HasInnerState/IsHybrid
interfaces, PluggableLayer) was read directly from that commit, not from
training-data recollection -- see the accompanying architecture report for
citations to the exact source files.
"""

import json
import math
import os
import time
from collections.abc import Iterable
from pathlib import Path
from functools import partial
from itertools import islice

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor

# TP-aware output RMSNorm for the lightning layer. The reference HF model
# norms over the FULL (num_heads * head_dim) inner vector as a single group;
# under TP each rank holds only a (tp_heads * head_dim) shard, so a plain
# RMSNorm would (a) mis-shape its weight and (b) compute the RMS statistic
# over the local shard instead of the full vector. This is the exact same
# reason MiniMaxText01LinearAttention uses MiniMaxText01RMSNormTP for its
# output norm; at TP=1 it degrades to a standard RMSNorm.
from vllm.model_executor.layers.minimax_rms_norm import MiniMaxText01RMSNormTP
from vllm.model_executor.layers.mamba.abstract import MambaBase

# Reusing the SHARED prefill/decode dispatch helpers from
# minimax_linear_attn.py rather than duplicating them: these are the same
# functions the currently-ACTIVE `BailingMoELinearAttention`
# (vllm/model_executor/layers/mamba/linear/bailing_linear_attn.py) reuses
# from this module -- confirming this is the established, non-model-
# specific idiom for adding a new gated-linear-attention layer, not
# something specific to MiniMax. NOTE: the MiniMax *model* wrapper
# (`MiniMaxText01ForCausalLM`) is itself deprecated -- it appears in
# vllm/model_executor/models/registry.py's `_PREVIOUSLY_SUPPORTED_MODELS`
# as removed at v0.23.0 -- but this kernel-dispatch module is still live,
# actively-imported infrastructure, not dead code.
from vllm.model_executor.layers.mamba.linear.minimax_linear_attn import (
    clear_linear_attention_cache_for_new_sequences,
    linear_attention_decode,
    linear_attention_prefill_and_mix,
)
from vllm.model_executor.layers.lightning_attn import lightning_attention
from einops import rearrange


def _agent_debug_log(
    location: str,
    message: str,
    data: dict,
    hypothesis_id: str,
    run_id: str = "pre-fix",
) -> None:
    if os.environ.get("MINICPM_SALA_DEBUG_GLA", "") != "1":
        return
    layer_idx = data.get("layer_idx")
    if layer_idx is not None and layer_idx not in (1, -1):
        return
    # #region agent log
    log_path = os.environ.get("DEBUG_LOG_PATH", str(Path.cwd() / "debug-212a6e.log"))
    payload = {
        "sessionId": "212a6e",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except OSError:
        pass
    # #endregion


from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.linear_attn import LinearAttentionMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from .interfaces import HasInnerState, IsHybrid, SupportsPP
from .minicpm_sala_parity import ensure_native_rms_norm_kernels as _ensure_native_rms_norm_kernels
from .minicpm_sala_sparse_wiring import create_sparse_attention_if_available

# ---------------------------------------------------------------------------
# Layer-schedule helpers (pure functions, unit-testable without torch/CUDA --
# see tests/models/test_minicpm_sala_schedule.py)
# ---------------------------------------------------------------------------

_SPARSE_MIXER_NAMES = {"minicpm4"}
_LIGHTNING_MIXER_NAMES = {"lightning", "lightning_attn", "lightning-attn"}


def validate_mixer_schedule(mixer_types: list[str]) -> None:
    """Validate the per-layer mixer schedule against the invariant enforced
    by the reference implementation's `MiniCPMSALACache.__init__`:
    layer 0 must be a sparse ("minicpm4") layer, because the reference cache
    class piggybacks its global `_seen_tokens` counter on layer 0's update
    call. vLLM's own KV-cache manager tracks sequence length independently
    of any single layer, so this invariant is NOT load-bearing for
    correctness here -- but it is validated anyway as a config sanity check
    and an explicit acknowledgment of the coupling documented in the
    Phase-1 architecture report, section 4.
    """
    if not mixer_types:
        raise ValueError("mixer_types must be non-empty")
    if mixer_types[0] not in _SPARSE_MIXER_NAMES:
        raise ValueError(
            "MiniCPM-SALA reference invariant: layer 0 must be a "
            f"'minicpm4' (sparse) layer, got {mixer_types[0]!r}. This "
            "matches the upstream HF config for the released checkpoint; "
            "if you are configuring a hypothetical variant that violates "
            "this, the port's cache bookkeeping needs re-verification "
            "before trusting the result."
        )
    for idx, mixer in enumerate(mixer_types):
        if mixer not in _SPARSE_MIXER_NAMES | _LIGHTNING_MIXER_NAMES:
            raise ValueError(
                f"Unsupported mixer_types[{idx}] = {mixer!r}; expected one "
                f"of {_SPARSE_MIXER_NAMES | _LIGHTNING_MIXER_NAMES}"
            )


def is_sparse_layer(mixer_type: str) -> bool:
    return mixer_type in _SPARSE_MIXER_NAMES


def is_lightning_layer(mixer_type: str) -> bool:
    return mixer_type in _LIGHTNING_MIXER_NAMES


def _lightning_prefill_starts_at_position_zero(
    attn_metadata: LinearAttentionMetadata,
    positions: torch.Tensor,
) -> bool:
    """True when any prefill chunk in this forward starts at position 0."""
    offset = attn_metadata.num_decode_tokens
    for prefill_idx in range(attn_metadata.num_prefills):
        q_start = int(attn_metadata.query_start_loc[offset + prefill_idx].item())
        if int(positions[q_start].item()) == 0:
            return True
    return False


def _lightning_should_reset_qkv_history(
    attn_metadata: LinearAttentionMetadata,
    positions: torch.Tensor,
) -> bool:
    """True when q/k/v history must drop stale tokens for a fresh GLA slot.

    Mirrors ``clear_linear_attention_cache_for_new_sequences`` (``context_len
    == 0``) plus the position-0 engine-prefill guard in
    ``_clear_lightning_state_for_engine_prefill``. Resetting only on position
    0 misses new sequences whose inflated ``seq_lens`` skip the cache clear.
    """
    offset = attn_metadata.num_decode_tokens
    for prefill_idx in range(attn_metadata.num_prefills):
        q_start = int(attn_metadata.query_start_loc[offset + prefill_idx].item())
        q_end = int(attn_metadata.query_start_loc[offset + prefill_idx + 1].item())
        if int(positions[q_start].item()) == 0:
            return True
        query_len = q_end - q_start
        context_len = int(attn_metadata.seq_lens[offset + prefill_idx].item()) - query_len
        if context_len == 0:
            return True
    return False


def _lightning_target_hist_len(
    attn_metadata: LinearAttentionMetadata | None,
) -> int | None:
    """Expected q/k/v history length after syncing this forward's tokens."""
    if attn_metadata is None:
        return None
    if attn_metadata.num_decode_tokens > 0:
        return int(attn_metadata.seq_lens[0].item())
    offset = attn_metadata.num_decode_tokens
    if attn_metadata.num_prefills > 0:
        return int(attn_metadata.seq_lens[offset].item())
    return None


def _clear_lightning_state_for_engine_prefill(
    kv_cache: torch.Tensor,
    state_indices_tensor: torch.Tensor,
    attn_metadata: LinearAttentionMetadata,
    positions: torch.Tensor,
) -> None:
    """Clear recurrent GLA state for fresh prompt prefills in EngineCore.

    ``clear_linear_attention_cache_for_new_sequences`` only clears when
    ``seq_lens - query_len == 0``. The engine can report inflated ``seq_lens``
    on a first-chunk prefill, leaving stale slot data. Also clear when this
    prefill chunk starts at position 0 (new request), including chunked
    prefill's first chunk.
    """
    clear_linear_attention_cache_for_new_sequences(
        kv_cache, state_indices_tensor, attn_metadata
    )
    offset = attn_metadata.num_decode_tokens
    for prefill_idx in range(attn_metadata.num_prefills):
        q_start = int(attn_metadata.query_start_loc[offset + prefill_idx].item())
        if int(positions[q_start].item()) == 0:
            slot = int(state_indices_tensor[offset + prefill_idx].item())
            kv_cache[slot, ...] = 0


def build_alibi_slopes(num_heads: int) -> torch.Tensor:
    """Byte-for-byte port of `_build_slope_tensor` from the reference
    `modeling_minicpm_sala.py`. This is the SAME algorithm already used by
    vLLM's in-tree `MiniMaxText01LinearAttention._build_slope_tensor`
    (vllm/model_executor/layers/mamba/linear/minimax_linear_attn.py) --
    confirmed by direct comparison, not assumed. Kept as a local copy
    rather than importing the MiniMax version because MiniCPM-SALA does
    NOT apply MiniMax's additional per-layer decay scaling
    (`1 - layer_idx/(num_hidden_layers-1)`); reusing the MiniMax class
    directly would silently introduce that extra scaling. This divergence
    is exactly the kind of thing flagged in Phase 2/3 as a "cannot
    literally reuse, must verify" item.
    """

    def get_slopes_power_of_2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * start**i for i in range(n)]

    def get_slopes(n: int) -> list[float]:
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )

    return torch.tensor(get_slopes(num_heads), dtype=torch.float32)


def build_lightning_decay_rate(num_heads: int) -> torch.Tensor:
    """Per-head decay RATE fed to the lightning-attention kernels.

    Single source of truth for the decay SIGN convention. The reused vLLM
    kernels (``lightning_attention`` / ``linear_decode_forward_triton``)
    apply the decay internally as ``exp(-rate * distance)``, so ``rate``
    MUST be strictly positive for a bounded decay -- exactly matching how
    ``MiniMaxText01LinearAttention`` feeds its own positive slope.
    ``build_alibi_slopes`` already returns the positive slope magnitude, so
    this is a straight pass-through; it exists as a named, unit-tested seam
    (see ``test_lightning_decay_rate_is_strictly_positive``) specifically so
    that a stray ``* -1.0`` cannot silently reintroduce the ``exp(+|s|*d)``
    overflow-to-NaN regression this replaced.
    """
    return build_alibi_slopes(num_heads)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _build_rope_inv_freq(head_dim: int, rope_theta: float) -> torch.Tensor:
    return 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )


def _apply_hf_rotary_bhtd(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match HF ``apply_rotary_pos_emb`` on ``[num_tokens, heads, head_dim]``."""
    dtype = q.dtype
    seq_len = int(positions.max().item()) + 1
    t = torch.arange(seq_len, device=q.device, dtype=inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)
    q4 = q.transpose(0, 1).unsqueeze(0)
    k4 = k.transpose(0, 1).unsqueeze(0)
    pos_ids = positions.unsqueeze(0)
    cos_p = cos[pos_ids].unsqueeze(1)
    sin_p = sin[pos_ids].unsqueeze(1)
    q4f = q4.float()
    k4f = k4.float()
    q4 = (q4f * cos_p + _rotate_half(q4f) * sin_p).to(dtype)
    k4 = (k4f * cos_p + _rotate_half(k4f) * sin_p).to(dtype)
    return q4.squeeze(0).transpose(0, 1), k4.squeeze(0).transpose(0, 1)


def _minicpm_sala_lightning_forward_prefix(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_caches: torch.Tensor,
    slope_rate: torch.Tensor,
    block_size: int,
    layer_idx: int | None = None,
    scale: float | None = None,
    *,
    fresh_sequence: bool = False,
    **kwargs,
) -> torch.Tensor:
    """HF-matched lightning prefill via ``fla`` simple_gla kernels.

    The reference ``LightningAttention.attn_fn`` uses
    ``fused_recurrent_simple_gla`` when ``seqlen < 64`` and
    ``chunk_simple_gla`` otherwise, with fp32 q/k/v, ``g_gamma = -slope``,
    and ``scale = head_dim**-0.5``. vLLM's native ``lightning_attention``
    Triton kernel is retained only as a fallback when ``fla`` is absent.
    """
    debug_layer = layer_idx
    del layer_idx, kwargs
    try:
        from fla.ops.simple_gla import chunk_simple_gla, fused_recurrent_simple_gla
    except ImportError:
        fused_recurrent_simple_gla = None  # type: ignore[misc, assignment]
        chunk_simple_gla = None  # type: ignore[misc, assignment]

    should_pad_dim = q.dim() == 3
    if should_pad_dim:
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
    _b, h, n, d = q.shape
    e = d
    attn_scale = scale if scale is not None else d**-0.5

    if fused_recurrent_simple_gla is not None and chunk_simple_gla is not None:
        g_gamma = (-slope_rate.to(torch.float32)).reshape(h)
        q_bthd = rearrange(q, "b h t d -> b t h d").to(torch.float32)
        k_bthd = rearrange(k, "b h t d -> b t h d").to(torch.float32)
        v_bthd = rearrange(v, "b h t d -> b t h d").to(torch.float32)
        initial_state = kv_caches.reshape(1, h, d, e).contiguous().to(torch.float32)
        if fresh_sequence or initial_state.abs().sum().item() == 0.0:
            # HF reference passes ``initial_state=None`` on a fresh sequence
            # (no ``past_key_value``); zeros are not equivalent in fla.
            initial_state = None
        gla_fn = fused_recurrent_simple_gla if n < 64 else chunk_simple_gla
        # #region agent log
        _agent_debug_log(
            "minicpm_sala.py:_minicpm_sala_lightning_forward_prefix",
            "gla recompute",
            {
                "layer_idx": debug_layer if debug_layer is not None else -1,
                "n": int(n),
                "fresh_sequence": fresh_sequence,
                "initial_state_none": initial_state is None,
                "g_gamma0": float(g_gamma[0].item()),
                "scale": float(attn_scale),
            },
            "C",
            run_id=os.environ.get("DEBUG_RUN_ID", "pre-fix"),
        )
        # #endregion
        o, final_state = gla_fn(
            q=q_bthd,
            k=k_bthd,
            v=v_bthd,
            g_gamma=g_gamma,
            scale=attn_scale,
            initial_state=initial_state,
            output_final_state=True,
        )
        kv_caches.copy_(final_state.reshape(h, d, e).to(kv_caches.dtype))
        o = rearrange(o.to(q.dtype), "b t h d -> b h t d")
        assert o.shape[0] == 1, "batch size must be 1"
        return rearrange(o.squeeze(0), "h n d -> n (h d)")

    slope_rate = slope_rate.to(q.dtype)
    kv_history = kv_caches.reshape(1, h, d, e).contiguous()
    output, kv_history = lightning_attention(
        q, k, v, slope_rate, block_size=block_size, kv_history=kv_history
    )
    kv_caches.copy_(kv_history[:, :, -1, :, :].reshape(h, d, e))
    assert output.shape[0] == 1, "batch size must be 1"
    return rearrange(output.squeeze(0), "h n d -> n (h d)")


# ---------------------------------------------------------------------------
# MLP (unchanged SwiGLU -- identical to LlamaMLP, kept local for clarity of
# the residual-scaling call site in the decoder layer)
# ---------------------------------------------------------------------------


class MiniCPMSALAMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.intermediate_size = intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _minicpm_mlp_forward(self, x)


def _minicpm_mlp_forward(mlp: MiniCPMSALAMLP, x: torch.Tensor) -> torch.Tensor:
    """SwiGLU MLP with separate gate/up matmuls at TP=1 (HF parity).

    vLLM's ``MergedColumnParallelLinear`` fuses gate+up into one bf16 GEMM; HF
    runs ``gate_proj`` and ``up_proj`` separately.
    """
    if mlp.gate_up_proj.tp_size != 1:
        gate_up, _ = mlp.gate_up_proj(x)
        hidden = mlp.act_fn(gate_up)
        out, _ = mlp.down_proj(hidden)
        return out

    w = mlp.gate_up_proj.weight
    inter = mlp.intermediate_size
    gate_w, up_w = w[:inter], w[inter : 2 * inter]
    use_fp32 = os.environ.get("MINICPM_SALA_FP32_MLP", "").lower() in (
        "1",
        "true",
        "yes",
    )
    xin = x.float() if use_fp32 else x
    wf = w.float() if use_fp32 else w
    gate_w, up_w = wf[:inter], wf[inter : 2 * inter]
    gate = torch.nn.functional.linear(xin, gate_w)
    up = torch.nn.functional.linear(xin, up_w)
    hidden = torch.nn.functional.silu(gate) * up
    if use_fp32:
        hidden = hidden.to(x.dtype)
    out, _ = mlp.down_proj(hidden)
    return out


# ---------------------------------------------------------------------------
# Dense GQA attention for "minicpm4" mixer layers (PR1)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Dense GQA attention (minicpm4 / sparse-index layers below dense_len)
# ---------------------------------------------------------------------------


def _dense_o_proj(
    o_proj: RowParallelLinear,
    attn_output: torch.Tensor,
) -> torch.Tensor:
    """Attention output projection; fp32 accumulation at TP=1 for HF parity."""
    if o_proj.tp_size == 1 and os.environ.get(
        "MINICPM_SALA_BF16_O_PROJ", ""
    ).lower() not in ("1", "true", "yes"):
        bias = o_proj.bias
        out = torch.nn.functional.linear(
            attn_output.float(),
            o_proj.weight.float(),
            bias.float() if bias is not None else None,
        )
        return out.to(dtype=attn_output.dtype)
    output, _ = o_proj(attn_output)
    return output


def _minicpm_qkv_proj(
    qkv_proj: QKVParallelLinear,
    hidden_states: torch.Tensor,
    q_size: int,
    k_size: int,
    v_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Q/K/V as three separate linears to match HF accumulation order (TP=1).

    vLLM's fused ``QKVParallelLinear`` uses one bf16 GEMM; HF runs separate
    ``q_proj`` / ``k_proj`` / ``v_proj``. Optional fp32 via
    ``MINICPM_SALA_FP32_QKV_PROJ=1``.
    """
    if qkv_proj.tp_size != 1:
        qkv, _ = qkv_proj(hidden_states)
        return qkv.split([q_size, k_size, v_size], dim=-1)

    w = qkv_proj.weight
    b = qkv_proj.bias
    use_fp32 = os.environ.get("MINICPM_SALA_FP32_QKV_PROJ", "").lower() in (
        "1",
        "true",
        "yes",
    )
    x = hidden_states.float() if use_fp32 else hidden_states
    wf = w.float() if use_fp32 else w
    off_k = q_size
    off_v = q_size + k_size
    q_w, k_w, v_w = wf[:q_size], wf[off_k:off_v], wf[off_v : off_v + v_size]
    if b is not None:
        bf = b.float() if use_fp32 else b
        q_b, k_b, v_b = bf[:q_size], bf[off_k:off_v], bf[off_v : off_v + v_size]
    else:
        q_b = k_b = v_b = None
    q = torch.nn.functional.linear(x, q_w, q_b)
    k = torch.nn.functional.linear(x, k_w, k_b)
    v = torch.nn.functional.linear(x, v_w, v_b)
    if use_fp32:
        dtype = hidden_states.dtype
        q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    return q, k, v


def _dense_qkv_proj(
    qkv_proj: QKVParallelLinear,
    hidden_states: torch.Tensor,
    q_size: int,
    kv_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _minicpm_qkv_proj(qkv_proj, hidden_states, q_size, kv_size, kv_size)


class MiniCPMSALADenseAttention(nn.Module):
    """Dense causal GQA for ``minicpm4`` layers (NoPE, optional output gate).

    Uses vLLM's standard ``Attention`` with auto-selected backend. This matches
    the reference model for contexts below ``sparse_config.dense_len``. PR2 adds
    optional InfLLM-V2 sparse execution via ``minicpm_sala_sparse_wiring.py``.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        attention_bias = getattr(config, "attention_bias", False)

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.use_output_gate = getattr(config, "attn_use_output_gate", True)
        if self.use_output_gate:
            # ColumnParallelLinear (NOT ReplicatedLinear): the attention
            # output this gate multiplies is TP-sharded to
            # (num_heads // tp) * head_dim per rank, so the gate must be
            # sharded the same way. ReplicatedLinear would emit the full
            # (total_num_heads * head_dim) on every rank and fail to
            # broadcast against the sharded attention output under TP>1.
            # At TP=1 this is identical to the previous behavior.
            self.o_gate = ColumnParallelLinear(
                self.hidden_size,
                self.total_num_heads * self.head_dim,
                bias=attention_bias,
                gather_output=False,
                quant_config=quant_config,
                prefix=f"{prefix}.o_gate",
            )

        sparse_attn = create_sparse_attention_if_available(
            config,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.attn = sparse_attn

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        q, k, v = _dense_qkv_proj(
            self.qkv_proj, hidden_states, self.q_size, self.kv_size
        )
        # No RoPE applied here -- see class docstring.
        attn_output = self.attn(q, k, v)
        if self.use_output_gate:
            gate, _ = self.o_gate(hidden_states)
            # HF applies sigmoid in fp32 before the bf16 gate multiply.
            attn_output = attn_output * torch.sigmoid(gate.float()).to(
                attn_output.dtype
            )
        return _dense_o_proj(self.o_proj, attn_output)


# ---------------------------------------------------------------------------
# Lightning Attention (gated linear attention) -- full Stage-1 implementation
# ---------------------------------------------------------------------------


class MiniCPMSALALightningAttention(PluggableLayer, MambaBase):
    """Gated linear attention, reusing vLLM's native `lightning_attention`
    (chunked prefill) / `linear_decode_forward_triton` (decode) kernels --
    the same kernel family as the in-tree `MiniMaxText01LinearAttention`
    (vllm/model_executor/layers/mamba/linear/minimax_linear_attn.py).

    Confirmed identical to the MiniMax layer:
      * `_build_slope_tensor` algorithm (see `build_alibi_slopes` above).
      * output path shape: norm(o) -> sigmoid-gate(hidden_states) * o -> o_proj.

    Confirmed DIFFERENT from the MiniMax layer (all three verified against
    `LightningAttention.forward` in the reference modeling file, Phase-1
    report section 2a):
      1. MiniCPM-SALA applies RoPE to q/k BEFORE the decay-linear-attention
         recurrence (`lightning_use_rope=true`); MiniMax's layer has no
         RoPE at all. Implemented below via `get_rope(...)`, applied prior
         to calling the lightning-attention kernels.
      2. MiniCPM-SALA applies RMSNorm independently to q and k
         (`qk_norm=true`) before use; MiniMax's layer does not norm q/k.
      3. MiniCPM-SALA's decay is NOT layer-idx-scaled -- every lightning
         layer uses the exact same `build_alibi_slopes(num_heads)`
         regardless of `layer_idx`. MiniMax scales its slope by
         `(1 - layer_idx/(num_hidden_layers-1) + 1e-5)`, which must NOT be
         reproduced here (see `build_alibi_slopes` docstring).

    KV heads: reference sets `lightning_nkv = 32 = lightning_nh` -- i.e.
    NO GQA on lightning layers, unlike the sparse layers' 16:1 GQA ratio.
    This is intentionally hardcoded from config rather than assumed equal
    to `num_attention_heads`, since a future checkpoint could set them
    independently.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = extract_layer_index(prefix)
        self.prefix = prefix
        self.cache_config = cache_config

        tp_size = get_tensor_model_parallel_world_size()
        self.tp_size = tp_size
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.lightning_nkv
        assert self.num_heads == self.num_kv_heads, (
            "Reference checkpoint has lightning_nh == lightning_nkv (no "
            "GQA on lightning layers); a config with them unequal would "
            "need a repeat_kv step this Stage-1 implementation does not "
            "yet have wired up -- flagging rather than silently assuming."
        )
        assert self.num_heads % tp_size == 0
        self.tp_heads = self.num_heads // tp_size
        self.head_dim = config.lightning_head_dim
        self.hidden_inner = self.head_dim * self.num_heads
        self.rms_norm_eps = config.rms_norm_eps
        self.qk_norm = getattr(config, "qk_norm", True)
        self.use_output_norm = getattr(config, "use_output_norm", True)
        self.use_output_gate = getattr(config, "use_output_gate", True)
        self.use_rope = getattr(config, "lightning_use_rope", True)
        self.block_size = 256
        assert getattr(config, "lightning_scale", "1/sqrt(d)") == "1/sqrt(d)", (
            "Only the '1/sqrt(d)' lightning_scale policy is implemented "
            "for Stage 1; the reference code also supports '1/d' and a "
            "no-op scale, neither exercised by the released checkpoint."
        )
        self.scale = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.hidden_inner,
            self.hidden_size,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        if self.use_output_gate:
            # ColumnParallelLinear (NOT ReplicatedLinear): mirrors the
            # reference MiniMaxText01LinearAttention.output_gate. The
            # attention output this gates is sharded to
            # (tp_heads * head_dim) per rank; a replicated full-size gate
            # would not broadcast against it under TP>1. Identical at TP=1.
            self.z_proj = ColumnParallelLinear(
                self.hidden_size,
                self.hidden_inner,
                bias=getattr(config, "attention_bias", False),
                gather_output=False,
                quant_config=quant_config,
                prefix=f"{prefix}.z_proj",
            )
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=self.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=self.rms_norm_eps)
        if self.use_output_norm:
            # TP-aware: normalizes over the full (num_heads * head_dim)
            # inner vector via an all-reduced variance, matching the
            # single-device reference. A plain RMSNorm(hidden_inner) here
            # would mis-shape its weight against the (tp_heads * head_dim)
            # shard and normalize over the shard only under TP>1.
            self.o_norm = MiniMaxText01RMSNormTP(
                self.hidden_inner, eps=self.rms_norm_eps
            )

        if self.use_rope:
            self.register_buffer(
                "rope_inv_freq",
                _build_rope_inv_freq(self.head_dim, config.rope_theta),
                persistent=False,
            )

        # Full (un-TP-sharded) per-head decay, computed identically on
        # every rank (matches MiniMaxText01LinearAttention's
        # `self.slope_rate = _build_slope_tensor(self.num_heads)`), then
        # sliced to this rank's shard of heads -- mirrors
        # `self.tp_slope = self.slope_rate[tp_rank*tp_heads:(tp_rank+1)*tp_heads]`
        # in that same reference layer. Verified correct under a real
        # TP=2 CPU distributed group (gloo backend, actual 2-process
        # init_distributed_environment/initialize_model_parallel, not
        # simulated) -- see docs/minicpm_sala_known_limitations.md §2.5.
        # NOT yet verified under real multi-GPU TP (nccl); the sharding
        # arithmetic itself is backend-independent, but this is worth
        # re-confirming once GPU access is available rather than assuming
        # gloo and nccl behave identically here.
        # CRITICAL: pass the slope with a POSITIVE sign. The reused vLLM
        # kernels (lightning_attention / linear_decode_forward_triton, via
        # linear_attention_prefill_and_mix / linear_attention_decode /
        # MiniMaxText01LinearKernel.jit_linear_forward_prefix) apply the
        # decay internally as exp(-s * distance) and therefore require
        # s > 0 -- this is exactly how MiniMaxText01LinearAttention feeds
        # its own (positive) `self.tp_slope`. Multiplying by -1.0 here
        # (as an earlier revision did) yields exp(+|s| * distance), i.e.
        # a term that GROWS with distance and overflows to Inf/NaN on any
        # non-trivial sequence. build_alibi_slopes already returns the
        # positive slope magnitude; do not negate it. Routed through the
        # named seam so the sign is guarded by a GPU-free regression test.
        full_decay = build_lightning_decay_rate(self.num_heads)
        tp_rank = get_tensor_model_parallel_rank()
        self.register_buffer(
            "tp_slope",
            full_decay[
                tp_rank * self.tp_heads : (tp_rank + 1) * self.tp_heads
            ].contiguous(),
            persistent=False,
        )

        # Register into the compilation static forward context so
        # `torch.ops.vllm.linear_attention` (registered once, at import
        # time, by minimax_linear_attn.py) can dispatch back to this
        # layer instance by prefix -- exact pattern used by both
        # `MiniMaxText01LinearAttention.__init__` and
        # `BailingMoELinearAttention.__init__`.
        _vllm_config = get_current_vllm_config()
        model_config = _vllm_config.model_config
        self._model_dtype = (
            model_config.dtype if model_config is not None else torch.bfloat16
        )
        compilation_config = _vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self._qkv_hist_q: torch.Tensor | None = None
        self._qkv_hist_k: torch.Tensor | None = None
        self._qkv_hist_v: torch.Tensor | None = None

    def _reset_qkv_history(self) -> None:
        """Drop accumulated q/k/v; used when a slot starts a fresh sequence."""
        self._qkv_hist_q = None
        self._qkv_hist_k = None
        self._qkv_hist_v = None

    def _sync_qkv_history(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        fresh: bool,
        target_hist_len: int | None = None,
    ) -> None:
        """Accumulate per-token q/k/v for HF-matched full GLA recompute on decode."""
        layer_idx = getattr(self, "layer_idx", -1)
        if not fresh and target_hist_len is not None:
            cur = 0 if self._qkv_hist_q is None else int(self._qkv_hist_q.shape[0])
            n_new = int(q.shape[0])
            if n_new == 0 and cur >= target_hist_len:
                # #region agent log
                _agent_debug_log(
                    "minicpm_sala.py:_sync_qkv_history",
                    "skip duplicate append",
                    {
                        "layer_idx": layer_idx,
                        "cur": cur,
                        "target_hist_len": target_hist_len,
                    },
                    "B",
                )
                # #endregion
                return
            if n_new > 0 and cur >= target_hist_len:
                n_rep = min(n_new, target_hist_len)
                qf, kf, vf = q.detach().float(), k.detach().float(), v.detach().float()
                self._qkv_hist_q = torch.cat(
                    [self._qkv_hist_q[: target_hist_len - n_rep], qf[-n_rep:]], dim=0
                )
                self._qkv_hist_k = torch.cat(
                    [self._qkv_hist_k[: target_hist_len - n_rep], kf[-n_rep:]], dim=0
                )
                self._qkv_hist_v = torch.cat(
                    [self._qkv_hist_v[: target_hist_len - n_rep], vf[-n_rep:]], dim=0
                )
                # #region agent log
                _agent_debug_log(
                    "minicpm_sala.py:_sync_qkv_history",
                    "history refresh tail",
                    {
                        "layer_idx": layer_idx,
                        "hist_len": int(self._qkv_hist_q.shape[0]),
                        "target_hist_len": target_hist_len,
                        "n_rep": n_rep,
                    },
                    "B",
                )
                # #endregion
                return
        if fresh or self._qkv_hist_q is None:
            self._qkv_hist_q = q.detach().float()
            self._qkv_hist_k = k.detach().float()
            self._qkv_hist_v = v.detach().float()
            # #region agent log
            _agent_debug_log(
                "minicpm_sala.py:_sync_qkv_history",
                "history reset",
                {
                    "layer_idx": layer_idx,
                    "hist_len": int(self._qkv_hist_q.shape[0]),
                    "fresh": fresh,
                },
                "B",
            )
            # #endregion
            return
        self._qkv_hist_q = torch.cat([self._qkv_hist_q, q.detach().float()], dim=0)
        self._qkv_hist_k = torch.cat([self._qkv_hist_k, k.detach().float()], dim=0)
        self._qkv_hist_v = torch.cat([self._qkv_hist_v, v.detach().float()], dim=0)
        # #region agent log
        _agent_debug_log(
            "minicpm_sala.py:_sync_qkv_history",
            "history append",
            {
                "layer_idx": layer_idx,
                "hist_len": int(self._qkv_hist_q.shape[0]),
                "target_hist_len": target_hist_len,
            },
            "B",
        )
        # #endregion

    def _qkv_sequence_for_recompute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_metadata: LinearAttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return q/k/v rows for a full-sequence GLA pass including the live decode token.

        ``_sync_qkv_history`` can skip appending when ``cur >= target_hist_len`` on
        prefill; on decode ``seq_lens`` may equal history length before the live
        token is stored. Recompute must still see the current q/k/v row.
        """
        if self._qkv_hist_q is None:
            return q.detach().float(), k.detach().float(), v.detach().float()
        hq, hk, hv = self._qkv_hist_q, self._qkv_hist_k, self._qkv_hist_v
        expected = int(attn_metadata.seq_lens[0].item())
        n_dec = int(attn_metadata.num_decode_tokens)
        qf, kf, vf = q.detach().float(), k.detach().float(), v.detach().float()
        if int(hq.shape[0]) < expected:
            hq = torch.cat([hq, qf], dim=0)
            hk = torch.cat([hk, kf], dim=0)
            hv = torch.cat([hv, vf], dim=0)
        elif n_dec > 0 and int(hq.shape[0]) >= n_dec:
            hq = torch.cat([hq[:-n_dec], qf], dim=0)
            hk = torch.cat([hk[:-n_dec], kf], dim=0)
            hv = torch.cat([hv[:-n_dec], vf], dim=0)
        if int(hq.shape[0]) > expected:
            hq, hk, hv = hq[:expected], hk[:expected], hv[:expected]
        return hq, hk, hv

    def _decode_infer_parity(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: torch.Tensor,
        state_indices_tensor: torch.Tensor,
        attn_metadata: LinearAttentionMetadata,
    ) -> torch.Tensor:
        """Decode with HF ``use_cache=False`` semantics when seq_len < 64.

        Incremental ``fused_recurrent_simple_gla`` state carry can diverge from
        a one-shot full-sequence GLA pass (gate1_decode_incremental_vs_oneshot).
        Below 64 tokens, recompute on accumulated q/k/v history instead.
        """
        hist_len = 0 if self._qkv_hist_q is None else int(self._qkv_hist_q.shape[0])
        num_decode = int(attn_metadata.num_decode_tokens)
        layer_idx = getattr(self, "layer_idx", -1)
        seq_len0 = int(attn_metadata.seq_lens[0].item())
        branch = (
            "incremental_empty"
            if hist_len <= 0
            else "recompute_chunk"
            if hist_len >= 64
            else "recompute_fused"
        )
        # #region agent log
        _agent_debug_log(
            "minicpm_sala.py:_decode_infer_parity",
            "decode branch",
            {
                "layer_idx": layer_idx,
                "hist_len": hist_len,
                "num_decode": num_decode,
                "seq_len0": seq_len0,
                "branch": branch,
                "hist_eq_seq": hist_len == seq_len0,
            },
            "A",
        )
        # #endregion
        if hist_len <= 0:
            return self._decode_infer(
                q, k, v, kv_cache, state_indices_tensor, attn_metadata
            )
        if hist_len >= 64:
            slot_id = int(state_indices_tensor[0].item())
            slice_cache = kv_cache[slot_id, ...]
            rq, rk, rv = self._qkv_sequence_for_recompute(
                q, k, v, attn_metadata
            )
            qs = rq.transpose(0, 1).unsqueeze(0).contiguous()
            ks = rk.transpose(0, 1).unsqueeze(0).contiguous()
            vs = rv.transpose(0, 1).unsqueeze(0).contiguous()
            out_all = _minicpm_sala_lightning_forward_prefix(
                qs,
                ks,
                vs,
                slice_cache,
                self.tp_slope,
                self.block_size,
                scale=self.scale,
                fresh_sequence=True,
            )
            return out_all[-num_decode:].to(dtype=q.dtype)
        slot_id = int(state_indices_tensor[0].item())
        slice_cache = kv_cache[slot_id, ...]
        rq, rk, rv = self._qkv_sequence_for_recompute(q, k, v, attn_metadata)
        qs = rq.transpose(0, 1).unsqueeze(0).contiguous()
        ks = rk.transpose(0, 1).unsqueeze(0).contiguous()
        vs = rv.transpose(0, 1).unsqueeze(0).contiguous()
        out_all = _minicpm_sala_lightning_forward_prefix(
            qs,
            ks,
            vs,
            slice_cache,
            self.tp_slope,
            self.block_size,
            scale=self.scale,
            fresh_sequence=True,
        )
        return out_all[-num_decode:].to(dtype=q.dtype)

    def get_state_shape(self) -> tuple[tuple[int, int, int], ...]:
        return MambaStateShapeCalculator.linear_attention_state_shape(
            num_heads=self.num_heads, tp_size=self.tp_size, head_dim=self.head_dim
        )

    def get_state_dtype(self) -> tuple[torch.dtype]:
        # vLLM's lightning_attention kernel accumulates recurrent state in
        # fp32 (see vllm/model_executor/layers/lightning_attn.py); bf16
        # state triggers Triton dtype mismatches on sm_89 with torch 2.11.
        return (torch.float32,)

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.LINEAR

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """In-place-output convention, NOT a return value: matches every
        other layer in the `LinearAttention`/`MambaBase` family (see
        `BailingMoeV25DecoderLayer.forward`,
        `MiniMaxText01LinearAttention.forward`) so this layer is
        piecewise-CUDA-graph-capturable and dispatchable through the
        shared `torch.ops.vllm.linear_attention` custom op by prefix
        rather than by direct Python call.
        """
        torch.ops.vllm.linear_attention(hidden_states, output, positions, self.prefix)

    def _forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """Real computation, invoked by the shared custom op via
        `get_forward_context().no_compile_layers[self.prefix]._forward(...)`
        -- identical dispatch mechanism to
        `BailingMoELinearAttention._forward` /
        `MiniMaxText01LinearAttention._forward`.
        """
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        attn_metadata: LinearAttentionMetadata | None = None
        if attn_metadata_raw is not None:
            assert isinstance(attn_metadata_raw, dict)
            attn_metadata = attn_metadata_raw[self.prefix]
            assert isinstance(attn_metadata, LinearAttentionMetadata)
            num_actual_tokens = (
                attn_metadata.num_prefill_tokens + attn_metadata.num_decode_tokens
            )
        else:
            num_actual_tokens = hidden_states.shape[0]

        qkv_size = self.tp_heads * self.head_dim
        q, k, v = _minicpm_qkv_proj(
            self.qkv_proj,
            hidden_states[:num_actual_tokens],
            qkv_size,
            qkv_size,
            qkv_size,
        )
        q = q.view(-1, self.tp_heads, self.head_dim)
        k = k.view(-1, self.tp_heads, self.head_dim)
        v = v.view(-1, self.tp_heads, self.head_dim)

        # Reference verified: qk_norm is applied BEFORE rope (Phase 1
        # section 2a / `LightningAttention.forward` in the HF reference:
        # `if self.qk_norm: q = self.q_norm(q); k = self.k_norm(k)` runs
        # strictly before the `if self.use_rope:` block). Order matters --
        # RMSNorm does not commute with rotation.
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.use_rope:
            # The released checkpoint's HF modules ship with ``inv_freq`` /
            # ``cos_cached`` / ``sin_cached`` buffers zeroed after
            # ``from_pretrained`` (verified on A100: ``cos.max()==0``), so
            # ``apply_rotary_pos_emb`` zeros q/k before ``attn_fn``. vLLM
            # must mirror that effective behavior for greedy parity (2132
            # vs 3566 if real RoPE is applied). Identity RoPE on HF yields
            # the same greedy as vLLM without this guard.
            q = torch.zeros_like(q)
            k = torch.zeros_like(k)

        if attn_metadata is not None:
            kv_cache = self.kv_cache[0]
            state_indices_tensor = attn_metadata.state_indices_tensor
            if attn_metadata.num_prefills > 0 and _lightning_should_reset_qkv_history(
                attn_metadata, positions
            ):
                self._reset_qkv_history()
            _clear_lightning_state_for_engine_prefill(
                kv_cache, state_indices_tensor, attn_metadata, positions
            )

        decode_only = (
            getattr(attn_metadata, "num_prefills", 0) == 0
            if attn_metadata is not None
            else False
        )
        fresh_sequence = False
        if attn_metadata is not None and not decode_only:
            fresh_sequence = _lightning_prefill_starts_at_position_zero(
                attn_metadata, positions
            )
        if attn_metadata is not None:
            # Prefill-only guard: skip duplicate sync when ``linear_attention_prefill_and_mix``
            # already filled history for this forward. On decode-only steps, never skip —
            # ``seq_lens`` can equal current history length and would drop the live token
            # (gate1_decode_incremental_vs_oneshot RED at step 14).
            self._sync_qkv_history(
                q,
                k,
                v,
                fresh=(not decode_only) and fresh_sequence,
                target_hist_len=_lightning_target_hist_len(attn_metadata),
            )

        if attn_metadata is None:
            hidden = torch.zeros(
                (q.shape[0], q.shape[1] * q.shape[2]),
                device=q.device,
                dtype=q.dtype,
            )
        elif not decode_only:
            hidden = linear_attention_prefill_and_mix(
                q=q,
                k=k,
                v=v,
                kv_cache=kv_cache,
                state_indices_tensor=state_indices_tensor,
                attn_metadata=attn_metadata,
                slope_rate=self.tp_slope,
                block_size=self.block_size,
                decode_fn=self._decode_infer_parity,
                prefix_fn=partial(
                    _minicpm_sala_lightning_forward_prefix,
                    scale=self.scale,
                    fresh_sequence=fresh_sequence,
                ),
                layer_idx=self.layer_idx,
            )
        else:
            hidden = self._decode_infer_parity(
                q, k, v, kv_cache, state_indices_tensor, attn_metadata
            )

        hidden = hidden.reshape(hidden.shape[0], -1)
        # Reference verified: output norm and output gate are applied
        # AFTER the attention kernel and BEFORE o_proj (Phase 1 section
        # 2a: `o = o_norm(o)` then `o = o * sigmoid(z_proj(hidden_states))`
        # then `y = o_proj(o)`), operating on the ORIGINAL (pre-qkv-split)
        # `hidden_states` for the gate projection input, not on `hidden`.
        if self.use_output_norm:
            hidden = self.o_norm._forward(hidden)
        if self.use_output_gate:
            z, _ = self.z_proj(hidden_states[:num_actual_tokens])
            hidden = hidden * torch.sigmoid(z)
        hidden = hidden.to(hidden_states.dtype)

        dense_out, _ = self.o_proj(hidden)
        output[:num_actual_tokens] = dense_out

    def _decode_infer(self, q, k, v, kv_cache, state_indices_tensor, attn_metadata):
        try:
            from fla.ops.simple_gla import fused_recurrent_simple_gla
        except ImportError:
            fused_recurrent_simple_gla = None  # type: ignore[misc, assignment]

        if fused_recurrent_simple_gla is not None:
            h = self.tp_heads
            d = self.head_dim
            g_gamma = (-self.tp_slope.to(torch.float32)).reshape(h)
            outs = []
            for i in range(attn_metadata.num_decodes):
                slot_id = int(state_indices_tensor[i].item())
                # fla expects [batch, time, heads, dim] — not [batch, heads, time, dim].
                qi = q[i : i + 1].unsqueeze(0).to(torch.float32)
                ki = k[i : i + 1].unsqueeze(0).to(torch.float32)
                vi = v[i : i + 1].unsqueeze(0).to(torch.float32)
                initial_state = (
                    kv_cache[slot_id].reshape(1, h, d, d).contiguous().to(torch.float32)
                )
                if initial_state.abs().sum().item() == 0.0:
                    initial_state = None
                o, final_state = fused_recurrent_simple_gla(
                    q=qi,
                    k=ki,
                    v=vi,
                    g_gamma=g_gamma,
                    scale=self.scale,
                    initial_state=initial_state,
                    output_final_state=True,
                )
                kv_cache[slot_id].copy_(final_state.reshape(h, d, d).to(kv_cache.dtype))
                outs.append(rearrange(o.to(q.dtype)[0, 0], "h d -> (h d)"))
            return torch.stack(outs, dim=0)

        return linear_attention_decode(
            q,
            k,
            v,
            kv_cache,
            self.tp_slope.float(),
            state_indices_tensor,
            q_start=0,
            q_end=attn_metadata.num_decode_tokens,
            slot_start=0,
            slot_end=attn_metadata.num_decodes,
            block_size=self.block_size,
        )


# ---------------------------------------------------------------------------
# Decoder layer -- per-layer dispatch by mixer_types[layer_idx], with the
# muP-derived residual scaling verified in Phase 1 section 3.
# ---------------------------------------------------------------------------


class MiniCPMSALADecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        self.mixer_type = config.mixer_types[layer_idx]

        if is_sparse_layer(self.mixer_type):
            self.self_attn = MiniCPMSALADenseAttention(
                config, cache_config, quant_config, prefix=f"{prefix}.self_attn"
            )
        elif is_lightning_layer(self.mixer_type):
            self.self_attn = MiniCPMSALALightningAttention(
                config, cache_config, quant_config, prefix=f"{prefix}.self_attn"
            )
        else:
            raise ValueError(f"Unsupported mixer type: {self.mixer_type!r}")

        self.mlp = MiniCPMSALAMLP(
            config.hidden_size,
            config.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # muP residual-branch scaling -- see Phase 1 report section 3.
        # NOT a learned parameter; applied identically to both branches on
        # every layer, verified against `MiniCPMSALADecoderLayer.forward`
        # in the reference code (both `hidden_states * (scale_depth /
        # sqrt(num_hidden_layers))` call sites).
        self.residual_scale = config.scale_depth / math.sqrt(config.num_hidden_layers)
        # Stage 5 opt-in: fused multiply-add path for the muP-scaled
        # residual branches. Defaults False -- see _add_scaled_residual
        # and docs/minicpm_sala_known_limitations.md.
        self.use_fused_residual = False

    def _add_scaled_residual(
        self, residual: torch.Tensor, branch: torch.Tensor
    ) -> torch.Tensor:
        """muP-scaled residual add shared by both decoder branches.

        Default path: ``residual + branch * residual_scale`` (matches the
        reference). Opt-in fused path uses ``torch.add(..., alpha=...)`` --
        mathematically equivalent, not bit-identical (see
        test_minicpm_sala_fused_residual.py).
        """
        if self.use_fused_residual:
            return torch.add(residual, branch, alpha=self.residual_scale)
        if os.environ.get("MINICPM_SALA_BF16_RESIDUAL", "").lower() not in (
            "1",
            "true",
            "yes",
        ):
            out = residual.float() + branch.float() * self.residual_scale
            return out.to(dtype=residual.dtype)
        return residual + branch * self.residual_scale

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Deliberately does NOT use vLLM's usual fused
        `hidden_states, residual = norm(hidden_states, residual)` pattern
        (see e.g. `LlamaDecoderLayer.forward`,
        `BailingMoeV25DecoderLayer.forward`). That fusion performs an
        UNSCALED residual add internally; MiniCPM-SALA's residual
        branches are SCALED (`residual + branch * scale_depth /
        sqrt(num_hidden_layers)`, Phase 1 report section 3) -- silently
        reusing the fused API here would drop the muP scale factor. This
        is a deliberate Stage-1 "architectural correctness over
        performance" choice per the mission brief's own staging
        (Stage 1: correctness, no shortcuts, no perf optimization;
        Stage 5: perf). A fused-scaled-residual custom op is a legitimate
        Stage-4/5 kernel-fusion target, tracked in
        docs/minicpm_sala_known_limitations.md, not implemented blind here.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if is_sparse_layer(self.mixer_type):
            attn_out = self.self_attn(hidden_states)
        else:
            # LinearAttention-family in-place-output convention -- see
            # `MiniCPMSALALightningAttention.forward` docstring.
            attn_out = torch.zeros_like(hidden_states)
            self.self_attn(
                hidden_states=hidden_states,
                output=attn_out,
                positions=positions,
            )
        hidden_states = self._add_scaled_residual(residual, attn_out)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self._add_scaled_residual(residual, hidden_states)

        return hidden_states


# ---------------------------------------------------------------------------
# Model + CausalLM wrapper
# ---------------------------------------------------------------------------


class MiniCPMSALAModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        validate_mixer_schedule(list(config.mixer_types))
        assert len(config.mixer_types) == config.num_hidden_layers, (
            f"len(mixer_types)={len(config.mixer_types)} != "
            f"num_hidden_layers={config.num_hidden_layers}"
        )

        self.config = config
        self.vocab_size = config.vocab_size
        self.scale_emb = config.scale_emb

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MiniCPMSALADecoderLayer(
                config, cache_config, quant_config, prefix
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Only "hidden_states" is threaded across pipeline-parallel stage
        # boundaries -- unlike vLLM's usual two-key
        # ("hidden_states", "residual") IntermediateTensors convention,
        # because this decoder layer does not use the fused-residual norm
        # API (see `MiniCPMSALADecoderLayer.forward` docstring) and so has
        # no separate "residual" tensor to carry between layers or across
        # PP ranks.
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        # muP embedding scale-in -- verified against
        # `inputs_embeds = embed_tokens(input_ids) * self.config.scale_emb`
        # in the reference `MiniCPMSALAModel.forward`.
        return self.embed_tokens(input_ids) * self.scale_emb

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.get_input_embeddings(input_ids)
            )
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states = layer(positions, hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(mapped_name, self):
                    continue
                if mapped_name not in params_dict:
                    continue
                param = params_dict[mapped_name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped_name)
                break
            else:
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
        return loaded_params


class MiniCPMSALAForCausalLM(nn.Module, HasInnerState, IsHybrid, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        _ensure_native_rms_norm_kernels(vllm_config)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.model = MiniCPMSALAModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        # NOT tied: reference config sets tie_word_embeddings=false and the
        # reference model allocates a fully independent lm_head.
        assert not getattr(config, "tie_word_embeddings", False), (
            "This Stage-1 implementation assumes an untied lm_head, "
            "matching the released checkpoint's tie_word_embeddings=false. "
            "A tied-embedding variant would need explicit handling, not "
            "silent reuse of this path."
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        # logit scale-out -- verified against
        # `logits = lm_head(hidden_states / (hidden_size/dim_model_base))`
        # in the reference `MiniCPMSALAForCausalLM.forward`.
        self.logit_scale = config.hidden_size / config.dim_model_base
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        scaled = hidden_states / self.logit_scale
        return self.logits_processor(self.lm_head, scaled)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int, int], ...]:
        hf_config = vllm_config.model_config.hf_config
        parallel_config = vllm_config.parallel_config
        return MambaStateShapeCalculator.linear_attention_state_shape(
            num_heads=hf_config.lightning_nkv,
            tp_size=parallel_config.tensor_parallel_size,
            head_dim=hf_config.lightning_head_dim,
        )

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.linear_attention_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple:
        return MambaStateCopyFuncCalculator.linear_attention_state_copy_func()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

