#!/usr/bin/env bash
# Install a minimal `flash_attn` shim backed by infllm_v2's kernels.
#
# Why: the HF reference modeling_minicpm_sala.py requires
# `attn_implementation="flash_attention_2"` and imports
# flash_attn.{flash_attn_func, flash_attn_varlen_func} +
# flash_attn.bert_padding.{index_first_axis, pad_input, unpad_input}.
# flash-attn's PyPI sdist fails to compile against CUDA 13 / torch 2.11
# (CUTLASS header breakage), but infllm_v2 IS a flash-attn fork built from
# the same lineage: `infllmv2_attn_varlen_func(..., topk_idx=None)` is
# flash_attn_varlen_func. This shim maps the names 1:1 so the HF reference
# runs unmodified. Use ONLY where a real flash-attn wheel is unavailable;
# it covers exactly the entry points the MiniCPM-SALA reference exercises.
set -euo pipefail

python3 -c "import infllm_v2" || {
  echo "ERROR: infllm_v2 must be installed first (scripts/install_infllm_v2.sh)" >&2
  exit 1
}
if python3 -c "import flash_attn" 2>/dev/null; then
  echo "flash_attn already importable -- not installing the shim."
  exit 0
fi

SITE="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
PKG="${SITE}/flash_attn"
mkdir -p "${PKG}"

cat > "${PKG}/__init__.py" <<'PY'
"""Minimal flash_attn shim backed by infllm_v2 (a flash-attn fork).

Provides exactly the surface modeling_minicpm_sala.py uses. Dense
semantics: infllmv2_attn_varlen_func with topk_idx=None IS the fork's
flash_attn_varlen_func.
"""

import torch
from infllm_v2 import infllmv2_attn_varlen_func as _varlen

__version__ = "2.999.0+infllmv2-shim"


def flash_attn_varlen_func(
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
    dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1),
    softcap=0.0, alibi_slopes=None, deterministic=False,
    return_attn_probs=False, block_table=None,
):
    return _varlen(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal,
        window_size=window_size, softcap=softcap, alibi_slopes=alibi_slopes,
        deterministic=deterministic, return_attn_probs=return_attn_probs,
        block_table=block_table,
    )


def flash_attn_func(
    q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs
):
    # Batched (b, s, h, d) dense attention with no padding: express as one
    # varlen call with uniform sequence lengths.
    b, s, h, d = q.shape
    hk = k.shape[2]
    cu = torch.arange(0, (b + 1) * s, s, dtype=torch.int32, device=q.device)
    out = _varlen(
        q.reshape(b * s, h, d), k.reshape(b * s, hk, d), v.reshape(b * s, hk, d),
        cu, cu, s, s, dropout_p=dropout_p, softmax_scale=softmax_scale,
        causal=causal,
    )
    return out.reshape(b, s, h, d)
PY

cat > "${PKG}/bert_padding.py" <<'PY'
"""flash_attn.bert_padding-compatible helpers (pure torch)."""

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


def index_first_axis(x, indices):
    other_shape = x.shape[1:]
    second_dim = other_shape.numel()
    return torch.gather(
        rearrange(x, "b ... -> b (...)"), 0,
        repeat(indices, "z -> z d", d=second_dim),
    ).reshape(-1, *other_shape)


def pad_input(hidden_states, indices, batch, seqlen):
    output = torch.zeros(
        batch * seqlen, *hidden_states.shape[1:],
        device=hidden_states.device, dtype=hidden_states.dtype,
    )
    output[indices] = hidden_states
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


def unpad_input(hidden_states, attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = int(seqlens_in_batch.max().item())
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
    )
    return (
        index_first_axis(
            rearrange(hidden_states, "b s ... -> (b s) ..."), indices
        ),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )
PY

python3 - <<'PY'
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
import flash_attn
print("flash_attn shim OK:", flash_attn.__version__)
PY
echo "flash_attn shim installed into ${PKG}"
