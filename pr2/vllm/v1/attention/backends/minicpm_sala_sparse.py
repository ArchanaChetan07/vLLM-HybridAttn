# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AttentionBackend/AttentionImpl for MiniCPM-SALA InfLLM-V2 sparse layers.

Kernel contracts are grounded in OpenBMB/infllmv2_cuda_impl. Page KV layout
matches FlashAttentionBackend: ``(num_blocks, 2, page_block_size, ...)``.
Sparse top-k scoring uses ``hf_config.sparse_config.block_size`` (typically 64),
which is distinct from the paged-attention page size (``cache_config.block_size``,
must be a multiple of 256 per infllm_v2).
"""

from typing import TYPE_CHECKING, Any, ClassVar

import copy
import os
from dataclasses import dataclass

import torch
from torch import nn

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

_SPARSE_DEBUG = os.environ.get("MINICPM_SALA_DEBUG_SPARSE", "").lower() in (
    "1",
    "true",
    "yes",
)


def _debug_tensor(name: str, t: torch.Tensor | None) -> None:
    """Log tensor stats when MINICPM_SALA_DEBUG_SPARSE=1."""
    if not _SPARSE_DEBUG:
        return
    if t is None:
        logger.info("[sparse-debug] %s: None", name)
        return
    with torch.no_grad():
        finite = torch.isfinite(t)
        t_f = t.float()
        logger.info(
            "[sparse-debug] %s shape=%s dtype=%s device=%s "
            "min=%.6g max=%.6g mean=%.6g abs_sum=%.6g nan=%d inf=%d",
            name,
            tuple(t.shape),
            t.dtype,
            t.device,
            float(t_f.min().item()) if t.numel() else 0.0,
            float(t_f.max().item()) if t.numel() else 0.0,
            float(t_f.mean().item()) if t.numel() else 0.0,
            float(t_f.abs().sum().item()) if t.numel() else 0.0,
            int((~finite).sum().item()) if t.numel() else 0,
            int(torch.isinf(t).sum().item()) if t.numel() else 0,
        )


try:
    from infllm_v2 import (
        infllmv2_attn_stage1,
        infllmv2_attn_varlen_func,
        infllmv2_attn_with_kvcache,
        max_pooling_1d_varlen,
    )

    INFLLM_V2_AVAILABLE = True
except ImportError:
    # Mirrors the reference HF modeling file's own
    # `try: from infllm_v2 import ...; except ImportError: pass` pattern
    # exactly -- this is not a Stage-1-only workaround, the reference
    # model itself is written to tolerate infllm_v2 being absent (and
    # presumably falls back to eager/dense attention in that case,
    # though the reference code path for that fallback was not
    # separately re-verified here beyond what Phase 1's report already
    # covers for the `_flash_attention_forward_dense` branch).
    INFLLM_V2_AVAILABLE = False
    infllmv2_attn_with_kvcache = None
    infllmv2_attn_varlen_func = None
    infllmv2_attn_stage1 = None
    max_pooling_1d_varlen = None


@dataclass(frozen=True)
class MiniCPMSALASparseConfig:
    """Runtime sparse-regime parameters from ``hf_config.sparse_config``.

    ``sparse_block_size`` is the top-k *scoring* block size (reference default 64).
    ``page_block_size`` is the paged KV page size from ``cache_config.block_size``
    (must be a multiple of 256 for infllm_v2); passed separately at construction.
    """

    kernel_size: int
    kernel_stride: int
    dense_len: int
    init_blocks: int
    topk: int
    window_size: int
    sparse_block_size: int

    @property
    def compress_k2_kernel_size(self) -> int:
        return self.kernel_size * 4

    @property
    def compress_k2_kernel_stride(self) -> int:
        return self.kernel_stride * 4

    @property
    def local_blocks(self) -> int:
        if self.window_size % self.sparse_block_size != 0:
            raise ValueError(
                f"sparse_config.window_size ({self.window_size}) must be "
                f"divisible by sparse_config.block_size ({self.sparse_block_size})"
            )
        return self.window_size // self.sparse_block_size


def _sparse_config_field(raw: Any, key: str) -> Any:
    if isinstance(raw, dict):
        if key not in raw:
            raise ValueError(f"sparse_config missing required field {key!r}")
        return raw[key]
    if not hasattr(raw, key):
        raise ValueError(f"sparse_config missing required attribute {key!r}")
    return getattr(raw, key)


def parse_sparse_config(hf_config: Any) -> MiniCPMSALASparseConfig:
    """Read and validate ``hf_config.sparse_config`` (no duplicated constants)."""
    raw = getattr(hf_config, "sparse_config", None)
    if raw is None:
        raise ValueError("MiniCPM-SALA requires hf_config.sparse_config; got None")
    cfg = MiniCPMSALASparseConfig(
        kernel_size=int(_sparse_config_field(raw, "kernel_size")),
        kernel_stride=int(_sparse_config_field(raw, "kernel_stride")),
        dense_len=int(_sparse_config_field(raw, "dense_len")),
        init_blocks=int(_sparse_config_field(raw, "init_blocks")),
        topk=int(_sparse_config_field(raw, "topk")),
        window_size=int(_sparse_config_field(raw, "window_size")),
        sparse_block_size=int(_sparse_config_field(raw, "block_size")),
    )
    if cfg.kernel_size <= 0 or cfg.kernel_stride <= 0:
        raise ValueError(
            f"sparse_config kernel_size/kernel_stride must be positive, "
            f"got {cfg.kernel_size}/{cfg.kernel_stride}"
        )
    if cfg.dense_len <= 0:
        raise ValueError(
            f"sparse_config.dense_len must be positive, got {cfg.dense_len}"
        )
    if cfg.topk <= 0:
        raise ValueError(f"sparse_config.topk must be positive, got {cfg.topk}")
    if cfg.sparse_block_size <= 0:
        raise ValueError(
            f"sparse_config.block_size must be positive, got {cfg.sparse_block_size}"
        )
    _ = cfg.local_blocks  # validates window_size divisibility
    return cfg


def validate_page_block_size(page_block_size: int) -> None:
    """infllmv2_attn_with_kvcache requires page_block_size % 256 == 0."""
    if page_block_size <= 0:
        raise ValueError(f"page block_size must be positive, got {page_block_size}")
    if page_block_size % 256 != 0:
        raise ValueError(
            f"MiniCPM-SALA sparse page block_size must be a multiple of 256 "
            f"(infllm_v2 constraint), got {page_block_size}"
        )


# Sparse-regime boundary matches HF reference: dense when kv_seq_len < dense_len,
# sparse when kv_seq_len >= dense_len (Phase-1 report §2b; modeling_minicpm_sala.py).
def sequence_sparse_mask(seq_lens: torch.Tensor, dense_len: int) -> torch.Tensor:
    """Per-sequence sparse-regime mask: True when ``seq_len >= dense_len``."""
    return seq_lens >= dense_len


def _assert_k_cache_page_size(k_cache: torch.Tensor, page_block_size: int) -> None:
    if k_cache.ndim != 4:
        raise ValueError(
            f"Expected k_cache shape (num_blocks, page_block_size, H, D), "
            f"got ndim={k_cache.ndim}"
        )
    if k_cache.shape[1] != page_block_size:
        raise ValueError(
            f"KV page size mismatch: k_cache page dim is {k_cache.shape[1]}, "
            f"expected page_block_size={page_block_size}. Check cache_config "
            f"propagation into Attention(..., block_size=...)."
        )


def calc_chunks_with_stride(
    cu_seqlen: torch.Tensor, chunk_size: int, kernel_stride: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Faithful, direct port of the reference `calc_chunks_with_stride`
    (modeling_minicpm_sala.py, fetched from the real source at commit
    9180fe1 -- copied line-for-line in logic, not reconstructed from
    the Phase 1 report's prose description of it). Computes the
    overlapping compression-window start offsets (stride=kernel_stride,
    width=chunk_size) within each packed sequence, and the resulting
    per-sequence compressed-row counts.

    NOTE: the reference decorates this with `@lru_cache(maxsize=16)`,
    keyed on the `cu_seqlen` tensor itself. Not reproduced here --
    `cu_seqlen` is a tensor (unhashable in the way `lru_cache` needs
    without `tensor.__hash__` support, which torch.Tensor does define
    but by identity, not value -- meaning the reference's caching only
    ever hits for the literal same tensor object, not equal-valued ones,
    a subtlety worth being aware of if reproducing the caching behavior
    is later judged worthwhile for performance; skipped here as a
    correctness-first-only concern per this project's own staging
    philosophy).
    """
    batch_sizes = cu_seqlen[1:] - cu_seqlen[:-1]
    max_seq_len = torch.max(batch_sizes)
    max_num_chunks_per_seq = (max_seq_len - chunk_size) // kernel_stride + 1
    chunk_start_offsets = torch.arange(
        0,
        max_num_chunks_per_seq * kernel_stride,
        kernel_stride,
        device=cu_seqlen.device,
    )
    seq_starts = cu_seqlen[:-1]
    chunk_start_in_seq = seq_starts[:, None] + chunk_start_offsets[None, :]

    chunk_end_in_seq = chunk_start_in_seq + chunk_size
    valid_chunk_mask = chunk_end_in_seq <= (seq_starts[:, None] + batch_sizes[:, None])

    valid_chunk_starts = chunk_start_in_seq[valid_chunk_mask]
    chunk_indices = torch.arange(0, chunk_size, device=cu_seqlen.device)[None, :]
    filtered_indices = (valid_chunk_starts[:, None] + chunk_indices).view(-1)

    num_filtered_chunks_per_batch = valid_chunk_mask.sum(dim=1)
    cu_seqlens_compressed = torch.zeros(
        len(cu_seqlen), dtype=torch.int32, device=cu_seqlen.device
    )
    cu_seqlens_compressed[1:] = num_filtered_chunks_per_batch.cumsum(dim=0)
    return filtered_indices, cu_seqlens_compressed


class CompressK(nn.Module):
    """Faithful, direct port of the reference `CompressK` module (same
    source as `calc_chunks_with_stride` above). Pure PyTorch, no
    `infllm_v2` dependency -- unlike the attention kernels themselves,
    this compression step has no custom CUDA kernel in the reference; it
    is plain `index_select` + `mean`, and is therefore fully portable and
    testable without any external package, GPU compilation step, or
    CUDA toolkit -- confirmed by reading the actual reference forward()
    body, which uses only `torch.Tensor.index_select`/`.view`/`.mean`.
    """

    def __init__(
        self, head_num_k: int, head_dim: int, kernel_size: int, kernel_stride: int = 16
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.head_num_k = head_num_k
        self.head_dim = head_dim
        self.kernel_stride = kernel_stride

    def forward(
        self, k: torch.Tensor, cu_seqlens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            k: (total_seq_len, num_heads, head_dim) -- packed/varlen keys,
                same layout vLLM's own attention metadata already uses
                (no reshaping needed at the call site beyond what any
                other varlen-format vLLM attention path already does).
            cu_seqlens: (batch_size + 1,) cumulative sequence lengths.
        Returns:
            compressed_k: (num_compressed_rows, num_heads, head_dim)
            cu_seqlens_compressed: (batch_size + 1,)
        """
        filtered_k_indices, cu_seqlens_compressed = calc_chunks_with_stride(
            cu_seqlens, self.kernel_size, self.kernel_stride
        )
        filtered_k = k.index_select(0, filtered_k_indices.view(-1))
        filtered_k = filtered_k.view(
            filtered_k.shape[0] // self.kernel_size,
            self.kernel_size,
            self.head_num_k,
            self.head_dim,
        )
        compressed_k = filtered_k.mean(dim=1)
        return compressed_k, cu_seqlens_compressed


def compressed_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    k2: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    topk: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_k2: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    init_blocks: int = 1,
    local_blocks: int = 2,
    cache_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Faithful, direct port of the reference `compressed_attention`
    function. Computes per-query-token top-k block indices over the
    compressed-key tiers.

    REAL DETAIL preserved exactly, not guessed independently (easy to
    get wrong without reading the actual source): `infllmv2_attn_stage1`
    is called with the tier-2 compressed keys `k2` passed as its `v`
    argument and `cu_seqlens_k2` passed as `cu_seqlens_v` -- i.e. the
    kernel's "value" input slot is being repurposed to carry the SECOND
    compression tier, not literal attention values. This is exactly what
    the reference does (`infllmv2_attn_stage1(q, k, k2, ...,
    cu_seqlens_v=cu_seqlens_k2, ...)`); reproduced verbatim here rather
    than "corrected" to look more conventional, since changing it would
    silently diverge from the real kernel contract.
    """
    if not INFLLM_V2_AVAILABLE:
        raise ImportError(
            "compressed_attention requires the infllm_v2 package "
            "(infllmv2_attn_stage1, max_pooling_1d_varlen) -- see "
            "MiniCPMSALASparseAttentionImpl's __init__ for the same "
            "check and its rationale."
        )
    with torch.no_grad():
        batch_size = cu_seqlens_q.shape[0] - 1
        is_prefilling = cache_lens is None or bool((cache_lens == 0).all().item())

        if is_prefilling:
            cache_lens = torch.zeros(batch_size, dtype=torch.int32, device=q.device)
            q_idx = torch.cat(
                [
                    (
                        torch.arange(
                            cu_seqlens_q[i + 1] - cu_seqlens_q[i], device=q.device
                        )
                        + max_seqlen_q
                        - (cu_seqlens_q[i + 1] - cu_seqlens_q[i])
                    )
                    // block_size
                    for i in range(batch_size)
                ],
                dim=0,
            )
        else:
            q_idx = cache_lens // block_size

        score = infllmv2_attn_stage1(
            q.contiguous(),
            k.contiguous(),
            k2.contiguous(),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_v=cu_seqlens_k2,  # k2 rides the "v" slot, see docstring
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=is_prefilling,
        )
        score = score[:, : q_idx.shape[0], :]

        block_score = max_pooling_1d_varlen(
            score.contiguous(),
            cu_seqlens_q,
            cu_seqlens_k,
            cache_lens,
            max_seqlen_q,
            max_seqlen_k,
            local_blocks=local_blocks,
            init_blocks=init_blocks,
            block_size=block_size,
            stride=kernel_stride,
        )

        topk = min(topk, block_score.shape[-1])
        topk_idx = block_score.topk(topk, dim=-1).indices.sort(-1).values
        topk_idx[topk_idx > q_idx[None, :, None]] = -1
        topk_idx = topk_idx.to(torch.int32)

    return topk_idx


@dataclass
class MiniCPMSALASparseAttentionMetadata:
    """Per-layer sparse-attention metadata."""

    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    dense_len: int
    page_block_size: int


class MiniCPMSALASparseAttentionMetadataBuilder(
    AttentionMetadataBuilder[MiniCPMSALASparseAttentionMetadata]
):
    """Translates vLLM's common metadata into sparse-layer field names."""

    _cudagraph_support = AttentionCGSupport.NEVER
    supports_update_block_table: bool = True

    def __init__(
        self,
        kv_cache_spec: "AttentionSpec",
        layer_names: list[str],
        vllm_config: "VllmConfig | None",
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_spec: "AttentionSpec",
    ) -> AttentionCGSupport:
        return AttentionCGSupport.NEVER

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MiniCPMSALASparseAttentionMetadata:
        dense_len = getattr(self.kv_cache_spec, "dense_len", None)
        if dense_len is None:
            raise ValueError(
                "HierarchicalCompressedAttentionSpec (or compatible spec) "
                "with dense_len is required for MiniCPM-SALA sparse layers"
            )
        page_block_size = self.kv_cache_spec.block_size
        validate_page_block_size(page_block_size)
        return MiniCPMSALASparseAttentionMetadata(
            query_start_loc=common_attn_metadata.query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            dense_len=int(dense_len),
            page_block_size=int(page_block_size),
        )

    def update_block_table(
        self,
        metadata: MiniCPMSALASparseAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> MiniCPMSALASparseAttentionMetadata:
        new_metadata = copy.copy(metadata)
        new_metadata.block_table = blk_table
        return new_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return False


class MiniCPMSALASparseAttentionBackend(AttentionBackend):
    """Real signatures throughout -- see module docstring for the source
    this was grounded against."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

    # From the real infllmv2_attn_with_kvcache docstring: "page_block_size
    # must be a multiple of 256" -- NOT the same constraint as
    # FlashAttentionBackend's `MultipleOf(16)`, deliberately not copied
    # from there.
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(256)]

    @staticmethod
    def get_name() -> str:
        return "MINICPM_SALA_INFLLM_V2"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        # Decoder-only causal attention -- this model has no
        # encoder/cross-attention layers (Phase 1 report: text-only
        # causal LM).
        return attn_type == AttentionType.DECODER

    @staticmethod
    def get_impl_cls() -> type["MiniCPMSALASparseAttentionImpl"]:
        return MiniCPMSALASparseAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["MiniCPMSALASparseAttentionMetadataBuilder"]:
        return MiniCPMSALASparseAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 256 != 0:
            raise ValueError(
                "MiniCPM-SALA sparse attention block_size must be a "
                "multiple of 256 (real constraint from infllm_v2's "
                "infllmv2_attn_with_kvcache docstring: 'page_block_size "
                "must be a multiple of 256'), got "
                f"block_size={block_size}."
            )
        # REVISED (previously reserved extra space for persistent
        # tier1/tier2 compressed-K storage; reverted back to full-K/V
        # only). Real design decision made while actually wiring up the
        # sparse forward path (see `_forward_sparse` below): rather than
        # persist compressed tier rows across decode steps (which would
        # need real incremental-update bookkeeping -- exactly the class
        # of stateful logic flagged as too risky to write blind since
        # Stage 3b), compression tiers are recomputed FRESH on every
        # call from the full K cache. This is the "measure before
        # optimizing" tradeoff already flagged in
        # `HierarchicalCompressedAttentionSpec`'s design note -- shipping
        # the simpler, more obviously-correct version first. Persistent
        # tier caching (requiring this shape to grow again, symmetric
        # with `HierarchicalCompressedAttentionSpec.page_size_bytes`,
        # which was ALSO reverted to full-KV-only for the same reason --
        # see that file's own updated design note) is legitimate future
        # work once real profiling shows the recompute cost matters, not
        # before.
        #
        # (num_blocks, 2, block_size, num_kv_heads, head_size) -- SAME
        # convention as FlashAttentionBackend.get_kv_cache_shape (the
        # "2" packs K and V into one tensor; confirmed by reading
        # vllm/v1/attention/backends/flash_attn.py directly).
        return (num_blocks, 2, block_size, num_kv_heads, head_size)


class MiniCPMSALASparseAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        *,
        block_size: int,
        sparse_config: MiniCPMSALASparseConfig,
    ) -> None:
        if not INFLLM_V2_AVAILABLE:
            raise ImportError(
                "MiniCPMSALASparseAttentionImpl requires the infllm_v2 package "
                "(OpenBMB/infllmv2_cuda_impl), which is not installed."
            )
        assert attn_type == AttentionType.DECODER
        validate_page_block_size(block_size)
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.page_block_size = block_size
        self.sparse_config = sparse_config
        self.compress_k1 = CompressK(
            head_num_k=self.num_kv_heads,
            head_dim=self.head_size,
            kernel_size=sparse_config.kernel_size,
            kernel_stride=sparse_config.kernel_stride,
        )
        self.compress_k2 = CompressK(
            head_num_k=self.num_kv_heads,
            head_dim=self.head_size,
            kernel_size=sparse_config.compress_k2_kernel_size,
            kernel_stride=sparse_config.compress_k2_kernel_stride,
        )
        assert sliding_window is None
        assert alibi_slopes is None
        assert logits_soft_cap is None

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        k_cache = kv_cache[:, 0]
        v_cache = kv_cache[:, 1]
        page_block_size = getattr(
            attn_metadata, "page_block_size", self.page_block_size
        )
        if page_block_size != self.page_block_size:
            raise ValueError(
                f"Metadata page_block_size ({page_block_size}) != "
                f"Impl page_block_size ({self.page_block_size})"
            )
        _assert_k_cache_page_size(k_cache, page_block_size)

        dense_len = attn_metadata.dense_len
        sparse_mask = sequence_sparse_mask(attn_metadata.seq_lens, dense_len)
        if _SPARSE_DEBUG:
            logger.info(
                "[sparse-debug] forward seq_lens=%s dense_len=%d sparse_mask=%s",
                attn_metadata.seq_lens.tolist(),
                dense_len,
                sparse_mask.tolist(),
            )
            _debug_tensor("query", query)
            _debug_tensor("key", key)
            _debug_tensor("value", value)
        if not sparse_mask.any():
            return self._forward_dense(
                query, key, value, k_cache, v_cache, attn_metadata, output
            )
        if sparse_mask.all():
            return self._forward_sparse(
                query, key, value, k_cache, v_cache, attn_metadata, output
            )
        return self._forward_mixed(
            query,
            key,
            value,
            k_cache,
            v_cache,
            attn_metadata,
            output,
            sparse_mask,
        )

    def _forward_dense(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        out = self._call_infllmv2_kvcache(
            query,
            key,
            value,
            k_cache,
            v_cache,
            attn_metadata,
            topk_idx=None,
        )
        output.copy_(out)
        return output

    def _forward_sparse(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        sc = self.sparse_config
        num_new_tokens = _num_new_tokens_per_seq(attn_metadata)
        full_k, cu_seqlens_full = _gather_full_k_with_new_tokens(
            k_cache=k_cache,
            new_key=key,
            block_table=attn_metadata.block_table,
            seq_lens_before=attn_metadata.seq_lens - num_new_tokens,
            query_start_loc=attn_metadata.query_start_loc,
            block_size=self.page_block_size,
        )
        compressed_k, cu_seqlens_k1 = self.compress_k1(full_k, cu_seqlens_full)
        compressed_k2, cu_seqlens_k2 = self.compress_k2(full_k, cu_seqlens_full)
        _debug_tensor("full_k", full_k)
        _debug_tensor("compressed_k", compressed_k)
        _debug_tensor("compressed_k2", compressed_k2)

        q_lens = attn_metadata.query_start_loc[1:] - attn_metadata.query_start_loc[:-1]
        q_for_topk, q_head_repeat = _maybe_repeat_q_heads_for_infllm(
            query, self.num_heads, self.num_kv_heads
        )
        topk_idx = compressed_attention(
            q=q_for_topk,
            k=compressed_k,
            k2=compressed_k2,
            kernel_size=sc.kernel_size,
            kernel_stride=sc.kernel_stride,
            block_size=sc.sparse_block_size,
            topk=sc.topk,
            cu_seqlens_q=attn_metadata.query_start_loc,
            cu_seqlens_k=cu_seqlens_k1,
            cu_seqlens_k2=cu_seqlens_k2,
            max_seqlen_q=int(q_lens.max().item()),
            max_seqlen_k=int(cu_seqlens_k1[1:].max().item()),
            init_blocks=sc.init_blocks,
            local_blocks=sc.local_blocks,
            cache_lens=attn_metadata.seq_lens - num_new_tokens,
        )
        _debug_tensor("topk_idx", topk_idx)

        out = self._call_infllmv2_varlen_sparse(
            query,
            key,
            value,
            attn_metadata,
            topk_idx=topk_idx,
            q_head_repeat=q_head_repeat,
        )
        _debug_tensor("sparse_attn_out", out)
        output.copy_(out)
        return output

    def _call_infllmv2_varlen_sparse(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata,
        *,
        topk_idx: torch.Tensor,
        q_head_repeat: int = 1,
    ) -> torch.Tensor:
        """Sparse attention via ``infllmv2_attn_varlen_func`` (HF reference path).

        ``infllmv2_attn_with_kvcache`` accepts ``topk_idx`` but expects a
        different layout than ``compressed_attention`` produces; the reference
        model calls ``infllmv2_attn_varlen_func`` for the sparse regime instead.
        """
        q_lens = attn_metadata.query_start_loc[1:] - attn_metadata.query_start_loc[:-1]
        max_seqlen_q = int(q_lens.max().item())
        max_seqlen_k = int(attn_metadata.seq_lens.max().item())
        cu_seqlens_k = torch.zeros(
            attn_metadata.seq_lens.shape[0] + 1,
            dtype=torch.int32,
            device=query.device,
        )
        cu_seqlens_k[1:] = torch.cumsum(
            attn_metadata.seq_lens.to(torch.int32), dim=0
        )

        q_attn = query
        if q_head_repeat > 1:
            q_attn = query.repeat_interleave(q_head_repeat, dim=1)

        num_new = _num_new_tokens_per_seq(attn_metadata)
        seq_lens_before = attn_metadata.seq_lens - num_new
        block_table = (
            attn_metadata.block_table
            if bool((seq_lens_before > 0).any().item())
            else None
        )

        out = infllmv2_attn_varlen_func(
            q_attn,
            key,
            value,
            attn_metadata.query_start_loc,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=0.0,
            softmax_scale=self.scale,
            causal=max_seqlen_q != 1,
            block_table=block_table,
            topk_idx=topk_idx,
        )
        if q_head_repeat > 1:
            out = (
                out.view(
                    out.shape[0],
                    out.shape[1] // q_head_repeat,
                    q_head_repeat,
                    out.shape[2],
                ).mean(dim=2)
            )
        return out

    def _call_infllmv2_kvcache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata,
        *,
        topk_idx: torch.Tensor | None,
        q_batched: torch.Tensor | None = None,
        k_batched: torch.Tensor | None = None,
        v_batched: torch.Tensor | None = None,
        q_head_repeat: int = 1,
    ) -> torch.Tensor:
        if q_batched is None:
            q_batched, k_batched, v_batched = _pack_varlen_qkv_for_infllm_kvcache(
                query, key, value, attn_metadata.query_start_loc
            )
        out_batched = infllmv2_attn_with_kvcache(
            q=q_batched,
            k_cache=k_cache,
            v_cache=v_cache,
            k=k_batched,
            v=v_batched,
            cache_seqlens=attn_metadata.seq_lens,
            block_table=attn_metadata.block_table,
            softmax_scale=self.scale,
            causal=True,
            topk_idx=topk_idx,
        )
        if q_head_repeat > 1:
            out_batched = (
                out_batched.view(
                    out_batched.shape[0],
                    out_batched.shape[1],
                    out_batched.shape[2] // q_head_repeat,
                    q_head_repeat,
                    out_batched.shape[3],
                ).mean(dim=3)
            )
        output = torch.empty_like(query)
        _unpack_batched_output_for_varlen(
            out_batched, attn_metadata.query_start_loc, output
        )
        return output

    def _forward_mixed(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor,
        sparse_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Per-sequence dense vs sparse dispatch for mixed-length batches."""
        dense_indices = (~sparse_mask).nonzero(as_tuple=False).flatten().tolist()
        sparse_indices = sparse_mask.nonzero(as_tuple=False).flatten().tolist()

        if dense_indices:
            sub_q, sub_k, sub_v, sub_meta, ranges = _select_varlen_sequences(
                dense_indices, query, key, value, attn_metadata
            )
            sub_out = torch.empty_like(sub_q)
            self._forward_dense(
                sub_q, sub_k, sub_v, k_cache, v_cache, sub_meta, sub_out
            )
            # per-sequence scatter-back: sub_out is packed in ranges order
            sub_offset = 0
            for start, end in ranges:
                n = end - start
                output[start:end].copy_(sub_out[sub_offset : sub_offset + n])
                sub_offset += n
            assert sub_offset == sub_out.shape[0]

        if sparse_indices:
            sub_q, sub_k, sub_v, sub_meta, ranges = _select_varlen_sequences(
                sparse_indices, query, key, value, attn_metadata
            )
            sub_out = torch.empty_like(sub_q)
            self._forward_sparse(
                sub_q, sub_k, sub_v, k_cache, v_cache, sub_meta, sub_out
            )
            # per-sequence scatter-back: sub_out is packed in ranges order
            sub_offset = 0
            for start, end in ranges:
                n = end - start
                output[start:end].copy_(sub_out[sub_offset : sub_offset + n])
                sub_offset += n
            assert sub_offset == sub_out.shape[0]

        return output


def _num_new_tokens_per_seq(attn_metadata) -> torch.Tensor:
    return attn_metadata.query_start_loc[1:] - attn_metadata.query_start_loc[:-1]


def _maybe_repeat_q_heads_for_infllm(
    query: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
) -> tuple[torch.Tensor, int]:
    """Match HF ``sparse_forward`` 16:1 Q:KV head ratio for infllm_v2."""
    required_ratio = 16
    current_ratio = num_heads // num_kv_heads
    if current_ratio >= required_ratio:
        return query, 1
    repeat_times = required_ratio // current_ratio
    return query.repeat_interleave(repeat_times, dim=1), repeat_times


def _pack_varlen_qkv_for_infllm_kvcache(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack varlen (total, H, D) tensors to (batch, seqlen, H, D)."""
    batch_size = query_start_loc.shape[0] - 1
    q_lens = query_start_loc[1:] - query_start_loc[:-1]
    max_seqlen = int(q_lens.max().item())
    num_heads = query.shape[1]
    head_dim = query.shape[2]
    num_kv_heads = key.shape[1]

    q_batched = query.new_zeros((batch_size, max_seqlen, num_heads, head_dim))
    k_batched = key.new_zeros((batch_size, max_seqlen, num_kv_heads, head_dim))
    v_batched = value.new_zeros((batch_size, max_seqlen, num_kv_heads, head_dim))
    for i in range(batch_size):
        start = int(query_start_loc[i].item())
        end = int(query_start_loc[i + 1].item())
        n = end - start
        q_batched[i, :n] = query[start:end]
        k_batched[i, :n] = key[start:end]
        v_batched[i, :n] = value[start:end]
    return q_batched, k_batched, v_batched


def _unpack_batched_output_for_varlen(
    out_batched: torch.Tensor,
    query_start_loc: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Scatter batched (batch, seqlen, H, D) output into varlen buffer."""
    batch_size = query_start_loc.shape[0] - 1
    for i in range(batch_size):
        start = int(query_start_loc[i].item())
        end = int(query_start_loc[i + 1].item())
        n = end - start
        output[start:end].copy_(out_batched[i, :n])


def _select_varlen_sequences(
    seq_indices: list[int],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata: MiniCPMSALASparseAttentionMetadata,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    MiniCPMSALASparseAttentionMetadata,
    list[tuple[int, int]],
]:
    """Extract a subset of sequences from a packed varlen batch."""
    if not seq_indices:
        raise ValueError("seq_indices must be non-empty")
    qsl = attn_metadata.query_start_loc.tolist()
    token_ranges: list[tuple[int, int]] = []
    q_parts: list[torch.Tensor] = []
    k_parts: list[torch.Tensor] = []
    v_parts: list[torch.Tensor] = []
    for i in seq_indices:
        start, end = qsl[i], qsl[i + 1]
        token_ranges.append((start, end))
        q_parts.append(query[start:end])
        k_parts.append(key[start:end])
        v_parts.append(value[start:end])

    sub_q = torch.cat(q_parts, dim=0)
    sub_k = torch.cat(k_parts, dim=0)
    sub_v = torch.cat(v_parts, dim=0)
    token_counts = [end - start for start, end in token_ranges]
    new_qsl = torch.zeros(len(seq_indices) + 1, dtype=torch.int32, device=query.device)
    new_qsl[1:] = torch.tensor(token_counts, dtype=torch.int32, device=query.device)
    new_qsl[1:] = new_qsl[1:].cumsum(dim=0)

    idx = torch.tensor(seq_indices, dtype=torch.long, device=query.device)
    sub_metadata = MiniCPMSALASparseAttentionMetadata(
        query_start_loc=new_qsl,
        seq_lens=attn_metadata.seq_lens.index_select(0, idx),
        block_table=attn_metadata.block_table.index_select(0, idx),
        dense_len=attn_metadata.dense_len,
        page_block_size=attn_metadata.page_block_size,
    )
    return sub_q, sub_k, sub_v, sub_metadata, token_ranges


def _gather_full_k_with_new_tokens(
    k_cache: torch.Tensor,
    new_key: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens_before: torch.Tensor,
    query_start_loc: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather cached + new K tokens into one contiguous varlen tensor."""
    _assert_k_cache_page_size(k_cache, block_size)
    num_seqs = seq_lens_before.shape[0]
    seq_lens_before_list = seq_lens_before.tolist()
    query_start_loc_list = query_start_loc.tolist()
    gathered_per_seq: list[torch.Tensor] = []
    for i in range(num_seqs):
        n_before = seq_lens_before_list[i]
        num_blocks_before = (n_before + block_size - 1) // block_size
        if num_blocks_before > 0:
            physical_blocks = block_table[i, :num_blocks_before]
            cached_k = k_cache[physical_blocks].reshape(
                num_blocks_before * block_size, *k_cache.shape[2:]
            )[:n_before]
        else:
            cached_k = k_cache.new_zeros((0, *k_cache.shape[2:]))

        new_start = query_start_loc_list[i]
        new_end = query_start_loc_list[i + 1]
        new_k_this_seq = new_key[new_start:new_end]
        gathered_per_seq.append(torch.cat([cached_k, new_k_this_seq], dim=0))

    full_k = torch.cat(gathered_per_seq, dim=0)
    seq_lens_after = [g.shape[0] for g in gathered_per_seq]
    cu_seqlens = torch.zeros(num_seqs + 1, dtype=torch.int32, device=full_k.device)
    cu_seqlens[1:] = torch.tensor(
        seq_lens_after, dtype=torch.int32, device=full_k.device
    ).cumsum(0)
    return full_k, cu_seqlens
