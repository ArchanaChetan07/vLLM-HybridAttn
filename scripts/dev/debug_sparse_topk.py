#!/usr/bin/env python3
"""Dev-only: diagnose sparse topk_idx and varlen kernel output on GPU."""
import torch
from transformers import PretrainedConfig

from vllm.v1.attention.backends.minicpm_sala_sparse import (
    MiniCPMSALASparseAttentionImpl,
    MiniCPMSALASparseAttentionMetadata,
    _gather_full_k_with_new_tokens,
    _maybe_repeat_q_heads_for_infllm,
    _num_new_tokens_per_seq,
    compressed_attention,
    parse_sparse_config,
)

device = torch.device("cuda:0")
num_heads, num_kv_heads, head_size = 32, 2, 128
block_size = 256
dense_len = 8192
seq_len = 8448
sc = parse_sparse_config(
    PretrainedConfig(
        sparse_config={
            "kernel_size": 32,
            "kernel_stride": 16,
            "init_blocks": 1,
            "block_size": 64,
            "window_size": 2048,
            "topk": 64,
            "dense_len": dense_len,
        }
    )
)
impl = MiniCPMSALASparseAttentionImpl(
    num_heads=num_heads,
    head_size=head_size,
    scale=head_size**-0.5,
    num_kv_heads=num_kv_heads,
    block_size=block_size,
    sparse_config=sc,
)
num_blocks = (seq_len + block_size - 1) // block_size + 1
kv_cache = torch.zeros(
    num_blocks, 2, block_size, num_kv_heads, head_size, device=device, dtype=torch.bfloat16
)
query = torch.randn(seq_len, num_heads, head_size, device=device, dtype=torch.bfloat16)
key = torch.randn(seq_len, num_kv_heads, head_size, device=device, dtype=torch.bfloat16)
value = torch.randn(seq_len, num_kv_heads, head_size, device=device, dtype=torch.bfloat16)
meta = MiniCPMSALASparseAttentionMetadata(
    query_start_loc=torch.tensor([0, seq_len], device=device, dtype=torch.int32),
    seq_lens=torch.tensor([seq_len], device=device, dtype=torch.int32),
    block_table=torch.arange(num_blocks, device=device, dtype=torch.int32).unsqueeze(0),
    dense_len=dense_len,
    page_block_size=block_size,
)
num_new = _num_new_tokens_per_seq(meta)
full_k, cu_full = _gather_full_k_with_new_tokens(
    kv_cache[:, 0], key, meta.block_table, meta.seq_lens - num_new, meta.query_start_loc, block_size
)
ck1, cu1 = impl.compress_k1(full_k, cu_full)
ck2, cu2 = impl.compress_k2(full_k, cu_full)
q_for_topk, _ = _maybe_repeat_q_heads_for_infllm(query, num_heads, num_kv_heads)
topk_idx = compressed_attention(
    q=q_for_topk,
    k=ck1,
    k2=ck2,
    kernel_size=sc.kernel_size,
    kernel_stride=sc.kernel_stride,
    block_size=sc.sparse_block_size,
    topk=sc.topk,
    cu_seqlens_q=meta.query_start_loc,
    cu_seqlens_k=cu1,
    cu_seqlens_k2=cu2,
    max_seqlen_q=seq_len,
    max_seqlen_k=int(cu1[1:].max().item()),
    init_blocks=sc.init_blocks,
    local_blocks=sc.local_blocks,
    cache_lens=meta.seq_lens - num_new,
)
valid = (topk_idx >= 0).sum().item()
print("topk_idx shape", tuple(topk_idx.shape), "valid", valid)
out_s = torch.zeros_like(query)
impl._forward_sparse(query, key, value, kv_cache[:, 0], kv_cache[:, 1], meta, out_s)
print("sparse out sum", out_s.abs().sum().item())