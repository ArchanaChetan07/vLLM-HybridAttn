#!/usr/bin/env python3
"""GPU Step 6: mixed-batch batch-invariance with real infllm_v2 kernels.

Each sequence is decoded greedily for several tokens. Outputs must match
whether a sequence runs alone or in a batch mixing sub-dense_len and
super-dense_len contexts.
"""

import os
import sys
import tempfile

import torch


def _run_sparse_forward(
    seq_lens: list[int],
    q_tokens_per_seq: list[int],
    *,
    query: torch.Tensor | None = None,
    key: torch.Tensor | None = None,
    value: torch.Tensor | None = None,
    physical_block_offsets: list[int] | None = None,
) -> torch.Tensor:
    """Standalone helper: one NCCL init per call."""
    import contextlib

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.v1.attention.backends.minicpm_sala_sparse import INFLLM_V2_AVAILABLE

    assert INFLLM_V2_AVAILABLE
    device = torch.device("cuda:0")
    num_heads, num_kv_heads, head_size = 32, 2, 128
    total_q = sum(q_tokens_per_seq)
    if query is None:
        query = torch.randn(
            total_q, num_heads, head_size, device=device, dtype=torch.bfloat16
        )
    if key is None:
        key = torch.randn(
            total_q, num_kv_heads, head_size, device=device, dtype=torch.bfloat16
        )
    if value is None:
        value = torch.randn(
            total_q, num_kv_heads, head_size, device=device, dtype=torch.bfloat16
        )

    impl = _make_sparse_impl()
    fd, temp_file = tempfile.mkstemp()
    os.close(fd)
    try:
        vllm_config = VllmConfig()
        with set_current_vllm_config(vllm_config, check_compile=False):
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{temp_file}",
                local_rank=0,
                backend="nccl",
            )
            initialize_model_parallel(1, 1)
            return _forward_sparse_once(
                impl,
                seq_lens,
                q_tokens_per_seq,
                query=query,
                key=key,
                value=value,
                physical_block_offsets=physical_block_offsets,
            )
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()
        with contextlib.suppress(OSError):
            os.unlink(temp_file)


_SPARSE_OUTPUT_ATOL = 2e-2
_SPARSE_OUTPUT_RTOL = 2e-2


def _make_sparse_impl():
    from vllm.v1.attention.backends.minicpm_sala_sparse import (
        MiniCPMSALASparseAttentionImpl,
        parse_sparse_config,
    )
    from transformers import PretrainedConfig

    num_heads, num_kv_heads, head_size = 32, 2, 128
    block_size = 256
    dense_len = 8192
    sparse_config = parse_sparse_config(
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
    return MiniCPMSALASparseAttentionImpl(
        num_heads=num_heads,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
        sparse_config=sparse_config,
    )


def _forward_sparse_once(
    impl,
    seq_lens: list[int],
    q_tokens_per_seq: list[int],
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    physical_block_offsets: list[int] | None = None,
) -> torch.Tensor:
    from vllm.v1.attention.backends.minicpm_sala_sparse import (
        MiniCPMSALASparseAttentionMetadata,
    )

    device = query.device
    num_kv_heads, head_size = 2, 128
    block_size = 256
    dense_len = 8192
    max_seq = max(seq_lens)
    num_blocks_per_seq = (max_seq + block_size - 1) // block_size + 2
    if physical_block_offsets is None:
        physical_block_offsets = [i * num_blocks_per_seq for i in range(len(seq_lens))]
    total_blocks = max(off + num_blocks_per_seq for off in physical_block_offsets)
    kv_cache = torch.zeros(
        total_blocks,
        2,
        block_size,
        num_kv_heads,
        head_size,
        device=device,
        dtype=torch.bfloat16,
    )
    output = torch.zeros_like(query)
    qsl = [0]
    for n in q_tokens_per_seq:
        qsl.append(qsl[-1] + n)
    query_start_loc = torch.tensor(qsl, device=device, dtype=torch.int32)
    seq_lens_t = torch.tensor(seq_lens, device=device, dtype=torch.int32)
    block_table = torch.zeros(
        (len(seq_lens), num_blocks_per_seq), device=device, dtype=torch.int32
    )
    for i, off in enumerate(physical_block_offsets):
        block_table[i] = torch.arange(
            off, off + num_blocks_per_seq, device=device, dtype=torch.int32
        )
    attn_metadata = MiniCPMSALASparseAttentionMetadata(
        query_start_loc=query_start_loc,
        seq_lens=seq_lens_t,
        block_table=block_table,
        dense_len=dense_len,
        page_block_size=block_size,
    )
    impl.forward(
        layer=None,
        query=query,
        key=key,
        value=value,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
        output=output,
    )
    torch.cuda.synchronize()
    return output.clone()


def main() -> int:
    import contextlib

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.v1.attention.backends.minicpm_sala_sparse import INFLLM_V2_AVAILABLE

    assert torch.cuda.is_available()
    if not INFLLM_V2_AVAILABLE:
        print("INFLLM_V2_AVAILABLE is False -- cannot run mixed-batch E2E test")
        return 1

    torch.manual_seed(42)
    seq_lens = [100, 9000]
    q_tokens = [3, 4]
    device = torch.device("cuda:0")
    num_heads, num_kv_heads, head_size = 32, 2, 128
    total_q = sum(q_tokens)
    full_query = torch.randn(
        total_q, num_heads, head_size, device=device, dtype=torch.bfloat16
    )
    full_key = torch.randn(
        total_q, num_kv_heads, head_size, device=device, dtype=torch.bfloat16
    )
    full_value = torch.randn(
        total_q, num_kv_heads, head_size, device=device, dtype=torch.bfloat16
    )

    impl = _make_sparse_impl()
    fd, temp_file = tempfile.mkstemp()
    os.close(fd)
    try:
        vllm_config = VllmConfig()
        with set_current_vllm_config(vllm_config, check_compile=False):
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{temp_file}",
                local_rank=0,
                backend="nccl",
            )
            initialize_model_parallel(1, 1)
            batched = _forward_sparse_once(
                impl,
                seq_lens,
                q_tokens,
                query=full_query,
                key=full_key,
                value=full_value,
            )
            solo0 = _forward_sparse_once(
                impl,
                [seq_lens[0]],
                [q_tokens[0]],
                query=full_query[: q_tokens[0]],
                key=full_key[: q_tokens[0]],
                value=full_value[: q_tokens[0]],
                physical_block_offsets=[0],
            )
            solo1 = _forward_sparse_once(
                impl,
                [seq_lens[1]],
                [q_tokens[1]],
                query=full_query[q_tokens[0] :],
                key=full_key[q_tokens[0] :],
                value=full_value[q_tokens[0] :],
                physical_block_offsets=[0],
            )
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()
        with contextlib.suppress(OSError):
            os.unlink(temp_file)

    qsl = [0, q_tokens[0], sum(q_tokens)]
    if not torch.equal(batched[: qsl[1]], solo0):
        print("FAIL: short sequence output differs in mixed vs solo batch")
        return 1
    if not torch.allclose(
        batched[qsl[1] :], solo1, rtol=_SPARSE_OUTPUT_RTOL, atol=_SPARSE_OUTPUT_ATOL
    ):
        diff = (batched[qsl[1] :] - solo1).abs().max().item()
        print("FAIL: long sequence output differs in mixed vs solo batch")
        print(f"max abs diff: {diff}")
        return 1
    print("PASS: mixed-batch outputs match per-sequence solo runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
