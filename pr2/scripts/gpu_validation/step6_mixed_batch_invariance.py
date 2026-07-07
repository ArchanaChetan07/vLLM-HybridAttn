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
) -> torch.Tensor:
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.v1.attention.backends.minicpm_sala_sparse import (
        INFLLM_V2_AVAILABLE,
        MiniCPMSALASparseAttentionImpl,
        MiniCPMSALASparseAttentionMetadata,
        parse_sparse_config,
    )
    from transformers import PretrainedConfig

    assert INFLLM_V2_AVAILABLE

    device = torch.device("cuda:0")
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
    impl = MiniCPMSALASparseAttentionImpl(
        num_heads=num_heads,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
        sparse_config=sparse_config,
    )

    total_q = sum(q_tokens_per_seq)
    max_seq = max(seq_lens)
    num_blocks = (max_seq + block_size - 1) // block_size + 2
    kv_cache = torch.zeros(
        num_blocks,
        2,
        block_size,
        num_kv_heads,
        head_size,
        device=device,
        dtype=torch.bfloat16,
    )
    query = torch.randn(total_q, num_heads, head_size, device=device, dtype=torch.bfloat16)
    key = torch.randn(total_q, num_kv_heads, head_size, device=device, dtype=torch.bfloat16)
    value = torch.randn(total_q, num_kv_heads, head_size, device=device, dtype=torch.bfloat16)
    output = torch.zeros_like(query)

    qsl = [0]
    for n in q_tokens_per_seq:
        qsl.append(qsl[-1] + n)
    query_start_loc = torch.tensor(qsl, device=device, dtype=torch.int32)
    seq_lens_t = torch.tensor(seq_lens, device=device, dtype=torch.int32)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).unsqueeze(0)
    block_table = block_table.expand(len(seq_lens), -1).contiguous()

    attn_metadata = MiniCPMSALASparseAttentionMetadata(
        query_start_loc=query_start_loc,
        seq_lens=seq_lens_t,
        block_table=block_table,
        dense_len=dense_len,
        page_block_size=block_size,
    )

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
    finally:
        import contextlib

        with contextlib.suppress(OSError):
            os.unlink(temp_file)
    return output.clone()


def main() -> int:
    assert torch.cuda.is_available()
    from vllm.v1.attention.backends.minicpm_sala_sparse import INFLLM_V2_AVAILABLE

    if not INFLLM_V2_AVAILABLE:
        print("INFLLM_V2_AVAILABLE is False -- cannot run mixed-batch E2E test")
        return 1

    torch.manual_seed(42)
    seq_lens = [100, 9000]
    q_tokens = [3, 4]
    batched = _run_sparse_forward(seq_lens, q_tokens)

    torch.manual_seed(42)
    solo0 = _run_sparse_forward([seq_lens[0]], [q_tokens[0]])
    torch.manual_seed(42)
    solo1 = _run_sparse_forward([seq_lens[1]], [q_tokens[1]])

    qsl = [0, q_tokens[0], sum(q_tokens)]
    if not torch.equal(batched[: qsl[1]], solo0):
        print("FAIL: short sequence output differs in mixed vs solo batch")
        return 1
    if not torch.equal(batched[qsl[1] :], solo1):
        print("FAIL: long sequence output differs in mixed vs solo batch")
        return 1
    print("PASS: mixed-batch outputs match per-sequence solo runs token-for-token")
    return 0


if __name__ == "__main__":
    sys.exit(main())
