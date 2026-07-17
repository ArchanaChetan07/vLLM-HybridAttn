# Performance

## First measured baseline (2026-07-17)

Host: NVIDIA A100-SXM4-80GB (sm_80), vLLM 0.25.0 + PR2 overlay, PyTorch
2.11+cu130, CUDA 13, bf16, `enforce_eager` (no CUDA graphs), greedy
decoding, `block_size=256`. Script:
`pr2/scripts/gpu_validation/bench_throughput.py` (after warmup; wall-clock
over `LLM.generate`). These are BASELINE numbers for the correctness-first
port — see the caveats below, not a leaderboard entry.

| Scenario | Throughput | Wall time |
|----------|-----------|-----------|
| decode bs=1, ctx 64 (dense regime) | 25.5 output tok/s | 10.06 s / 256 tok |
| decode bs=8, ctx 64 (dense regime) | 170.2 output tok/s aggregate | 6.02 s / 8×128 tok |
| prefill 4096 (dense regime) | 10,185 prefill tok/s | 0.40 s |
| prefill 8300 (sparse regime) | 9,175 prefill tok/s | 0.90 s |
| decode bs=1, ctx 8300 (sparse regime) | 15.2 output tok/s | 4.22 s / 64 tok |

Observations (honest):

- **Decode scales with batch** (25.5 → 170 tok/s at bs=8): the 2026-07-17
  batched fla decode (one `fused_recurrent_simple_gla` call per layer per
  step, replacing a per-sequence Python loop) is doing its job.
- **Sparse prefill costs only ~10% over dense** at 8.3k tokens — the
  compression + top-k selection overhead is amortized well in prefill.
- **Sparse decode drops to ~60% of dense decode** (15.2 vs 25.5 tok/s):
  the sparse path currently RECOMPUTES both compression tiers from a full
  K gather on every decode step (a deliberate correctness-first choice —
  see `HierarchicalCompressedAttentionSpec`'s design note). Incremental
  tier caching (the reference's ring-buffer scheme) is the top
  optimization target.

## Known optimization headroom (not yet done)

1. `enforce_eager` off / CUDA-graph capture for the lightning layers.
2. Incremental compressed-tier caching in the sparse decode path
   (removes the per-step full-K gather + CompressK recompute).
3. Python-loop K/V gather (`_gather_full_k_with_new_tokens`) →
   vectorized/Triton gather.
4. In-tree `lightning_attention` Triton prefill (chunked) instead of the
   fla fp32 path, once numerics are proven equivalent.

## Hardware floors (verified)

- **Lightning kernels:** compute capability >= 8.0 (confirmed on T1000 sm_7.5 failure)
- **Sparse InfLLM-V2 kernels:** compute capability >= 8.0 + `infllm_v2` installed

## Methodology

See [minicpm_sala_benchmark_plan.md](minicpm_sala_benchmark_plan.md).
Reproduce with:

```bash
export MINICPM_SALA_WEIGHTS=/path/to/openbmb/MiniCPM-SALA
python pr2/scripts/gpu_validation/bench_throughput.py
```
