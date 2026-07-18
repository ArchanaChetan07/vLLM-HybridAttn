# Roadmap

Honest status as of 2026-07-17.

## Phase 1 - Model integration (PR1)

| Item | Status |
|------|--------|
| Hybrid layer schedule | Done |
| Lightning Attention | Done (unit tests) |
| Dense GQA for minicpm4 layers | Done (unit tests) |
| PR1 independent import boundary | Done |
| HF greedy parity (short + long/sparse) | **Done** (A100 2026-07-17; check_logprobs_close in vLLM's own harness pending the upstream-tree PR) |

## Phase 2 - Sparse backend (PR2)

| Item | Status |
|------|--------|
| HierarchicalCompressedAttentionSpec | Done (unit tests) |
| Sparse attention backend | Done (unit tests) |
| Scheduler get_kv_cache_spec wiring | Done (unit tests) |
| Ampere+ kernel dispatch | **Done** (A100 2026-07-17) |
| Long-context sparse benchmark | **Baseline done** (performance.md); incremental tier caching pending |

## Phase 3 - Production validation

| Item | Status |
|------|--------|
| Docker PR1 gate (22 tests) | Done |
| Docker full stack (66 tests) | Done |
| T1000 sm_7.5 partial validation | Done |
| A100 (Ampere) validation | **Done** (2026-07-17, all gates + parity) |
| Multi-GPU TP | **Done** (4x RTX 4090, TP=2/4 nccl sharding + engine token parity, 2026-07-17) |
