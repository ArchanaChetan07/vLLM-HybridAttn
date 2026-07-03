# Roadmap

Honest status as of 2026-07-03.

## Phase 1 - Model integration (PR1)

| Item | Status |
|------|--------|
| Hybrid layer schedule | Done |
| Lightning Attention | Done (unit tests) |
| Dense GQA for minicpm4 layers | Done (unit tests) |
| PR1 independent import boundary | Done |
| check_logprobs_close HF parity | Pending |

## Phase 2 - Sparse backend (PR2)

| Item | Status |
|------|--------|
| HierarchicalCompressedAttentionSpec | Done (unit tests) |
| Sparse attention backend | Done (unit tests) |
| Scheduler get_kv_cache_spec wiring | Done (unit tests) |
| Ampere+ kernel dispatch | Pending |
| Long-context sparse benchmark | Pending |

## Phase 3 - Production validation

| Item | Status |
|------|--------|
| Docker PR1 gate (22 tests) | Done |
| Docker full stack (66 tests) | Done |
| T1000 sm_7.5 partial validation | Done |
| A40 / Ampere+ validation | Pending |
| Multi-GPU TP | Pending |
