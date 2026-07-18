# Compatibility

## vLLM versions
| Version | Status |
|---|---|
| 0.24.0 (pin) / **0.25.0** | Overlay install validated end-to-end (all gates + HF parity) |
| `main` (post-0.25) | PR1 tree patch applies cleanly and lints under upstream config: `docs/pull_requests/0001-minicpm-sala-pr1-vllm-main.patch`. One API move handled: `MiniMaxText01RMSNormTP` merged into `minimax_linear_attn` (repo copies carry a dual-path import; the tree patch uses the new path directly). `embed_input_ids` implemented with a `get_input_embeddings` alias for <= 0.24. Upstream CI is the final authority. |

## Dependencies
| Dependency | Required for | Notes |
|---|---|---|
| `einops` | PR1 + PR2 | hard |
| `flash-linear-attention` (fla) | Lightning parity path | **optional**: absent -> falls back to in-tree `lightning_attention` Triton kernels. HF parity was proven WITH fla; the fallback is not parity-proven (flagged for maintainers). |
| `infllm_v2` (OpenBMB/infllmv2_cuda_impl) | PR2 sparse only | **optional**: absent -> dense fallback on all `minicpm4` layers. Build: `scripts/install_infllm_v2.sh` (CUTLASS patch applied; sm_80+; validated builds on sm_80 A100 and sm_89 RTX 4090, CUDA 13). |
| `flash-attn` | HF reference side of parity only | not needed to serve; shim available (`scripts/install_flash_attn_shim.sh`). |
| transformers | serving: 5.x fine | The HF *reference* modeling file requires 4.56 (parity harness runs it in a separate venv: `scripts/setup_hf_reference_env.sh`). |

## Hardware
- Lightning + sparse kernels: compute capability **>= sm_80** (T1000/sm_75 confirmed to fail). Validated: A100 (sm_80), RTX 4090 (sm_89).
- Sparse path requires paged-KV `block_size` that is a **multiple of 256** (`LLM(..., block_size=256)`); vLLM's hybrid page-size unification may pad it further (e.g. 2048) -- any multiple of 256 is accepted.

## Tensor parallelism
- TP=1/2/4 validated: nccl slope sharding (Step 5) and engine token parity (Step 5b) -- greedy tokens identical across TP degrees, including the sparse regime, and identical across sm_80/sm_89.
- TP=2 is the natural fit (16:1 q:kv per rank). TP=4 replicates kv heads (ratio 8:1) and uses the reference's q-head repeat-to-16; validated but with the extra repeat cost. TP > 4 untested.
- Pipeline parallel: interface implemented (`SupportsPP`), not validated on multi-GPU.

## Known limitations
- CUDA graphs disabled for the sparse backend (`enforce_eager` recommended); prefix caching disabled for sparse layers (Mamba-style rationale).
- Sparse decode recomputes compression tiers per step (correctness-first; ~60% of dense decode throughput -- see docs/performance.md headroom list).
