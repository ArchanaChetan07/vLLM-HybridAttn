# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Fixed (2026-07-16) — correctness audit against the real HF reference

All items below were verified line-by-line against the actual
`openbmb/MiniCPM-SALA` `config.json` + `modeling_minicpm_sala.py` on the Hub
(not recollection). GPU parity re-run still required before any parity claim.

- **CRITICAL — lightning RoPE restored.** Both model copies previously
  ZEROED q/k under `lightning_use_rope` ("HF-effective RoPE policy"). The
  reference applies real rotary embeddings; its `cos_cached`/`sin_cached`
  are non-persistent buffers rebuilt in `__init__` and never read from
  safetensors — the "zeroed cos after load" bisect observation was a
  loading-harness artifact (meta-device/empty-weights load), not model
  behavior. Zeroed q/k silence all 24 lightning layers. Now applies the
  HF-exact fp32 rotation (`_apply_hf_rotary_bhtd`), pinned by
  `tests/.../test_minicpm_sala_rope.py` (bit-exact vs an independent
  transcription of HF `apply_rotary_pos_emb`).
- **CRITICAL — fla decode crash/layout.** The fused-recurrent decode loop
  fed `(b, h, t, d)` tensors to `fused_recurrent_simple_gla`, which takes
  `(b, t, h, d)` (initial-state head-count mismatch), and used an einops
  pattern (`"b t h d -> t (h d)"`) that drops an axis — an immediate
  runtime error on the first decode step with `fla` installed.
- **Sparse top-k off by `local_blocks`.** The reference budgets the local
  window ON TOP of the configured top-k
  (`topk = sparse_config.topk + window_size // block_size` = 96, not 64).
  Added `MiniCPMSALASparseConfig.effective_topk` and use it in
  `_forward_sparse`; unit-tested.
- **Recurrent-state dtype mismatch.** `get_mamba_state_dtype_from_config`
  (allocator) returned the model dtype while `get_state_dtype` (layer)
  returned fp32 — bf16-allocated state would be silently downcast every
  decode step. Both now return fp32 (matches the reference's fp32 GLA
  recurrence).
- **PR1/PR2 model drift resolved.** The PR2 copy had diverged (old vLLM
  `get_rope` path, MiniMax prefill kernel, unregistered `tp_slope` plain
  attribute that would stay on CPU when the module moves device,
  calculator-based state dtype). Regenerated from the fixed PR1 file +
  sparse-wiring deltas; `scripts/check_pr1_pr2_lightning_sync.py` (new,
  pure stdlib, wired into CI) now gates the drift.
- **`install_pr2_overlay.sh` post-check** asserted the removed zero-RoPE
  policy (and would have crashed on `AttributeError` against the drifted
  PR2 copy). Now asserts the fla prefill AND the real-RoPE helper, and
  fails if a zeroing policy ever returns.
- **CI never ran on the default branch** (`master` missing from workflow
  triggers). Added, plus the new sync-gate job.

### Added (2026-07-16)

- Previously referenced-but-missing validation tooling:
  `pr2/scripts/gpu_validation/assert_sparse_live.py` (Step 0),
  `step6_mixed_batch_invariance.py` (Step 6),
  `run_parity_sequential.py` (Step B harness with the TokensPrompt/BOS
  rules baked in), `scripts/install_infllm_v2.sh`,
  `scripts/verify_fresh_clone.sh`, `scripts/remote/a100_validation.sh`,
  and a `Makefile` (`verify-fresh`, `sync-check`, `pr1-gate`,
  `integration`, `lint`).
- `run_all_gpu_validation.sh` now runs Steps 0 and 6, and Step B when
  `MINICPM_SALA_WEIGHTS` is set.

### Added

- MiniCPM-SALA model integration for vLLM 0.24.0 (pinned commit `8cfeb84`)
- Lightning Attention layers via vLLM linear-attention infrastructure
- Dense GQA fallback for `minicpm4` mixer layers (PR1)
- InfLLM-V2 sparse attention backend overlay (PR2, optional)
- `HierarchicalCompressedAttentionSpec` and scheduler wiring (PR2)
- 22 PR1 unit tests and 44 PR2 unit tests (66 total in full stack)
- `docker_run_pr1.sh` and `docker_run_integration.sh` validation gates
- GPU validation suite under `pr2/scripts/gpu_validation/`

### Changed

- Split implementation into independent PR1 (model) and PR2 (sparse) branches
- Moved all sparse-only code under `pr2/` in the monorepo layout

### Verified (2026-07-03)

- Docker PR1 gate: import OK, ruff OK, 22/22 tests pass
- Docker full stack: ruff OK, 66/66 tests pass
- T1000 (sm_7.5): diagnostic PASS, gather PASS; kernel dispatch FAIL (Ampere floor)

### Known gaps

- `check_logprobs_close` against real HF weights not yet executed
- Ampere+ (sm_80+) end-to-end sparse validation pending
- Long-context sparse benchmark pending
- Multi-GPU tensor parallelism validation pending
