# Merge Readiness Checklist

Evidence must come from a **single gated run** on Ampere+ GPU with weights present.
Do not mark PASS without log artifacts under `/tmp/phase2_logs/` (or equivalent).

## Branch policy (2026-07-07)

| Branch | Role |
|--------|------|
| `main` | **PR1 dense path only** — validated CPU gate; no PR2 sparse overlay |
| `feature/minicpm-sala-sparse` | Sparse (PR2) work in progress — **not GPU-validated for production** |

PR #5 was merged prematurely and **reverted on `main`** (see revert commit).
Re-merge sparse into `main` only after the gated GPU run + HF parity below.
Because the un-merge used `git revert -m 1`, a future merge requires **reverting
that revert first** (or merging via a fresh branch rebased on current `main`).

## Gates (required before re-merge to main)

| # | Gate | Command | Pass criteria |
|---|------|---------|---------------|
| 0 | Sparse LIVE | `assert_sparse_live.py` | `INFLLM_V2_AVAILABLE=True`, fail-loud wiring |
| 1 | Diagnostic | `step1_diagnostic.py` | Imports, backend resolution |
| 2 | Lightning kernel | `step2_kernel_dispatch.py` | Prefill + decode, finite non-zero output |
| 3 | Paged gather | `step3_real_gather_test.py` | Real block_table gather |
| 4 | Sparse e2e | `step4_sparse_e2e_test.py` | seq_len > dense_len, non-zero output |
| 6 | Mixed impl | `step6_mixed_batch_invariance.py` | Solo == batched per sequence |
| B | HF parity | `run_parity_sequential.py` | Token + logprob match (short + long) |
| C | Full-model batch | `step_c_mixed_batch_greedy.py` | Greedy invariance solo vs batch |
| 5 | TP | `step5_multi_gpu_tp_test.py` | `tp_slope` shards match (2+ GPUs) |

## Repository quality

- [x] CPU pack test fixed (`test_minicpm_sala_infllm_pack.py` — shape follows max(q_lens))
- [x] `bash scripts/verify_fresh_clone.sh` — CPU-only copy set; git check optional
- [ ] `make lint` PASS on feature branch
- [ ] CPU pytest suite PASS on feature branch (fresh clone)
- [ ] No debug prints in production paths (`MINICPM_SALA_DEBUG_SPARSE` only)

## Correctness (GPU + weights — not done)

- [ ] `test_minicpm_sala.py` — dense regime logprobs
- [ ] `test_minicpm_sala_long_context.py` — sparse regime logprobs
- [x] Boundary `seq_len == dense_len` covered by unit test (CPU)

## Documentation

- [ ] `docs/minicpm_sala_known_limitations.md` updated with gated run date + hardware
- [ ] Performance benchmarks executed per `docs/minicpm_sala_benchmark_plan.md`

## Current status (2026-07-07)

| Item | Status |
|------|--------|
| `main` | PR1 dense only; PR #5 merge **reverted** |
| `feature/minicpm-sala-sparse` | Sparse code preserved; CPU gate fixes applied |
| Steps 1, 3, 4 on RTX 4090 | PASS (pre-gated session — **not re-run in single gated suite**) |
| Step 2 | Fix applied; **re-run required** |
| Steps 0, B, C, 6, 5 | **Not completed** in gated run |
| Merge-ready | **NO** — sparse production still **NO-GO** |

## One-command gated run (GPU host)

```bash
export MINICPM_SALA_WEIGHTS=/workspace/models/openbmb/MiniCPM-SALA
export MINICPM_SALA_DOWNLOAD_WEIGHTS=1   # optional if weights missing
mkdir -p /tmp/phase2_logs

pip install "vllm==0.24.0"
bash scripts/install_pr2_overlay.sh
bash scripts/install_infllm_v2.sh

bash pr2/scripts/gpu_validation/run_all_gpu_validation.sh \
  2>&1 | tee /tmp/phase2_logs/full_gpu_validation.log
```
