# Merge Readiness Checklist

Evidence must come from a **single gated run** on Ampere+ GPU with weights present.
Do not mark PASS without log artifacts under `/tmp/phase2_logs/` (or equivalent).

## Gates

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

- [ ] All fixes committed (no site-packages-only overlay)
- [ ] `bash scripts/verify_fresh_clone.sh` PASS
- [ ] `make lint` PASS
- [ ] CPU pytest suite PASS (includes `test_minicpm_sala_infllm_pack.py`)
- [ ] No debug prints in production paths (`MINICPM_SALA_DEBUG_SPARSE` only)

## Correctness

- [ ] `test_minicpm_sala.py` — dense regime logprobs
- [ ] `test_minicpm_sala_long_context.py` — sparse regime logprobs
- [ ] Boundary `seq_len == dense_len` covered by unit test

## Documentation

- [ ] `docs/minicpm_sala_known_limitations.md` updated with run date + hardware
- [ ] Performance benchmarks executed per `docs/minicpm_sala_benchmark_plan.md`

## Current status (2026-07-07)

| Item | Status |
|------|--------|
| Steps 1, 3, 4 on RTX 4090 | PASS (pre-gated session) |
| Step 2 | Fix applied; **re-run required** |
| Steps 0, B, C, 6, 5 | **Not completed** in gated run |
| Fresh clone script | Added; **run required** |
| Merge-ready | **NO** |

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
