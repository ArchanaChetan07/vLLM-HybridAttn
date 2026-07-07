# Validation report

**Last updated:** 2026-07-07  
**Hardware reference:** NVIDIA A100 80 GB (sm_80), vLLM 0.24.0, PyTorch 2.11+cu130  
**Weights:** `openbmb/MiniCPM-SALA` (~19 GB)

This document is the evidence bundle linked from the README. It separates **validated**
from **pending**. Nothing here is claimed green without a log path or reproducible command.

---

## Executive summary

| Gate | Status | Blocks upstream PR1? |
|------|--------|----------------------|
| CPU unit tests (PR1 Docker gate) | **PASS** (22 tests) | No |
| CPU unit tests (full overlay) | **PASS** (74 tests on `feature/minicpm-sala-sparse`) | PR2 only |
| Gated GPU Steps 0–4, 6 (sparse LIVE) | **PASS** (A100, 2026-07-07) | PR2 pipeline only |
| HF parity short prompts | **PENDING RE-RUN** (KV-cache + decode fixes landed 2026-07-07; last run **FAIL**) | **Yes** |
| HF parity long (≥8192, sparse regime) | **NOT COMPLETED** | **Yes** |
| `check_logprobs_close` in upstream harness | **NOT RUN** | **Yes** |

**Verdict:** PR1 is **not** numerically verified. PR2 sparse path **runs** on real kernels but
**correctness is not proven** until parity passes.

---

## Toolchain (observed on A100 validation host)

| Component | Version / note |
|-----------|----------------|
| GPU | NVIDIA A100 80 GB, sm_80 |
| Driver / CUDA | CUDA 13 runtime drift observed (`torch 2.11+cu130`) |
| vLLM | 0.24.0 (installed) |
| infllm_v2 | Built for sm_80 from OpenBMB/infllmv2_cuda_impl |
| flash-attn | 2.x (HF reference path) |
| flash-linear-attention (`fla`) | Required for HF `chunk_simple_gla` reference |

**Reproduction:** `bash scripts/remote/a100_validation.sh` (requires weights at
`MINICPM_SALA_WEIGHTS`).

---

## CPU gates (reproducible without GPU)

```bash
bash docker_run_pr1.sh          # PR1: 22 pytest cases, ruff
bash scripts/verify_fresh_clone.sh   # overlay + CPU suite (sparse branch)
```

| Suite | Count | Branch | Status |
|-------|-------|--------|--------|
| `test_minicpm_sala_schedule.py` | 17 | PR1 | PASS |
| `test_minicpm_sala_fused_residual.py` | 4 | PR1 | PASS |
| `test_minicpm_sala_mamba_helpers.py` | 2 | PR1 | PASS |
| PR2 pack / KV / sparse unit tests | 52+ | sparse overlay | PASS (74 total on sparse branch) |

---

## Gated GPU suite (`feature/minicpm-sala-sparse`)

Script: `pr2/scripts/gpu_validation/run_all_gpu_validation.sh`

| Step | Script | What it proves | A100 result |
|------|--------|----------------|-------------|
| 0 | `assert_sparse_live.py` | Sparse backend LIVE (not dense fallback) | **PASS** |
| 1 | `step1_environment_check.py` | vLLM + infllm_v2 + arch | **PASS** |
| 2 | `step2_kernel_dispatch.py` | Real `linear_attention_prefill_and_mix` | **PASS** |
| 3 | `step3_real_gather_test.py` | Tier gather on real KV | **PASS** |
| 4 | `step4_sparse_e2e_test.py` | Sparse path past `dense_len` | **PASS** (runs, not correct) |
| 6 | `step6_mixed_batch_invariance.py` | Mixed dense/sparse batch | **PASS** |
| B | `run_parity_sequential.py` | HF vs vLLM greedy + logprobs | **PENDING RE-RUN** (last run **FAIL**) |

Log artifacts (on validation host): `/tmp/phase2_logs/gated_run.log`, `step_b_parity.log`.

**Important:** Step 4’s own log states that end-to-end execution does **not** imply numerical
correctness. Only parity (Step B) converts execution into a correctness claim.

---

## HF parity (Stage 0 — blocking)

Harness: `pr2/scripts/gpu_validation/run_parity_sequential.py`

### Short prompts (2026-07-07, last full run before triage fixes)

| Prompt | HF greedy (token 0) | vLLM greedy | `logprobs_ok` |
|--------|---------------------|-------------|---------------|
| `Hello, my name is` | 2132 | 1709 → 3566 after partial fixes | False |
| `The capital of France is` | 3019 | mismatch | False |
| `Briefly explain gravity:` | 1420 | mismatch | False |

`short_max_delta = inf` (disjoint top-k sets).

### Long prompt (≥8192 tokens)

Earlier run failed with `TypeError: LLM.generate() got an unexpected keyword argument
'prompt_token_ids'`. Fixed locally to `llm.generate([long_ids], ...)`. **Re-run pending**
after parity fixes land on branch.

### Bisect findings (A100, reproducible)

| Checkpoint | `max_abs_diff` | Notes |
|------------|----------------|-------|
| embed | 0.0 | Weight load OK |
| layer-1 q after q_norm | 0.0 | Projections OK |
| layer-1 q after RoPE | 26.25 | HF `cos_cached` zeroed after load → q/k zeroed |
| layer-1 attn (HF h₀ input, fla kernels) | 0.0 | Lightning path can match HF |
| full-model greedy | HF 2132 vs vLLM 3566/1709 | Harness + kernel gaps (see below) |

### Parity fixes landed (2026-07-07, pending GPU re-run)

1. **Harness:** `run_parity_sequential.py` now feeds vLLM `TokensPrompt(prompt_token_ids=…)` using the same `tokenizer.encode(..., add_special_tokens=True)` ids as HF (BOS token `1` was previously dropped when vLLM received raw strings). Short prompts run **one at a time** (`max_num_seqs=1`).
2. **Lightning (PR1 + PR2):** `fla` `chunk_simple_gla` / `fused_recurrent_simple_gla` for prefill **and decode**; `g_gamma = -slope`; `initial_state=None` on fresh sequences; decode uses `[batch, time, heads, dim]` layout for fla.
3. **RoPE policy:** zero q/k on lightning layers to match HF effective behavior on the released checkpoint (greedy 2132 vs 3566 with vLLM real RoPE).
4. **Dense minicpm4:** FlashAttention below `dense_len` in sparse backend (not infllm kvcache).
5. **KV cache (critical):** `MiniCPMSALASparseAttentionBackend.forward_includes_kv_cache_update=False` + `do_kv_cache_update()` delegates to FlashAttention — fixes decode-after-prefill on layer-0 `minicpm4` (stale cache caused greedy token `59360`).

**Re-run required:** `bash scripts/install_pr2_overlay.sh && python3 pr2/scripts/gpu_validation/run_parity_sequential.py` on A100 with weights at `MINICPM_SALA_WEIGHTS`.

**Open work:** Confirm Step B PASS after re-run; long-context (≥8192) sparse-regime parity still needs infllm_v2 validated end-to-end.

---

## Clean-clone reproduction

```bash
git clone https://github.com/ArchanaChetan07/vLLM-HybridAttn.git
cd vLLM-HybridAttn
git checkout feature/minicpm-sala-sparse   # or feature/pr1-upstream-staging for PR1 only
pip install vllm==0.24.0
bash scripts/install_pr2_overlay.sh          # sparse branch only
# GPU host: build infllm_v2, set MINICPM_SALA_WEIGHTS, run gated suite
```

---

## Related documents

- [DESIGN_RFC.md](DESIGN_RFC.md) — architecture intent
- [minicpm_sala_known_limitations.md](minicpm_sala_known_limitations.md) — detailed limits
- [merge_readiness_checklist.md](merge_readiness_checklist.md) — maintainer objection matrix
