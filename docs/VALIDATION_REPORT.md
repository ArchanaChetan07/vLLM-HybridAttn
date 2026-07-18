# Validation report

**Last updated:** 2026-07-17  
**Hardware reference:** NVIDIA A100-SXM4-80GB (sm_80), vLLM 0.25.0, PyTorch 2.11+cu130, CUDA 13.0  
**Weights:** `openbmb/MiniCPM-SALA` (~19 GB, local safetensors)

## 2026-07-17 A100 session (fixed code, vLLM 0.25.0)

All gates below were re-run on the post-audit code (real lightning RoPE,
fixed fla decode, effective_topk=96, fp32 state) against **vLLM 0.25.0**
(one minor ahead of the repo's 0.24 pin -- the overlay imports and all CPU
suites pass unchanged; five 0.25 integration fixes were required and are in
the changelog).

| Gate | Result |
|------|--------|
| CPU suites (PR1 34 + PR2 46) on vLLM 0.25.0 | **PASS** |
| Step 0 sparse LIVE (infllm_v2 built for sm_80, CUDA 13) | **PASS** |
| Step 1 diagnostic / Step 2 lightning prefill + decode (fla) | **PASS** |
| Step 3 paged gather / Step 4 sparse e2e (varlen + topk) | **PASS** |
| Step 6 mixed dense/sparse batch invariance | **PASS** (max diff 0.0) |
| Engine smoke: `LLM()` load + greedy, short prompts | **PASS** -- coherent ("The capital of France is" -> " Paris.") |
| Engine smoke: 8418-token prompt (sparse regime, chunked prefill) | **PASS** -- correct long-context answer (" The fox.") |
| **Step B HF parity, short prompts (3)** | **PASS** -- greedy tokens IDENTICAL for all 16 steps on every prompt; vLLM greedy inside HF top-5 at every step |
| **Step B HF parity, long (8306 tokens, sparse regime)** | **PASS** -- greedy tokens IDENTICAL |

Evidence: [validation_logs/2026-07-17/step_b_parity_raw.txt](validation_logs/2026-07-17/step_b_parity_raw.txt), [validation_logs/2026-07-17/tp_matrix_4x4090.txt](validation_logs/2026-07-17/tp_matrix_4x4090.txt).
First-ever coherent end-to-end generation AND first-ever HF parity PASS for
this port, in both the dense and sparse regimes.

Parity-host notes (full setup scripted, see `scripts/`): the HF reference
phase runs in its own venv (`setup_hf_reference_env.sh`: transformers
4.56 as the checkpoint declares; the reference file does not import under
transformers 5.x) via `MINICPM_SALA_HF_PYTHON`; `flash_attn` on the HF side
is a shim over infllm_v2's kernels (`install_flash_attn_shim.sh`) because
flash-attn's sdist does not build against CUDA 13/torch 2.11; the installed
fla's removed `head_first` kwarg is patched to tolerate the reference's
explicit `head_first=False` (a no-op layout-wise). vLLM side: stock 0.25.0
+ the PR2 overlay, `block_size=256`.

This document is the evidence bundle linked from the README. It separates **validated**
from **pending**. Nothing here is claimed green without a log path or reproducible command.

---

## Executive summary

| Gate | Status | Blocks upstream PR1? |
|------|--------|----------------------|
| CPU unit tests (PR1 34 + PR2 46, vLLM 0.25.0) | **PASS** (2026-07-17) | No |
| Gated GPU Steps 0–4, 6 (sparse LIVE, fixed code) | **PASS** (A100, 2026-07-17) | No |
| Engine-level generation, dense + sparse regimes | **PASS** (2026-07-17, coherent output) | No |
| **HF parity short prompts** | **PASS** (2026-07-17, greedy tokens identical, 3 prompts × 16 steps) | Cleared |
| **HF parity long (≥8192, sparse regime)** | **PASS** (2026-07-17, 8306-token prompt, greedy identical) | Cleared |
| `check_logprobs_close` in vLLM's own test harness | **NOT RUN** here (needs the vLLM-tree PR checkout; the equivalent greedy + top-k-containment check passes above) | For the upstream PR itself |
| Multi-GPU TP (nccl) | **PASS** (4x RTX 4090, 2026-07-17: Step 5 sharding at TP=2/4; Step 5b engine parity — TP=1/2/4 token-identical incl. 8306-token sparse; also matches the A100 tokens cross-arch) | Cleared |
| Throughput/latency benchmarks | **NOT RUN** | No (docs/performance.md stays empty) |

**Verdict (2026-07-17):** the numerical-correctness merge blocker is
**cleared**: the port matches the HF reference greedily, token-for-token, in
both the dense and the InfLLM-V2 sparse regime, on A100 against vLLM 0.25.0.
Remaining work for upstream submission is packaging (a real vllm-tree PR
branch, `check_logprobs_close` in their harness, TP validation), not
correctness.

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
| 0 | `assert_sparse_live.py` | Sparse backend LIVE (not dense fallback) | **PASS** (2026-07-07) |
| 1 | `step1_diagnostic.py` | vLLM + infllm_v2 + arch | **PASS** |
| 2 | `step2_kernel_dispatch.py` | Real `linear_attention_prefill_and_mix` | **PASS** |
| 3 | `step3_real_gather_test.py` | Tier gather on real KV | **PASS** |
| 4 | `step4_sparse_e2e_test.py` | Sparse path past `dense_len` | **PASS** (runs, not correct) |
| 6 | `step6_mixed_batch_invariance.py` | Mixed dense/sparse batch | **PASS** (2026-07-07) — **re-run needed** (script re-created 2026-07-16 with real-kernel invariance check + `effective_topk` fix) |
| B | `run_parity_sequential.py` | HF vs vLLM greedy + logprobs | **PENDING RE-RUN** (last run **FAIL**; RoPE/topk/decode fixes landed 2026-07-16) |

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
| layer-1 q after RoPE | 26.25 | ⚠ HF side had zeroed `cos_cached` — a **loading-harness artifact**, see below |
| layer-1 attn (HF h₀ input, fla kernels) | 0.0 | Lightning path can match HF |
| full-model greedy | HF 2132 vs vLLM 3566/1709 | Harness + kernel gaps (see below) |

**Correction (2026-07-16):** the bisect run's HF reference had a zeroed
`cos_cached`, which was misread as "HF effectively zeroes q/k". The reference
`MiniCPMRotaryEmbedding` registers `cos_cached`/`sin_cached` as
**non-persistent buffers rebuilt in `__init__`** — they are never loaded from
safetensors and are never zero in a correct `from_pretrained` load. The zeroed
buffers came from the bisect harness's meta-device/empty-weights load path.
The HF greedy token `2132` recorded above is therefore itself suspect and the
whole bisect must be re-run with a clean HF load.

### Parity fixes landed (2026-07-07 harness/kernels, 2026-07-16 audit; pending GPU re-run)

1. **Harness:** `run_parity_sequential.py` feeds vLLM `TokensPrompt(prompt_token_ids=…)` using the same `tokenizer.encode(..., add_special_tokens=True)` ids as HF (BOS token `1` was previously dropped when vLLM received raw strings).
2. **Lightning (PR1 + PR2):** `fla` `chunk_simple_gla` / `fused_recurrent_simple_gla` for prefill **and decode**; `g_gamma = -slope`; `initial_state=None` on fresh sequences. 2026-07-16: decode loop fixed to the kernel's real `(b, t, h, d)` layout (previous revision passed `(b, h, t, d)` and crashed on an einops axis-drop).
3. **RoPE policy (REVERSED 2026-07-16):** lightning layers now apply **real** HF-exact RoPE (`_apply_hf_rotary_bhtd`: fp32 cos/sin, fp32 rotation, cast back — bit-exact vs HF `apply_rotary_pos_emb`, see `tests/.../test_minicpm_sala_rope.py`). The earlier "zero q/k to match HF effective behavior" policy was based on the harness artifact described above and silenced all 24 lightning layers.
4. **Dense minicpm4 (< `dense_len`) in the sparse backend:** `infllmv2_attn_with_kvcache(..., topk_idx=None)` — the kernel's dense mode against the paged cache (the flash-attn-fork equivalent of the reference's `_flash_attention_forward_dense`).
5. **Sparse top-k (2026-07-16):** `compressed_attention` now receives `topk + window_size // block_size` (= 96), matching `MiniCPMInfLLMv2Attention.__init__`; the raw config `topk` (64) under-selected by the whole local-window budget.

**Re-run required:** `bash scripts/remote/a100_validation.sh` on an A100 with `MINICPM_SALA_WEIGHTS` set (runs CPU gates → infllm_v2 build → Steps 0–6 → Step B short + long).

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
