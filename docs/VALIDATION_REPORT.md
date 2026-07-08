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
| HF parity short prompts | **PARTIAL** (token-1 match 3/3; France 16-token greedy match; Hello/Briefly diverge token 2+) | **Yes** |
| HF parity long (≥8192, sparse regime) | **FAIL** (improved; first tokens closer) | **Yes** |
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
| B | `run_parity_sequential.py` | HF vs vLLM greedy + logprobs | **FAIL** (2026-07-07 A100; token-1 OK, token-2+ drift) |

Log artifacts (on validation host): `/tmp/phase2_logs/gated_run.log`, `step_b_parity.log`.

**Important:** Step 4’s own log states that end-to-end execution does **not** imply numerical
correctness. Only parity (Step B) converts execution into a correctness claim.

---

## HF parity (Stage 0 — blocking)

Harness: `pr2/scripts/gpu_validation/run_parity_sequential.py`

### Short prompts (2026-07-07 A100, post KV-cache + fla decode fixes)

| Prompt | Token 1 (HF=vLLM) | Token 2+ | `logprobs_ok` |
|--------|-------------------|----------|---------------|
| `Hello, my name is` | **2132=2132** | diverges token 2 (1417 vs 1358) | False |
| `The capital of France is` | **3019=3019** | **16-token greedy seq match** | False |
| `Briefly explain gravity:` | **1420=1420** | diverges token 2 (7670 vs 1527) | False |

`short_max_delta = inf` (disjoint top-k sets even when greedy tokens match).

### Long prompt (8200 ctx)

| HF greedy (8 tok) | vLLM greedy | Notes |
|-------------------|-------------|-------|
| `[49712, 59342, …]` | `[5330, 1367, …]` | sparse-regime drift; first token still wrong |

### Token-2+ bisect (2026-07-07 A100)

| Finding | Detail |
|---------|--------|
| Token 1 | Fixed by dense KV `do_kv_cache_update()` — all 3 short prompts match |
| Token 2 Hello | HF top-5: 1417, 2258, **1358** — vLLM picks 1358 (close logits, wrong argmax) |
| France token 2 | Full prefill on `prompt+t1` **matches** HF; lightning layers OK when HF layer-0 input used |
| Layer-1 attn (7 tok, HF h₀) | `max_abs_diff=0` — lightning kernels match when layer-0 input matches |
| `mamba_cache_mode` | `none` / `align` / `all` — **no change** to token-2 Hello |
| Dense eager prefill | In-memory `flash_attn_varlen` on fresh prefill (HF-aligned); **no change** to Hello token-2 |

**Working hypothesis:** residual drift accumulates above layer 0 on multi-token prefill/decode
(prompt-dependent); France has wider logit margins so greedy tokens still match.

### Parity fixes landed (2026-07-07)

1. **Harness:** `run_parity_sequential.py` now feeds vLLM `TokensPrompt(prompt_token_ids=…)` using the same `tokenizer.encode(..., add_special_tokens=True)` ids as HF (BOS token `1` was previously dropped when vLLM received raw strings). Short prompts run **one at a time** (`max_num_seqs=1`).
2. **Lightning (PR1 + PR2):** `fla` `chunk_simple_gla` / `fused_recurrent_simple_gla` for prefill **and decode**; `g_gamma = -slope`; `initial_state=None` on fresh sequences; decode uses `[batch, time, heads, dim]` layout for fla.
3. **RoPE policy:** zero q/k on lightning layers to match HF effective behavior on the released checkpoint (greedy 2132 vs 3566 with vLLM real RoPE).
4. **Dense minicpm4:** FlashAttention below `dense_len` in sparse backend (not infllm kvcache).
5. **KV cache (critical):** `MiniCPMSALASparseAttentionBackend.forward_includes_kv_cache_update=False` + `do_kv_cache_update()` delegates to FlashAttention — fixes decode-after-prefill on layer-0 `minicpm4` (stale cache caused greedy token `59360`).
6. **Dense eager prefill:** below `dense_len`, fresh prefills (`seq_lens_before==0`) use in-memory `flash_attn_varlen` to mirror HF `_flash_attention_forward_dense` (env `MINICPM_SALA_DENSE_EAGER_PREFILL=0` disables).
7. **Parity harness:** `enable_prefix_caching=False`, `mamba_cache_mode="none"` for deterministic hybrid state.

**Diagnostics added:** `pr2/scripts/gpu_validation/diagnostics/gate1_prefill_plus_one.py`, `gate1_two_token_logits.py`, `gate1_l1_seqlen7.py`, `gate1_mamba_mode_probe.py`.

**Open work:** layer-0 engine hidden vs HF on `prompt+t1` for failing prompts; upper-layer drift; long-context sparse parity; `check_logprobs_close`.

---

## Session status (2026-07-07 late — decode bisect)

| Area | Status | Notes |
|------|--------|-------|
| Prefill parity (engine vs HF) | **GREEN** | Native RMSNorm hook (ix: force native RMSNorm, 2026-07-07); prefill hidden/logits align on probed prompts |
| Decode parity (greedy + logprobs) | **RED** | Multi-step decode still diverges; not a prefill-only bug |
| check_logprobs_close (upstream harness) | **NOT RUN** | Overlay 	ests.models incomplete on validation host |
| 
un_parity_sequential.py (Step B) | **FAIL** | Short prompts: token-1 often matches; long / multi-step greedy still fails |

**Decode investigation (partial, 2026-07-07 A100):**

- Layer-0 hidden drift appears around **decode step ~11** on probed Hello-style runs (gate1_decode_l0_per_step.py, gate1_layer0_compare.py).
- **Hello** greedy sequence can **flip at generated token 14** (HF vs vLLM) even when early tokens match.
- Dense decode KV gather: **lock_table vs slot_mapping lead** under investigation; landed partial fixes (slot_mapping anchor, full QKV history flash, lightning GLA recompute) — parity still **RED**.
- Trace artifacts: pr2/scripts/gpu_validation/diagnostics/traces/decode_meta_latest.json, pr2/scripts/gpu_validation/traces/run_parity_sequential.log.

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
