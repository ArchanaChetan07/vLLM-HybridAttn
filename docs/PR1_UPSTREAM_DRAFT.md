# Upstream PR1 — MiniCPM-SALA model support (draft)

> **Status:** Staged on branch `feature/pr1-upstream-staging`.  
> **Target:** `vllm-project/vllm` (paste this body when opening the upstream PR).  
> **Do not merge** until HF `check_logprobs_close` passes on maintainer CI hardware.

This draft supersedes the shorter notes in `docs/UPSTREAM_PR1.md` for the actual GitHub PR body.
Integration-repo validation evidence lives in `docs/VALIDATION_REPORT.md`.

---

## Summary

Adds inference support for [MiniCPM-SALA](https://huggingface.co/openbmb/MiniCPM-SALA) to vLLM:
a 32-layer hybrid model combining **gated linear (Lightning) attention** on 24 layers and
**dense GQA** on 8 `minicpm4` layers.

**PR1 scope (mergeable, CPU-verifiable):**

- In-tree `MiniCPMSALAForCausalLM` with dense GQA for all contexts in this PR
- Lightning layers via existing vLLM `linear_attention` custom op
- CPU unit tests (schedule, residual, mamba helpers)
- HF parity **scaffold** (`test_minicpm_sala.py` + `check_logprobs_close`)

**Explicitly out of scope for PR1:**

- `infllm_v2` / InfLLM-V2 sparse backend
- Custom hierarchical KV-cache spec
- Long-context sparse-regime parity (follow-up PR2)

## Motivation

MiniCPM-SALA requires per-layer mixer dispatch, NoPE dense GQA, gated linear attention with
RoPE and Mamba-compatible state, and muP residual scaling. None of this exists as a single
upstream model today.

## Architecture

| Layer type | Count | PR1 implementation |
|------------|-------|-------------------|
| `lightning-attn` | 24 | `MiniCPMSALALightningAttention` → `torch.ops.vllm.linear_attention` |
| `minicpm4` | 8 | `MiniCPMSALADenseAttention` → standard `Attention` (dense GQA, NoPE) |

Below `dense_len=8192` the checkpoint uses dense attention — PR1 matches that regime.
At/above `dense_len`, PR1 continues dense GQA; sparse InfLLM-V2 is deferred to PR2.

## Files to add / patch

```
vllm/model_executor/models/minicpm_sala.py          # new (PR1 canonical; no sparse imports)
vllm/model_executor/models/registry.py              # +1 entry (patches/registry.py.patch)
tests/models/registry.py                            # +1 entry (patches/tests_registry.py.patch)
tests/models/language/generation/test_minicpm_sala_schedule.py
tests/models/language/generation/test_minicpm_sala_fused_residual.py
tests/models/language/generation/test_minicpm_sala_mamba_helpers.py
tests/models/language/generation/test_minicpm_sala.py
```

Apply `patches/registry.py.patch` and `patches/tests_registry.py.patch` when porting to a vLLM fork.
See `upstream/README.md` for copy-paste snippets.

**PR1 / PR2 boundary (integration repo):** `vllm/model_executor/models/minicpm_sala.py` is the
PR1-facing canonical file. The `pr2/` overlay replaces it at install time for sparse work — that
overlay is **not** part of this upstream PR.

## Tests

| Test file | Cases | What it covers | Runs on CPU? |
|-----------|-------|----------------|--------------|
| `test_minicpm_sala_schedule.py` | 17 | Mixer schedule, layer dispatch | Yes |
| `test_minicpm_sala_fused_residual.py` | 4 | muP residual math | Yes |
| `test_minicpm_sala_mamba_helpers.py` | 2 | State shape / dtype contracts | Yes |
| `test_minicpm_sala.py` | 1 | HF `check_logprobs_close` | **No** (GPU + weights) |

```bash
# CPU gate (integration repo: docker_run_pr1.sh)
pytest tests/models/language/generation/test_minicpm_sala_*.py -m "not hybrid_model"

# GPU parity (maintainer CI only — not claimed green yet)
pytest tests/models/language/generation/test_minicpm_sala.py -m hybrid_model
```

**CPU today:** 22/22 pass in `docker_run_pr1.sh` (integration repo, 2026-07-07).  
**GPU parity:** scaffold collects; execution **not green** — do not claim otherwise.

## Validation evidence (integration repo)

| Claim | Status | Notes |
|-------|--------|-------|
| Model registry patch | Ready | `patches/registry.py.patch` |
| Test registry patch | Ready | `patches/tests_registry.py.patch` |
| CPU unit tests | **22/22 PASS** | `docker_run_pr1.sh` |
| Weight loading | Validated | sparse-branch overlay host |
| Lightning kernel dispatch | Validated | A100 sparse branch only — **not a PR1 merge claim** |
| HF `check_logprobs_close` | **NOT GREEN** | Last run FAIL; fixes pending A100 re-run |

We do **not** claim numerical equivalence until parity passes.

## Known limitations (PR1)

- `minicpm4` layers: dense GQA only; contexts ≥8192 stay dense until PR2
- Lightning kernels require Ampere+ (sm_80+) at **runtime**; CPU tests cover schedule/residual only
- `check_logprobs_close`: harness scaffold present, execution blocked on GPU parity

## Follow-up PR2 (sparse, separate)

| Topic | Plan |
|-------|------|
| Dependency | `infllm_v2` from OpenBMB/infllmv2_cuda_impl |
| Integration | `minicpm_sala_sparse_wiring.py` + sparse attention backend |
| KV cache | `MiniCPMSALAKVCacheSpec` |
| Degradation | Import guard → dense `Attention` when extension absent |

PR2 is **not** required to merge PR1.

## Checklist for reviewers

- [x] Single PR1 `minicpm_sala.py`; no sparse/infllm imports
- [x] Registry patches provided for model + test registries
- [x] `HasInnerState`, `IsHybrid`, `SupportsPP` declared accurately
- [x] CPU unit tests (22) pass in Docker gate
- [ ] `check_logprobs_close` green on GPU CI
- [ ] Multi-GPU TP smoke (deferred)

## Suggested commit message

```
Add MiniCPM-SALA hybrid attention model

Introduces MiniCPMSALAForCausalLM with lightning linear attention layers
(reusing vLLM MiniMax kernels) and dense NoPE GQA for minicpm4 layers.
Includes schedule tests and HF parity scaffold; sparse InfLLM-V2 path
is follow-up work.
```

Signed-off-by: Archana Chetan <archana@example.com>
