# PR Split Plan — MiniCPM-SALA upstream merge

Two independent PRs. **PR1 must merge with zero PR2 files present.**
PR2 depends on PR1 landing first and adds sparse backend + wiring only.

## Repository layout (this deliverable)

```
minicpm_sala_stage1_pr/
├── vllm/model_executor/models/minicpm_sala.py     # PR1 — dense + lightning only
├── tests/models/language/generation/              # PR1 tests (~24)
├── patches/registry.py.patch
├── patches/tests_registry.py.patch
├── docker_run_pr1.sh                              # PR1-only CI gate
│
└── pr2/                                           # entire PR2 overlay (not in PR1 branch)
    ├── vllm/model_executor/models/
    │   ├── minicpm_sala.py                        # PR2 delta: imports sparse wiring
    │   └── minicpm_sala_sparse_wiring.py          # MiniCPMSALASparseAttention + factory
    ├── vllm/v1/attention/backends/minicpm_sala_sparse.py
    ├── vllm/v1/core/minicpm_sala_kv_cache_spec.py
    ├── tests/v1/                                  # PR2 tests (~42)
    └── scripts/gpu_validation/
```

Full-stack local validation: `docker_run_integration.sh` overlays `pr2/` on
top of vLLM 0.24.0. PR1 gate: `docker_run_pr1.sh` overlays **only** PR1 model.

---

## PR 1: MiniCPM-SALA dense model (mergeable independently)

**Upstream files to add/modify:**

```
vllm/model_executor/models/minicpm_sala.py
vllm/model_executor/models/registry.py                (+1 line via patch)
tests/models/registry.py                               (+entry via patch)
tests/models/language/generation/test_minicpm_sala_schedule.py
tests/models/language/generation/test_minicpm_sala_fused_residual.py
tests/models/language/generation/test_minicpm_sala_mamba_helpers.py
tests/models/language/generation/test_minicpm_sala.py
docs/minicpm_sala_phase1_architecture_report.md       (optional)
```

**Scope:**

- Lightning Attention layers (`lightning-attn` mixer) — full implementation
- `minicpm4` mixer layers — **dense causal GQA** via vLLM's standard
  `Attention(..., attn_backend=None)` (NoPE, optional output gate)
- Hybrid schedule helpers, fused residual, Mamba state dtype/shape hooks
- **No** imports of `minicpm_sala_sparse`, `minicpm_sala_kv_cache_spec`, or
  `minicpm_sala_sparse_wiring`
- **No** `MiniCPMSALASparseAttention`, scheduler KV spec wiring, or InfLLM code

**Merge gate:**

```bash
docker_run_pr1.sh
# or: pytest tests/models/language/generation/test_minicpm_sala_*.py
# plus GPU: test_minicpm_sala.py (check_logprobs_close) with real weights
```

**Verified property:** With PR2 files physically absent from the installed
vLLM package, `from vllm.model_executor.models.minicpm_sala import
MiniCPMSALAForCausalLM` succeeds. Run `docker_run_pr1.sh` to confirm.

---

## PR 2: InfLLM-V2 sparse attention backend (follow-up)

**Upstream files to add/modify (on top of merged PR1):**

```
vllm/v1/core/minicpm_sala_kv_cache_spec.py
vllm/v1/attention/backends/minicpm_sala_sparse.py
vllm/model_executor/models/minicpm_sala_sparse_wiring.py
vllm/model_executor/models/minicpm_sala.py              (~15-line delta, see pr2/)
cmake/external_projects/infllm_v2.cmake                     (when ready)
tests/v1/core/test_minicpm_sala_kv_cache_spec.py
tests/v1/core/test_minicpm_sala_kv_cache_manager.py
tests/v1/core/test_minicpm_sala_scheduler_spec.py
tests/v1/attention/test_minicpm_sala_*.py
docs/minicpm_sala_phase2_3_hybrid_infra_mapping.md
docs/minicpm_sala_benchmark_plan.md
scripts/gpu_validation/                                     (maintainer GPU suite)
```

**PR2 delta to `minicpm_sala.py` (exact changes in `pr2/vllm/.../minicpm_sala.py`):**

1. Import `create_sparse_attention_if_available` from
   `minicpm_sala_sparse_wiring`
2. In `MiniCPMSALADenseAttention.__init__`, call the factory first; fall back
   to plain `Attention` when `infllm_v2` is unavailable

**Scope:** Hierarchical compressed KV cache, sparse attention backend,
scheduler integration (`get_kv_cache_spec`), InfLLM-V2 kernels. Graceful
dense fallback when `infllm_v2` missing or compute capability &lt; 8.0.

**Merge gate:** `docker_run_integration.sh` (66 tests + ruff) + Ampere+ GPU
validation (`pr2/scripts/gpu_validation/step4_sparse_e2e_test.py`) + benchmark
past `dense_len`.

---

## Dependency boundary (no hidden coupling)

| Component | PR1 | PR2 |
|-----------|-----|-----|
| `minicpm_sala.py` (root) | Self-contained | Replaced/ patched by PR2 copy |
| `minicpm_sala_sparse.py` | **Absent** | Added |
| `minicpm_sala_kv_cache_spec.py` | **Absent** | Added |
| `minicpm_sala_sparse_wiring.py` | **Absent** | Added |
| PR1 tests (`tests/models/...`) | Yes | Unchanged |
| PR2 tests (`pr2/tests/v1/`) | **Not run** | Yes |

PR1 → PR2: **one-way**. PR2 imports PR1 symbols (`MiniCPMSALADenseAttention`
structure, layer schedule). PR1 never imports PR2.

---

## Verification commands

```bash
# PR1-only (no PR2 files on disk in upstream branch)
bash docker_run_pr1.sh

# Full stack (this monorepo)
bash docker_run_integration.sh
```

Last updated: 2026-07-03 — PR1 independence refactor complete.
