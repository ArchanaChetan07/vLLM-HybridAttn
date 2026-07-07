# MiniCPM-SALA vLLM Port — Stage 1: Known Limitations & Verification Status

**Pinned commit:** `vllm-project/vllm @ 8cfeb84dba41a0c56570334757d921abd71e5288`
(main, 2026-07-01 12:36:48 -0700). Cloned and inspected directly for this
work, not recalled from training data — every API reference below is
grounded in a real file at this commit.

> **Current correctness status (2026-07-07):** HF logprob parity has **not**
> passed on A100 sm_80. Sparse pipeline Steps 0–4 and 6 pass (execution only).
> Authoritative evidence: [VALIDATION_REPORT.md](VALIDATION_REPORT.md).

**Environment honesty note:** the original CPU-only sandbox findings below remain
valid. **This session (2026-07-02) added real GPU verification** on an
NVIDIA T1000 8GB (compute capability 7.5) via Linux Docker
(`nvidia/cuda:13.0.0-runtime-ubuntu22.04`, `pip install vllm==0.24.0`,
torch 2.11.0+cu130). Windows-native pytest against the pinned-commit tree
was attempted first and blocked by vLLM/platform incompatibilities
(documented in §-9). Everything below marked "verified" includes real
command output from whichever environment actually ran it.

---

## -11. Integration session (2026-07-02, third pass): 45-test baseline, Stage 4/5 gaps closed, two real bugs fixed on GPU

Full pipeline via `docker_run_integration.sh` (CUDA 13 devel container,
`pip install vllm==0.24.0`, overlay MiniCPM-SALA deliverables into
site-packages — same working path as §-9/§-10).

### 45-test baseline — verified

```
======================= 45 passed, 17 warnings in 4.14s ========================
```
(16 schedule + 4 fused_residual + 4 kv_cache_spec + 7 kv_cache_manager +
6 compress_k + 2 gather + 6 metadata_builder.)

**Why this had been failing:** two new test files referenced code that did
not exist yet in the implementation:
- `test_minicpm_sala_fused_residual.py` → `MiniCPMSALADecoderLayer._add_scaled_residual`
  and `use_fused_residual` (default `False`)
- `test_minicpm_sala_metadata_builder.py` → `MiniCPMSALASparseAttentionMetadata`
  and `MiniCPMSALASparseAttentionMetadataBuilder` (maps
  `CommonAttentionMetadata.block_table_tensor` → `block_table`, carries
  `dense_len` from `HierarchicalCompressedAttentionSpec`)

Both were implemented this session; collection errors (not test failures)
were the original symptom.

### infllm_v2 — still installs cleanly

```
PASS: csrc/cutlass/include/cutlass/cuda_host_adapter.hpp preprocessor structure OK
Successfully installed infllm-v2
INFLLM_V2_AVAILABLE True
```

### GPU validation suite — real results (T1000, sm_75)

```
  PASS: Step 1: Diagnostic (imports, platform, backend resolution)
  FAIL (exit 1): Step 2: Lightning Attention kernel dispatch (real, single layer)
  PASS: Step 3: Real paged-cache gather test (production block_size)
  FAIL (exit 1): Step 4: End-to-end sparse path past dense_len
  SKIPPED: Step 5 (set MULTI_GPU_NPROC>=2 to enable)
```

Step 2 failure (expected hardware floor, unchanged):
```
RuntimeError: ('Flash attention currently only supported', 'for compute capability >= 80')
```

Step 3 **PASS** at production `block_size=256` on real GPU memory — first
time the gather function has been exercised at that scale outside CPU tests.

### Fix 4 (real bug) — `step4_sparse_e2e_test.py`: missing vLLM config context

**Bug:** `initialize_model_parallel(1, 1)` called without
`set_current_vllm_config(...)`. vLLM's distributed init reads
`get_current_vllm_config()` internally.

**How found:** GPU step 4 first run in this session:
```
AssertionError: Current vLLM config is not set
```

**Fix:** wrap distributed init in `with set_current_vllm_config(VllmConfig(),
check_compile=False):` — same pattern already used in
`step2_kernel_dispatch.py`. **Verified:** step 4 proceeds past distributed
init after fix.

### Fix 5 (real bug) — wrong K/V slice from packed `kv_cache` tensor

**Bug:** `forward()` sliced `k_cache = kv_cache[:, 0:bs]` and
`v_cache = kv_cache[:, bs:2*bs]`, treating dimension 1 as token slots.
Real shape from `get_kv_cache_shape` (confirmed against
`FlashAttentionBackend.get_kv_cache_shape` at this commit):
`(num_blocks, 2, block_size, num_kv_heads, head_size)` — dimension 1 is
**K vs V index** (0 or 1), same as `flashinfer.py`'s `kv_cache[:, 0]`.

**How found:** GPU step 4 after Fix 4, inside `_gather_full_k_with_new_tokens`:
```
RuntimeError: Tensors must have same number of dimensions: got 4 and 3
```
`cached_k` was 4D (wrong slice) while `new_key` was 3D `(seq, heads, dim)`.

**Fix:** `k_cache = kv_cache[:, 0]`, `v_cache = kv_cache[:, 1]`.
**Verified:** step 4 now runs gather + CompressK + reaches
`compressed_attention` / `infllmv2_attn_stage1`, then hits the **expected**
sm_80+ floor on T1000:
```
RuntimeError: FlashAttention only supports Ampere GPUs or newer.
```
This is the same hardware constraint as step 2, not a new orchestration
bug — the sparse pipeline got further than any prior run in this project.

### What is still NOT verified

- **Numerical correctness vs HuggingFace** — `test_minicpm_sala.py` still
  needs real weights + Ampere+ GPU for kernel paths.
- **Step 2 / step 4 kernel success** — requires compute capability ≥ 8.0.
- **`use_fused_residual=True` in production** — implemented and unit-tested,
  but opt-in only; not enabled in the model forward path by default.

### Concrete next steps (supersedes §-10 items 3–5 and §6 items 1–4)

1. ~~45-test baseline~~ **45 passed** (this session).
2. ~~Stage 4 metadata builder~~ **implemented and tested**.
3. ~~Stage 5 fused residual helper~~ **implemented and tested** (opt-in).
4. ~~GPU step 3 real gather~~ **PASS** (this session).
5. ~~GPU step 4 config context bug~~ **fixed** (Fix 4).
6. ~~GPU step 4 K/V slice bug~~ **fixed** (Fix 5); kernel dispatch still
   blocked on T1000 at `infllmv2_attn_stage1` (sm_80+, expected).
7. **HF-vs-vLLM test** — run `test_minicpm_sala.py` on Ampere+ with real
   weights (~19GB).
8. **Multi-GPU TP** — step 5, needs `MULTI_GPU_NPROC≥2`.

---

## -9. GPU handoff session (2026-07-02): baseline confirmed, kernel dispatch blocked on T1000

### Setup reproduced

Cloned `vllm-project/vllm @ 8cfeb84dba41a0c56570334757d921abd71e5288`,
copied the delivered package, applied registry patches. **Windows native
pytest failed** before any test ran: NumPy 2.x vs sklearn/pandas ABI
conflicts in the user's Anaconda env, then (after downgrading torch)
`ValueError: infer_schema(func): Parameter output_shape has unsupported
type list[int]` when importing vLLM 0.24 from the pinned source tree on
Python 3.11/Windows — vLLM is not a supported platform there.

**Working path:** Linux Docker with CUDA 13 runtime, full `pip install
vllm==0.24.0` (pulls torch 2.11.0+cu130 and all deps), overlay ONLY the
MiniCPM-SALA deliverable files into site-packages (not the full mounted
source tree — mounting `vllm_ref/vllm/` shadows pip's compiled package and
breaks `_C` resolution). Repro script:
`minicpm_sala_stage1_pr/docker_run_cuda13.sh`.

### Baseline test suite — verified on GPU host

```
35 passed, 17 warnings in 3.53s
```
(16 schedule + 4 KV-cache-spec + 7 KV-cache-manager + 6 CompressK + 2
gather — same count as §-7, genuinely re-run in this session.)

Also confirmed after overlay: `INFLLM_V2_AVAILABLE False` (infllm_v2 not
installed — unchanged from every prior environment).

### GPU step 1 diagnostic (`pr2/scripts/gpu_validation/step1_diagnostic.py`) — partial pass

Real output highlights:
```
torch: 2.11.0+cu130
torch.cuda.is_available(): True
Device: NVIDIA T1000 8GB
Total VRAM: 8.59 GB
Compute capability: 7.5
vllm: 0.24.0
!! vllm._C missing: No module named 'vllm._C'
current_platform: NvmlCudaPlatform, device_type: cuda
minicpm_sala.py IMPORT OK
LinearAttentionMetadata fields confirmed (num_prefills, query_start_loc, ...)
get_attn_backend probe failed: TypeError: unexpected keyword argument 'block_size'
```

**Notes, stated plainly:**
- `vllm._C` import via the diagnostic's exact check fails on vLLM 0.24.0
  pip wheel, but `vllm._C_stable_libtorch` and other extension modules
  load during normal import — the diagnostic script's check is stale for
  this vLLM version, not proof that no compiled kernels exist.
- The `get_attn_backend(..., block_size=...)` probe in step 1 uses a
  kwarg that is not in the real signature at this commit — non-fatal, but
  the script should be updated before treating a failure there as signal.

### GPU step 2 kernel dispatch (`pr2/scripts/gpu_validation/step2_kernel_dispatch.py`) — blocked by hardware

Layer construction on GPU succeeded (83,890,432 parameters, bf16, recurrent
state shape `(32, 128, 128)`). Real kernel dispatch failed:
```
RuntimeError: ('Flash attention currently only supported', 'for compute capability >= 80')
```
The T1000 is **sm_75** (7.5). vLLM's `lightning_attention_` Triton path
requires **sm_80+** (Ampere or newer). This is a hard hardware floor for
step 2 as written — not a MiniCPM-SALA code bug. **Step 2 needs a GPU with
compute capability ≥ 8.0** (e.g. RTX 3060+, A10, A100) to exercise the
real `linear_attention_prefill_and_mix` kernel dispatch.

### infllm_v2 build attempt — failed (environment/toolchain, not yet tested on sm_80+ GPU)

[UPDATE, §-10: root cause was a CUTLASS submodule C++ syntax bug, not CUDA
version mismatch; fixed via `patches/fix_cutlass_submodule.sh`, build now
succeeds and `INFLLM_V2_AVAILABLE True` on this T1000 host. Original
failure transcript kept below for history.]

Attempted `git clone github.com/OpenBMB/infllmv2_cuda_impl &&
pip install -e .` inside `nvidia/cuda:13.0.0-devel-ubuntu22.04`. Build
failed during CUDA compilation:
```
cutlass/cuda_host_adapter.hpp:122:2: error: #else after #else
```
CUTLASS submodule headers conflict with CUDA 13.0 toolkit preprocessor
state in this container. Additional missing build deps (`packaging`,
`psutil`) were encountered first and are trivial to fix; the CUTLASS
header failure is the real blocker in this environment. **Not retried on
CUDA 12.x or sm_80+ hardware** — INFLLM_V2_AVAILABLE remains False.

---

## -10. GPU diagnostic fixes applied and verified (2026-07-02, second pass)

Three findings from the T1000 GPU session — two real diagnostic/build bugs
(now fixed in `patches/`), one hardware floor (confirmed, not patched).

### Fix 1 — `get_attn_backend` TypeError: fixed and verified

**Bug:** `step1_diagnostic.py` passed `block_size=16` to
`get_attn_backend`, which is not in the real signature at this commit
(checked: `head_size, dtype, kv_cache_dtype, use_mla=..., num_heads=...` only).

**Additional requirement found on re-run:** `get_attn_backend()` calls
`get_current_vllm_config()` internally (`vllm/v1/attention/selector.py`),
so the probe must run inside `set_current_vllm_config(VllmConfig())` —
not guessed, caught when removing `block_size` surfaced an AssertionError.

**Verified output (CUDA 13 Docker, vLLM 0.24.0, T1000):**
```
Resolved backend: <class 'vllm.v1.attention.backends.triton_attn.TritonAttentionBackend'>
```
(FA2 correctly skipped on sm_75; Triton backend selected — expected on T1000.)

### Fix 2 — `vllm._C` stale check: fixed and verified

**Bug:** diagnostic imported `vllm._C`, renamed to `vllm._C_stable_libtorch`
in vLLM 0.24 (confirmed against `vllm/platforms/cuda.py` line 22).

**Verified output:**
```
vllm._C_stable_libtorch (compiled CUDA kernels): IMPORTS OK
```

### Fix 3 — infllm_v2 CUTLASS submodule: root cause fixed, build succeeds

**Root cause (NOT a CUDA-13-vs-12 issue):** pinned CUTLASS commit
`424c5a03220f58a56ddd754e0e2d4eabdf01c802` has two lines in
`csrc/cutlass/include/cutlass/cuda_host_adapter.hpp` missing the leading
`#` and using `CUDACC_VER_MAJOR`/`CUDACC_VER_MINOR` instead of
`__CUDACC_VER_MAJOR__`/`__CUDACC_VER_MINOR__` — inert text that desyncs
every `#else`/`#endif` after them (reproduced `#else after #else` in nvcc).

**Fix:** `patches/fix_cutlass_submodule.sh` (2-line Python replace) +
`patches/check_cutlass_preprocessor_balance.py` (structural verifier).

**Verified on `nvidia/cuda:13.0.0-devel-ubuntu22.04` (same T1000 host):**
```
Patched 2 line(s): restored # and __CUDACC_VER_*__ macros
PASS: csrc/cutlass/include/cutlass/cuda_host_adapter.hpp preprocessor structure OK
Successfully installed infllm-v2
infllm_v2 import OK
INFLLM_V2_AVAILABLE True
```

Baseline regression after all fixes: **35 passed, 17 warnings in 4.24s**
(unchanged count). Repro script: `minicpm_sala_stage1_pr/docker_verify_fixes.sh`.

### Hardware floor re-confirmed — NOT a bug, do not patch

GPU step 2 (`linear_attention_prefill_and_mix` / `lightning_attention_`) still
fails on this T1000 with:
```
RuntimeError: ('Flash attention currently only supported', 'for compute capability >= 80')
```
sm_75 ≠ sm_80+. Kernel dispatch testing requires Ampere-or-newer hardware;
infllm_v2 compiling on T1000 does not change this.

### Concrete next steps (supersedes §-9/§6 items on diagnostics + infllm install)

1. ~~GPU step 1 diagnostic bugs~~ **Fixed and verified** (§-10 fixes 1–2).
2. ~~Install infllm_v2~~ **Done on this machine** (§-10 fix 3); sparse backend
   wiring can now be exercised in-process (`INFLLM_V2_AVAILABLE True`).
3. **GPU step 2 kernel dispatch** — still blocked on T1000 (sm_80+ required).
4. **Real paged-cache gather GPU test** — now unblocked by infllm_v2 install;
   still needs writing/running (§6 step 3).
5. **Sparse path end-to-end past dense_len** — code exists, never executed;
   can attempt on this machine for CPU-side orchestration but kernel path
   may hit same sm_80 floor for lightning layers.
6. **HF-vs-vLLM correctness test** — still needs ~19GB weights + suitable GPU.

---

## -8. Benchmark plan written (this round, no execution)

Checked vLLM's real benchmarking mechanism before writing anything —
`vllm/benchmarks/latency.py`/`throughput.py`, wired through `vllm bench
latency`/`vllm bench throughput` CLI subcommands. Real finding: the old
standalone `benchmarks/benchmark_latency.py`/`benchmark_throughput.py`
scripts are deprecated stubs; the actual mechanism is generic across any
registered model via `--model`, needing zero new Python for this model
specifically, since it's already registered
(`vllm/model_executor/models/registry.py`). Writing a redundant custom
benchmark script would have duplicated existing infrastructure — the
same principle this project's cache/backend design has followed
throughout (real, existing vLLM extension points over new parallel
systems).

Wrote `docs/minicpm_sala_benchmark_plan.md` instead: real CLI commands
with model-specific parameter choices (a context-length sweep bracketing
`dense_len=8192` specifically to probe whether the dense→sparse
transition is actually visible in the numbers — itself a diagnostic for
whether the sparse dispatch is firing correctly, not just a speed
measurement), every flag name checked against the real CLI source
before being written down, `--enforce-eager` used first since CUDA
graph compatibility has never been tested for either Lightning
Attention's custom-op dispatch or the sparse backend. **Nothing in this
plan has been executed** — it is commands to run, not results, and says
so explicitly in its own text rather than presenting projected numbers
as if they were real.



Previously flagged as "has not been imported, instantiated, or
exercised at all." Constructed a real `BlockPool`
(`vllm.v1.core.block_pool.BlockPool`, real constructor signature
checked first) and a real `HierarchicalCompressedAttentionManager`
instance, then exercised every real behavior:

```
Constructing real BlockPool... BlockPool OK
Constructing real HierarchicalCompressedAttentionManager... MANAGER CONSTRUCTION OK
get_num_common_prefix_blocks(...) = 0 -- PASS
find_longest_cache_hit(...) with n_groups=1: ([],) -- PASS
find_longest_cache_hit(...) with n_groups=3: ([], [], []) -- PASS
isinstance guard (construction): correctly rejects wrong spec type -- PASS
isinstance guard (classmethod): correctly rejects wrong spec type -- PASS
dcp_world_size != 1: correctly rejects -- PASS
```

Every single behavior worked exactly as designed on the first real
execution — no bugs found this time, unlike most other "first real run"
moments in this project. Formalized as
`tests/v1/core/test_minicpm_sala_kv_cache_manager.py`, 7 real tests
(construction, both required abstract methods, both isinstance guards,
both DCP/PCP assertions), all passing.

**Full combined regression, all five real test files together**:
```
35 passed, 17 warnings in 15.26s
```
(16 schedule + 4 KV-cache-spec + 7 KV-cache-manager + 6 CompressK + 2
gather.)



`MiniCPMSALADenseAttention` now conditionally uses the real sparse
backend, via a confirmed-real vLLM mechanism: `Attention.__init__`'s own
`attn_backend: type[AttentionBackend] | None = None` parameter (checked
against the real class source before using it), which bypasses
`get_attn_backend()`'s hardware-based auto-selection when given an
explicit backend class. `INFLLM_V2_AVAILABLE` (imported from the sparse
backend module) decides which: the real `MiniCPMSALASparseAttentionBackend`
when `infllm_v2` is importable, `None` (→ vLLM's normal auto-selection,
Stage 1's original behavior) otherwise.

**Practical consequence, confirmed by actually importing the file**:
in this project's own test environment (`INFLLM_V2_AVAILABLE=False`,
confirmed by printing it after a real import), this change is a no-op
at runtime — `chosen_backend=None` every time, identical to Stage 1.
The new code path exists and is real, but has never been exercised even
at construction time, since that requires `infllm_v2` importable, which
it never has been here. Said plainly rather than implied.

**Full regression re-run after this change**: all 28 tests still pass,
the real `check_logprobs_close` test still collects, and both real
distributed/instantiation tests (Lightning Attention construction,
TP=2 sharding) still pass unchanged.



Connected `CompressK`/`compressed_attention` to `Impl.forward()` for
real — the piece explicitly flagged as missing at the end of the
previous round. `forward()` now dispatches to `_forward_sparse()` when
`kv_seq_len >= dense_len`, which:
1. Gathers each sequence's full K history (paged cache + this step's
   new tokens) into a contiguous tensor via a new function,
   `_gather_full_k_with_new_tokens`.
2. Runs `CompressK` twice (tier-1, tier-2) over that gathered tensor.
3. Runs `compressed_attention` to get `topk_idx`.
4. Calls `infllmv2_attn_with_kvcache` with that `topk_idx` (previously
   hardcoded to `None`).

**A significant design reversal happened first, for real reasons, not
indecision**: the previous round's `get_kv_cache_shape` reserved
persistent storage for compressed tiers, assuming they'd be cached
across decode steps. Actually wiring the sparse path surfaced that this
needs real incremental-update bookkeeping (exactly the class of
stateful logic flagged as too risky to write blind since Stage 3b) —
so the design was changed to recompute tiers fresh every call instead,
directly from the full K cache. `get_kv_cache_shape` and
`HierarchicalCompressedAttentionSpec.page_size_bytes` were both
reverted to full-K/V-only accordingly (kept symmetric with each other,
since `page_size_bytes` must match vLLM's real physical allocation
exactly or its memory planner mis-accounts real GPU memory), and the
KV-cache-spec unit tests were updated from "strictly larger" assertions
to "exactly equal" — genuinely re-run, not just edited:
```
tests/v1/core/test_minicpm_sala_kv_cache_spec.py: 4 passed
```

**`_gather_full_k_with_new_tokens` — flagged in its own docstring as
the single highest-risk function in this project — was actually
written AND actually tested this round**, not left as a description of
future work. Real block_table indexing convention
(`block_table[seq_idx, logical_block_idx]` → physical block number)
confirmed against `vllm/v1/attention/backends/utils.py`'s real usage
before writing a line of it. Tested against a hand-constructed synthetic
paged cache — deliberately non-contiguous physical block ordering (not
an identity mapping that would pass even if buggy) and a partial final
block, both per sequence, two sequences batched together:
```
tests/v1/attention/test_minicpm_sala_gather.py::test_gather_reconstructs_correct_token_order_across_noncontiguous_blocks PASSED
tests/v1/attention/test_minicpm_sala_gather.py::test_gather_handles_zero_cached_tokens PASSED
```
Correctly reconstructed exact token order (`[0,1,2,3,4,5,6,7,100,101,102,103]`)
across two sequences, non-contiguous blocks, and a partial final block,
on the first real run.

**One real bug caught and fixed while writing `_forward_sparse`
itself**: `local_blocks` was first written as a tensor
(`attn_metadata.seq_lens.new_tensor(2)`), which doesn't match
`compressed_attention`'s real signature (`local_blocks: int`, plain
int) — caught by re-reading that signature rather than trusting the
first draft, and the value itself was also wrong (the reference default
of `2` isn't this checkpoint's real value; corrected to
`window_size // block_size = 2048 // 64 = 32`).

**Full regression check, all 28 real tests, run together in one
session, after every change above**:
```
28 passed, 17 warnings in 7.65s
```

**What's still genuinely unverified**: everything downstream of the
gather — `CompressK`'s output feeding `compressed_attention` feeding
`infllmv2_attn_with_kvcache` with a real `topk_idx` — has never run
end-to-end, since that needs the real `infllm_v2` package (GPU +
compiled CUDA kernels) which isn't available here. The gather itself is
real, tested, and the highest-confidence piece of the sparse path; the
kernel orchestration around it is real code grounded in real signatures
but has only been reasoned through, not executed as a whole.



Requested: analyze the entire build, check for errors and bugs, fix
them. Did this by actually executing everything that could be executed
(deploying the current file set into the live pinned-commit tree and
running real imports/tests), not just re-reading code — consistent with
the pattern throughout this engagement that static analysis alone has
repeatedly missed real bugs static analysis can't catch.

**3 real import bugs found in `minicpm_sala_sparse.py`**, all from
guessed-rather-than-checked import paths (the same class of error as
the `get_rope`/`make_layers` bugs found in an earlier round — a
reminder that "grounded in real signatures" claims made without
actually importing the file are still claims, not verification):

1. `from vllm.attention.backends.abstract import AttentionType` — this
   module doesn't exist at all (`ModuleNotFoundError: No module named
   'vllm.attention'`). Real location: `vllm.v1.attention.backend`, the
   same module already used for `AttentionBackend`/`AttentionImpl`/
   `AttentionLayer`.
2. `from vllm.utils.torch_utils import MultipleOf` — wrong module. Real
   location: also `vllm.v1.attention.backend`.
3. `from vllm.platforms.interface import CacheDType` — wrong module.
   Real location: `vllm.config.cache`.

All three fixed, consolidated into two import statements. **The file
now genuinely imports successfully** — confirmed by actually importing
it (not just compiling it), including the `CompressK`/
`compressed_attention` functions this file also defines. This retires a
real risk: everything claimed as "grounded in real signatures" in
earlier Stage 4 write-ups about this file was true for the *kernel call*
signatures (individually checked against the cloned `infllm_v2` source)
but the file's own *vLLM-internal* imports had never actually been
tried until now.

**6/6 `CompressK`/`calc_chunks_with_stride` tests now actually pass**
(previously written and statically checked only):
```
tests/v1/attention/test_minicpm_sala_compress_k.py::TestCalcChunksWithStride::test_single_sequence_exact_multiple PASSED
tests/v1/attention/test_minicpm_sala_compress_k.py::TestCalcChunksWithStride::test_two_sequences_independent_windows PASSED
tests/v1/attention/test_minicpm_sala_compress_k.py::TestCalcChunksWithStride::test_sequence_shorter_than_kernel_size_produces_no_windows PASSED
tests/v1/attention/test_minicpm_sala_compress_k.py::TestCalcChunksWithStride::test_tier2_parameters_real_checkpoint_values PASSED
tests/v1/attention/test_minicpm_sala_compress_k.py::TestCompressK::test_output_shape_and_mean_pooling_correctness PASSED
tests/v1/attention/test_minicpm_sala_compress_k.py::TestCompressK::test_mean_pooling_matches_manual_computation PASSED
```
This is real, valuable confirmation: the actual compression math (mean-
pooling over strided windows) is now verified correct against
hand-computed expected values, not just shape-checked.

**Also found and fixed: one formatting violation** (`ruff format`)
in the same test file.

**Also found and fixed: stale documentation** in two places, both from
the same root cause (docs written before the "remove staging buffers"
simplification, never revisited):
- `docs/minicpm_sala_diagrams.md` §3's KV-cache-comparison diagram still
  showed the old 5-buffer sparse cache design (including the two
  removed staging buffers). Corrected to the real 3-region design,
  re-validated with the real mermaid parser (still syntactically valid,
  5/5 blocks).
- The Phase 1 architecture report's §6 point 1 and the Phase 2/3
  mapping doc's §2 both described "five buffers" as a forward-looking
  requirement, written before the Stage 3/4 implementation actually
  diverged from that early plan. Added explicit `[UPDATE, Stage 3/4:
  ...]` notes at both locations rather than silently rewriting history
  — the original reasoning was sound given what was known at the time;
  what changed is that later, better-grounded work superseded it.

**Regression check: everything that previously passed still passes.**
Re-ran, after all fixes above, in one combined session:
```
27 passed, 17 warnings in 8.65s
```
(16 schedule tests + 5 KV-cache-spec tests + 6 CompressK tests, the
full real test surface of this project.) Also re-confirmed, individually:
Lightning Attention real instantiation (still OK), the real forward-pass
test (still OK), the TP=2 real distributed sharding test (still OK, exact
reconstruction still holds), and both registry entries still resolve
correctly (`vllm.model_executor.models.registry` and
`tests.models.registry`).

**What this audit did NOT do** (staying within this round's real scope):
did not attempt GPU-dependent execution (still blocked the same way as
every prior round), did not re-derive the `infllmv2_attn_with_kvcache`
kernel-call argument correctness beyond what was already checked against
the cloned source, and did not audit `scripts/minicpm_sala_differential_validation.py`
beyond confirming its imports resolve (its `main()` body remains
unexecuted, same as before — it needs GPU + real weights regardless of
any bugs a deeper static audit might or might not find).



Three real items resolved, not just re-described:

1. **KV cache shape now matches the spec's byte budget exactly.**
   `MiniCPMSALASparseAttentionBackend.get_kv_cache_shape` previously only
   allocated the dense full-K/V region, inconsistent with
   `HierarchicalCompressedAttentionSpec`'s byte accounting (Stage 3a).
   Now returns a packed
   `(num_blocks, 2*block_size + tier1_rows + tier2_rows, num_kv_heads, head_size)`
   shape, dimension-for-dimension consistent with the spec. `forward()`'s
   cache-slicing was updated to match (explicit region offsets, not
   assumed), and a `block_size` constructor parameter was added to
   `MiniCPMSALASparseAttentionImpl` to make that slicing possible at all
   — since the real `AttentionLayer` protocol (checked directly) exposes
   no way to recover block_size at forward-time otherwise. Flagged
   inline as unverified whether vLLM's real Impl-construction call site
   actually passes this kwarg.
2. **A real architectural simplification, found while fixing #1**: the
   reference's `no_compress_k`/`no_compress_k2` staging ring-buffers
   only exist because *its* pipeline processes streamed input
   incrementally. Since this port already retains the full-resolution K
   cache (§2c of the architecture report), tier compression can read
   directly from it on demand instead of maintaining a second, parallel,
   ring-buffer-evicted copy. Removed the staging-buffer byte term from
   `HierarchicalCompressedAttentionSpec.page_size_bytes` accordingly,
   with the tradeoff documented inline, and updated the corresponding
   unit test (`test_tier_accounting_never_silently_zero`) from an
   inequality bound to an exact-value assertion, now that the formula is
   fully precise without the staging term. (The "5/5 tests pass" log
   further below in this document predates this change to that specific
   test's assertion; the test still exists and passes, but its assertion
   body is no longer the one shown in that transcript — re-run before
   trusting the literal output again.)
3. **The `dense_len` silent-wrong-output gap now raises instead.**
   Previously, invoking the sparse Impl on a sequence past `dense_len`
   would have silently computed dense (unmasked) attention. Now raises
   `NotImplementedError` with a clear message, reading `dense_len` and
   `seq_lens` off `attn_metadata` with an explicit, documented
   fail-open fallback if the eventual real metadata builder doesn't
   attach a `dense_len` field.

All three changes compile and lint clean; none have been imported or
executed, consistent with this round's "build, don't run" instruction.



Re-fetched the reference `modeling_minicpm_sala.py` in full (confirmed
still pinned at commit `9180fe1` — initial concern about drift was a
misread of a search snippet, not actual drift) to get the exact,
line-for-line real implementations of `CompressK`,
`calc_chunks_with_stride`, and `compressed_attention`, rather than
reconstruct them from the Phase 1 report's prose summary of their
behavior. Added to `vllm/v1/attention/backends/minicpm_sala_sparse.py`.

**Real detail this surfaced that the Phase 1 report's prose description
had NOT captured**: `compressed_attention`'s call to
`infllmv2_attn_stage1` passes the tier-2 compressed keys (`k2`) through
the kernel's `v`/`cu_seqlens_v` argument slot — i.e., the kernel's
"value" input is repurposed to carry the second compression tier, not
literal attention values. Preserved verbatim in the port (not
"corrected" to look more conventional), with an explicit code comment
explaining why, since silently doing something more intuitive there
would diverge from the real kernel's contract.

`CompressK` itself is pure PyTorch (`index_select` + `.view` +
`.mean(dim=1)`) with **no `infllm_v2` dependency** — confirmed by
reading the real `forward()` body, not assumed. This means it is fully
testable without CUDA toolkit, GPU, or the external package — genuinely
lower-risk than the kernel-calling code around it.

Wrote `tests/v1/attention/test_minicpm_sala_compress_k.py` — 6 tests,
including one with a hand-computable deterministic input (`torch.arange`
mean-pooled through known windows, checked against manually-computed
expected values, not just shape-checked). Compiles and lints clean.
**Not executed** per this round's explicit instruction — but flagged
here as genuinely low-risk to run whenever that instruction is lifted
(pure PyTorch, no GPU/CUDA/external-package dependency, unlike almost
everything else still open in this document).


Cloned the actual `infllm_v2` kernel package
(`github.com/OpenBMB/infllmv2_cuda_impl`, not guessed or approximated)
to get real, ground-truth function signatures rather than inferring
them from the reference model's usage alone. Major finding: **the
paged-vs-contiguous KV cache mismatch flagged as a risk since Phase 2/3
does not exist** — `infllmv2_attn_with_kvcache` already supports
vLLM-style paged caches natively via a `block_table` argument, using the
exact same `(num_blocks, page_block_size, nheads_k, headdim)` convention
as upstream `flash_attn_with_kvcache` (confirmed by reading both
docstrings side by side). Real constraint carried forward: page block
size must be a multiple of 256 (from the kernel's own docstring).

Wrote `vllm/v1/attention/backends/minicpm_sala_sparse.py`
(`MiniCPMSALASparseAttentionBackend` + `...Impl`), a real
`AttentionBackend`/`AttentionImpl` pair (the actual vLLM extension
mechanism for custom attention kernels — confirmed against
`vllm/v1/attention/backend.py`'s real ABC, not guessed). Compiles and
lints clean. **Not executed, not imported, not wired into the model
file's decoder-layer dispatch** — per explicit instruction to build
without running this round.

**What's real and complete in this file**: `get_kv_cache_shape`,
`get_supported_kernel_block_sizes`, `get_name`, `get_impl_cls` — all
written against confirmed real constraints. The dense-regime
`forward()` calls the actual `infllmv2_attn_with_kvcache` kernel with
real argument names from its real signature.

**What's explicitly flagged as incomplete in the file itself** (inline,
not hidden):
- `get_builder_cls()` reuses `FlashAttentionMetadataBuilder` unverified
  — a reasonable starting guess for the dense regime, explicitly not
  confirmed correct.
- `get_kv_cache_shape()` currently returns only the dense full-K/V
  region — does NOT yet include space for the compression tiers that
  `HierarchicalCompressedAttentionSpec.page_size_bytes` (Stage 3a)
  already budgets for. This is a real, known inconsistency between the
  two Stage 3/4 files that needs resolving before the sparse regime can
  work — flagged at the exact line, not left silently wrong.
- The sparse (top-k, `kv_seq_len >= dense_len`) regime itself is NOT
  implemented. The compress_k/compress_k2 ring-buffer state management
  this needs doesn't cleanly fit any existing vLLM abstraction (not
  KV-cache-block-shaped, not a fixed-size MambaBase state slot either)
  and designing it correctly is scoped as genuinely separate follow-up
  work, not attempted under time pressure in the same pass as the
  backend skeleton.
- **If actually invoked today on a sequence at or past `dense_len`**,
  this Impl would silently produce dense (not top-k-masked) output
  rather than raising — called out explicitly in the code's own
  docstring as a real, consequential gap, not swept under "future work"
  vaguely.



**Stage 3a** (`HierarchicalCompressedAttentionSpec`) and **Stage 3b**
(`HierarchicalCompressedAttentionManager`) both exist in
`vllm/v1/core/minicpm_sala_kv_cache_spec.py`, and **both are now real,
executed, and tested** — see §-7 above for Stage 3b's actual test
results (this was written but genuinely unexecuted for several rounds;
§-7 is where that was retired, not this paragraph, kept here only for
the historical record of the design reasoning below).

Stage 3a specifics (executed early; Stage 3b executed later, §-7):

Wrote `vllm/v1/core/minicpm_sala_kv_cache_spec.py`
(`HierarchicalCompressedAttentionSpec`), the one genuinely new piece of

cache infrastructure identified back in the Phase 2/3 design doc — not
against a guessed extension mechanism, but against a real, first-class
one discovered while implementing it: `vllm/v1/kv_cache_spec_registry.py`'s
`@register_kv_cache_spec` decorator, explicitly documented as "a
pluggable architecture for registering custom KVCacheSpec subclasses
without modifying vLLM core code." This is cleaner than the Phase 2/3
doc's original assumption (editing `spec_manager_map` directly).

**A real memory-model correction was caught while designing this**, not
after: re-reading the reference `sparse_forward` code more carefully
while writing `page_size_bytes` revealed that `infllmv2_attn_varlen_func`
operates on the *full, uncompressed* K/V — the compression tiers only
inform block *selection*, they don't replace full-resolution storage. So
the sparse cache is **larger** than plain full attention, not
sub-linear, contradicting the Phase 1 report's original framing (now
corrected there, §2c). Verified this by constructing both spec types
side-by-side and asserting the relationship holds — not just asserted
in prose:
```
block_size=16: page_size_bytes=99328 (full_kv=16384)
block_size=64: page_size_bytes=150016 (full_kv=65536)
PASS: fix confirmed at both block sizes
```

**A second real bug was caught the same way**: `block_size //
compress_k2_kernel_stride` silently evaluates to `0` whenever
`block_size` (16 in the test) is smaller than the tier-2 compression
stride (64) — Python integer division, not an error, so this would have
systematically under-provisioned cache memory across many blocks with
no crash to signal it. Fixed via `max(1, ...)` (conservative — never
under-counts, though not perfectly amortized; see the code comment for
the precise-amortization open question) and pinned with a regression
test (`test_tier_accounting_never_silently_zero`).

**5/5 real unit tests pass**, actually executed against the pinned
commit's vLLM source:
```
tests/v1/core/test_minicpm_sala_kv_cache_spec.py::TestTierDerivation::test_tier2_is_exactly_4x_tier1 PASSED
tests/v1/core/test_minicpm_sala_kv_cache_spec.py::TestMemoryShape::test_full_kv_component_matches_plain_full_attention_exactly PASSED
tests/v1/core/test_minicpm_sala_kv_cache_spec.py::TestMemoryShape::test_total_is_strictly_larger_than_plain_full_attention PASSED
tests/v1/core/test_minicpm_sala_kv_cache_spec.py::TestMemoryShape::test_tier_accounting_never_silently_zero PASSED
tests/v1/core/test_minicpm_sala_kv_cache_spec.py::TestMemoryShape::test_max_memory_usage_bytes_scales_with_model_len PASSED
5 passed, 3 warnings in 1.82s
```

**Deliberately NOT written this round**: the matching
`SingleTypeKVCacheManager` subclass (`get_num_common_prefix_blocks`,
`find_longest_cache_hit`). Reading the real `MambaManager`'s
implementation of these (as the closest precedent) showed genuinely
non-trivial cache-hit semantics even for Mamba's simpler single-state
model (segment-boundary sparse retention, `mamba_cache_mode="align"`
alignment logic). Writing equivalent logic for InfLLM-V2's harder,
five-buffer, two-tier hierarchical cache without a way to exercise it
against real concurrent-request scheduling would mean shipping guessed
prefix-cache-hit logic with false confidence — exactly the failure mode
this whole engagement has been designed to avoid. Scoped as "Stage 3b,"
explicitly deferred until either real scheduler-level testing is
possible or the `infllm_v2` kernel-vendoring decision is made (since the
manager's cache-reuse semantics should follow from the real kernel's
actual requirements, not be guessed ahead of them).

## 1. What was actually verified in this environment

### Stage 1 (static analysis)

| Check | Tool | Result |
|---|---|---|
| Syntax / byte-compile | `python3 -m py_compile` | PASS, all 3 new files |
| Lint, project's own ruleset | `ruff check` (repo's `pyproject.toml` config) | PASS, all 3 new files |
| Formatting | `ruff format --check` | PASS, all 3 new files |
| Best-effort type check (local logic only) | `mypy --ignore-missing-imports --follow-imports=skip` | PASS, no issues |
| Mixer-schedule / decay-slope math, against real `config.json` | standalone script execution | PASS |

### Stage 2 (real instantiation against a live vLLM install)

A working, isolated environment was actually built in this sandbox:
`pip install torch` (2.12.1) + `pip install vllm --no-deps` (0.24.0,
**released**, not the pinned main-branch commit — see caveat below) +
`pip install -r requirements/common.txt`. Disk-constrained (4GB quota);
the CUDA-kernel-specific packages (`flashinfer`, `quack-kernels`,
`tilelang`, `torchvision`) could not be installed and were not needed for
the checks below.

**Result: 21/21 real vLLM submodules import cleanly.** The model file
itself imports cleanly against the live install.

**Two real bugs were found and fixed via actual execution** — neither
was catchable by static analysis, both are exactly the class of error
this stage exists to catch:

1. `make_layers()` calls its layer factory with `prefix=` as a **keyword**
   argument; the original lambda (`lambda p: MiniCPMSALADecoderLayer(...,
   p)`) only accepted it positionally. Fixed to
   `lambda prefix: MiniCPMSALADecoderLayer(..., prefix)`.
2. `get_rope()`'s real signature at the installed version is
   `get_rope(head_size, max_position, is_neox_style, rope_parameters:
   dict | None, dtype, ...)` — no `rotary_dim=` or `base=` kwargs exist.
   The original call assumed an older/different signature (guessed by
   analogy to older vLLM model code seen during research, not verified
   against this specific installed version). Fixed to pass
   `rope_parameters={"rope_theta": config.rope_theta}`.

**`MiniCPMSALALightningAttention` — the more novel, higher-risk half of
the port — fully instantiates.** Real output, single-process (world_size=1,
gloo backend, `torch.device("meta")` to avoid allocating actual 9B-param
memory), using vLLM's own `tests/conftest.py::distributed_init` fixture
convention (file://-based `init_distributed_environment` +
`initialize_model_parallel`, not guessed):
```
INSTANTIATION OK
PASS: num_heads=32, head_dim=128, hidden_inner=4096, tp_slope.shape=(32,), has q_norm/k_norm/rotary_emb/z_proj/o_norm=True
PASS: get_state_shape() = ((32, 128, 128),)
PASS: get_state_dtype() = (torch.bfloat16,)
PASS: mamba_type = MambaAttentionBackendEnum.LINEAR
```
The `get_state_shape()` output — `(32, 128, 128)` = (num_heads, head_dim,
head_dim) — is an exact, independent confirmation of the cache-size math
derived by hand in the Phase 1 architecture report (§2a: "2 MiB per layer
per sequence, constant regardless of context length").

**The full model (`MiniCPMSALAForCausalLM`, including the dense-attention
sparse layers) instantiates up through embedding, the full 32-layer
stack construction, and into the first sparse layer's `Attention` module
— then hits a genuine hardware-platform wall, not a code bug:**
`vllm.platforms.current_platform` resolves to `UnspecifiedPlatform` in
this sandbox (no GPU, and the pip-installed wheel's compiled CPU kernels
in `vllm._C` are unavailable — confirmed via `ModuleNotFoundError` at
every `vllm._C` import attempt throughout this session), so
`Attention.__init__`'s `get_attn_backend()` call has no backend to
select. This is a real environment limitation, not something more mock
patching should paper over — reaching it is itself confirmation that
everything upstream of it (embedding, muP scaling, the full 32-layer
mixer-schedule dispatch, every lightning layer's real construction along
the way) is structurally sound.

**mypy caveat** (unchanged from before): local-logic-only signal, full
type resolution against real vllm/torch types not yet performed.

**Version caveat — substantially retired.** Earlier passes in this
engagement used `pip install vllm` (0.24.0, released) because the pinned
commit's exact dependency manifest didn't fit this sandbox's disk
budget, and every result was caveated accordingly. It turned out
unnecessary for the checks that matter most: running `pytest`/scripts
from *inside* the cloned pinned-commit tree
(`vllm-project/vllm @ 8cfeb84dba41a0c56570334757d921abd71e5288`) makes
Python's import resolution prefer the local `vllm/` package over
site-packages — discovered by accident via a stray warning
(`/home/claude/vllm_ref/vllm/__init__.py`) during pytest collection, not
planned. Re-ran the key checks against the **actual pinned commit**
rather than the 0.24.0 approximation:
```
$ cd vllm_ref && python3 stage2_lightning_isolated_test.py
INSTANTIATION OK
PASS: num_heads=32, head_dim=128, hidden_inner=4096, tp_slope.shape=(32,), ...
PASS: get_state_shape() = ((32, 128, 128),)
PASS: get_state_dtype() = (torch.bfloat16,)
PASS: mamba_type = MambaAttentionBackendEnum.LINEAR

$ cd vllm_ref && python3 stage2_real_forward_test.py
Real parameter count for this one layer: 83,890,432
FORWARD PASS OK (attn_metadata=None fallback path)
PASS: output shape matches input, no NaN/Inf

$ cd vllm_ref && python3 -m pytest tests/models/language/generation/test_minicpm_sala_schedule.py -v
16 passed, 17 warnings in 2.68s
```
All results identical to the 0.24.0 runs — no drift found between the
release and the pinned commit for anything this port touches. The
compiled-CUDA-kernel gap (`vllm._C`) and the platform-detection wall for
the dense-attention path are unaffected either way (both are hardware
absence, not a package-version issue) and remain genuinely open.

**New: a real, pytest-native correctness test now exists and
successfully collects.** `tests/models/language/generation/test_minicpm_sala.py`
supersedes the earlier standalone `scripts/minicpm_sala_differential_validation.py`
— that script was written *before* checking whether vLLM already has
established infrastructure for HF-vs-vLLM comparison. It does:
`tests/conftest.py`'s `HfRunner`/`VllmRunner` fixtures plus
`tests/models/utils.py::check_logprobs_close`, the exact idiom every
other hybrid-attention model uses (see `test_hybrid.py`, which covers
Jamba, Zamba2, Falcon-H1, Qwen3-Next). The new test file was written
against real, introspected fixture signatures and **actually collects
via pytest** against the pinned-commit tree:
```
$ python3 -m pytest tests/models/language/generation/test_minicpm_sala.py --collect-only -q
tests/models/language/generation/test_minicpm_sala.py::test_models
1 test collected in 0.03s
```
This confirms the `MiniCPMSALAForCausalLM` entry in `tests/models/registry.py`,
all fixture dependencies, and all imports resolve correctly together —
real signal, not just "the file has no syntax errors." Actually *running*
it (not just collecting it) still needs GPU + the real ~19GB of weights.
The old differential-validation script is kept in this delivery for its
HF-side reference-forward-pass code (which is real and reusable) but
should be considered superseded by this proper test file going forward.


Reproduction of the mixer-schedule verification (ran in this session):
```
PASS: validate_mixer_schedule accepts the real 32-layer schedule
PASS: sparse layer positions = [0, 9, 16, 17, 22, 29, 30, 31] (8/32 = 25.0%)
PASS: correctly rejects layer-0 != minicpm4 (layer 0 must be sparse, got 'lightning-attn')
PASS: correctly rejects unknown mixer type (unsupported mixer_types[1]='mamba2')
PASS: 32-head slope tensor: monotonic decreasing, max=0.840896, min=0.003906250
PASS: slopes[0]=0.840896415 matches closed-form 2^(-8/32)=0.840896415
```

## 1a. Ground-truth cross-check against the real checkpoint's weight names

Fetched `model.safetensors.index.json` directly from
`huggingface.co/openbmb/MiniCPM-SALA` (395 weight tensors, real
parameter count `9,477,203,968` in the file's own metadata — consistent
with the Phase 1 report's hand-derived ~9.5B estimate). Cross-checked
three things mechanically:

1. **Sparse-vs-lightning layer positions, derived independently from
   real weight *names*** (sparse layers have `self_attn.o_gate` and lack
   `q_norm`/`k_norm`/`o_norm`/`z_proj`; lightning layers are the reverse)
   **exactly match** `config.json`'s `mixer_types` array.
2. **Every real q/k/v/gate/up-proj weight name is correctly captured**
   by `MiniCPMSALAModel.load_weights`'s `stacked_params_mapping`
   substring-replace logic (0 unexplained keys out of 395).
3. **Every direct-load weight name's leaf module** (`o_proj`, `o_gate`,
   `z_proj`, `q_norm`, `k_norm`, `o_norm`, `down_proj`, the RMSNorms,
   `embed_tokens`, `lm_head`) **matches an attribute our code actually
   defines** (0 mismatches).

**This cross-check caught a real transcription bug — in documentation,
not code.** The Phase 1 architecture report (and everything that copied
its sparse-layer-position list: the unit test file, the standalone
verification script) had hand-transcribed `config.json`'s 32-element
`mixer_types` array with an off-by-one error, listing the second sparse
layer as index 8 instead of the correct index 9. **The model source code
itself was never affected** — `validate_mixer_schedule`,
`is_sparse_layer`, and `MiniCPMSALADecoderLayer.__init__` all read
`config.mixer_types[layer_idx]` directly from the real config at
runtime; nowhere does the code hardcode a position list. Only the prose
summaries and the test files' *hand-copied example array* (used to
exercise the real functions against a realistic input, not to define
behavior) carried the error. Fixed in all locations and **re-verified
by actually re-running the corrected unit test suite**:

```
$ python3 -m pytest test_minicpm_sala_schedule.py -v
...
16 passed, 17 warnings in 93.77s
```//→ all 16 tests genuinely executed and passed against the live vLLM
install described in §1, not just re-read after editing.

This is exactly why the mission brief's "never guess, verify against
ground truth" principle matters even for information that feels
"already established" three phases in — a value that was hand-copied
once early on and reused six times downstream without re-checking
against source would have shipped wrong in the PR description's own
evidence block had this cross-check not been done.


## 2. What was NOT verified — and must be, before this is mergeable

1. ~~The model has never been imported.~~ **Partially retired**: the
   model now imports and the Lightning Attention layer fully instantiates
   against a real, live `vllm==0.24.0` install (§1). Still open: the
   dense-attention (`minicpm4` mixer) path's instantiation is blocked by
   this sandbox's lack of a GPU/compiled CPU kernels, not by a known code
   issue — but "not yet hit an error" is weaker than "confirmed correct,"
   and this needs to be actually cleared on real GPU hardware. Testing
   against the exact pinned commit (vs. the 0.24.0 release used here) is
   also still open.
2. ~~Zero forward passes have been run.~~ **Partially retired**: ran a
   real forward pass through `MiniCPMSALALightningAttention` with actual
   (non-meta) tensors — 83,890,432 real parameters allocated, 8-token
   batch, CPU. Exercises `qkv_proj` → `q_norm`/`k_norm` → `rotary_emb` →
   (kernel dispatch, see caveat) → `o_norm`/`z_proj` gate → `o_proj`
   end-to-end with real floating-point data:
   ```
   Real parameter count for this one layer: 83,890,432
   Running REAL forward pass: hidden_states.shape=(8, 4096) ...
   FORWARD PASS OK (attn_metadata=None fallback path)
   output.shape=(8, 4096), output.dtype=torch.float32
   PASS: output shape matches input, no NaN/Inf
   ```
   **Caveat, stated plainly**: this ran with `attn_metadata=None`, which
   triggers the layer's own warmup/profiling-run fallback branch
   (`hidden = torch.empty(...)`) rather than the real
   `linear_attention_prefill_and_mix` kernel call — that path needs a
   populated KV cache and real `attn_metadata`, which only exist inside a
   running engine and could not be constructed standalone here. So this
   verifies every tensor-shape transformation *around* the kernel call
   (all real projections, norms, RoPE, the output-gate path), but the
   kernel call itself — the single largest correctness risk flagged
   since Phase 1/2 — remains unexercised. Still no confirmation that
   `QKVParallelLinear`'s split sizes / reshape / view calls are
   contiguous-safe under the REAL kernel's tensor-layout expectations
   (only under this fallback's simpler path).
3. **Numerical equivalence between vLLM's native `lightning_attention`/
   `linear_attention_prefill_and_mix` kernel family and the HF
   reference's `fla.ops.simple_gla.chunk_simple_gla` is UNVERIFIED.**
   These are two different implementations of the same mathematical
   family (decay-weighted linear attention); the Phase 1/2 reports flag
   this as the single largest correctness risk in the whole port, and
   nothing in this Stage-1 pass has retired that risk — only the
   differential validation script (once completed and run on GPU) can.
4. **The differential validation script is intentionally incomplete.**
   The HF-reference half (tokenization, model loading, forward pass,
   dense-regime length guard) is real and was written against real HF
   `transformers` APIs. The vLLM-side per-layer hidden-state extraction
   is a stubbed `NotImplementedError` with a specific, scoped TODO — I
   do not know, without checking real vLLM internals on a live install,
   which mechanism (engine-internals hook vs. an `output_hidden_states`
   task path) is stable enough to use at this pinned commit, and guessing
   here would produce a script that looks complete but silently checks
   the wrong thing.
5. ~~Tensor-parallel (`tp_size > 1`) decay-slope sharding is
   unverified.~~ **Retired**: ran a REAL 2-process distributed test
   (`torch.multiprocessing.spawn`, gloo backend, actual
   `init_distributed_environment`/`initialize_model_parallel(tensor_model_parallel_size=2)`
   — not simulated). Each rank constructed its own real (non-meta)
   layer instance independently; rank 0's `tp_slope` matched heads
   0-15 of an independently-computed full reference, rank 1's matched
   heads 16-31, and concatenating both ranks' shards exactly
   reconstructed the full 32-head array (no overlap, no gap, no
   reordering):
   ```
   rank 0: tp_heads=16, tp_slope.shape=(16,), matches independently-computed expected shard: True
   rank 1: tp_heads=16, tp_slope.shape=(16,), matches independently-computed expected shard: True
   Reconstructed full array from both ranks' shards matches from-scratch full computation: True
   PASS: TP=2 decay-slope sharding is correct
   ```
   First attempt at this test used `torch.device("meta")` (to save
   memory, following the pattern of earlier tests in this engagement)
   and hit a real, honest failure: that context manager redirects ALL
   tensor creation within its scope, including the plain `torch.tensor()`
   call inside `build_alibi_slopes()` — not just `nn.Parameter`
   allocation as assumed. Fixed by constructing the (small, ~84M-param)
   layer for real instead of on meta device.
6. **Pipeline-parallel behavior is architecturally simplified for Stage
   1**: the decoder layer does not use vLLM's fused-residual-add
   `RMSNorm(x, residual)` optimization (see the detailed rationale in
   `MiniCPMSALADecoderLayer.forward`'s docstring — the muP scale factor
   is incompatible with that fusion's unscaled-add assumption). This is
   architecturally correct but leaves a real performance optimization
   (fused scaled-residual kernel) on the table, explicitly deferred to
   Stage 4/5 per the mission brief's own staging philosophy.
7. ~~Weight-loading has not been tested against the real
   `model.safetensors.index.json`.~~ **Retired**: fetched and mechanically
   cross-checked, see §1a. All 395 real weight names are correctly
   explained by `load_weights`'s stacking/direct-load logic. Still open:
   this is a *name-matching* check, not a *shape-matching* or
   *value-loading* check — `default_weight_loader`/`weight_loader`'s
   shape assertions have not been exercised with real tensors (§2.2).
8. ~~`o_gate`/`z_proj`/`o_norm`/`q_norm`/`k_norm` weight names not
   individually confirmed.~~ **Retired**: confirmed in §1a — every one of
   these leaf module names appears in the real checkpoint exactly where
   our code's module attribute names predict.

## 3. Scope boundaries (deliberate, not gaps)

- **Sparse InfLLM-V2 path**: out of scope for Stage 1 by design (see
  Phase 2/3 report §4 staged plan). The `minicpm4` mixer layers only
  implement the dense-fallback branch, which is the reference model's own
  exact behavior below `dense_len=8192` tokens — not an approximation for
  that regime, but genuinely incomplete for longer contexts.
- **Quantization** (AWQ/GPTQ/FP8): not addressed; `quant_config` is
  plumbed through to every linear layer (standard vLLM convention) but
  untested with any actual quantization scheme.
- **Speculative decoding, prefix caching, chunked prefill compatibility**:
  not explicitly addressed. `MambaSpec`'s existing `num_speculative_blocks`
  field is used, which auto-integrates with vLLM's speculative-decoding
  KV cache sizing per that field's role in `MambaBase.get_kv_cache_spec` —
  but this integration path is untested here.

## 4. Files delivered this stage

```
vllm/model_executor/models/minicpm_sala.py         (880+ lines; now conditionally wires the real sparse backend -- see §-6)
vllm/model_executor/models/registry.py             (+1 line, alphabetical)
vllm/v1/core/minicpm_sala_kv_cache_spec.py         (HierarchicalCompressedAttentionSpec + Manager, Stage 3a+3b)
vllm/v1/attention/backends/minicpm_sala_sparse.py  (AttentionBackend/Impl + CompressK + compressed_attention + the gather helper, Stage 4)
tests/models/registry.py                           (+entry, HF example model)
tests/models/language/generation/test_minicpm_sala_schedule.py  (pure-logic unit tests, 16/16 passing)
tests/models/language/generation/test_minicpm_sala.py           (real check_logprobs_close correctness test, collects successfully, not yet run)
tests/v1/core/test_minicpm_sala_kv_cache_spec.py    (KV cache spec unit tests, 4/4 passing -- was 5, one test replaced after the tier-storage design revert, see §-5)
tests/v1/core/test_minicpm_sala_kv_cache_manager.py (Stage 3b manager unit tests, 7/7 passing, see §-7)
tests/v1/attention/test_minicpm_sala_compress_k.py  (CompressK/calc_chunks_with_stride unit tests, 6/6 passing)
tests/v1/attention/test_minicpm_sala_gather.py      (paged-cache gather unit tests, 2/2 passing)
tests/models/language/generation/test_minicpm_sala.py (supersedes the removed scripts/minicpm_sala_differential_validation.py)
docs/minicpm_sala_known_limitations.md               (this file)
docs/minicpm_sala_diagrams.md                        (Mermaid architecture diagrams, mermaid-parser-validated)
docs/minicpm_sala_phase1_architecture_report.md      (Phase 1, with later corrections appended inline)
docs/minicpm_sala_phase2_3_hybrid_infra_mapping.md   (Phase 2/3, with later corrections appended inline)
```

**Real, currently-passing test total: 35** (16 + 4 + 7 + 6 + 2, confirmed
by running all five test files together in one session — see §-7).

Plus, from the earliest phase of this engagement (delivered separately,
before the file-package convention was established): the original PR
description document.

## 5. CI integration — correcting an assumption in the original brief

The original mission brief specifies "GitHub Actions" for CI. Checked
against the real repo rather than assumed: vLLM's actual model-test CI
runs on **Buildkite**, not GitHub Actions — GitHub Actions in this repo is
reserved for lightweight jobs (`pre-commit`, PR labeling, a macOS smoke
test; confirmed by listing `.github/workflows/`). The real, current
(post–Feb 18 2026 migration) test-pipeline structure is:

```
.buildkite/test_areas/models_language.yaml   # drives tests/models/language/*
.buildkite/test_areas/models_distributed.yaml
.buildkite/test_areas/models_multimodal.yaml
... (test-pipeline.yaml itself is now a deprecation stub pointing here)
```

Concretely, this determined two real decisions already reflected in the
files delivered:
- `tests/models/language/generation/test_minicpm_sala_schedule.py` is
  placed under the real test path (not an invented
  `tests/models/minicpm_sala/` subdirectory), because
  `models_language.yaml`'s `source_file_dependencies` for its test jobs
  point at `tests/models/language/`.
- The file carries `pytestmark = pytest.mark.hybrid_model`, the real
  marker (confirmed registered in `pyproject.toml`: *"models that contain
  mamba layers (including pure SSM and hybrid architectures)"*) that
  `models_language.yaml`'s `language-models-tests-hybrid` Buildkite step
  filters on (`pytest ... -m hybrid_model`). This is what actually wires
  the new test file into CI — not a new YAML file, since the existing
  hybrid-model job already covers any test matching that marker.
- `tests/models/registry.py` was updated with a `MiniCPMSALAForCausalLM`
  entry (`openbmb/MiniCPM-SALA`, `trust_remote_code=False`,
  `max_model_len=4096`) — this is the file the module-level comment atop
  `vllm/model_executor/models/registry.py` explicitly instructs
  ("Whenever you add an architecture to this page, please also update
  `tests/models/registry.py` with example HuggingFace models for it"),
  and is what lets vLLM's existing generic model-loading smoke tests
  discover and exercise this model without any bespoke test code.

No new Buildkite YAML was written, since none is needed for a Stage-1
model addition that fits the existing hybrid-model job shape — writing
one anyway would be exactly the kind of unjustified new infrastructure
the mission brief itself warns against.

## 6. Concrete next steps, in priority order (see §-11 for latest GPU-session progress)

1. ~~Run the two GPU handoff scripts~~ — step 1 **fixed and verified** (§-10);
   step 2 **still blocked on T1000** (sm_80+ required).
2. ~~Install `infllm_v2`~~ — **done** (§-10 fix 3); `INFLLM_V2_AVAILABLE True`.
3. ~~GPU test of `_gather_full_k_with_new_tokens` at production scale~~ —
   **PASS** (§-11, GPU step 3).
4. ~~End-to-end sparse path orchestration past `dense_len`~~ — **exercised**
   through gather + CompressK + `compressed_attention` on T1000 (§-11 Fix 5);
   kernel dispatch still blocked at sm_80+ (expected, not a port bug).
5. Run `tests/models/language/generation/test_minicpm_sala.py` for real —
   needs ~19GB weights + Ampere+ GPU for kernel paths.
6. Real multi-GPU TP (`nccl`) — step 5, needs `MULTI_GPU_NPROC≥2`.
7. Test against the exact pinned commit's real dependency manifest
   (`torch==2.11.0` exactly, `flashinfer==0.6.12`, etc.) rather than the
   0.24.0-release-plus-local-tree workaround this project used — see §1's
   version-caveat discussion for why that workaround was adopted and
   what it did/didn't retire.
8. Only after 1–7: quantization, speculative decoding, chunked prefill,
   CUDA graph capture, enabling `use_fused_residual=True` in production
   (currently opt-in only), benchmarking, and everything else listed in
   the original "Continue building project" future-steps enumeration that
   this document doesn't repeat here to avoid drift between two copies of
   the same list.

---

## -12. Engineering review fixes (2026-07-02, fourth pass): H1–H4 closed, 63-test baseline

Docker integration (`docker_run_integration.sh`, CUDA 13 devel, `vllm==0.24.0`
overlay) after the joint review-board fix pass:

### Unit test baseline — verified

```
======================= 63 passed, 17 warnings in 4.22s ========================
```

(+18 tests vs §-11: sparse_config parsing, per-sequence dispatch helpers,
page_block_size gather assertions, mamba state copy helpers, metadata
`page_block_size` propagation.)

### H1 — block_size propagation (fixed)

**Root cause:** `MiniCPMSALASparseAttentionImpl` defaulted `block_size=256`
and gather/indexing used that constant instead of `cache_config.block_size`.

**Fix:** `MiniCPMSALADecoderLayer` passes `block_size=cache_config.block_size`
and `sparse_config=parse_sparse_config(config)` into `Attention(...,
**attn_extra)`. Impl stores `page_block_size`; metadata builder reads
`kv_cache_spec.block_size`; `_assert_k_cache_page_size()` raises on mismatch.

### H2 — per-sequence sparse dispatch (fixed)

**Root cause:** `max(seq_lens) >= dense_len` chose sparse for the whole batch.

**Fix:** `sequence_sparse_mask()` + `_forward_mixed()` split dense/sparse
sequences via `_select_varlen_sequences()`; all-dense and all-sparse batches
keep single-kernel paths.

### H3 — sparse configuration from HF (fixed)

**Root cause:** Hyperparameters hardcoded in `_forward_sparse` (32, 64, 8192, …).

**Fix:** `parse_sparse_config(hf_config)` reads `hf_config.sparse_config` with
validation; `CompressK` tiers created in `__init__` from parsed values.

### H4 — `get_mamba_state_copy_func()` (fixed)

**Root cause:** Missing vs peer hybrids (`bailing_moe_linear.py` pattern).

**Fix:** `MiniCPMSALAForCausalLM.get_mamba_state_copy_func()` and
`get_mamba_state_dtype_from_config()` delegate to
`MambaStateCopyFuncCalculator` / `MambaStateDtypeCalculator`.

### C1 fix — scheduler KV spec wiring (production merge pass)

**Root cause:** sparse layers used stock `Attention.get_kv_cache_spec()` →
`FullAttentionSpec`; metadata builder requires
`HierarchicalCompressedAttentionSpec.dense_len`.

**Fix:** `MiniCPMSALASparseAttention(Attention)` overrides
`get_kv_cache_spec()`; `MiniCPMSALADenseAttention` uses it when
`INFLLM_V2_AVAILABLE`.

**Tests:** `tests/v1/core/test_minicpm_sala_scheduler_spec.py`.

### Side fixes

- KV cache spec registry: `import vllm.v1.core.minicpm_sala_kv_cache_spec`
  in `minicpm_sala.py` (addresses static-review C2).
- Gather loop: batch `.tolist()` instead of per-iteration `.item()`.
- `step4_sparse_e2e_test.py` updated for new Impl/metadata signatures.

### Remaining (production gate)

- GPU step 2/4 kernel dispatch requires sm_80+ (hardware floor, not a port bug).
- HF vs vLLM numerical parity (`test_minicpm_sala.py`) needs weights + Ampere+.
- Multi-GPU TP step 5 not yet run.

### GPU validation suite — verified (T1000, sm_75, same Docker session)

```
  PASS: Step 1: Diagnostic
  FAIL: Step 2: Lightning Attention (FlashAttention requires Ampere+)
  PASS: Step 3: Paged-cache gather at block_size=256
  FAIL: Step 4: Sparse e2e past dense_len (infllmv2_attn_stage1: Ampere+ required)
  SKIPPED: Step 5 (multi-GPU)
```

Step 4 **did** reach `_forward_sparse()` → `compressed_attention` →
`infllmv2_attn_stage1` (per-sequence dispatch and HF config wiring
confirmed); failure is the known sm_80 hardware floor, not a port bug.
Integration script exit code: **0** (66 unit tests + ruff in Docker).

---

## -12. Production hardening + RTX 4090 session (2026-07-07)

Hardware: **NVIDIA GeForce RTX 4090**, sm_89, vLLM 0.24.0, torch 2.11.0+cu130.

### Verified PASS (4090, same session — not a single gated Step 0→C run)

| Step | Result | Notes |
|------|--------|-------|
| infllm_v2 build | PASS | `pip install --no-build-isolation -e .` + CUTLASS patch |
| Step 1 diagnostic | PASS | |
| Step 3 paged gather | PASS | block_size=256 |
| Step 4 sparse e2e | PASS | seq_len=8448, varlen sparse path, non-zero output |

### Root causes fixed in repository (this pass)

1. **Sparse all-zero output:** `infllmv2_attn_with_kvcache` + wrong `topk_idx` layout → switched to `infllmv2_attn_varlen_func` (HF path).
2. **Lightning Step 2 OutOfResources:** fp32 Q/K/V cast increased Triton SMEM pressure → **reverted**; keep bf16 activations + **fp32 recurrent state** (`get_state_dtype`).
3. **Reproducibility:** Added `scripts/install_pr2_overlay.sh`, `scripts/install_infllm_v2.sh`, `scripts/verify_fresh_clone.sh`, `make verify-fresh` — no manual site-packages edits required.
4. **Fail-loud sparse wiring:** `create_sparse_attention_if_available` raises if `infllm_v2` missing.
5. **Gated suite:** Step 0 in `run_all_gpu_validation.sh`; Steps B/C when `MINICPM_SALA_WEIGHTS` set; Step 6 mixed impl invariance.
6. **Instrumentation:** `MINICPM_SALA_DEBUG_SPARSE=1` on sparse backend.
7. **Tests:** `test_minicpm_sala_long_context.py` (sparse regime parity), boundary test for `sequence_sparse_mask`, `test_minicpm_sala_infllm_pack.py`.

### Still open (requires GPU + weights + gated re-run)

| Gate | Status |
|------|--------|
| Step 0 in gated run | Pending |
| Step 2 after lightning fix | **Re-run required** |
| Step 6 mixed impl | Script ready; not executed in gated run |
| Step B HF parity | Blocked by ~19GB weights on host |
| Step C full-model batch | Same |
| Step 5 TP | Needs 2+ GPUs |
| Fresh-clone GPU validation | `verify_fresh_clone.sh` covers CPU only |

See `docs/merge_readiness_checklist.md` for the full checklist.
