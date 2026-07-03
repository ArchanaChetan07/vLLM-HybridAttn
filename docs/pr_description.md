# [Model] Add MiniCPM-SALA support

**Pinned base commit:** `vllm-project/vllm @ 8cfeb84dba41a0c56570334757d921abd71e5288`

**Read `docs/minicpm_sala_known_limitations.md` first.** It is the living,
authoritative record of what's actually been verified vs. what's real
code with unverified behavior — this file is a snapshot summary and
will drift out of date faster than that one does, since the limitations
doc gets updated every round and this file does not.

## What this is

Adds `MiniCPMSALAForCausalLM`, OpenBMB's hybrid sparse/linear-attention
9B model, to vLLM. The model interleaves two structurally different
attention mechanisms across its 32 layers:

- **Lightning Attention** (24/32 layers, gated linear attention) —
  fully implemented, real, and the most-tested part of this
  project. Reuses vLLM's existing linear-attention kernel dispatch and
  `MambaSpec`-family KV cache infrastructure (the same infrastructure
  the in-tree `BailingMoELinearAttention` model uses). Real
  instantiation, a real forward pass, and a real 2-process distributed
  TP-sharding test all pass (see the limitations doc §1).
- **InfLLM-V2 sparse attention** (8/32 layers) — has TWO real code
  paths now, not one: a dense fallback (vLLM's generic `Attention`
  class, used when the real `infllm_v2` kernel package isn't
  installed) and a real sparse top-k path (a custom `AttentionBackend`
  calling the actual `infllm_v2` kernels, used when it is). Which one
  runs is decided automatically at model-construction time. The sparse
  path's orchestration is real, grounded in the actual cloned
  `infllm_v2` source — but has never run end-to-end (needs GPU +
  compiled kernels neither available in the environment this was
  built in).

## Current status, honestly, in one paragraph

Everything checkable without a GPU has been checked: static analysis
(compile/lint/format) across every file, real imports against a live
vLLM install (a pip release, the actual pinned commit, and — as of this
update — an independent Docker install on completely different real
hardware), 35 real unit/integration tests passing together (reconfirmed
identically on that separate machine), several real bugs found and
fixed by actually running things (not just reading them), a real
2-process distributed test, and a real hand-verified test of the
highest-risk new function (paged-cache gathering). **A first real GPU
report has now come in** (T1000, 8GB, Turing/sm_75): construction and
import-level checks pass on real hardware, and found 2 more real bugs
in the diagnostic scripts themselves (now fixed). It also surfaced a
genuine, previously-unknown hardware requirement: Lightning Attention's
kernel path needs **sm_80+ (Ampere or newer)** — a T1000 cannot exercise
real kernel dispatch regardless of software fixes. See
`docs/minicpm_sala_known_limitations.md` §-9 for full detail. Still
unchecked: anything needing Ampere+ GPU (the real kernel calls, TP under
`nccl`, PP, CUDA graphs) or the actual ~19GB of model weights
(numerical correctness, `check_logprobs_close` against HF).

## Files in this PR

See `docs/minicpm_sala_known_limitations.md` §4 for the current,
maintained list — not repeated here to avoid two copies of the same
list drifting apart, which already happened once to this file's
earlier version.

## Verification performed (see limitations doc for full detail and reproduction commands)

- Static analysis: 100% clean (compile, ruff lint, ruff format) across
  every file, re-verified after every round of changes, most recently
  after this round's backend-wiring change.
- Real execution: 35/35 tests passing, run together in one session,
  most recently including a hand-verified test of paged-cache gather
  logic against a deliberately non-trivial (non-contiguous blocks,
  partial final block) synthetic cache.
- Real bugs found and fixed by actually running code, not just
  reading it: 3 wrong import paths, 1 `make_layers` calling-convention
  mismatch, 1 `get_rope` signature mismatch, 1 wrong-type argument
  (`local_blocks` passed as a tensor when a plain int was required), 1
  silent-zero arithmetic bug in cache-memory accounting, and 1
  documentation transcription error (traced back through the report
  history and fixed everywhere it had propagated, not just at the
  source).
- Two real design reversals, made honestly when the reasoning changed:
  the sparse cache's memory model went from "sub-linear" (Phase 1's
  original framing) to "larger than plain full attention" (the
  corrected, verified framing) to "compression tiers aren't persisted
  in cache memory at all, recomputed fresh each call" (the final
  design, after actually trying to wire persistent tier storage and
  finding it needed guessed stateful logic this project explicitly
  avoids).

## What this PR does NOT prove

The model has never produced a single real logit. Every "passing test"
in this project checks shapes, control flow, index arithmetic, or
memory accounting — none of them check that the *numbers* coming out of
this implementation match the reference model's numbers. That check
(`tests/models/language/generation/test_minicpm_sala.py`, written and
collecting successfully) needs real GPU hardware and the real weights,
neither of which have been available in the environment this PR was
developed in. Treat every claim in this repository as "structurally
sound, numerically unverified" until that test has actually run.

## Architecture & design documents (for reviewer context)

- `docs/minicpm_sala_phase1_architecture_report.md` — layer-by-layer
  math, KV cache structure, parameter count, derived from the real
  `config.json`/`modeling_minicpm_sala.py`. Contains an inline
  correction (§2c) to its own original memory-model claim.
- `docs/minicpm_sala_phase2_3_hybrid_infra_mapping.md` — maps this
  model onto vLLM's existing Hybrid KV Cache Manager. Contains an
  inline correction describing how the actual Stage 3/4 implementation
  diverged from this document's original speculation, and why.
- `docs/minicpm_sala_diagrams.md` — Mermaid architecture diagrams,
  actually validated against mermaid's real parser (not just assumed
  to render), updated when the underlying design changed rather than
  left showing a stale cache layout.
