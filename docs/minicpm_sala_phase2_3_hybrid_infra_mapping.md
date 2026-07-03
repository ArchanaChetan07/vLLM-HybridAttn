# MiniCPM-SALA → vLLM: Phase 2/3 — Mapping onto vLLM's Real Hybrid Infrastructure

**Correction to the brief's Phase 3 ask.** The mission requests a new `BaseAttentionBackend` /
`LinearAttentionBackend` / `HybridCache` framework be *designed from scratch*. Before writing a
line of that, I checked whether it already exists — it does, it's called the **Hybrid KV Cache
Manager (HMA)**, and three production model families already run on it. Inventing a parallel
system here would fail the brief's own stated principles ("never copy blindly" cuts both ways —
it also forbids *reinventing* blindly; "prefer reusable infrastructure over one-off solutions").
This document is the corrected Phase 2/3: map MiniCPM-SALA onto what's real, identify the actual
gap, and scope only the genuinely new pieces.

---

## 1. What already exists (verified against real vLLM source, not memory)

### 1a. `KVCacheSpec` hierarchy (`vllm/v1/kv_cache_interface.py`)
A base class with subclasses already in production:
```
KVCacheSpec (ABC)
├── FullAttentionSpec        (num_kv_heads, head_size, dtype, sliding_window)
├── SlidingWindowSpec
├── ChunkedLocalAttentionSpec
├── MLAAttentionSpec / SlidingWindowMLASpec
├── CrossAttentionSpec       (encoder-decoder)
├── SinkFullAttentionSpec / TQFullAttentionSpec
└── MambaSpec                (shapes: tuple[tuple[int,...],...], dtypes, mamba_type,
                               mamba_cache_mode, page_size_bytes = Σ prod(shape)×dtype_size)
```
Each spec is a declarative description of "what does one layer's cache look like," not an
implementation — dispatch to behavior happens via:
```
spec_manager_map = {
    FullAttentionSpec: FullAttentionManager,
    MLAAttentionSpec:  FullAttentionManager,
    SlidingWindowSpec: SlidingWindowManager,
    ChunkedLocalAttentionSpec: ChunkedLocalAttentionManager,
    MambaSpec: MambaManager,
    CrossAttentionSpec: CrossAttentionManager,
}
```
This **is** the `BaseAttentionBackend` framework the brief asks for — it's called
`SingleTypeKVCacheManager` and it's already an ABC with the allocate/evict/checkpoint lifecycle
the brief specifies (`get_num_blocks_to_allocate`, `save_new_computed_blocks`,
`remove_skipped_blocks`, `find_longest_cache_hit`, ...).

### 1b. `KVCacheGroupSpec` — the actual answer to "heterogeneous cache in one model"
Layers are partitioned into groups that share a block table and physical page size. For models with only one type of attention there is one group containing all layers; for models with multiple attention types there are multiple groups, each independently sized and allocated from a shared physical block pool. This *is* the "CacheManager" the brief asks for.

The hard engineering problem — different layer types wanting different physical page sizes but sharing one block pool — is already solved: vLLM automatically tunes the "logical" block size of the full-attention layers so that full-attention and linear-attention layer state occupy the same amount of physical GPU memory per block, which is exactly my Phase 1 §6.1 concern (heterogeneous page sizes), already answered upstream.

### 1c. Per-layer mixer-type dispatch — Qwen3.5 is a near-exact structural precedent
Qwen3.5 ships `"layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention", ...]` — a hybrid architecture with both full attention (GQA) and linear attention (GDN/Mamba-like) layers, where the linear attention layers produce a KV cache spec that is not an AttentionSpec but a recurrent state spec. This is *structurally isomorphic* to MiniCPM-SALA's `mixer_types` array (§Phase-1 report) — a per-layer string tag read from config, used both to pick the attention module and to pick the KV cache spec class. Qwen3-Next's decoder layer constructor literally takes `layer_type="full_attention"` as a parameter. The "LayerScheduler" the brief asks me to design is, concretely, this pattern — I should follow it, not invent a new one.

### 1d. Linear attention kernels are already vendored
vLLM ships `vllm/model_executor/layers/fla/` (Flash Linear Attention ops) for Qwen3-Next's Gated DeltaNet. MiniCPM-SALA's Lightning Attention uses the *same upstream `fla` library* (`fla.ops.simple_gla.chunk_simple_gla` / `fused_recurrent_simple_gla`) — different recurrence (simple GLA vs. gated delta net) but the same tensor-layout conventions. This is a real reuse opportunity for kernel work (Stage 4), not a place to write CUDA from scratch.

## 2. The gap — what MiniCPM-SALA actually needs that doesn't exist yet

**Lightning Attention → mostly already covered by `MambaSpec`.** Its recurrent state is
`(batch, num_heads=32, head_dim=128, head_dim=128)` fp32, constant size — this is exactly the
shape/dtype-tuple model `MambaSpec` already represents (`shapes: tuple[tuple[int,...],...]`,
`dtypes: tuple[torch.dtype]`). Plausibly needs zero new cache-spec code, only a `mamba_type` value
(or equivalent) wired to the right manager, plus a model-side module that calls `chunk_simple_gla`
the way Qwen3-Next's decoder layer calls its own GDN kernel. **This is the single most valuable
finding of this phase**: the "linear attention" half of the port may require no new cache
infrastructure at all.

**InfLLM-V2 sparse layers → genuinely novel, needs a new `KVCacheSpec` subclass.** Nothing in the
existing hierarchy models "five buffers per layer with two-tier hierarchical incremental
compression and sub-linear-but-growing size" (Phase 1 §2b/§4). This needs a new spec, tentatively
`HierarchicalCompressedAttentionSpec`, and a matching `SingleTypeKVCacheManager` subclass. This is
the one piece of real, justified new abstraction — not a whole framework, one new spec class plus
one new manager, added to `spec_manager_map` exactly like the six that already exist.

[UPDATE, Stage 3/4: the actual implementation diverged from this early speculation in two ways,
both corrections made against real evidence, not further guessing. (1) The cache is NOT sub-linear
— re-reading the reference kernel call while implementing `page_size_bytes` showed it operates on
the full, uncompressed K/V; the compression tiers are additive overhead for cheap block
*selection*, not a smaller replacement (see Phase 1 report §2c). (2) The "five buffers" collapsed
to three regions — the two ring-buffer staging buffers turned out to be unnecessary for a from-scratch
port that already retains the full K cache (they exist in the reference only because *its* pipeline
processes streamed input incrementally). And the registration mechanism was a real, better one
than assumed here: `@register_kv_cache_spec` from `vllm/v1/kv_cache_spec_registry.py`, not a
`spec_manager_map` dict edit. See `docs/minicpm_sala_known_limitations.md` for the full detail.]

## 3. Known landmines — real, currently-open bugs in exactly this code path

I pulled current GitHub issues rather than assuming the hybrid path is bug-free:

- **#38041** — vLLM's newer "V2 model runner" crashes on Qwen3.5's mixed linear+full attention: `_reshape_kv_cache` asserts `isinstance(kv_cache_spec, AttentionSpec)`, which is false for recurrent-state specs, and `unify_kv_cache_spec_page_size` raises `NotImplementedError` when linear-attention layers (`block_size=None`) mix with full-attention layers outside the hybrid-grouping path. **Implication**: I must target the V1 (stable) model runner path for the hybrid grouping, and explicitly test against the V2 runner rather than assume it's a drop-in — this is presently a known-broken combination upstream, not a hypothetical edge case.
- **#38643** — Qwen3.5's FLA layer produces silent gibberish output (not a crash) from a `head_first` tensor-layout mismatch where inputs are passed in head-first format [B,H,T,...] when head_first=False was specified. This is directly relevant: MiniCPM-SALA's reference code explicitly calls `chunk_simple_gla(..., head_first=False)` (Phase 1 §2a). If vLLM's own FLA call-site convention doesn't match, the port will *run without error and produce silently wrong tokens* — exactly the failure mode Phase 9 (numerical validation) exists to catch, but cheaper to design around now than discover after a failed logit diff.

## 4. Revised staged plan (supersedes the generic 6-stage plan in the brief with a concrete one)

| Stage | Scope | Cache infra needed |
|---|---|---|
| **1. Correctness, dense-only** | All 32 layers real math; sparse (`minicpm4`) layers run their own dense-FlashAttention branch (`kv_seq_len < dense_len=8192`, already the reference model's own fallback) instead of InfLLM-V2 sparse kernels | `FullAttentionSpec` (sparse layers, dense branch) + `MambaSpec`-family (Lightning layers) — **zero new infra** |
| **2. Numerical validation** | Layer-by-layer diff vs. HF reference, bf16 tolerance, both mixer types independently | none (test harness only) |
| **3. Long-context correctness** | Implement `HierarchicalCompressedAttentionSpec` + manager for the true InfLLM-V2 sparse path above `dense_len` | **one new spec + one new manager** (the real new-infra deliverable) |
| **4. Kernel optimization** | Vendor or adapt `infllm_v2`'s Triton kernels; verify FLA `head_first` convention against landmine #38643 | kernel work, not cache-architecture work |
| **5–6. Perf + hardening** | As per brief | — |

This directly answers my own open question from the prior turn: **dense-only-first is not just
easier, it's the only version of Stage 1 that requires zero new cache infrastructure**, since it
reuses `FullAttentionSpec` + the existing Mamba-family manager untouched. The sparse cache spec
(the genuinely hard, novel part) becomes an isolated, independently-testable Stage 3 deliverable
instead of a blocking dependency for getting *anything* running and diff-tested against HF.

## 5. What I need from you to proceed into real Stage-1 code

Two things I can't responsibly guess:
1. **Which vLLM version/branch to target** — the V1-vs-V2 model runner split in landmine #38041 means the hybrid-grouping code path differs by version, and I'd rather pin this than write against a moving target.
2. **Whether you want me to actually write and run this** (I have a Linux sandbox with no GPU — I can write, lint, and structurally test the model file and cache-spec code, but I cannot run CUDA kernels or do the bf16-tolerance logit diff against real HF weights here; that step needs your GPU environment). If you want the Stage-1 code now, I'll write it as a real PR-shaped diff against a pinned vLLM commit and hand you a test script to run the HF-comparison locally, being explicit in the PR description about what I could/couldn't verify in this environment.
