# MiniCPM-SALA → vLLM Port: Phase 1 Architecture Report

**Source of truth:** `openbmb/MiniCPM-SALA` commit `9180fe1` — `config.json` and `modeling_minicpm_sala.py` fetched directly, not inferred from the model card. Every claim below is either a direct quote of config values or a derivation from the actual `forward()` code.

---

## 1. Config ground truth

```
hidden_size            = 4096
num_hidden_layers      = 32
num_attention_heads    = 32        head_dim = 128   (hidden_size / num_heads = 128, consistent)
num_key_value_heads    = 2         (sparse/InfLLM-v2 layers: GQA ratio 16:1)
intermediate_size      = 16384
vocab_size             = 73448
max_position_embeddings= 524288
rope_theta             = 10000.0
attn_use_rope          = FALSE     <-- sparse ("minicpm4") layers get NO RoPE by default
rms_norm_eps           = 1e-6
tie_word_embeddings    = FALSE     <-- separate lm_head, NOT tied
qk_norm                = true
use_output_gate         = true (mlp-parallel gate)     attn_use_output_gate = true
use_output_norm         = true (Lightning layers only)

# MiniCPM "mup"-style scaling (inherited from MiniCPM4 line, NOT optional)
scale_emb          = 12      → embeddings multiplied by 12 at input
scale_depth        = 1.4     → residual branches scaled by scale_depth/sqrt(num_hidden_layers)
dim_model_base     = 256     → logits divided by (hidden_size/dim_model_base) = 16 before lm_head... 
                                 actually hidden_states is DIVIDED by (hidden_size/dim_model_base)=16
                                 before the lm_head matmul (see §5)
mup_denominator     = 32     (present in config; not referenced in visible forward path —
                                 flag for Phase 2 grep against configuration_minicpm_sala.py)

# Lightning Attention (linear-attention layers)
lightning_nh          = 32
lightning_nkv         = 32   (NO GQA on lightning layers — full MHA head count for K/V)
lightning_head_dim    = 128
lightning_scale       = "1/sqrt(d)"
lightning_use_rope    = TRUE  <-- opposite of the sparse layers

# Sparse (InfLLM-V2) config block
sparse_config = {
  kernel_size: 32, kernel_stride: 16,     # compress_k: mean-pool every 32 tokens, stride 16 (50% overlap)
  init_blocks: 1, block_size: 64,
  window_size: 2048,                       # local dense window = 2048/64 = 32 local blocks
  topk: 64,                                # + local_blocks → 64+32=96 effective blocks selected
  use_nope: false,                         # dense/full path also has no separate NoPE toggle set
  dense_len: 8192                          # below this context length, sparse layers run DENSE flash-attn
}

mixer_types (32 entries, exact layer-by-layer schedule — this is NOT 25/75 uniformly interleaved,
it's front-loaded with sparse layers at specific positions):
  L0:  minicpm4 (sparse)      L11: lightning
  L1:  lightning              L12: lightning
  L2:  lightning              L13: lightning
  L3:  lightning              L14: lightning
  L4:  lightning              L15: lightning
  L5:  lightning              L16: minicpm4 (sparse)
  L6:  lightning              L17: minicpm4 (sparse)
  L7:  lightning              L18: lightning
  L8:  lightning              L19: lightning
  L9:  minicpm4 (sparse)      L20: lightning
  L10: lightning              L21: lightning
                               L22: minicpm4 (sparse)
                               L23: lightning
                               L24: lightning
                               L25: lightning
                               L26: lightning
                               L27: lightning
                               L28: lightning
                               L29: minicpm4 (sparse)
                               L30: minicpm4 (sparse)
                               L31: minicpm4 (sparse)

Sparse-layer positions: {0, 9, 16, 17, 22, 29, 30, 31} → 8/32 = 25% exactly, matches "25% InfLLM-V2". CORRECTION (caught during Stage 2 real-weight cross-check, see docs/minicpm_sala_known_limitations.md): the original count in this report incorrectly listed the second sparse layer as index 8 instead of 9 -- a hand-transcription error, re-verified by careful re-count of the raw config.json array AND independently cross-checked against the real model.safetensors.index.json weight-name patterns (sparse layers have self_attn.o_gate and lack q_norm/k_norm/o_norm/z_proj). The model CODE was never affected since it reads config.mixer_types directly at runtime -- only this prose summary and downstream test/doc transcriptions were wrong.
Note the clustering: layers 29-31 are three consecutive sparse layers at the very end, and layer 0 is
always sparse (the code enforces this — see §4, cache invariant). This clustering has direct
consequences for pipeline-parallel partitioning (§8) and for scheduling granularity in vLLM
(you cannot treat "sparse layer" as uniformly distributed for load-balancing heuristics).
```

## 2. Two structurally different attention mechanisms — exact math

### 2a. Lightning Attention (75% of layers) — this is **gated linear attention**, not "attention" in the vLLM PagedAttention sense

From `LightningAttention.forward` + `attn_fn`, the actual computation is `chunk_simple_gla` / `fused_recurrent_simple_gla` from the `fla` (flash-linear-attention) library — i.e. **scalar-decay Gated Linear Attention (GLA)**, the same family as RetNet / Mamba's "simple GLA" variant, *not* softmax attention at all.

Per-head decay (fixed, non-learned, computed once):
```
slopes[h] = ALiBi-style geometric slopes(num_attention_heads)     # _build_slope_tensor()
decay[h]  = -slopes[h]                                            # negative log-decay per head
```
This is literally the ALiBi slope construction repurposed as a **per-head exponential decay rate** — there is no learned gate/forget parameter, decay is static and head-indexed. This matters for the vLLM port: decay is a compile-time-constant tensor of shape `(num_heads,)`, not part of the KV cache, not data-dependent.

State-space recurrence (the actual "attention" math, chunked linear form):
```
For chunk c, head h:
  S_h ← decay_h · S_h + K_c^T V_c                      # (d_k × d_v) recurrent state, decayed each step
  O_c = Q_c S_h + (intra-chunk causal linear term)      # chunk-parallel form via chunk_simple_gla
scale = head_dim^(-0.5)   ("1/sqrt(d)", from lightning_scale)
```
- `q, k` are **RMSNorm'd** before use (`qk_norm=true`, separate `q_norm`/`k_norm`, shape `(head_dim,)`).
- **RoPE IS applied** here (`lightning_use_rope=true`) — rotary is applied to q/k *before* they enter the GLA recurrence, meaning position information is injected via rotation of the linear-attention keys/queries, not via the decay term itself.
- KV heads = 32 = Q heads → **no GQA on Lightning layers** (`lightning_nkv=32`), unlike the sparse layers.
- Output path: `o = o_norm(o)` (RMSNorm on `(h·d)`-flattened output, gated separately from decay) then `o = o * sigmoid(z_proj(hidden_states))` (a **second**, independent output gate on top of the norm) then `o_proj`.
- **State size per layer**: `(batch, num_heads=32, head_dim=128, head_dim=128)` in fp32 → `32×128×128×4 bytes = 2 MiB` per sequence per lightning layer, **constant regardless of sequence length**. This is the entire point of linear attention and the key departure from PagedAttention's O(seq_len) cache growth.

### 2b. InfLLM-V2 Sparse Attention (25% of layers, the `minicpm4` mixer type)

Two regimes selected by `kv_seq_len < dense_len (8192)`:

**Dense regime (seq < 8192):** plain causal FlashAttention-2, GQA 16:1, **no RoPE** (`attn_use_rope=False`, confirmed at config level — this is the "NoPE" the model card alludes to).

**Sparse regime (seq ≥ 8192):** three-tier key compression + top-k block selection:
1. **`CompressK` (kernel_size=32, stride=16):** mean-pool every 32 contiguous keys (50%-overlapping windows, stride 16) → `compressed_k`. A second, coarser compressor (`compress_k2`, kernel_size=128, stride=64) produces `compressed_k2` — this is a **two-level hierarchical compression**, not documented in the model card at all; only visible in code.
2. **`compressed_attention`**: computes attention scores between (possibly no-RoPE) queries and the compressed K tiers via `infllmv2_attn_stage1` (a custom Triton/CUDA kernel from the `infllm_v2` package — **external dependency, not in vLLM's kernel set**), then max-pools scores into per-block importance (`block_size=64`) via `max_pooling_1d_varlen`.
3. **Top-k block selection**: `topk = config topk (64) + local_blocks (window_size/block_size = 32) = 96` blocks selected per query, sorted, with an always-included local window (`init_blocks=1` + local causal window). `topk_idx` is masked to `-1` for any block index exceeding the query's own position (causal-safe top-k).
4. **`infllmv2_attn_varlen_func`**: the actual sparse FlashAttention pass restricted to only the top-k selected KV blocks, given `topk_idx`. Requires a **16:1 Q:K head ratio internally** (there's an explicit `repeat_interleave` if the natural GQA ratio is lower — here it's exactly 16:1 already, `32/2=16`, so this is a no-op in this specific checkpoint but the code path exists generically).

**KV cache for sparse layers is NOT a simple K/V pair — it's five parallel buffers per layer:**
```
compress_k_cache        # tier-1 compressed keys, varlen list-of-tensors per batch item
no_compress_k_cache      # tier-1 pending (not-yet-32-tokens) raw keys, ring-buffer-like eviction
compress_k2_cache        # tier-2 compressed keys (coarser)
no_compress_k2_cache     # tier-2 pending raw keys
no_rope_keys              # full-resolution K without RoPE applied (used_nope path only; here use_nope=False
                             at sparse_config level, so this buffer is unused for this checkpoint — but the
                             code and cache class always allocate it)
```
Every decode step does **incremental mean-pool compression**: when `no_compress_k_cache` accumulates ≥32 tokens, it emits one new compressed-k row and slides the window by `kernel_stride=16` (i.e. it keeps the last 16 raw tokens as overlap for the next compression window). This is a genuinely stateful, non-trivial cache update — **not expressible as append-only PagedAttention block writes**.

## 2c. CORRECTION (caught during Stage 3 design work): the sparse cache is NOT sub-linear in memory

Section 2b above frames InfLLM-V2's cache as growing "sub-linearly via compression tiers." Re-reading the reference `sparse_forward` more carefully while designing the actual `KVCacheSpec` for it: `infllmv2_attn_varlen_func` is called on the **full, uncompressed** `key_layer`/`value_layer` (see the reference `MiniCPMInfLLMv2Attention.sparse_forward`, the actual attention call) — the compression tiers (`compress_k`, `compress_k2`, and their `no_compress_*` staging buffers) are used **only** to compute `topk_idx`, i.e., which blocks the real attention should look at. They do not replace the full-resolution K/V storage.

So the real per-layer memory shape for a sparse layer is:

```
full K/V cache:        O(seq_len) — same order as plain FullAttentionSpec
+ compress_k tier:      O(seq_len / kernel_stride)      = O(seq_len / 16)
+ compress_k2 tier:     O(seq_len / (kernel_stride*4))  = O(seq_len / 64)
+ no_compress_k buffer: O(kernel_size) = O(32), bounded, does not grow
+ no_compress_k2 buffer:O(kernel_size*4) = O(128), bounded, does not grow
```

All five terms are still O(seq_len) or smaller — the sparse layer's cache is **strictly larger** than a plain full-attention layer's cache, not smaller. The compute savings (only attending to top-k selected blocks instead of the full causal mask) are real and are the actual point of the mechanism, but "sub-linear memory growth" was the wrong framing and is corrected here. This changes the `max_memory_usage_bytes` formula for the `KVCacheSpec` design relative to what §2b's prose would have implied.

## 3. Residual / normalization math (MiniCPM "mup"-derived scaling — easy to silently get wrong)

Every decoder layer (`MiniCPMSALADecoderLayer.forward`, confirmed verbatim):
```
residual = h
h = input_layernorm(h)                     # RMSNorm, pre-norm
h = self_attn(h)                            # sparse or lightning, per mixer_types[layer_idx]
h = residual + h * (scale_depth / sqrt(num_hidden_layers))     # = 1.4 / sqrt(32) ≈ 0.2475

residual = h
h = post_attention_layernorm(h)
h = mlp(h)                                   # standard SwiGLU: down(silu(gate(h)) * up(h))
h = residual + h * (scale_depth / sqrt(num_hidden_layers))     # same constant, both branches
```
This constant (`≈0.2475`) is **not a learned parameter** — it's a fixed depth-scaling factor baked into the architecture (inherited from MiniCPM's muP parameterization), applied identically to both attention and MLP residual branches, on *every* layer regardless of mixer type. Any vLLM port that reuses a generic `LlamaDecoderLayer`-style residual add (`h = residual + h`) will silently produce wrong activations — this is exactly the kind of "no hacks" line-item the mission brief calls out, and it's cheap to get right if caught in Phase 1 (here) rather than in Phase 9 (numerical validation) after a failed logit comparison.

Embedding and LM-head scaling (`MiniCPMSALAModel.forward` / `...ForCausalLM.forward`, verbatim):
```
inputs_embeds = embed_tokens(input_ids) * scale_emb        # scale_emb = 12
...
logits = lm_head(hidden_states / (hidden_size / dim_model_base))   # divide by 4096/256 = 16
```
So the embedding is scaled **up** by 12× on the way in and the final hidden state is scaled **down** by 16× on the way into the unembedding — two independent muP-style constants, both config-driven (`scale_emb`, `dim_model_base`), neither is the "1/sqrt(d)" you'd guess by analogy to standard transformer embedding scaling.

## 4. Cache-layer type invariant (enforced at runtime, must be preserved in vLLM's cache manager)

```python
if self.mixer_type[0] != "minicpm4":
    raise ValueError("The first layer must be 'minicpm4' to track seen tokens.")
```
Layer 0 is always sparse in this checkpoint (confirmed above) and the cache class uses `layers[0]`'s update call to increment `_seen_tokens`, i.e. **the global sequence-length counter for the whole model is piggybacked on the sparse-cache layer's update**, not tracked independently. A vLLM `KVCacheManager` design that treats all layers symmetrically for "tokens processed" bookkeeping will diverge from this reference unless layer-0's special role is preserved or the counter is re-derived independently (safer: don't inherit this coupling, track sequence length independently in vLLM's own scheduler — but the invariant needs to be *known*, which is why it's flagged here rather than discovered during correctness validation).

## 5. Parameter count (bottom-up, from real config, not the "9B" marketing figure)

```
embed_tokens:            73448 × 4096                          = 300,908,528
32 × decoder layers, each:
  attn (lightning, 24 layers):
    q/k/v/o_proj: 4 × (4096×4096)                              = 67,108,864
    z_proj (gate): 4096×4096                                    = 16,777,216
    q_norm/k_norm/o_norm: negligible (~4096+128+128)
  attn (sparse, 8 layers):
    q_proj: 4096×4096, k/v_proj: 4096×256 each (2 kv heads×128), o/o_gate_proj: 4096×4096 each
                                                                  ≈ 4096×4096×3 + 4096×256×2 ≈ 52,428,800
  mlp (all 32 layers): gate+up+down = 3 × (4096×16384)          = 201,326,592
  norms: input_layernorm + post_attn_layernorm ≈ 8192, negligible

  lightning layer total ≈ 67.1M + 16.8M + 201.3M ≈ 285.2M params × 24 layers ≈ 6.85B
  sparse layer total    ≈ 52.4M + 201.3M ≈ 253.7M params × 8 layers ≈ 2.03B
final norm: negligible
lm_head: 73448 × 4096 (NOT tied)                                = 300,908,528

TOTAL ≈ 6.85B + 2.03B + 0.30B (embed) + 0.30B (lm_head) ≈ 9.5B parameters
```
Consistent with the "9B" figure OpenBMB advertises and with the four `.safetensors` shards totaling ~19GB at bf16 (2 bytes/param × 9.5B ≈ 19GB — checks out exactly against the file listing).

## 6. Why this is genuinely a "Medium→Expert" vLLM port, not a template job

1. **The KV cache is not one thing.** The *reference HF model* uses five buffers per sparse layer (full K/V, two compressed tiers, two ring-buffer staging buffers) plus a recurrent-state tensor per lightning layer. [UPDATE, Stage 3/4: this port's own `HierarchicalCompressedAttentionSpec` simplifies the sparse side to three regions — full K/V + two compressed tiers, no separate staging buffers — since the port already retains the full K cache and can compute compression windows on demand rather than double-buffering the same recent tokens. This is a port-side design choice, not a claim about the reference implementation, which genuinely does use five. See `docs/minicpm_sala_known_limitations.md`'s "Stage 3/4 consistency fixes" for the full reasoning.] Either way, vLLM's `KVCacheManager`/paged-block abstraction assumes a single K/V tensor pair per layer with block-granular append; this model needs a **custom cache spec per mixer type**, analogous to how vLLM already handles Mamba-style constant-size state (Jamba/Zamba) alongside PagedAttention blocks for full-attention layers — this is the closest existing precedent (see Phase 2 compatibility matrix, to follow).
2. **Two attention math families in one model**, selected per-layer by a static schedule (`mixer_types`), not by input — this is a **scheduling problem at the layer level**, not the token level, so it's simpler than data-dependent MoE routing but still means the decoder's `forward` loop must dispatch per-layer to different kernel families.
3. **External kernel dependency**: sparse attention's real speed comes from `infllm_v2`'s custom Triton kernels (`infllmv2_attn_stage1`, `infllmv2_attn_varlen_func`, `max_pooling_1d_varlen`) — these do not exist in vLLM's kernel set and are not FlashInfer/FlashAttention-upstream kernels. Options: (a) vendor the `infllm_v2` kernels as an optional dependency (precedent: vLLM already vendors FlashMLA, TRTLLM-gen kernels for DeepSeek), (b) implement a PagedAttention-compatible re-expression of top-k block-sparse attention using vLLM's existing sparse/MoE-adjacent kernel infra, or (c) fall back to dense attention below `dense_len=8192` (already the reference model's own behavior) and only invest in the sparse kernel path for long-context serving. This is a real design decision requiring a trade-off writeup, not a guess — flagging for Phase 3.
4. **Two RoPE policies in one model** (sparse layers: NoPE; lightning layers: RoPE) — this is unusual enough that vLLM's per-model RoPE initialization (usually one `rotary_emb` shared across all layers) needs to become per-layer-type, and the "HyPE" (Hybrid Positional Embedding) length-generalization claim in the README rests specifically on this split, so getting it backwards would silently produce a model that runs but degrades on long context — exactly the failure mode that's expensive to catch without the correctness-validation harness (Phase 9).

## 7. What's next (Phase 2 — not yet done, correctly deferred)

Before writing any vLLM code I need to establish the **compatibility matrix**: which existing vLLM model implementation is the right base to fork from. Candidates worth checking against real vLLM source (not memory):
- **Qwen3-Next** (vLLM has hybrid gated-linear-attention + full-attention support already — closest precedent for the "static per-layer mixer schedule + heterogeneous cache" problem)
- **Jamba / Zamba2** (vLLM's existing Mamba-hybrid cache management — precedent for O(1)-size recurrent state living alongside paged full-attention cache)
- **DeepSeek-V3.2/V4** (precedent for *this exact repo's* "hybrid attention, heterogeneous KV cache, multiple page-size buckets" engineering pattern, per the vLLM blog writeup on DSV4 — worth reading their actual cache-bucketing design before inventing one from scratch)

I'll pull the real vLLM source for whichever of these is closest (not guess from training-data familiarity) before proposing a module-reuse plan, since guessing here is exactly the kind of unverified-assumption the mission brief prohibits.

One clarifying question before I go further: given the external-kernel dependency issue (§6.3), do you want me to scope the **first vLLM PR** as (a) dense-only support (correct, simpler, works for ≤8192 tokens, defers sparse-kernel vendoring to a follow-up PR — this is how vLLM typically lands new architectures, "initial release" then hardening passes per the DeepSeek-V4 precedent in §7), or (b) full sparse+linear hybrid from the start? (a) is a much more tractable and honestly-scoped Phase-3 target; (b) requires vendoring or reimplementing the InfLLM-V2 Triton kernels before any correctness testing can even begin.
