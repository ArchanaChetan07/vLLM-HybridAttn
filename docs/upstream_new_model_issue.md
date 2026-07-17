# Draft: vllm-project/vllm "New Model" issue

Post at: https://github.com/vllm-project/vllm/issues/new?template=400-new-model.yml

Before posting, search existing issues/PRs for "MiniCPM-SALA", "SALA",
"lightning attention", "InfLLM" — if OpenBMB already has one in flight,
comment there instead of opening a duplicate.

---

**Title:** `[New Model]: MiniCPM-SALA (openbmb) — hybrid Lightning Attention + InfLLM-V2 sparse GQA`

## The model to consider

[openbmb/MiniCPM-SALA](https://huggingface.co/openbmb/MiniCPM-SALA) — ~9.5B-param,
32-layer hybrid causal LM from OpenBMB (Apache-2.0, weights public,
`trust_remote_code` reference implementation in the model repo).

Architecture (from `config.json` / `modeling_minicpm_sala.py`):

- **24/32 layers**: gated linear attention ("lightning-attn") — per-head
  ALiBi-slope decay, q/k RMSNorm, RoPE on q/k before the recurrence, fp32
  recurrent state, sigmoid output gate. O(1) state per sequence.
- **8/32 layers** (`minicpm4` mixer): GQA 32:2, **NoPE**
  (`attn_use_rope=false`), sigmoid output gate. Dense FlashAttention below
  `dense_len=8192`; at/above it, InfLLM-V2 block-sparse attention (two-tier
  compressed-K scoring -> top-k block selection -> varlen sparse attention).
- muP scaling throughout: `scale_emb=12`, residual scale `scale_depth/sqrt(32)`,
  logits divided by `hidden_size/dim_model_base = 16`. 524k max positions.

## The closest model vLLM already supports

- Lightning layers: same kernel family as `MiniMaxText01LinearAttention`
  (in-tree `lightning_attention` / `linear_decode_forward_triton`,
  `MambaBase` state management) — with three verified differences: RoPE on
  q/k, q/k RMSNorm, and no per-layer decay scaling.
- Dense side: `MiniCPM3ForCausalLM` lineage; hybrid scheduling like
  Jamba / Qwen3-Next (`IsHybrid` / `HasInnerState`).

## What's your difficulty of supporting the model you want?

I have a working integration I'd like to upstream, split into two PRs, and
I'm asking for guidance on two design points before opening PR1.

Existing work: https://github.com/ArchanaChetan07/vLLM-HybridAttn

- **PR1** (model + lightning + dense GQA): single model file reusing
  in-tree linear-attention infra; ~30 CPU unit tests green in Docker CI;
  `check_logprobs_close` test written in the standard harness.
  **Honest status: the HF-parity GPU run is the current gate — an earlier
  run failed, root causes were found and fixed (verified line-by-line
  against the reference modeling file), A100 re-run is scheduled. Green
  parity logs will be attached to the PR before requesting review.**
- **PR2** (follow-up): InfLLM-V2 sparse backend —
  `AttentionBackend`/`AttentionImpl`, a `HierarchicalCompressedAttentionSpec`
  KV spec (prefix caching disabled, Mamba-style rationale), 44 unit tests,
  gated GPU validation suite.

Questions for maintainers:

1. **Lightning kernel path**: the HF reference computes the recurrence with
   `fla` (`chunk_simple_gla`). Preference for (a) proving parity of the
   in-tree `lightning_attention` Triton kernel and shipping only that, or
   (b) an optional `fla` path? I assume (a) and am validating accordingly.
2. **Sparse kernels (PR2)**: InfLLM-V2 kernels live in OpenBMB's
   `infllmv2_cuda_impl` (external CUDA package, sm_80+). Would you want
   these vendored into `csrc/`, taken as an optional dependency, or should
   PR2 wait for an OpenBMB-published wheel? Happy to write this up as an
   RFC if useful.

I'm committed to maintaining this model (validation harness + parity
scripts are in the repo above).
