# Architecture (Summary)

Full design notes: [docs/minicpm_sala_phase1_architecture_report.md](minicpm_sala_phase1_architecture_report.md)

HF reference: [openbmb/MiniCPM-SALA](https://huggingface.co/openbmb/MiniCPM-SALA)

## Model

32-layer hybrid causal LM (~9.5B params):

- **75%** `lightning-attn` — gated linear attention (vLLM `lightning_attention` Triton, recurrent fp32 state)
- **25%** `minicpm4` — GQA; dense below `dense_len=8192`, InfLLM-V2 sparse at or above

## PR split

| PR | Files | Role |
|----|-------|------|
| PR1 | `vllm/model_executor/models/minicpm_sala.py` | Model, dense attention, lightning layers |
| PR2 | `pr2/vllm/...` | Sparse backend, wiring, KV cache spec |

Install PR2 via `bash scripts/install_pr2_overlay.sh` after `pip install vllm`.

## Sparse path (PR2)

```
forward 뿯↽ sparse_mask(seq_len >= dense_len)
  뿯↽ dense: infllmv2_attn_with_kvcache (packed batched q/k/v)
  뿯↽ sparse: gather K 뿯↽ CompressK 뿯½2 뿯↽ compressed_attention 뿯↽ topk_idx
            뿯↽ infllmv2_attn_varlen_func (HF reference API)
  뿯↽ mixed: per-sequence dense/sparse + scatter-back
```

Debug: `MINICPM_SALA_DEBUG_SPARSE=1`

## KV cache

- Lightning: `(num_heads/tp, head_dim, head_dim)` fp32 recurrent state per slot
- Sparse: paged full K/V; `block_size` multiple of 256; compressed tiers recomputed each forward

## Validation

Gated GPU suite: `pr2/scripts/gpu_validation/run_all_gpu_validation.sh`

- Step 0: `assert_sparse_live.py`
- Steps 1–4, 6: kernel / gather / sparse e2e / mixed impl
- Steps B/C: parity + full-model batch (needs `MINICPM_SALA_WEIGHTS`)

Diagrams: [docs/minicpm_sala_diagrams.md](minicpm_sala_diagrams.md)
