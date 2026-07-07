# Upstream vLLM fork integration snippets
#
# Copy these into a vLLM fork at the paths shown. The `patches/` directory
# contains equivalent unified diffs for review.

## vllm/model_executor/models/registry.py

Add inside `_TEXT_GENERATION_MODELS` after `MiniCPM3ForCausalLM`:

```python
    "MiniCPMSALAForCausalLM": ("minicpm_sala", "MiniCPMSALAForCausalLM"),
```

## tests/models/registry.py

Add inside `_TEXT_GENERATION_EXAMPLE_MODELS` after `MiniCPM3ForCausalLM`:

```python
    "MiniCPMSALAForCausalLM": _HfExamplesInfo(
        "openbmb/MiniCPM-SALA",
        trust_remote_code=False,
        max_model_len=4096,
    ),
```

## New in-tree files (PR1 + PR2)

| Path | Source in this repo |
|------|---------------------|
| `vllm/model_executor/models/minicpm_sala.py` | `vllm/...` (PR1) or `pr2/vllm/...` (full stack) |
| `vllm/model_executor/models/minicpm_sala_sparse_wiring.py` | `pr2/vllm/...` only |
| `vllm/v1/core/minicpm_sala_kv_cache_spec.py` | `pr2/vllm/...` |
| `vllm/v1/attention/backends/minicpm_sala_sparse.py` | `pr2/vllm/...` |

## Tests to include in upstream PR

- `tests/models/language/generation/test_minicpm_sala*.py`
- `tests/v1/core/test_minicpm_sala_*.py`
- `tests/v1/attention/test_minicpm_sala_*.py`

## External dependency (document in PR description)

Sparse layers require `infllm_v2` from OpenBMB/infllmv2_cuda_impl, built with
`patches/fix_cutlass_submodule.sh` and `pip install --no-build-isolation -e .`.
Ampere+ (sm_80+) required for lightning and InfLLM-V2 kernels.
