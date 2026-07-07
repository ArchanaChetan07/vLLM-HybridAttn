# Known Limitations (Summary)

This file is a short index. The **authoritative, evidence-backed log** is:

[docs/minicpm_sala_known_limitations.md](minicpm_sala_known_limitations.md)

## Pinned vLLM commit

`vllm-project/vllm @ 8cfeb84dba41a0c56570334757d921abd71e5288` (v0.24.0 API surface)

## Hardware requirements

| Capability | Requirement |
|------------|-------------|
| Lightning attention kernels (step 2) | CUDA sm_80+ (Ampere or newer) |
| InfLLM-V2 sparse kernels | sm_80+ + `infllm_v2` package |
| HF parity / full model | ~19 GB weights + 48 GB GPU recommended (sequential HF/vLLM load) |

## Verification status (2026-07-07)

| Check | RTX 4090 | CPU/Docker |
|-------|----------|------------|
| Unit tests + ruff | — | 45+ passed |
| Step 0 sparse LIVE gate | Pending gated run | N/A |
| Steps 1, 3, 4 sparse e2e | PASS | N/A |
| Step 2 lightning | FAIL 뿯↽ fix applied (bf16 q/k/v, fp32 state) | sm_75 blocked |
| HF parity | BLOCKED (weights) | N/A |
| Mixed batch | Not run | N/A |
| Fresh clone | `scripts/verify_fresh_clone.sh` | Use `make verify-fresh` |

## Reproducible install (no manual site-packages edits)

```bash
pip install "vllm==0.24.0"
bash scripts/install_pr2_overlay.sh
bash scripts/install_infllm_v2.sh   # GPU sparse path
bash scripts/verify_fresh_clone.sh  # CPU tests
bash pr2/scripts/gpu_validation/run_all_gpu_validation.sh
```

## PR boundary

PR1 (`vllm/model_executor/models/minicpm_sala.py`) has **zero** sparse imports.
PR2 overlay adds sparse wiring under `pr2/vllm/`.

## Merge readiness

**Not merge-ready** until Step 0–C, HF parity, TP, and fresh-clone GPU validation pass in one clean run. See [docs/merge_readiness_checklist.md](merge_readiness_checklist.md).
