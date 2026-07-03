# vLLM-HybridAttn

Production-grade vLLM integration for [MiniCPM-SALA](https://huggingface.co/openbmb/MiniCPM-SALA) — OpenBMB 9B hybrid-attention language model combining Lightning Attention and InfLLM-V2 sparse attention.

[![Tests](https://img.shields.io/badge/unit%20tests-66%2F66%20pass-brightgreen)](#testing)
[![PR1](https://img.shields.io/badge/PR1-independent-blue)](#repository-structure)
[![vLLM](https://img.shields.io/badge/vLLM-0.24.0-6c5ce7)](https://github.com/vllm-project/vllm)
[![License](https://img.shields.io/badge/license-Apache%202.0-lightgrey)](LICENSE)

## Motivation

MiniCPM-SALA interleaves two attention mechanisms across 32 layers:

- **Lightning Attention** (75%) — O(1) gated linear attention
- **Sparse GQA** (25%) — InfLLM-V2 top-k block sparse past 8192 tokens

Integrating this into vLLM requires model code, optional sparse backends, custom KV cache specs, and scheduler wiring — delivered as **two independent upstream PRs**.

## Features

| Feature | PR | Status |
|---------|-----|--------|
| Hybrid layer schedule | PR1 | Verified (unit tests) |
| Lightning Attention kernels | PR1 | Verified (unit tests) |
| Dense GQA fallback (NoPE) | PR1 | Verified (unit tests) |
| Weight loading + registry | PR1 | Verified |
| InfLLM-V2 sparse backend | PR2 | Verified (unit tests) |
| Hierarchical KV cache spec | PR2 | Verified (unit tests) |
| HF logprob parity | PR1 | Pending |
| Ampere+ sparse e2e | PR2 | Pending |

## Architecture

See [docs/architecture.md](docs/architecture.md) for Mermaid diagrams covering forward pass, KV cache lifecycle, and module dependencies.

## Repository Structure

```
vllm/model_executor/models/minicpm_sala.py    # PR1 — no sparse imports
tests/models/language/generation/             # PR1 tests (22)
pr2/                                          # PR2 overlay (not in PR1 branch)
docker_run_pr1.sh                             # PR1 CI gate
docker_run_integration.sh                   # Full stack gate
docs/                                         # Architecture, testing, limitations
```

Branch layout:

| Branch | Contents |
|--------|----------|
| `main` | Full monorepo (PR1 + PR2 overlay + docs) |
| `feature/minicpm-sala-model` | PR1 only — no `pr2/` directory |
| `feature/minicpm-sala-sparse` | PR1 + PR2 full stack |

## Installation

```bash
git clone https://github.com/ArchanaChetan07/vLLM-HybridAttn.git
cd vLLM-HybridAttn
pip install vllm==0.24.0 pytest tblib einops ruff
```

Overlay files into your vLLM install — see [docs/developer_guide.md](docs/developer_guide.md).

## Docker

```bash
# PR1 only — no sparse files required
bash docker_run_pr1.sh

# Full stack — PR1 + PR2 + infllm_v2 build
bash docker_run_integration.sh
```

## Testing

| Gate | Tests | Verified |
|------|-------|----------|
| `docker_run_pr1.sh` | 22 | 2026-07-03 |
| `docker_run_integration.sh` | 66 | 2026-07-03 |
| ruff check + format | all | 2026-07-03 |

Details: [docs/testing.md](docs/testing.md)

## GPU Validation

| Hardware | Step 1 | Step 2 | Step 3 | Step 4 |
|----------|--------|--------|--------|--------|
| T1000 sm_7.5 | Pass | Fail (Ampere) | Pass | Fail (Ampere) |
| A40 Ampere+ | Pending | Pending | Pending | Pending |

```bash
bash pr2/scripts/gpu_validation/run_all_gpu_validation.sh
```

## Supported Hardware

| Path | Minimum GPU |
|------|-------------|
| PR1 import + unit tests | CPU (Docker) |
| Dense GQA inference | vLLM default backend |
| Lightning kernels | Ampere+ (sm_80+) |
| Sparse InfLLM-V2 | Ampere+ (sm_80+) + infllm_v2 |

## Current Limitations

- Sparse path not validated on Ampere+ yet (A40 pending)
- `check_logprobs_close` not executed (needs GPU + weights)
- No published benchmark numbers
- Multi-GPU TP not validated

Full list: [docs/known_limitations.md](docs/known_limitations.md)

## Roadmap

[ROADMAP.md](ROADMAP.md)

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) — branch strategy, coding standards, PR workflow.

Upstream PR templates: [docs/pull_requests/PR1_model.md](docs/pull_requests/PR1_model.md), [docs/pull_requests/PR2_sparse.md](docs/pull_requests/PR2_sparse.md)

## Citation

```bibtex
@misc{vllm-hybridattn2026,
  title={vLLM-HybridAttn: MiniCPM-SALA Integration for vLLM},
  year={2026},
  url={https://github.com/ArchanaChetan07/vLLM-HybridAttn}
}
```

Reference model: [OpenBMB/MiniCPM-SALA](https://huggingface.co/openbmb/MiniCPM-SALA)

## License

Apache License 2.0 — see [LICENSE](LICENSE).
