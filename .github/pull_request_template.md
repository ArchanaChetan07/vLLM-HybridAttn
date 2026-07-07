## Summary

<!-- What does this PR change and why? -->

## PR scope

- [ ] PR1 — model only (`vllm/model_executor/models/minicpm_sala.py`, no sparse imports)
- [ ] PR2 — sparse overlay (`pr2/`, requires `infllm_v2`)
- [ ] Docs / CI / tooling only

## Validation

| Check | Result | Notes |
|-------|--------|-------|
| `ruff check` / `ruff format` | | |
| PR1 Docker gate (`docker_run_pr1.sh`) | | |
| HF `check_logprobs_close` (short) | | Pending until GPU parity passes |
| HF `check_logprobs_close` (long ≥8192) | | Sparse regime |
| Gated GPU Steps 0–4, 6 | | See `docs/VALIDATION_REPORT.md` |

## Limitations

<!-- Link to docs/minicpm_sala_known_limitations.md if behavior is intentionally incomplete -->
