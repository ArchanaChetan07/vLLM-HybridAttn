# Dev automation plan (prioritized)

## Phase A — Implemented (this pass)

| Item | Path | Purpose |
|------|------|---------|
| Idempotent init | `scripts/dev/init-dev-env.sh` | Zero-touch SSH/local setup |
| Health check | `scripts/dev/health-check.sh` | Read-only repo validation |
| Quality gates | `scripts/dev/run-gates.sh` + `Makefile` | quick / pr1 / full |
| Git hooks | `scripts/dev/hooks/*` | ruff pre-commit, PR1 pre-push |
| Cursor rules | `.cursor/rules/*.mdc` | Architecture + constraints for AI |
| Agent guide | `AGENTS.md` | Cursor agent entrypoint |
| Workspace auto-init | `.vscode/tasks.json` | `runOn: folderOpen` |
| Dev deps pin | `requirements-dev.txt` | Reproducible venv |
| Remote bootstrap | `scripts/dev/bootstrap-remote.sh` | Fresh GPU VM clone + install |

## Phase B — Next (recommended)

1. **systemd / tmux session** for long-running vLLM servers on SSH hosts.
2. **pre-commit framework** (optional) — migrate from shell hooks if team grows.
3. **CI parity** — GitHub Actions calling same `make gate-*` targets as local.
4. **infllm_v2 cache** — bake Ampere-compatible wheel into init on known A40 images.
5. **Cursor Cloud Agent** profile — document branch + gate policy for cloud runs.

## Phase C — Production hardening

- HF `check_logprobs_close` in init health when weights path set (`VLLM_TEST_MODEL`).
- NCCL / multi-GPU probe in `checks.sh` when `WORLD_SIZE>1`.
- Benchmark discovery registry (`docs/performance.md` → machine-readable manifest).
- Merge conflict predictor in pre-push (compare base branch file sets PR1 vs PR2).

## Architecture review (summary)

**Strengths:** Clear PR1/PR2 split, Docker gates, dense documentation, GPU validation scripts.

**Gaps addressed:** No unified init, no Cursor rules, `.vscode` gitignored, CRLF fragility on Windows.

**Remaining risks:** Sparse path never validated on hardware in CI; main branch may contain full `pr2/` — cut PR1-only branches for upstream vLLM submission.

## Success metrics

- New engineer: SSH connect → open folder → `make health` green in <10 min (excluding vLLM pip time).
- Every PR: `make gate-quick` locally; `make gate-pr1` before PR1 merge.
- AI edits: respect `.cursor/rules` without manual reminder.