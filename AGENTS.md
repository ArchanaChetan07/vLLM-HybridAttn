# AGENTS.md — Cursor / AI agent instructions

## Project
MiniCPM-SALA integration for vLLM (vLLM-HybridAttn). Hybrid Lightning + dense/sparse GQA.

## Before editing
1. Read `docs/pr_split_plan.md` for PR1 vs PR2 scope.
2. Run `make health` after environment changes.
3. Never claim test counts from a dirty tree — use `make gate-pr1` or fresh clone.

## Quality gates (layered)
| Level | Command |
|-------|---------|
| Quick static | `make gate-quick` |
| PR1 | `make gate-pr1` |
| Full stack | `make gate-full` |

## Remote SSH
On folder open, VS Code/Cursor runs `scripts/dev/init-dev-env.sh` automatically (see `.vscode/tasks.json`).
Manual: `make init`.

## Non-negotiables
- PR1: zero sparse imports
- muP scaling preserved
- bf16-only for supported dtypes
- infllm_v2 page_block_size % 256 == 0
- Do not force-push main; sparse path not production-certified until A40 GPU validation