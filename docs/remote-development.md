# Remote development (Cursor SSH + GPU)

This repository is set up for **zero-touch initialization** on Linux GPU hosts
(local A40 cloud instances, RunPod, etc.) via Cursor Remote SSH.

## One-time: connect Cursor to the GPU host

1. Install [Cursor](https://cursor.com) and the Remote SSH extension (built-in).
2. Add a host to `~/.ssh/config` (example — **do not commit credentials**):

```sshconfig
Host hybridattn-gpu
    HostName YOUR_HOST_IP
    User root
    Port 26515
    IdentityFile ~/.ssh/id_ed25519
    LocalForward 8080 localhost:8080
    ServerAliveInterval 60
```

3. In Cursor: **Remote-SSH: Connect to Host** → `hybridattn-gpu`.
4. **File → Open Folder** → clone path, e.g. `/root/vLLM-HybridAttn`.

## Zero-touch after open

When the workspace folder opens, Cursor runs **HybridAttn: Init Dev Environment**
(`.vscode/tasks.json`, `runOn: folderOpen`). This executes:

```bash
bash scripts/dev/init-dev-env.sh
```

Idempotent steps:
- validates OS, disk, git, Python, GPU (nvidia-smi)
- creates/updates `.venv` from `requirements-dev.txt`
- installs optional git hooks (ruff pre-commit, PR1 boundary pre-push)
- writes `.dev/environment.json` and init stamp

Manual re-run: `make init` or `make init-force`.

## Daily workflow

```bash
make health        # repo health (non-destructive)
make gate-quick    # ruff + py_compile
make gate-pr1      # Docker PR1 gate (26 tests)
make gate-full     # Docker full stack (72 unit tests)
make lint / make fmt
```

## GPU validation (A40)

After `make init` on a real GPU with vLLM installed:

```bash
pip install "vllm==0.24.0"
bash scripts/install_pr2_overlay.sh
bash scripts/install_infllm_v2.sh
bash pr2/scripts/gpu_validation/run_all_gpu_validation.sh
```

Or: `make overlay && make install-infllm && make gpu-validation`

Requires `infllm_v2` build (see `scripts/install_infllm_v2.sh`).

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Init task did not run | Command Palette → **Tasks: Run Task** → HybridAttn: Init Dev Environment |
| `vllm` import fails | `make init-force`; confirm `.venv` selected in Cursor |
| CRLF shell errors | `file pr2/scripts/gpu_validation/*.sh` must not say CRLF; `.gitattributes` enforces LF |
| PR1 import fails | Ensure `pr2/` not copied into site-packages for PR1-only work |
| CUDA mismatch | Match PyTorch CUDA build to driver; see `init` log at `.dev/init.log` |

## Security

- Never commit `.env`, PATs, or private keys.
- Revoke any token exposed in chat logs; use SSH keys + `gh auth login`.