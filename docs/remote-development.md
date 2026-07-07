# Remote development (Cursor SSH + GPU)

This repository is set up for **zero-touch initialization** on Linux GPU hosts
(local A40 cloud instances, RunPod, etc.) via Cursor Remote SSH.

## One-time: connect Cursor to the GPU host (Vast.ai RTX 4090)

1. Install [Cursor](https://cursor.com) — Remote SSH extension is built-in.
2. Windows SSH config (`C:\Users\archa\.ssh\config`):

```sshconfig
Host vast4090
    HostName 23.158.136.85
    User root
    Port 26515
    IdentityFile C:\Users\archa\.ssh\id_ed25519
    IdentitiesOnly yes
    ServerAliveInterval 30
    ServerAliveCountMax 10
    StrictHostKeyChecking accept-new
```

3. Cursor user settings should include:
   - `remote.SSH.path`: `C:\Windows\System32\OpenSSH\ssh.exe`
   - `remote.SSH.configFile`: `C:\Users\archa\.ssh\config`
   - `remote.SSH.remotePlatform.vast4090`: `linux`

4. In Cursor: **Remote-SSH: Connect to Host** → `vast4090`.
5. **File → Open Folder** → `/workspace/hybridattn`.

### One-command preflight (from local Git Bash)

```bash
bash scripts/dev/connect_cursor.sh
```

This verifies SSH, runs sync + bootstrap + health check on the remote host, and prints connection instructions.

## Zero-touch after open

When the workspace folder opens, Cursor runs **HybridAttn: Init Dev Environment**
(`.vscode/tasks.json`, `runOn: folderOpen`). This executes:

```bash
bash scripts/dev/init-dev-env.sh   # sync-repo + bootstrap-vast on Remote SSH
```

Idempotent steps:
- **sync** with `origin/feature/minicpm-sala-sparse` (if clean tree)
- **bootstrap** vLLM overlay + infllm_v2 when missing
- validates OS, disk, git, Python, GPU (nvidia-smi), weights path
- creates/updates `.venv` from `requirements-dev.txt`
- installs optional git hooks (ruff pre-commit, PR1 boundary pre-push)
- writes `.dev/environment.json` and init stamp

Manual re-run: `make init` or `make init-force`.

## Daily workflow

```bash
make health              # scripts/dev/health_check.sh
make connect-cursor      # SSH preflight + remote pipeline
make sync-repo           # git fetch/pull on feature branch
make gate-quick          # ruff + py_compile
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