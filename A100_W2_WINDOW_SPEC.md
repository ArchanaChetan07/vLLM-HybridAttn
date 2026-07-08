# A100 W2 one-shot window spec (do not run here)

This repo is currently prepared for a **single A100 session** to close the remaining
`token14` parity gap. **Do not run GPU gates on the local T1000/CPU-only setup.**

## Hard constraints

- **Commit before trust**: fresh-clone and verify `origin/feature/minicpm-sala-sparse` head.
- **token14 stays RED** until the GPU log shows **GREEN**.
- **`v` is the truth probe** (per-position `v` parity is the primary signal).
- **Do not touch** lightning bookkeeping or dense KV gather, and **do not change** the fix.
- **One window only**: 45-minute hard stop.

## One-shot procedure (A100)

### 0) Fresh clone + pin to `a839fd1+`

```bash
git clone https://github.com/ArchanaChetan07/vLLM-HybridAttn.git
cd vLLM-HybridAttn
git fetch origin
git checkout feature/minicpm-sala-sparse
git rev-parse HEAD
```

**Expected:** `HEAD` is `a839fd1` (or a descendant, i.e. `a839fd1+`).

### 1) Install overlay

```bash
./scripts/install_pr2_overlay.sh
```

### 2) Set weights

```bash
export MINICPM_SALA_WEIGHTS=/path/to/MiniCPM-SALA
```

### 3) HARD STOP sanity checks

Overlay sanity (must report `OVERLAY_OK`):

```bash
python3 -c "import vllm.v1.attention.backends.minicpm_sala_sparse as m; print(m.__file__); assert m._DENSE_HISTORY_DECODE_MAX_SEQ == 64; assert hasattr(m, '_flash_dense_varlen_causal'); print('OVERLAY_OK')"
```

Sparse live (must be **LIVE**):

```bash
python3 pr2/scripts/gpu_validation/assert_sparse_live.py
```

If either check fails, **STOP** (do not start W2).

### 4) Run the one-shot W2 harness

```bash
bash pr2/scripts/gpu_validation/diagnostics/run_a100_w2_final.sh
```

**Expected outputs:** 4 trace logs written to:

- `pr2/scripts/gpu_validation/diagnostics/traces/assert_sparse_live_w2_final.log`
- `pr2/scripts/gpu_validation/diagnostics/traces/per_position_v_parity_w2_final.log`
- `pr2/scripts/gpu_validation/diagnostics/traces/hello_token14_w2_final.log`
- `pr2/scripts/gpu_validation/diagnostics/traces/lightning_state_w2_final.log`

## Pass criteria (GREEN)

- **Per-position `v` parity**: tol \(1e-5\) at **L1/L6**, **step 7 + step 14**.
- **token14**: token id **16091** and log indicates **GREEN**.
- **Lightning peak**: `overall_peak` approximately **0** (near-zero drift).

If any of the above fails, treat the run as **RED**.

## 45-minute hard stop

If W2 does not finish cleanly, or results are inconclusive within 45 minutes:

- Archive the `*_w2_final.log` files and any `traces/*.ndjson`.
- Stop the session; do **not** start exploratory reruns in the same window.

