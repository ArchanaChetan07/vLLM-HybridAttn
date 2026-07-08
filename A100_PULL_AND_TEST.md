# A100 pull and test (token14 / L0→L1 boundary)

Run before shutdown:

```bash
git pull && bash scripts/install_pr2_overlay.sh
python3 pr2/scripts/gpu_validation/diagnostics/gate1_hello_token14_parity.py
```

Optional bisect if token14 still RED:

```bash
python3 pr2/scripts/gpu_validation/diagnostics/gate1_l0_qkv_pos19_bisect.py
python3 pr2/scripts/gpu_validation/diagnostics/gate1_l0_layer_out_pos.py
python3 pr2/scripts/gpu_validation/diagnostics/gate1_l0_l1_layer_in_pos.py
```

Fix in this push:
- `history_decode` flash uses fp32 below seq cap (matches one-shot prefill).
- L0 `o_gate` sigmoid + `o_proj` default to fp32 at TP=1.
- muP residual add defaults to fp32 (opt out: `MINICPM_SALA_BF16_RESIDUAL=1`).
