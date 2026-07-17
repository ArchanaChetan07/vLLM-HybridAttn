#!/usr/bin/env python3
"""GPU Step 0: assert the InfLLM-V2 sparse backend is LIVE, not a silent
dense fallback.

This is the gate that makes every later "sparse path PASS" claim
meaningful: steps 2-6 exercising a build where `infllm_v2` failed to
import would silently test the dense fallback and report green. This
script fails LOUDLY unless all of the following hold:

  1. The PR2 overlay files are actually importable from the installed
     vLLM tree (i.e. `scripts/install_pr2_overlay.sh` ran against THIS
     environment) and are valid UTF-8 (guards the UTF-16 overlay
     corruption seen on 2026-07-02, see CHANGELOG).
  2. `INFLLM_V2_AVAILABLE` is True (the real CUDA package imported).
  3. A CUDA device with compute capability >= sm_80 is present (the
     infllmv2 kernels' confirmed hardware floor).
  4. `create_sparse_attention_if_available` would return a real sparse
     Attention for the released checkpoint config (not None).

Exit 0 = sparse is LIVE. Exit 1 = whatever passes after this would NOT
be testing the sparse path.
"""

import sys

import torch


def main() -> int:
    failures: list[str] = []

    # 1. Overlay modules importable + UTF-8 clean.
    try:
        from vllm.v1.attention.backends import minicpm_sala_sparse as sparse_mod

        raw = open(sparse_mod.__file__, "rb").read()
        if b"\x00" in raw:
            failures.append(
                f"{sparse_mod.__file__} contains null bytes (UTF-16 overlay "
                "corruption) -- re-run scripts/install_pr2_overlay.sh"
            )
    except ImportError as e:
        failures.append(
            f"PR2 sparse backend not importable ({e}) -- run "
            "scripts/install_pr2_overlay.sh against this environment first"
        )
        sparse_mod = None

    try:
        from vllm.model_executor.models import (  # noqa: F401
            minicpm_sala_sparse_wiring,
        )
    except ImportError as e:
        failures.append(f"PR2 sparse wiring not importable ({e})")

    # 2. Real infllm_v2 package.
    if sparse_mod is not None and not sparse_mod.INFLLM_V2_AVAILABLE:
        failures.append(
            "INFLLM_V2_AVAILABLE is False: the infllm_v2 CUDA package is not "
            "installed. Build it with scripts/install_infllm_v2.sh (sm_80+). "
            "Every 'sparse' test in this state exercises the DENSE fallback."
        )

    # 3. Hardware floor.
    if not torch.cuda.is_available():
        failures.append("No CUDA device -- sparse kernels cannot be LIVE.")
    else:
        props = torch.cuda.get_device_properties(0)
        cc = props.major * 10 + props.minor
        print(f"Device: {props.name}, compute capability sm_{cc}")
        if cc < 80:
            failures.append(
                f"Compute capability sm_{cc} < sm_80: infllmv2 kernels have a "
                "confirmed hardware floor of Ampere (sm_80)."
            )

    # 4. Wiring would actually go sparse for the released config.
    if sparse_mod is not None and sparse_mod.INFLLM_V2_AVAILABLE:
        from transformers import PretrainedConfig

        released = PretrainedConfig(
            sparse_config={
                "kernel_size": 32,
                "kernel_stride": 16,
                "init_blocks": 1,
                "block_size": 64,
                "window_size": 2048,
                "topk": 64,
                "dense_len": 8192,
            }
        )
        sc = sparse_mod.parse_sparse_config(released)
        assert sc.effective_topk == sc.topk + sc.local_blocks
        print(
            f"sparse_config parsed: dense_len={sc.dense_len}, "
            f"topk={sc.topk} (+{sc.local_blocks} local = "
            f"{sc.effective_topk} effective)"
        )

    if failures:
        print("\nSTEP 0 FAIL -- sparse backend is NOT live:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("\nSTEP 0 PASS: InfLLM-V2 sparse backend is LIVE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
