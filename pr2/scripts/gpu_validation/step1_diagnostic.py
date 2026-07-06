#!/usr/bin/env python3
"""Step 1: run this FIRST on your T1000. Confirms the environment, deploys
minicpm_sala.py into a real vllm install, and introspects the exact real
signatures needed for step 2 (LinearAttentionMetadata's fields, whether
vllm._C compiled kernels are available). Paste the full output back --
step 2 will be written against what this actually reports, not guessed.

Setup (run before this script):
    pip install vllm          # full install; a real GPU box should pull
                               # the CUDA wheel cleanly, unlike the
                               # CPU-only sandbox this was developed in
    pip install einops

Then copy vllm/model_executor/models/minicpm_sala.py from the delivered
zip to wherever `python3 -c "import vllm.model_executor.models, os;
print(os.path.dirname(vllm.model_executor.models.__file__))"` reports,
and run:
    python3 step1_diagnostic.py
"""

import dataclasses
import inspect
import sys


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> int:
    section("1. Python / CUDA basics")
    print(f"Python: {sys.version}")
    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Device: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"Total VRAM: {props.total_memory / 1e9:.2f} GB")
            print(f"Compute capability: {props.major}.{props.minor}")
        else:
            print(
                "!! CUDA not available -- stop here, fix the torch/driver "
                "install before continuing."
            )
            return 1
    except ImportError as e:
        print(f"!! torch not installed: {e}")
        return 1

    section("2. vLLM import + compiled kernels")
    try:
        import vllm

        print(f"vllm: {vllm.__version__}")
    except ImportError as e:
        print(f"!! vllm not installed: {e}")
        return 1

    try:
        # FIXED (real bug found on real GPU hardware, T1000, first live
        # run of this script): checked `vllm._C`, which no longer exists
        # in this vLLM version -- the compiled extension was renamed to
        # `vllm._C_stable_libtorch` (confirmed by grepping
        # vllm/platforms/cuda.py directly: `import vllm._C_stable_libtorch`
        # is the real, current import site). The original check's
        # "missing" result was itself real information (a stale check
        # genuinely fails to find a module that was renamed), but it
        # doesn't mean compiled kernels are actually absent -- checking
        # the real name instead.
        import vllm._C_stable_libtorch  # noqa: F401

        print("vllm._C_stable_libtorch (compiled CUDA kernels): IMPORTS OK")
    except ImportError as e:
        print(f"!! vllm._C_stable_libtorch missing: {e}")
        print(
            "   This means the pip wheel didn't include compiled "
            "kernels for this platform -- may need to build from "
            "source, or check CUDA version compatibility."
        )

    section("3. Platform detection")
    from vllm.platforms import current_platform

    print(f"current_platform: {current_platform}")
    print(f"device_type: {getattr(current_platform, 'device_type', 'N/A')}")

    section("4. minicpm_sala.py import")
    try:
        import vllm.model_executor.models.minicpm_sala as m

        print("IMPORT OK")
        print(f"Classes: {[n for n in dir(m) if n.startswith('MiniCPM')]}")
    except Exception as e:
        print(f"!! IMPORT FAILED: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return 1

    section("5. LinearAttentionMetadata -- real fields (needed for step 2)")
    from vllm.v1.attention.backends.linear_attn import LinearAttentionMetadata

    if dataclasses.is_dataclass(LinearAttentionMetadata):
        for f in dataclasses.fields(LinearAttentionMetadata):
            print(
                f"  {f.name}: {f.type}"
                + (f" = {f.default}" if f.default is not dataclasses.MISSING else "")
            )
    else:
        print(inspect.signature(LinearAttentionMetadata.__init__))

    section("6. Attention backend resolution (the thing that failed on CPU)")
    try:
        from vllm.v1.attention.selector import get_attn_backend

        # FIXED, in two real steps, both confirmed by actually running
        # this script on real GPU hardware (T1000):
        # (1) the original call included `block_size=16`, which is NOT a
        #     real parameter of get_attn_backend -- confirmed against
        #     its actual signature (head_size, dtype, kv_cache_dtype,
        #     use_mla=False, has_sink=False, use_sparse=False,
        #     use_mm_prefix=False, use_per_head_quant_scales=False,
        #     attn_type=None, num_heads=None).
        # (2) get_attn_backend() internally calls
        #     get_current_vllm_config() (confirmed by reading
        #     vllm/v1/attention/selector.py directly) -- it genuinely
        #     requires an active set_current_vllm_config(...) context,
        #     which the original probe didn't provide. Found on the
        #     first live GPU run, fixed by wrapping the call as below.
        from vllm.config import VllmConfig, set_current_vllm_config

        with set_current_vllm_config(VllmConfig()):
            backend = get_attn_backend(
                head_size=128,
                dtype=torch.bfloat16,
                kv_cache_dtype="auto",
            )
        print(f"Resolved backend: {backend}")
    except Exception as e:
        print(
            f"get_attn_backend probe failed (may need more args -- "
            f"non-fatal, just diagnostic): {type(e).__name__}: {e}"
        )

    section("DONE -- paste this entire output back")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
