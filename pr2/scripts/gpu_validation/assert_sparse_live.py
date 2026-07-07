#!/usr/bin/env python3
"""Step 0 gate: sparse path must be live before any validation is trusted."""
import sys

def main() -> int:
    try:
        from vllm.v1.attention.backends.minicpm_sala_sparse import (
            INFLLM_V2_AVAILABLE,
            infllmv2_attn_varlen_func,
        )
    except ImportError as e:
        print(f"FAIL: cannot import minicpm_sala_sparse backend: {e}")
        return 1

    if not INFLLM_V2_AVAILABLE:
        print("FAIL: INFLLM_V2_AVAILABLE is False — infllm_v2 not installed")
        return 1

    try:
        import infllm_v2  # noqa: F401
    except ImportError as e:
        print(f"FAIL: infllm_v2 import failed: {e}")
        return 1

    if infllmv2_attn_varlen_func is None:
        print("FAIL: infllmv2_attn_varlen_func is None")
        return 1

    from vllm.model_executor.models.minicpm_sala_sparse_wiring import (
        create_sparse_attention_if_available,
    )
    import inspect
    src = inspect.getsource(create_sparse_attention_if_available)
    if "if not INFLLM_V2_AVAILABLE" in src and "return None" in src:
        print("FAIL: create_sparse_attention_if_available silently returns None when infllm_v2 missing")
        return 1
    if "if not INFLLM_V2_AVAILABLE" in src and "raise RuntimeError" not in src and "raise ImportError" not in src:
        print("FAIL: create_sparse_attention_if_available does not fail loud when infllm_v2 missing")
        return 1

    print("PASS: sparse path LIVE (INFLLM_V2_AVAILABLE=True, infllm_v2 importable, fail-loud wiring)")
    return 0

if __name__ == "__main__":
    sys.exit(main())