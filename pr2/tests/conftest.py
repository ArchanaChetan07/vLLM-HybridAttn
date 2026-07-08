# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expose upstream vLLM test harness fixtures to PR2 tests.

PR2 tests live under ``pr2/tests`` and are often executed with
``PYTHONPATH=pr2:...``. This file bridges to the upstream vLLM test harness
(``vllm_ref/tests/conftest.py``) lazily so that PR2-only test runs can still
use fixtures like ``hf_runner`` and ``vllm_runner`` without importing the full
upstream test suite at collection time.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_UPSTREAM_MOD: Any | None = None


def _resolve_upstream_conftest() -> Path | None:
    candidates: list[Path] = []
    env_root = os.environ.get("VLLM_REF_ROOT", "").strip()
    if env_root:
        candidates.append(Path(env_root))

    # Common layouts:
    # - <repo_root>/../vllm_ref (this repo vendored beside vllm_ref)
    # - <repo_root>/vllm_ref (vllm_ref vendored inside repo)
    candidates.extend([_REPO_ROOT.parent / "vllm_ref", _REPO_ROOT / "vllm_ref"])

    for root in candidates:
        cf = root.resolve() / "tests" / "conftest.py"
        if cf.is_file():
            return cf
    return None


def _load_upstream_conftest() -> Any | None:
    global _UPSTREAM_MOD
    if _UPSTREAM_MOD is not None:
        return _UPSTREAM_MOD

    cf = _resolve_upstream_conftest()
    if cf is None:
        return None

    vllm_root = cf.parents[1]
    repo_s = os.path.normpath(str(_REPO_ROOT.resolve()))
    vllm_s = str(vllm_root.resolve())

    # Prevent our slim `tests.*` shim from shadowing upstream `tests.*` imports.
    saved_modules = {
        k: sys.modules[k]
        for k in list(sys.modules)
        if k == "tests" or k.startswith("tests.")
    }
    for k in saved_modules:
        del sys.modules[k]

    old_path = sys.path[:]
    sys.path = [p for p in sys.path if os.path.normpath(p) != repo_s]
    if vllm_s not in sys.path:
        sys.path.insert(0, vllm_s)

    mod_name = "_minicpm_sala_pr2_vllm_ref_conftest"
    try:
        spec = importlib.util.spec_from_file_location(mod_name, cf)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        _UPSTREAM_MOD = mod
        return mod
    except Exception:
        return None
    finally:
        sys.path[:] = old_path
        for k, v in saved_modules.items():
            sys.modules[k] = v


def _require_upstream():
    mod = _load_upstream_conftest()
    if mod is None:
        pytest.skip(
            "vLLM test harness unavailable: clone vllm_ref beside the repo "
            "or set VLLM_REF_ROOT, then ensure vllm_ref is on PYTHONPATH"
        )
    return mod


@pytest.fixture(scope="session")
def hf_runner():
    return _require_upstream().HfRunner


@pytest.fixture(scope="session")
def vllm_runner():
    return _require_upstream().VllmRunner


@pytest.fixture(scope="session")
def example_prompts():
    return _require_upstream().example_prompts()

