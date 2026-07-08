# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal test-model registry for MiniCPM-SALA integration overlay.

On a full vLLM fork, apply ``patches/tests_registry.py.patch`` to the upstream
``tests/models/registry.py`` instead of using this slim module.
"""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass, field
from typing import Any, Literal

import pytest
from packaging.version import Version
from transformers import __version__ as TRANSFORMERS_VERSION


@dataclass(frozen=True)
class _HfExamplesInfo:
    default: str
    extras: Mapping[str, str] = field(default_factory=dict)
    tokenizer: str | None = None
    tokenizer_mode: str = "auto"
    min_transformers_version: str | None = None
    max_transformers_version: str | None = None
    transformers_version_reason: dict[Literal["vllm", "hf"], str] | None = None
    dtype: str = "auto"
    enforce_eager: bool = False
    enable_prefix_caching: bool = True
    is_available_online: bool = True
    trust_remote_code: bool = False
    hf_overrides: dict[str, Any] = field(default_factory=dict)
    max_model_len: int | None = None

    def check_transformers_version(
        self,
        *,
        on_fail: Literal["error", "skip", "return"],
        check_version_reason: Literal["vllm", "hf"] = "hf",
        check_min_version: bool = True,
        check_max_version: bool = True,
    ) -> str | None:
        if (
            self.min_transformers_version is None
            and self.max_transformers_version is None
        ):
            return None

        current_version = TRANSFORMERS_VERSION
        cur_base_version = Version(current_version).base_version
        min_version = self.min_transformers_version
        max_version = self.max_transformers_version
        msg = f"`transformers=={current_version}` installed, but `transformers"
        if min_version and Version(cur_base_version) < Version(min_version):
            is_version_valid = False
            should_check_version = check_min_version
            msg += f">={min_version}` is required to run this model."
        elif max_version and Version(cur_base_version) > Version(max_version):
            is_version_valid = False
            should_check_version = check_max_version
            msg += f"<={max_version}` is required to run this model."
        else:
            is_version_valid = True
            should_check_version = False

        is_reason_applicable = (
            not is_version_valid
            and self.transformers_version_reason is not None
            and check_version_reason in self.transformers_version_reason
        )
        is_transformers_valid = is_version_valid or (
            not should_check_version and not is_reason_applicable
        )
        if is_transformers_valid:
            return None
        elif self.transformers_version_reason:
            for reason_type, reason in self.transformers_version_reason.items():
                msg += f" Reason({reason_type}): {reason}"

        if on_fail == "error":
            raise RuntimeError(msg)
        if on_fail == "skip":
            pytest.skip(msg)
        return msg

    def check_available_online(
        self,
        *,
        on_fail: Literal["error", "skip"],
    ) -> None:
        if not self.is_available_online:
            msg = "Model is not available online"
            if on_fail == "error":
                raise RuntimeError(msg)
            pytest.skip(msg)


_TEXT_GENERATION_EXAMPLE_MODELS = {
    "MiniCPMSALAForCausalLM": _HfExamplesInfo(
        "openbmb/MiniCPM-SALA",
        trust_remote_code=False,
        max_model_len=4096,
    ),
}


class HfExampleModels:
    def __init__(self, hf_models: Mapping[str, _HfExamplesInfo]) -> None:
        self.hf_models = hf_models

    def get_supported_archs(self) -> Set[str]:
        return self.hf_models.keys()

    def get_hf_info(self, model_arch: str) -> _HfExamplesInfo:
        try:
            return self.hf_models[model_arch]
        except KeyError as exc:
            raise ValueError(
                f"No example model defined for {model_arch}; please update this file."
            ) from exc

    def find_hf_info(self, model_id: str) -> _HfExamplesInfo:
        for info in self.hf_models.values():
            if info.default == model_id:
                return info
        for info in self.hf_models.values():
            if any(extra == model_id for extra in info.extras.values()):
                return info
        raise ValueError(
            f"No example model defined for {model_id}; please update this file."
        )


HF_EXAMPLE_MODELS = HfExampleModels(_TEXT_GENERATION_EXAMPLE_MODELS)
