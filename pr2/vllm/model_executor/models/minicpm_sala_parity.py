# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-SALA engine parity hooks (native RMSNorm under enforce_eager).

Installed at import time (see ``install_pr2_overlay.sh`` registry hook) so
``LLM()`` gets correct kernel priorities even when model-info cache skips
loading ``minicpm_sala.py`` in the parent process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def _is_minicpm_sala_vllm_config(vllm_config: VllmConfig) -> bool:
    model_config = vllm_config.model_config
    if model_config is None:
        return False
    archs = getattr(model_config, "architectures", None) or []
    if any("MiniCPMSALA" in a for a in archs):
        return True
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is not None:
        hf_archs = getattr(hf_config, "architectures", None) or []
        return any("MiniCPMSALA" in a for a in hf_archs)
    return False


def ensure_native_rms_norm_kernels(vllm_config: VllmConfig) -> None:
    """Pin RMSNorm to native kernels for greedy parity under enforce_eager.

    EngineCore defaults to ``vllm_c`` RMSNorm when compilation is off; that
    path drifts ~0.25 at layer-0 vs HF/get_model_loader and compounds through
    lightning layers. ``optimization_level`` may enable ``fuse_norm_quant``
    after platform defaults, so this must run at the end of ``VllmConfig``
    initialization.
    """
    for op_name in ("rms_norm", "fused_add_rms_norm"):
        setattr(vllm_config.kernel_config.ir_op_priority, op_name, ["native"])
    vllm_config.compilation_config.pass_config.fuse_norm_quant = False


def _install_vllm_config_parity_patch() -> None:
    from vllm.config import VllmConfig

    if getattr(VllmConfig, "_minicpm_sala_parity_patched", False):
        return

    _orig_post_init = VllmConfig.__post_init__

    def _patched_post_init(self: VllmConfig) -> None:
        _orig_post_init(self)
        if _is_minicpm_sala_vllm_config(self):
            ensure_native_rms_norm_kernels(self)

    VllmConfig.__post_init__ = _patched_post_init  # type: ignore[method-assign]
    VllmConfig._minicpm_sala_parity_patched = True  # type: ignore[attr-defined]


_install_vllm_config_parity_patch()
