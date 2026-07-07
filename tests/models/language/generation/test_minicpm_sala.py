# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness test for MiniCPM-SALA: HF reference vs. this vLLM port.

Supersedes the earlier standalone `scripts/minicpm_sala_differential_validation.py`
approach (custom cosine-similarity hidden-state hooks). That approach was
written before checking whether vLLM already has established
infrastructure for exactly this comparison -- it does:
`tests/conftest.py`'s `HfRunner`/`VllmRunner` fixtures plus
`tests/models/utils.py::check_logprobs_close`, the same idiom used by
every other hybrid-attention model in
`tests/models/language/generation/test_hybrid.py` (Jamba, Zamba2,
Falcon-H1, Qwen3-Next, ...). Rewritten against that real, reviewed
pattern instead of a bespoke script -- smaller, more idiomatic, and
exactly what a vLLM reviewer expects to see for a new-model PR.

STATUS: written against real, introspected fixture signatures
(`HfRunner.generate_greedy_logprobs_limit`,
`VllmRunner.generate_greedy_logprobs`, `check_logprobs_close`'s real
kwargs) -- not executed. This sandbox has no GPU and no network access
to huggingface.co for the real ~19GB of weights (see
docs/minicpm_sala_known_limitations.md). Requires a real GPU + weights
to actually run; that is the concrete next step, not something
achievable further in this environment.
"""

import pytest

from tests.models.registry import HF_EXAMPLE_MODELS

from ...utils import check_logprobs_close

pytestmark = pytest.mark.hybrid_model

MODEL = "openbmb/MiniCPM-SALA"

# Stage-1 scope: dense-regime correctness for short prompts.
# Long-context sparse-regime parity: see test_minicpm_sala_long_context.py
MAX_TOKENS = 32
NUM_LOGPROBS = 5


def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
) -> None:
    """Dense-regime correctness: HF greedy-decoding logprobs vs. this
    vLLM port's, following the exact pattern used by every other model
    in this test suite (see test_hybrid.py::test_models for the
    template this was copied from).
    """
    try:
        model_info = HF_EXAMPLE_MODELS.find_hf_info(MODEL)
        model_info.check_available_online(on_fail="skip")
        model_info.check_transformers_version(on_fail="skip")
    except ValueError:
        pass

    # HF reference needs trust_remote_code=True (custom
    # modeling_minicpm_sala.py); this vLLM port does not (it's in-tree --
    # see tests/models/registry.py's MiniCPMSALAForCausalLM entry, which
    # sets trust_remote_code=False deliberately). HfRunner is given the
    # kwarg explicitly since its default may differ from what the
    # registry entry implies for the vLLM side.
    with hf_runner(MODEL, trust_remote_code=True) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts, MAX_TOKENS, NUM_LOGPROBS
        )

    with vllm_runner(MODEL, max_model_len=4096) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, MAX_TOKENS, NUM_LOGPROBS
        )

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
