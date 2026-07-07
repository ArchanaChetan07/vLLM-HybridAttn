# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Long-context (sparse-regime) HF vs vLLM logprob parity.

Requires GPU + ~19GB weights. Exercises minicpm4 layers past dense_len=8192.
Run via vLLM test harness: pytest tests/models/language/generation/test_minicpm_sala_long_context.py
"""

import pytest

from tests.models.registry import HF_EXAMPLE_MODELS

from ...utils import check_logprobs_close

pytestmark = [pytest.mark.hybrid_model, pytest.mark.gpu_models]

MODEL = "openbmb/MiniCPM-SALA"
DENSE_LEN = 8192
MAX_TOKENS = 8
NUM_LOGPROBS = 5
LONG_INPUT_TOKENS = DENSE_LEN + 256


@pytest.fixture
def long_sparse_prompt() -> list[int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    chunk = tok.encode(
        "The quick brown fox jumps over the lazy dog. ", add_special_tokens=False
    )
    if not chunk:
        pytest.skip("tokenizer returned empty chunk")
    reps = LONG_INPUT_TOKENS // len(chunk) + 2
    return (chunk * reps)[:LONG_INPUT_TOKENS]


def test_sparse_regime_logprobs(
    hf_runner,
    vllm_runner,
    long_sparse_prompt: list[int],
) -> None:
    try:
        model_info = HF_EXAMPLE_MODELS.find_hf_info(MODEL)
        model_info.check_available_online(on_fail="skip")
        model_info.check_transformers_version(on_fail="skip")
    except ValueError:
        pytest.skip("Model unavailable")

    prompts = [long_sparse_prompt]
    max_model_len = LONG_INPUT_TOKENS + MAX_TOKENS + 32

    with hf_runner(
        MODEL, trust_remote_code=True, max_model_len=max_model_len
    ) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            prompts, MAX_TOKENS, NUM_LOGPROBS
        )

    with vllm_runner(MODEL, max_model_len=max_model_len) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            prompts, MAX_TOKENS, NUM_LOGPROBS
        )

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
