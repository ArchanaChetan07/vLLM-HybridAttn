# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal ``check_logprobs_close`` shim for MiniCPM-SALA overlay tests.

Copied from upstream ``tests/models/utils.py`` (vLLM) with only the helper
needed by ``test_minicpm_sala.py``. On a full vLLM fork, use the upstream
module instead of this slim copy.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

TokensTextLogprobs = tuple[list[int], str, list[dict[int, float]] | Any | None]
TextTextLogprobs = tuple[list[str], str, list[dict[str, float]] | list[dict[str, Any]] | None]
TokensTextLogprobsPromptLogprobs = tuple[
    list[int],
    str,
    list[dict[int, float]] | Any | None,
    list[dict[int, float] | None] | Any | None,
]


def check_logprobs_close(
    *,
    outputs_0_lst: Sequence[
        TokensTextLogprobs | TokensTextLogprobsPromptLogprobs | TextTextLogprobs
    ],
    outputs_1_lst: Sequence[
        TokensTextLogprobs | TokensTextLogprobsPromptLogprobs | TextTextLogprobs
    ],
    name_0: str,
    name_1: str,
    num_outputs_0_skip_tokens: int = 0,
    warn_on_mismatch: bool = True,
    always_check_logprobs: bool = False,
) -> None:
    """Compare greedy-decoding logprobs from two model runners."""
    assert len(outputs_0_lst) == len(outputs_1_lst)

    for prompt_idx, (outputs_0, outputs_1) in enumerate(
        zip(outputs_0_lst, outputs_1_lst)
    ):
        assert len(outputs_0) == len(outputs_1)
        if len(outputs_0) == 3:
            assert len(outputs_1) == 3
            output_ids_0, output_str_0, logprobs_0 = outputs_0
            output_ids_1, output_str_1, logprobs_1 = outputs_1
        elif len(outputs_0) == 4:
            assert len(outputs_1) == 4
            (
                output_ids_0,
                output_str_0,
                logprobs_0,
                prompt_logprobs_0,
            ) = outputs_0
            (
                output_ids_1,
                output_str_1,
                logprobs_1,
                prompt_logprobs_1,
            ) = outputs_1

            if prompt_logprobs_0 is not None and prompt_logprobs_1 is not None:
                for idx, (logprobs_elem_0, logprobs_elem_1) in enumerate(
                    zip(prompt_logprobs_0, prompt_logprobs_1)
                ):
                    fail_msg = (
                        f"Prompt logprobs test:"
                        f"\n{name_0}:\tPrompt index {idx}\t{logprobs_elem_0}"
                        f"\n{name_1}:\tPrompt index {idx}\t{logprobs_elem_1}"
                    )
                    if logprobs_elem_0 is None:
                        assert logprobs_elem_1 is None, fail_msg
                    else:
                        assert logprobs_elem_1 is not None, fail_msg
                        assert set(logprobs_elem_0.keys()) == set(
                            logprobs_elem_1.keys()
                        ), fail_msg
            else:
                fail_msg = (
                    f"Prompt logprobs test:"
                    f"\n{name_0}:\tlogprobs\t{prompt_logprobs_0}"
                    f"\n{name_1}:\tlogprobs\t{prompt_logprobs_1}"
                )
                assert prompt_logprobs_0 is None and prompt_logprobs_1 is None, fail_msg
        else:
            raise ValueError(
                f"Outputs tuple must have 3 or 4 elements but "
                f"{len(outputs_0)} elements were provided: {outputs_0}"
            )

        if logprobs_0 is None:
            logprobs_0 = [None] * len(output_ids_0)
        if logprobs_1 is None:
            logprobs_1 = [None] * len(output_ids_1)

        if num_outputs_0_skip_tokens < 0:
            raise ValueError("num_outputs_0_skip_tokens must be non-negative")
        output_ids_0 = output_ids_0[num_outputs_0_skip_tokens:]
        logprobs_0 = logprobs_0[num_outputs_0_skip_tokens:]

        for idx, (output_id_0, output_id_1) in enumerate(
            zip(output_ids_0, output_ids_1)
        ):
            is_tok_mismatch = output_id_0 != output_id_1
            if is_tok_mismatch or always_check_logprobs:
                logprobs_elem_0 = logprobs_0[idx]
                logprobs_elem_1 = logprobs_1[idx]
                fail_msg = (
                    f"Test{prompt_idx}:"
                    f"\nMatched tokens:\t{output_ids_0[:idx]}"
                    f"\n{name_0}:\t{output_str_0!r}\t{logprobs_elem_0}"
                    f"\n{name_1}:\t{output_str_1!r}\t{logprobs_elem_1}"
                )
                assert logprobs_elem_0 is not None, fail_msg
                assert logprobs_elem_1 is not None, fail_msg
                assert output_id_0 in logprobs_elem_1, fail_msg
                assert output_id_1 in logprobs_elem_0, fail_msg

                if warn_on_mismatch and is_tok_mismatch:
                    with warnings.catch_warnings():
                        warnings.simplefilter("always")
                        warnings.warn(fail_msg, stacklevel=2)
                break
        else:
            if output_str_0 != output_str_1 and warn_on_mismatch:
                fail_msg = (
                    f"Test{prompt_idx}:"
                    f"\n{name_0}:\t{output_str_0!r}"
                    f"\n{name_1}:\t{output_str_1!r}"
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("always")
                    warnings.warn(fail_msg, stacklevel=2)
