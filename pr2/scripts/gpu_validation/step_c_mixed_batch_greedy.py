#!/usr/bin/env python3
"""Step C: greedy batch-invariance through full vLLM (short + long)."""

import os
import sys

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
HF_REPO = os.environ.get("MINICPM_SALA_HF_REPO", "openbmb/MiniCPM-SALA")
LONG_PROMPT_TOKENS = 8200
MAX_TOKENS = 8


def _ensure_weights(path: str) -> bool:
    if os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json")):
        return True
    if os.environ.get("MINICPM_SALA_DOWNLOAD_WEIGHTS", "").lower() not in (
        "1",
        "true",
        "yes",
    ):
        return False
    from huggingface_hub import snapshot_download

    print(f"Downloading {HF_REPO} -> {path}", flush=True)
    snapshot_download(HF_REPO, local_dir=path)
    return os.path.isdir(path)


def main() -> int:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    if not _ensure_weights(WEIGHTS):
        print(f"FAIL: weights not found at {WEIGHTS}", flush=True)
        return 1

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    short = "Hello, my name is"
    chunk = tok.encode("The quick brown fox jumps. ", add_special_tokens=False)
    long_ids = (chunk * (LONG_PROMPT_TOKENS // len(chunk) + 2))[:LONG_PROMPT_TOKENS]

    llm = LLM(
        model=WEIGHTS,
        trust_remote_code=False,
        dtype="bfloat16",
        max_model_len=9000,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
    )
    sp = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)

    solo_short = list(llm.generate([short], sp)[0].outputs[0].token_ids)
    solo_long = list(
        llm.generate(prompt_token_ids=[long_ids], sampling_params=sp)[0]
        .outputs[0]
        .token_ids
    )

    batch = llm.generate(
        [short, TokensPrompt(prompt_token_ids=long_ids)],
        sp,
    )
    batch_short = list(batch[0].outputs[0].token_ids)
    batch_long = list(batch[1].outputs[0].token_ids)

    ok = True
    if solo_short != batch_short:
        ok = False
        print(f"FAIL short: solo={solo_short} batch={batch_short}", flush=True)
    else:
        print(f"PASS short: {solo_short}", flush=True)
    if solo_long != batch_long:
        ok = False
        print(f"FAIL long: solo={solo_long} batch={batch_long}", flush=True)
    else:
        print(f"PASS long: {solo_long}", flush=True)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
