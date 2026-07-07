#!/usr/bin/env python3
"""Quick bisect: chunked prefill on/off for greedy first token."""

from __future__ import annotations

import gc
import os
import subprocess
import sys

import torch
from vllm import LLM, SamplingParams

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = "Hello, my name is"


def main() -> int:
    script = "/workspace/hybridattn/scripts/remote/patch_hf_transformers_compat.py"
    if os.path.isfile(script):
        subprocess.run([sys.executable, script], check=False)

    from transformers import AutoConfig

    c = AutoConfig.from_pretrained(WEIGHTS, trust_remote_code=True)
    print(
        "scale_depth",
        c.scale_depth,
        "dim_model_base",
        c.dim_model_base,
        "hidden",
        c.hidden_size,
    )
    print("mixer0", c.mixer_types[0], "mixer1", c.mixer_types[1])

    for chunked in (True, False):
        llm = LLM(
            model=WEIGHTS,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=4096,
            block_size=256,
            gpu_memory_utilization=0.45,
            enforce_eager=True,
            enable_chunked_prefill=chunked,
        )
        sp = SamplingParams(temperature=0, max_tokens=1)
        g = llm.generate([PROMPT], sp)[0].outputs[0].token_ids[0]
        print("chunked_prefill", chunked, "greedy", int(g))
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
