#!/usr/bin/env python3
import os

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
os.environ.setdefault("MINICPM_SALA_DEBUG_GLA", "1")
os.environ.setdefault("DEBUG_LOG_PATH", "/tmp/gla_debug.log")

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

W = os.environ.get("MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA")
tok = AutoTokenizer.from_pretrained(W, trust_remote_code=True)
p = tok.encode("Hello, my name is", add_special_tokens=True)
llm = LLM(
    model=W,
    trust_remote_code=True,
    dtype="bfloat16",
    max_model_len=4096,
    block_size=256,
    gpu_memory_utilization=0.45,
    enforce_eager=True,
    max_num_seqs=1,
    enable_prefix_caching=False,
    mamba_cache_mode="none",
    enable_chunked_prefill=False,
)
out = llm.generate(
    [TokensPrompt(prompt_token_ids=p)], SamplingParams(temperature=0, max_tokens=15)
)[0].outputs[0].token_ids
print("out14", out[14] if len(out) > 14 else None)
