#!/usr/bin/env python3
"""Gate 1: HF vs vLLM top-10 next-token logprobs."""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import tempfile

import torch

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = os.environ.get("MINICPM_SALA_PROMPT", "Hello, my name is")


def main() -> int:
    script = "/workspace/hybridattn/scripts/remote/patch_hf_transformers_compat.py"
    if os.path.isfile(script):
        subprocess.run([sys.executable, script], check=False)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids = tok.encode(PROMPT, return_tensors="pt").to("cuda")

    hf = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    with torch.no_grad():
        hf_logits = hf(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[
            0, -1
        ].float()
    hf_top = torch.topk(hf_logits, 10)
    print(
        "HF top10",
        [(int(i), float(v)) for i, v in zip(hf_top.indices, hf_top.values)],
    )
    del hf
    gc.collect()
    torch.cuda.empty_cache()

    llm = LLM(
        model=WEIGHTS,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
        block_size=256,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
    )
    sp = SamplingParams(temperature=0, max_tokens=1, logprobs=20)
    out = llm.generate([PROMPT], sp)[0]
    lps = out.outputs[0].logprobs[0]
    top = sorted(
        lps.items(),
        key=lambda kv: float(kv[1].logprob if hasattr(kv[1], "logprob") else kv[1]),
        reverse=True,
    )[:10]
    print(
        "vLLM top10",
        [
            (int(k), float(v.logprob if hasattr(v, "logprob") else v))
            for k, v in top
        ],
    )
    print("vLLM greedy", int(out.outputs[0].token_ids[0]))
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    import vllm.config as vconfig
    from vllm.config import CacheConfig, ModelConfig, VllmConfig
    from vllm.config.device import DeviceConfig
    from vllm.config.load import LoadConfig
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.model_loader import get_model_loader

    mc = ModelConfig(
        model=WEIGHTS, trust_remote_code=True, dtype="bfloat16", max_model_len=2048
    )
    lc = LoadConfig()
    cc = CacheConfig(block_size=256)
    vc = VllmConfig(
        model_config=mc,
        load_config=lc,
        cache_config=cc,
        device_config=DeviceConfig(device="cuda"),
    )
    fd, tf = tempfile.mkstemp()
    os.close(fd)
    try:
        with vconfig.set_current_vllm_config(vc, check_compile=False):
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{tf}",
                local_rank=0,
                backend="nccl",
            )
            initialize_model_parallel(1, 1)
            m = get_model_loader(lc).load_model(vllm_config=vc, model_config=mc)
            lh = dict(m.named_parameters())["lm_head.weight"].float()
            print("lm_head shape", tuple(lh.shape))
            print("lm_head padded tail max_abs", lh[73448:73472].abs().max().item())
            destroy_model_parallel()
            destroy_distributed_environment()
    finally:
        os.unlink(tf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
