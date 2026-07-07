#!/usr/bin/env python3
"""Compare HF vs vLLM layer-0 output via real LLM generate hooks."""

from __future__ import annotations

import gc
import os
import sys

import torch

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = "Hello, my name is"


def main() -> int:
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
        emb = hf.model.embed_tokens(ids) * hf.config.scale_emb
        pos = torch.arange(ids.shape[1], device="cuda").unsqueeze(0)
        h0_hf = hf.model.layers[0](
            emb, attention_mask=torch.ones_like(ids), position_ids=pos, use_cache=False
        )[0][0, -1].float().cpu()
    del hf
    gc.collect()
    torch.cuda.empty_cache()

    # vLLM: compare first-token logits only; layer0 hidden needs worker hook.
    # Use direct model load instead.
    import vllm.config as vconfig
    import tempfile
    from vllm.config import CacheConfig, ModelConfig, VllmConfig
    from vllm.config.device import DeviceConfig
    from vllm.config.load import LoadConfig
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.model_loader import get_model_loader

    seq_len = ids.shape[1]
    positions = torch.arange(seq_len, device="cuda", dtype=torch.long)
    model_config = ModelConfig(
        model=WEIGHTS, trust_remote_code=True, dtype="bfloat16", max_model_len=4096
    )
    load_config = LoadConfig()
    vllm_config = VllmConfig(
        model_config=model_config,
        load_config=load_config,
        cache_config=CacheConfig(block_size=256),
        device_config=DeviceConfig(device="cuda"),
    )
    fd, temp = tempfile.mkstemp()
    os.close(fd)
    h0_v = None
    try:
        with vconfig.set_current_vllm_config(vllm_config, check_compile=False):
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{temp}",
                local_rank=0,
                backend="nccl",
            )
            initialize_model_parallel(1, 1)
            vm = get_model_loader(load_config).load_model(
                vllm_config=vllm_config, model_config=model_config
            )
            vm.eval().cuda()
            # Need attention metadata for sparse layer0 — use LLM engine instead
            del vm
            destroy_model_parallel()
            destroy_distributed_environment()
    finally:
        os.unlink(temp)

    llm = LLM(
        model=WEIGHTS,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
        block_size=256,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
    )
    out = llm.generate([PROMPT], SamplingParams(temperature=0, max_tokens=1))[0]
    print("vLLM greedy", int(out.outputs[0].token_ids[0]))
    print("HF layer0 last norm", h0_hf.norm().item())
    del llm
    return 0


if __name__ == "__main__":
    sys.exit(main())
