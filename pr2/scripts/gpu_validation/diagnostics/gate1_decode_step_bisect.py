#!/usr/bin/env python3
"""Compare HF vs vLLM at exact decode mismatch prefix (single engine load)."""

from __future__ import annotations

import gc
import os
import subprocess
import sys

import torch

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = os.environ.get("MINICPM_SALA_PROMPT", "Hello, my name is")
MISMATCH_STEP = int(os.environ.get("MINICPM_SALA_MISMATCH_STEP", "14"))


def _patch_hf() -> None:
    script = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "remote",
        "patch_hf_transformers_compat.py",
    )
    script = os.path.normpath(script)
    if os.path.isfile(script):
        subprocess.run([sys.executable, script], check=False)


def _hf_ref(ids: list[int]) -> tuple[int, torch.Tensor]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    with torch.no_grad():
        out = model(
            torch.tensor([ids], device="cuda"),
            attention_mask=torch.ones(1, len(ids), device="cuda"),
        )
        logits = out.logits[0, -1].float()
        greedy = int(logits.argmax().item())
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return greedy, logits


def main() -> int:
    _patch_hf()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    prompt_ids = tok.encode(PROMPT, add_special_tokens=True)

    # HF greedy prefix through mismatch-1
    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    cur = prompt_ids[:]
    for _ in range(MISMATCH_STEP):
        with torch.no_grad():
            nxt = int(
                model(
                    torch.tensor([cur], device="cuda"),
                    attention_mask=torch.ones(1, len(cur), device="cuda"),
                )
                .logits[0, -1]
                .argmax()
                .item()
            )
        cur.append(nxt)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    prefix = cur
    hf_greedy, hf_logits = _hf_ref(prefix)
    print(
        f"prefix_len={len(prefix)} mismatch_step={MISMATCH_STEP} "
        f"hf_greedy={hf_greedy}",
        flush=True,
    )

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    llm = LLM(
        model=WEIGHTS,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=max(len(prefix) + 8, 4096),
        block_size=256,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        max_num_seqs=1,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        enable_chunked_prefill=False,
    )

    # One-shot generate from prefix (tests carried KV/lightning state)
    one_shot = int(
        llm.generate(
            [TokensPrompt(prompt_token_ids=prefix)],
            SamplingParams(temperature=0, max_tokens=1),
        )[0]
        .outputs[0]
        .token_ids[0]
    )

    # Fresh generate from prompt only for mismatch_step+1 tokens
    fresh = list(
        llm.generate(
            [TokensPrompt(prompt_token_ids=prompt_ids)],
            SamplingParams(temperature=0, max_tokens=MISMATCH_STEP + 1),
        )[0]
        .outputs[0]
        .token_ids
    )

    print(f"vllm_one_shot_from_prefix={one_shot}", flush=True)
    print(
        f"vllm_fresh_gen[{MISMATCH_STEP}]={fresh[MISMATCH_STEP] if len(fresh) > MISMATCH_STEP else -1}",
        flush=True,
    )
    print(f"hf_logits_top5={hf_logits.topk(5).indices.tolist()}", flush=True)
    print(f"match_one_shot={one_shot == hf_greedy}", flush=True)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return 0 if one_shot == hf_greedy else 1


if __name__ == "__main__":
    raise SystemExit(main())
