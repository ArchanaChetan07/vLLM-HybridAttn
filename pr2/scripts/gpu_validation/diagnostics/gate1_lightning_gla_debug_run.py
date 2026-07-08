#!/usr/bin/env python3
"""Instrumented GLA decode bisect — writes NDJSON to debug-212a6e.log.

Run on A100 after overlay install::

  export MINICPM_SALA_WEIGHTS=/workspace/models/openbmb/MiniCPM-SALA
  export VLLM_ALLOW_INSECURE_SERIALIZATION=1
  export MINICPM_SALA_DEBUG_GLA=1
  export DEBUG_LOG_PATH=/workspace/hybridattn/debug-212a6e.log
  cd /workspace/hybridattn
  bash scripts/install_pr2_overlay.sh
  pkill -9 -f 'EngineCore|VLLM::' 2>/dev/null || true
  python3 pr2/scripts/gpu_validation/diagnostics/gate1_lightning_gla_debug_run.py
"""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = os.environ.get("MINICPM_SALA_PROMPT", "Hello, my name is")
MAX_STEP = int(os.environ.get("MINICPM_SALA_MAX_STEP", "15"))
LOG_PATH = os.environ.get(
    "DEBUG_LOG_PATH",
    str(Path(__file__).resolve().parents[5] / "debug-212a6e.log"),
)


def _log(message: str, data: dict, hypothesis_id: str = "runner") -> None:
    payload = {
        "sessionId": "212a6e",
        "runId": os.environ.get("DEBUG_RUN_ID", "pre-fix"),
        "hypothesisId": hypothesis_id,
        "location": "gate1_lightning_gla_debug_run.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def hf_greedy(prompt_ids: list[int], steps: int) -> list[int]:
    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    cur = prompt_ids[:]
    out: list[int] = []
    with torch.no_grad():
        for _ in range(steps):
            nxt = int(model(torch.tensor([cur], device="cuda")).logits[0, -1].argmax())
            out.append(nxt)
            cur.append(nxt)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main() -> int:
    os.environ["MINICPM_SALA_DEBUG_GLA"] = "1"
    os.environ.setdefault("DEBUG_RUN_ID", "post-fix")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    _log(
        "run_start",
        {"log_path": LOG_PATH, "max_step": MAX_STEP, "prompt": PROMPT},
        "runner",
    )

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    prompt_ids = tok.encode(PROMPT, add_special_tokens=True)
    hf = hf_greedy(prompt_ids, MAX_STEP)
    _log("hf_greedy", {"prompt_len": len(prompt_ids), "hf": hf}, "runner")

    llm = LLM(
        model=WEIGHTS,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
        block_size=256,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        max_num_seqs=1,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        enable_chunked_prefill=False,
    )

    for step in range(MAX_STEP):
        out = llm.generate(
            [TokensPrompt(prompt_token_ids=prompt_ids)],
            SamplingParams(temperature=0, max_tokens=step + 1),
        )[0].outputs[0].token_ids
        vllm_t = int(out[step])
        hf_t = hf[step]
        _log(
            "incremental_step",
            {
                "step": step,
                "decode_idx": step,
                "seq_len": len(prompt_ids) + step + 1,
                "vllm_token": vllm_t,
                "hf_token": hf_t,
                "match": vllm_t == hf_t,
            },
            "runner",
        )
        print(
            f"step={step} seq_len={len(prompt_ids)+step+1} "
            f"hf={hf_t} vllm={vllm_t} ok={vllm_t == hf_t}",
            flush=True,
        )
        if vllm_t != hf_t:
            break

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    _log("run_end", {}, "runner")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
