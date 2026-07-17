#!/usr/bin/env python3
"""Engine-level smoke test: load real weights under `LLM()` and generate.

This is the fastest end-to-end sanity check between the unit/kernel gates
(Steps 0-6) and full HF parity (Step B): it exercises weight loading, the
registry, the hybrid KV-cache planner, and greedy decoding through the real
vLLM engine -- first in the dense regime (short prompts), then, with
--long, in the sparse regime (a >= 8192-token prompt ending in a reading-
comprehension question whose answer requires retrieving from the long
context through InfLLM-V2 sparse attention).

First green run: 2026-07-17, A100-SXM4-80GB, vLLM 0.25.0 --
  "The capital of France is" -> " Paris."
  8418-token prompt, "What animal jumps over the dog?" -> " The fox."

Env:
  MINICPM_SALA_WEIGHTS  path or hub id (default: openbmb/MiniCPM-SALA)

Usage:
  python engine_smoke.py          # short prompts (dense regime)
  python engine_smoke.py --long   # 8418-token prompt (sparse regime)
"""

import os
import sys
import time


def main() -> int:
    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    model = os.environ.get("MINICPM_SALA_WEIGHTS", "openbmb/MiniCPM-SALA")
    run_long = "--long" in sys.argv

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    t0 = time.time()
    llm = LLM(
        model=model,
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        # infllm_v2 paged-KV floor (the sparse wiring refuses the default 16).
        block_size=256,
        max_model_len=16384,
    )
    print(f"LOAD OK in {time.time() - t0:.0f}s", flush=True)

    if run_long:
        base = tok.encode(
            "The quick brown fox jumps over the lazy dog. ",
            add_special_tokens=False,
        )
        ids = [tok.bos_token_id or 1] + base * (8400 // len(base) + 1)
        ids = ids[:8500] + tok.encode(
            "\nQuestion: What animal jumps over the dog? Answer:",
            add_special_tokens=False,
        )
        prompts = {f"<long {len(ids)} tokens>": ids}
    else:
        texts = ["Hello, my name is", "The capital of France is"]
        prompts = {t: tok.encode(t, add_special_tokens=True) for t in texts}

    outs = llm.generate(
        [TokensPrompt(prompt_token_ids=i) for i in prompts.values()],
        SamplingParams(temperature=0.0, max_tokens=16 if not run_long else 24),
    )
    ok = True
    for label, out in zip(prompts, outs):
        comp = out.outputs[0]
        print(f"PROMPT {label!r} -> tokens {list(comp.token_ids)}", flush=True)
        print(f"  text: {comp.text!r}", flush=True)
        ok &= len(comp.token_ids) > 0
    print("ENGINE SMOKE " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
