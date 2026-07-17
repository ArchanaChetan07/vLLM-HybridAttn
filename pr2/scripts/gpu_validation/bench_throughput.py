#!/usr/bin/env python3
"""Honest throughput measurements for the MiniCPM-SALA port.

Implements the core scenarios of docs/minicpm_sala_benchmark_plan.md.
Prints a markdown table for docs/performance.md. Numbers are wall-clock
over `LLM.generate` (greedy), after a warmup call; enforce_eager (no CUDA
graphs), correctness-first kernels -- these are BASELINE numbers for the
unoptimized port, not a leaderboard entry.

Env: MINICPM_SALA_WEIGHTS (default openbmb/MiniCPM-SALA)
"""

import os
import sys
import time


def main() -> int:
    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    model = os.environ.get("MINICPM_SALA_WEIGHTS", "openbmb/MiniCPM-SALA")
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    llm = LLM(
        model=model,
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        block_size=256,
        max_model_len=16384,
    )

    filler = tok.encode(
        "The quick brown fox jumps over the lazy dog. ", add_special_tokens=False
    )

    def make_prompt(n_tokens: int) -> list[int]:
        ids = [tok.bos_token_id or 1] + filler * (n_tokens // len(filler) + 1)
        return ids[:n_tokens]

    def run(prompt_lens: list[int], max_tokens: int) -> tuple[float, int, int]:
        prompts = [
            TokensPrompt(prompt_token_ids=make_prompt(n)) for n in prompt_lens
        ]
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        t0 = time.time()
        outs = llm.generate(prompts, params, use_tqdm=False)
        dt = time.time() - t0
        gen = sum(len(o.outputs[0].token_ids) for o in outs)
        pre = sum(prompt_lens)
        return dt, pre, gen

    # Warmup (compiles Triton kernels, touches both regimes).
    run([64], 8)
    run([8300], 4)

    scenarios = [
        ("decode bs=1, ctx 64 (dense regime)", [64], 256),
        ("decode bs=8, ctx 64 (dense regime)", [64] * 8, 128),
        ("prefill 4096 (dense regime)", [4096], 1),
        ("prefill 8300 (sparse regime)", [8300], 1),
        ("decode bs=1, ctx 8300 (sparse regime)", [8300], 64),
    ]

    rows = []
    for name, lens, max_toks in scenarios:
        dt, pre, gen = run(lens, max_toks)
        decode_dominant = max_toks > 4
        toks = gen if decode_dominant else pre
        kind = "output tok/s" if decode_dominant else "prefill tok/s"
        rows.append((name, f"{toks / dt:.1f} {kind}", f"{dt:.2f}s"))
        print(f"[bench] {name}: {toks / dt:.1f} {kind} ({dt:.2f}s wall)", flush=True)

    print("\n| Scenario | Throughput | Wall time |")
    print("|----------|-----------|-----------|")
    for name, thr, wall in rows:
        print(f"| {name} | {thr} | {wall} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
