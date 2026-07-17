#!/usr/bin/env python3
"""GPU Step 5b: engine-level tensor-parallel parity.

Runs the standard parity prompts greedily under `LLM(tensor_parallel_size=N)`
and writes the token ids to a JSON file. Run once per TP degree (separate
processes -- vLLM engines are not designed to be torn down and rebuilt for
a different world size in one process), then compare the JSON files:
TP>1 must reproduce the TP=1 tokens exactly on the same host.

    export MINICPM_SALA_WEIGHTS=/path/to/openbmb/MiniCPM-SALA
    python step5b_tp_engine_parity.py --tp 1 --out /tmp/tp1.json
    python step5b_tp_engine_parity.py --tp 2 --out /tmp/tp2.json
    python step5b_tp_engine_parity.py --tp 2 --long --out /tmp/tp2_long.json
    python step5b_tp_engine_parity.py --compare /tmp/tp1.json /tmp/tp2.json

The TP=1 tokens can additionally be compared against the committed A100
Step B logs (docs/validation_logs/); note cross-hardware bf16 greedy
outputs may legitimately differ in rare tie cases, while same-host
TP=1-vs-TP>1 must match.
"""

import argparse
import json
import os
import sys

MAX_TOKENS = 16

SHORT_PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
    "Briefly explain gravity:",
]


def generate(tp: int, run_long: bool) -> dict:
    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    model = os.environ.get("MINICPM_SALA_WEIGHTS", "openbmb/MiniCPM-SALA")
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    prompts = {p: tok.encode(p, add_special_tokens=True) for p in SHORT_PROMPTS}
    if run_long:
        base = tok.encode(
            "The quick brown fox jumps over the lazy dog. ",
            add_special_tokens=False,
        )
        ids = [tok.bos_token_id or 1] + base * (8300 // len(base) + 1)
        ids = ids[:8306]
        prompts[f"<long {len(ids)} tokens>"] = ids

    llm = LLM(
        model=model,
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        block_size=256,
        max_model_len=16384,
        tensor_parallel_size=tp,
    )
    outs = llm.generate(
        [TokensPrompt(prompt_token_ids=i) for i in prompts.values()],
        SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS),
    )
    result = {}
    for label, out in zip(prompts, outs):
        comp = out.outputs[0]
        result[label] = list(comp.token_ids)
        print(f"[tp={tp}] {label!r}: {list(comp.token_ids)}", flush=True)
    return result


def compare(paths: list[str]) -> int:
    runs = {p: json.load(open(p)) for p in paths}
    ref_path, ref = paths[0], runs[paths[0]]
    ok = True
    for p in paths[1:]:
        for label, ref_tokens in ref.items():
            got = runs[p].get(label)
            match = got == ref_tokens
            ok &= match
            print(
                f"[{'PASS' if match else 'FAIL'}] {label!r}: "
                f"{os.path.basename(ref_path)} vs {os.path.basename(p)}"
                + ("" if match else f"\n    ref {ref_tokens}\n    got {got}")
            )
    print("STEP 5b " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--long", action="store_true", dest="run_long")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--compare", nargs="+", default=None)
    args = ap.parse_args()

    if args.compare:
        return compare(args.compare)

    result = generate(args.tp, args.run_long)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
