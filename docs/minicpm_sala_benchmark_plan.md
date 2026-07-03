# MiniCPM-SALA — Benchmark Plan

## Why this is a plan, not a script

vLLM's benchmarking is a generic CLI (`vllm bench latency`, `vllm bench
throughput`) that works for any registered model via `--model`, with no
per-model code required — confirmed by reading `vllm/benchmarks/latency.py`
and `vllm/benchmarks/throughput.py` directly (the old standalone
`benchmarks/benchmark_latency.py`/`benchmark_throughput.py` scripts are
now deprecated stubs that just print a redirect). Writing a new custom
benchmark script would duplicate infrastructure that already exists and
already works for this model the moment it's registered — which it is
(see `vllm/model_executor/models/registry.py`'s `MiniCPMSALAForCausalLM`
entry). This document is real commands, not new Python, following the
same "don't build what vLLM already has" principle used throughout this
project's cache/backend design.

**None of the commands below have been run.** Every flag name was
checked against the real CLI source (`vllm/engine/arg_utils.py`,
`vllm/benchmarks/latency.py`, `vllm/benchmarks/throughput.py`) before
being written here — not guessed — but "the flag exists" and "the
command runs successfully and produces a meaningful number" are
different claims. This is a plan to execute once GPU access exists, not
a report of results.

## Prerequisites

- Real GPU (see `docs/minicpm_sala_known_limitations.md` for VRAM
  sizing — the full model needs ~19GB+ just for bf16 weights).
- Real weights reachable (`openbmb/MiniCPM-SALA` on the Hub, or a local
  path).
- Steps 1–6 of `known_limitations.md` §6 ideally done first — benchmark
  numbers from a model whose numerical correctness hasn't been verified
  are not meaningful, only structurally interesting (does it run, what
  shape are the numbers).

## 1. Latency, dense regime (context < dense_len=8192)

The model's own natural, unmodified fallback path — the most likely to
actually work first, and a reasonable baseline before touching the
sparse path at all.

```bash
vllm bench latency \
  --model openbmb/MiniCPM-SALA \
  --input-len 512 \
  --output-len 128 \
  --batch-size 1 \
  --num-iters-warmup 3 \
  --num-iters 10 \
  --enforce-eager \
  --output-json minicpm_sala_latency_dense_bs1.json
```

`--enforce-eager` first (disables CUDA graphs) — this project has never
tested CUDA graph compatibility for either the Lightning Attention
custom-op dispatch or the sparse backend, and a first benchmark run is
not the place to find out both things simultaneously if something goes
wrong. Re-run without `--enforce-eager` only after the eager-mode number
looks sane.

Batch size sweep, still dense regime (matches the shape of the
`generate_greedy_logprobs`-style sweeps used elsewhere in vLLM's own
test suite, and reasonable given Lightning Attention's O(1)-per-sequence
cache is exactly the mechanism that should make batching cheap — worth
confirming, not assuming):

```bash
for bs in 1 2 4 8 16 32; do
  vllm bench latency \
    --model openbmb/MiniCPM-SALA \
    --input-len 512 --output-len 128 --batch-size $bs \
    --num-iters-warmup 3 --num-iters 10 --enforce-eager \
    --output-json "minicpm_sala_latency_dense_bs${bs}.json"
done
```

## 2. Latency, crossing into the sparse regime (context >= dense_len=8192)

The real point of interest for this specific model — confirming the
sparse path is actually exercised (not silently falling back to dense
in a way that would hide behind a merely-slower-than-expected number)
and, once confirmed working, whether it's actually faster than dense at
this context length (the entire premise of InfLLM-V2's design).

```bash
for ctx in 4096 8192 16384 32768; do
  vllm bench latency \
    --model openbmb/MiniCPM-SALA \
    --input-len $ctx --output-len 64 --batch-size 1 \
    --num-iters-warmup 3 --num-iters 5 --enforce-eager \
    --output-json "minicpm_sala_latency_ctx${ctx}.json"
done
```

`4096` and `8192` (the `dense_len` boundary itself) bracket the
dense→sparse transition; `16384`/`32768` exercise the sparse path more
substantially. **If the 8192 and 4096 numbers scale suspiciously
similarly to the 16384/32768 numbers** (i.e., no visible transition),
that is itself useful diagnostic signal that the sparse dispatch might
not be firing — worth checking against a debug log or breakpoint in
`MiniCPMSALASparseAttentionImpl.forward()`'s dispatch branch before
trusting the throughput numbers below.

## 3. Throughput (serving-realistic, many concurrent requests)

```bash
vllm bench throughput \
  --model openbmb/MiniCPM-SALA \
  --dataset-name random \
  --input-len 512 \
  --output-len 128 \
  --num-prompts 200 \
  --enforce-eager
```

Repeat with `--input-len 8192` (crossing `dense_len`) once step 2 above
confirms the sparse path is actually engaging correctly — a throughput
run over many concurrent long-context requests is where InfLLM-V2's
design is supposed to pay off most, and also where the untested
multi-sequence cache-bookkeeping paths (block_table indexing across
many concurrent sequences, not just the two-sequence case
`test_minicpm_sala_gather.py` covers) are most likely to reveal a bug
the CPU unit tests couldn't reach.

## 4. What NOT to benchmark yet

- **Multi-GPU (TP/PP) throughput** — TP under real `nccl` has never
  been tested at all (only CPU `gloo`, see `known_limitations.md` §1);
  benchmark correctness-under-TP before benchmark speed-under-TP.
- **Quantized variants** — no quantization support has been written for
  this model yet (see the future-steps list).
- **Speculative decoding / chunked prefill throughput** — neither
  integration has been attempted.

## 5. Recording results

Save every `--output-json` file's contents into
`docs/minicpm_sala_benchmark_results.md` (not yet created — create it
when real numbers exist, following this project's own established
pattern of only writing "results" documents once something has actually
run, per the Phase 1 report and `known_limitations.md`'s consistent
practice of citing real command output rather than projected numbers).
