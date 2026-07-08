#!/usr/bin/env python3
"""Probe whether dense KV history is used on short-seq decode."""

from __future__ import annotations

import os

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)


def main() -> int:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    import vllm.v1.attention.backends.minicpm_sala_sparse as m

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    hits = {
        "append": 0,
        "hist_ok": 0,
        "hist_miss": 0,
        "n_before": [],
        "hist_len": [],
    }
    _orig_append = m._append_dense_kv_history
    _orig_prefix = m._dense_kv_history_prefix

    def _append(layer, query, key, value, n):
        hits["append"] += 1
        return _orig_append(layer, query, key, value, n)

    def _prefix(layer, n_before):
        hist = _orig_prefix(layer, n_before)
        hq = getattr(layer, "_sala_dense_kv_q", None)
        hl = 0 if hq is None else int(hq.shape[0])
        hits["n_before"].append(n_before)
        hits["hist_len"].append(hl)
        if hist is None:
            hits["hist_miss"] += 1
        else:
            hits["hist_ok"] += 1
        return hist

    m._append_dense_kv_history = _append
    m._dense_kv_history_prefix = _prefix

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    ids = tok.encode("Hello, my name is", add_special_tokens=True)
    llm = LLM(
        model=WEIGHTS,
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
    llm.generate(
        [TokensPrompt(prompt_token_ids=ids)],
        SamplingParams(temperature=0, max_tokens=3),
    )
    print(
        f"append={hits['append']} hist_ok={hits['hist_ok']} "
        f"hist_miss={hits['hist_miss']}",
        flush=True,
    )
    print(f"n_before={hits['n_before']}", flush=True)
    print(f"hist_len={hits['hist_len']}", flush=True)
    return 0 if hits["hist_ok"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
