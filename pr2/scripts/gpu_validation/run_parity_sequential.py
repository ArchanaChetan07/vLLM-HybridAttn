#!/usr/bin/env python3
"""Step B: HF vs vLLM parity — sequential load (never both on GPU)."""

import gc
import os
import sys

import torch

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
HF_REPO = os.environ.get("MINICPM_SALA_HF_REPO", "openbmb/MiniCPM-SALA")
NUM_LOGPROBS = 5
SHORT_MAX_TOKENS = 16
LONG_MAX_TOKENS = 8
LONG_PROMPT_TOKENS = 8200
DENSE_LEN = 8192


def _max_logprob_delta(hf_steps, vllm_logprobs_list, vllm_ids):
    max_delta = 0.0
    for i, (hf_id, hf_lp) in enumerate(hf_steps):
        vlp = vllm_logprobs_list[i]
        if vlp is None:
            continue
        hf_top = set(hf_lp.keys())
        v_top = set(int(k) for k in vlp.keys())
        if hf_top != v_top:
            max_delta = max(max_delta, float("inf"))
            continue
        for tid in hf_top:
            d = abs(hf_lp[tid] - float(vlp[tid]))
            max_delta = max(max_delta, d)
    return max_delta


def hf_greedy(model, tokenizer, input_ids, max_tokens, num_logprobs):
    steps = []
    ids = input_ids.clone()
    for _ in range(max_tokens):
        with torch.no_grad():
            out = model(input_ids=ids)
        logits = out.logits[0, -1].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        topv, topi = torch.topk(logprobs, num_logprobs)
        nxt = int(topi[0].item())
        lp = {int(topi[j].item()): float(topv[j].item()) for j in range(num_logprobs)}
        steps.append((nxt, lp))
        ids = torch.cat(
            [ids, torch.tensor([[nxt]], device=ids.device, dtype=ids.dtype)], dim=1
        )
    return steps, ids


def run_hf_suite():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        import fla  # noqa: F401 — required by HF MiniCPM-SALA reference code
    except ImportError as e:
        raise SystemExit(
            "FAIL: flash-linear-attention (fla) required for HF reference: "
            "pip install flash-linear-attention"
        ) from e
    try:
        import flash_attn  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "FAIL: flash-attn required for HF sparse layers: pip install flash-attn"
        ) from e

    print("=== HF load ===", flush=True)
    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    short_prompts = [
        "Hello, my name is",
        "The capital of France is",
        "Briefly explain gravity:",
    ]
    chunk = tok.encode("The quick brown fox jumps. ", add_special_tokens=False)
    long_ids = (chunk * (LONG_PROMPT_TOKENS // len(chunk) + 2))[:LONG_PROMPT_TOKENS]
    print(f"long prompt tokens: {len(long_ids)} (dense_len={DENSE_LEN})", flush=True)

    hf_short = []
    for p in short_prompts:
        ids = tok.encode(p, return_tensors="pt").to("cuda")
        steps, _ = hf_greedy(model, tok, ids[0], SHORT_MAX_TOKENS, NUM_LOGPROBS)
        hf_short.append((p, steps))

    long_ids_t = torch.tensor([long_ids], device="cuda")
    hf_long_steps, _ = hf_greedy(
        model, tok, long_ids_t[0], LONG_MAX_TOKENS, NUM_LOGPROBS
    )
    hf_long = (long_ids, hf_long_steps)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("=== HF unloaded ===", flush=True)
    return tok, hf_short, hf_long


def run_vllm_suite(tok, hf_short, hf_long):
    from vllm import LLM, SamplingParams

    print("=== vLLM load ===", flush=True)
    llm = LLM(
        model=WEIGHTS,
        trust_remote_code=False,
        dtype="bfloat16",
        max_model_len=max(LONG_PROMPT_TOKENS + LONG_MAX_TOKENS + 64, 9000),
        gpu_memory_utilization=0.90,
        enforce_eager=True,
    )
    sp_short = SamplingParams(
        temperature=0, max_tokens=SHORT_MAX_TOKENS, logprobs=NUM_LOGPROBS
    )
    sp_long = SamplingParams(
        temperature=0, max_tokens=LONG_MAX_TOKENS, logprobs=NUM_LOGPROBS
    )

    short_max = 0.0
    short_ok = True
    for (prompt, hf_steps), out in zip(
        hf_short, llm.generate([p for p, _ in hf_short], sp_short)
    ):
        v_ids = list(out.outputs[0].token_ids)
        v_lps = out.outputs[0].logprobs
        hf_ids = [t for t, _ in hf_steps]
        if v_ids != hf_ids:
            short_ok = False
            print(
                f"SHORT token mismatch prompt={prompt[:40]!r} hf={hf_ids} vllm={v_ids}",
                flush=True,
            )
        d = _max_logprob_delta(hf_steps, v_lps, v_ids)
        short_max = max(short_max, d)
        print(f"short prompt delta={d}", flush=True)

    long_ids, hf_steps = hf_long
    out = llm.generate(prompt_token_ids=[long_ids], sampling_params=sp_long)[0]
    v_ids = list(out.outputs[0].token_ids)
    hf_ids = [t for t, _ in hf_steps]
    long_ok = v_ids == hf_ids
    long_max = _max_logprob_delta(hf_steps, out.outputs[0].logprobs, v_ids)
    if not long_ok:
        print(f"LONG token mismatch hf={hf_ids} vllm={v_ids}", flush=True)
    print(f"long prompt ({len(long_ids)} ctx) delta={long_max}", flush=True)

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return short_ok and long_ok, short_max, long_max


def _ensure_weights() -> bool:
    if os.path.isdir(WEIGHTS) and os.path.isfile(os.path.join(WEIGHTS, "config.json")):
        return True
    if os.environ.get("MINICPM_SALA_DOWNLOAD_WEIGHTS", "").lower() not in (
        "1",
        "true",
        "yes",
    ):
        return False
    print(f"Downloading {HF_REPO} -> {WEIGHTS}", flush=True)
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(HF_REPO, local_dir=WEIGHTS)
        return os.path.isdir(WEIGHTS)
    except ImportError:
        print(
            "FAIL: huggingface_hub not installed; pip install huggingface_hub",
            flush=True,
        )
        return False


def main():
    if not _ensure_weights() and not os.path.isdir(WEIGHTS):
        print(
            f"FAIL: weights not found at {WEIGHTS}. "
            "Set MINICPM_SALA_DOWNLOAD_WEIGHTS=1 to auto-download (~19GB).",
            flush=True,
        )
        return 1
    tok, hf_short, hf_long = run_hf_suite()
    ok, sm, lm = run_vllm_suite(tok, hf_short, hf_long)
    print(f"PARITY short_max_delta={sm} long_max_delta={lm} pass={ok}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
