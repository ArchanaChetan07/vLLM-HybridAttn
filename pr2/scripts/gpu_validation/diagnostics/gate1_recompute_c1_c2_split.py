#!/usr/bin/env python3
"""Stage-1 split: C1 (live hidden) vs C2 (recompute math) at L1 seq=21.

Feeds identical Δ=0 q/k/v history into manual HF GLA and vLLM prefix_fn;
diffs outputs, g_gamma, fresh_sequence, and L1 input hidden (incremental vs one-shot).
"""

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import torch
from einops import rearrange
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = os.environ.get("MINICPM_SALA_PROMPT", "Hello, my name is")
STEP = int(os.environ.get("MINICPM_SALA_MISMATCH_STEP", "14"))
# HF greedy for Hello prompt (16 tokens) — avoids loading HF beside vLLM.
HF_GREEDY = [
    2132, 1417, 1523, 7089, 1520, 1606, 5, 1975, 19020, 59324,
    59342, 63, 59377, 59320, 16091, 1525,
]
LIGHTNING_LAYERS = tuple(
    int(x) for x in os.environ.get("MINICPM_SALA_LIGHTNING_LAYERS", "1,6,9").split(",")
)


def _log(msg: str, data: dict) -> None:
    if os.environ.get("MINICPM_SALA_DEBUG_GLA", "") != "1":
        return
    path = os.environ.get("DEBUG_LOG_PATH", "debug-212a6e.log")
    payload = {
        "sessionId": "212a6e",
        "runId": os.environ.get("DEBUG_RUN_ID", "c1c2-split"),
        "hypothesisId": "C",
        "location": "gate1_recompute_c1_c2_split.py",
        "message": msg,
        "data": data,
        "timestamp": int(__import__("time").time() * 1000),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _reset_hist(model: torch.nn.Module) -> int:
    for layer in model.model.layers:
        reset = getattr(layer.self_attn, "_reset_qkv_history", None)
        if callable(reset):
            reset()
    return 0


def _install_capture(model: torch.nn.Module) -> int:
    model._cap: dict = {"l1_in": None, "layers": {}}

    def _layer1_pre(_mod, args):
        if len(args) < 2:
            return
        hs = args[1]
        if isinstance(hs, torch.Tensor) and hs.shape[0] >= 1:
            model._cap["l1_in"] = hs[-1].detach().float().cpu().clone()

    model._cap_hooks = [
        model.model.layers[1].register_forward_pre_hook(_layer1_pre),
    ]
    return 0


def _read_hist(model: torch.nn.Module) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for idx in LIGHTNING_LAYERS:
        attn = model.model.layers[idx].self_attn
        row: dict = {"hist_len": 0}
        if getattr(attn, "_qkv_hist_q", None) is not None:
            row["hist_len"] = int(attn._qkv_hist_q.shape[0])
            row["q"] = attn._qkv_hist_q.detach().float().cpu().clone()
            row["k"] = attn._qkv_hist_k.detach().float().cpu().clone()
            row["v"] = attn._qkv_hist_v.detach().float().cpu().clone()
            row["slope"] = attn.tp_slope.detach().float().cpu().clone()
            row["scale"] = float(attn.scale)
        out[idx] = row
    out["l1_in"] = model._cap.get("l1_in")
    return out


def _manual_gla(
    q_hist: torch.Tensor,
    k_hist: torch.Tensor,
    v_hist: torch.Tensor,
    slope: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from fla.ops.simple_gla import fused_recurrent_simple_gla

    qs = q_hist.transpose(0, 1).unsqueeze(0).contiguous()
    ks = k_hist.transpose(0, 1).unsqueeze(0).contiguous()
    vs = v_hist.transpose(0, 1).unsqueeze(0).contiguous()
    h = qs.shape[1]
    n = qs.shape[2]
    g_gamma = (-slope.to(torch.float32)).reshape(h)
    q_b = rearrange(qs, "b h t d -> b t h d").to(torch.float32)
    k_b = rearrange(ks, "b h t d -> b t h d").to(torch.float32)
    v_b = rearrange(vs, "b h t d -> b t h d").to(torch.float32)
    o, fin = fused_recurrent_simple_gla(
        q=q_b,
        k=k_b,
        v=v_b,
        g_gamma=g_gamma,
        scale=scale,
        initial_state=None,
        output_final_state=True,
    )
    last = rearrange(o[0, -1], "h d -> (h d)").float().cpu()
    return last, fin.reshape(h, q_hist.shape[-1], q_hist.shape[-1]).float().cpu()


def _prefix_fn_gla(
    model: torch.nn.Module,
    layer_idx: int,
    q_hist: torch.Tensor,
    k_hist: torch.Tensor,
    v_hist: torch.Tensor,
) -> torch.Tensor:
    from vllm.model_executor.models.minicpm_sala import (
        _minicpm_sala_lightning_forward_prefix,
    )

    attn = model.model.layers[layer_idx].self_attn
    kv = torch.zeros(
        *attn.get_state_shape()[0],
        device="cuda",
        dtype=attn.get_state_dtype()[0],
    )
    qs = q_hist.transpose(0, 1).unsqueeze(0).contiguous().cuda()
    ks = k_hist.transpose(0, 1).unsqueeze(0).contiguous().cuda()
    vs = v_hist.transpose(0, 1).unsqueeze(0).cuda()
    flat = _minicpm_sala_lightning_forward_prefix(
        qs,
        ks,
        vs,
        kv,
        attn.tp_slope,
        attn.block_size,
        scale=attn.scale,
        fresh_sequence=True,
    )
    return flat[-1].float().cpu()


def _diff_tensors(a: torch.Tensor | None, b: torch.Tensor | None, label: str) -> float:
    if a is None or b is None:
        print(f"{label}: missing", flush=True)
        return float("nan")
    d = (a - b).abs().max().item()
    print(f"{label} peak={d:.6g}", flush=True)
    return d


def main() -> int:
    from transformers import AutoTokenizer

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    prompt_ids = tok.encode(PROMPT, add_special_tokens=True)
    prefix_ids = prompt_ids + HF_GREEDY[:STEP]
    seq_len = len(prefix_ids) + 1
    print(f"prompt_len={len(prompt_ids)} step={STEP} seq_len={seq_len}", flush=True)
    print(f"hf_next@{STEP}={HF_GREEDY[STEP]}", flush=True)

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
    llm.apply_model(_install_capture)

    inc_tok = int(
        llm.generate(
            [TokensPrompt(prompt_token_ids=prompt_ids)],
            SamplingParams(temperature=0, max_tokens=STEP + 1),
        )[0]
        .outputs[0]
        .token_ids[STEP]
    )
    inc = llm.apply_model(_read_hist)[0]
    print(f"incremental_token@{STEP}={inc_tok}", flush=True)

    llm.apply_model(_reset_hist)
    llm.apply_model(lambda m: setattr(m, "_cap", {"l1_in": None}) or 0)

    one_tok = int(
        llm.generate(
            [TokensPrompt(prompt_token_ids=prefix_ids)],
            SamplingParams(temperature=0, max_tokens=1),
        )[0]
        .outputs[0]
        .token_ids[0]
    )
    one = llm.apply_model(_read_hist)[0]
    print(f"oneshot_token@{STEP}={one_tok}", flush=True)

    # --- L1 input hidden (C1 probe) ---
    hdiff = _diff_tensors(inc.get("l1_in"), one.get("l1_in"), "L1_input_hidden")

    # --- per-lightning-layer q/k/v history ---
    for layer_idx in LIGHTNING_LAYERS:
        ih, oh = inc.get(layer_idx, {}), one.get(layer_idx, {})
        print(f"L{layer_idx} inc_hist={ih.get('hist_len')} one_hist={oh.get('hist_len')}", flush=True)
        if "q" in ih and "q" in oh:
            n = min(ih["q"].shape[0], oh["q"].shape[0])
            _diff_tensors(ih["q"][:n], oh["q"][:n], f"L{layer_idx}_q_hist")
            _diff_tensors(ih["k"][:n], oh["k"][:n], f"L{layer_idx}_k_hist")
            _diff_tensors(ih["v"][:n], oh["v"][:n], f"L{layer_idx}_v_hist")
            _diff_tensors(ih["q"][-1:], oh["q"][-1:], f"L{layer_idx}_q_last")
            _diff_tensors(ih["v"][-1:], oh["v"][-1:], f"L{layer_idx}_v_last")

    # --- C2: manual GLA on incremental vs one-shot L1 tensors ---
    l1_inc, l1_one = inc.get(1, {}), one.get(1, {})
    if "q" in l1_inc and "q" in l1_one:
        # Cross-feed: inc q/k/v through manual GLA
        inc_gla, inc_state = _manual_gla(
            l1_inc["q"], l1_inc["k"], l1_inc["v"], l1_inc["slope"], l1_inc["scale"]
        )
        one_gla, one_state = _manual_gla(
            l1_one["q"], l1_one["k"], l1_one["v"], l1_one["slope"], l1_one["scale"]
        )
        gla_diff = _diff_tensors(inc_gla, one_gla, "manual_gla_last_inc_vs_one")
        state_diff = _diff_tensors(inc_state, one_state, "manual_gla_state_inc_vs_one")

        # Same tensors cross-check: inc q/k/v vs one q/k/v fed to ONE manual run
        q_same = (l1_inc["q"] - l1_one["q"]).abs().max().item() == 0.0
        if q_same:
            print("L1 q/k/v hist IDENTICAL — manual_gla should match if deterministic", flush=True)
        cross_gla_on_inc, _ = _manual_gla(
            l1_one["q"], l1_one["k"], l1_one["v"], l1_one["slope"], l1_one["scale"]
        )
        _diff_tensors(inc_gla, cross_gla_on_inc, "manual_gla_inc_hist_vs_one_hist_tensors")

        # vLLM prefix_fn with fresh_sequence=True
        def _run_prefix(hist: dict) -> torch.Tensor:
            def fn(m: torch.nn.Module) -> torch.Tensor:
                return _prefix_fn_gla(m, 1, hist["q"], hist["k"], hist["v"])

            return llm.apply_model(fn)[0]

        pf_inc = _run_prefix(l1_inc)
        pf_one = _run_prefix(l1_one)
        pf_diff = _diff_tensors(pf_inc, pf_one, "prefix_fn_fresh_inc_vs_one")

        g_gamma = (-l1_inc["slope"].to(torch.float32)).reshape(-1)
        print(
            f"g_gamma[:4]={g_gamma[:4].tolist()} fresh_sequence=True n={l1_inc['q'].shape[0]}",
            flush=True,
        )
        _log(
            "split_result",
            {
                "step": STEP,
                "seq_len": seq_len,
                "hdiff": hdiff,
                "gla_diff": gla_diff,
                "state_diff": state_diff,
                "pf_diff": pf_diff,
                "inc_tok": inc_tok,
                "one_tok": one_tok,
                "q_hist_identical": q_same,
            },
        )

        if q_same and gla_diff == 0.0 and pf_diff == 0.0 and inc_tok != one_tok:
            print("VERDICT: C1 — inputs to GLA match, GLA matches, but tokens differ → upstream of recompute", flush=True)
        elif not q_same:
            print("VERDICT: C1 — q/k/v or hidden differ between paths", flush=True)
        elif gla_diff > 0 or pf_diff > 0:
            print("VERDICT: C2 — recompute math/wiring differs on matched inputs", flush=True)
        else:
            print("VERDICT: paths agree on tensors and tokens", flush=True)

    print(f"token_match_hf={inc_tok == HF_GREEDY[STEP]} inc={inc_tok} one={one_tok}", flush=True)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return 0 if inc_tok == one_tok == HF_GREEDY[STEP] else 1


if __name__ == "__main__":
    raise SystemExit(main())
