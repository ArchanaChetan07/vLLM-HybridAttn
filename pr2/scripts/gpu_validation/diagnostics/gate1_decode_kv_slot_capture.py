#!/usr/bin/env python3
"""Capture L0 sparse KV slot vs block_table at decode steps 10-15 (Hello prompt)."""

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

WEIGHTS = os.environ.get(
    "MINICPM_SALA_WEIGHTS", "/workspace/models/openbmb/MiniCPM-SALA"
)
PROMPT = os.environ.get("MINICPM_SALA_PROMPT", "Hello, my name is")
BLOCK_SIZE = int(os.environ.get("MINICPM_SALA_BLOCK_SIZE", "256"))
TARGET_STEPS = frozenset(
    int(x)
    for x in os.environ.get("MINICPM_SALA_DECODE_STEPS", "10,11,12,13,14,15").split(",")
    if x.strip()
)
MAX_DECODE = max(TARGET_STEPS) + 2
TRACE_JSON = Path(__file__).resolve().parent / "traces" / "decode_kv_slot_capture_latest.json"
TRACE_LOG = Path(__file__).resolve().parent / "traces" / "decode_kv_slot_capture_latest.log"

# Pickle-safe worker globals (no closures in apply_model lambdas).
_DECODE_IDX = 0
_PROMPT_LEN = 0
_SPARSE_KEY: str = ""


def _tensor_list(t: torch.Tensor | None) -> list | None:
    if t is None or not isinstance(t, torch.Tensor):
        return None
    return t.detach().cpu().tolist()


def _slot_analysis(
    seq_len: int,
    slot_mapping: list[int],
    block_table: list[list[int]],
) -> dict[str, Any]:
    slot = int(slot_mapping[0]) if slot_mapping else None
    bt_head = int(block_table[0][0]) if block_table and block_table[0] else None
    pos = seq_len - 1 if seq_len > 0 else 0
    derived: dict[str, Any] = {
        "seq_len": seq_len,
        "kv_write_pos_index": pos,
        "slot_mapping0": slot,
        "block_table_row0": block_table[0] if block_table else None,
    }
    if slot is not None:
        derived["slot_phys_block"] = slot // BLOCK_SIZE
        derived["slot_block_offset"] = slot % BLOCK_SIZE
        derived["slot_ge_2048"] = slot >= 2048
    if bt_head is not None:
        derived["table_phys_block0"] = bt_head
        derived["expected_slot_if_table0"] = bt_head * BLOCK_SIZE + (pos % BLOCK_SIZE)
    if slot is not None and bt_head is not None:
        derived["table_vs_slot_phys_mismatch"] = (slot // BLOCK_SIZE) != bt_head
        derived["expected_vs_actual_slot_delta"] = (
            derived.get("expected_slot_if_table0", 0) - slot
        )
    derived["crosses_block256_boundary"] = (
        seq_len > 0 and (seq_len - 1) // BLOCK_SIZE != pos // BLOCK_SIZE
    )
    return derived


def _sparse_row(meta: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"kind": "sparse"}
    for attr in (
        "num_prefills",
        "num_decodes",
        "num_prefill_tokens",
        "num_decode_tokens",
        "dense_len",
        "num_actual_tokens",
        "max_query_len",
        "max_seq_len",
    ):
        if hasattr(meta, attr):
            row[attr] = getattr(meta, attr)
    qsl = _tensor_list(getattr(meta, "query_start_loc", None))
    sl = _tensor_list(getattr(meta, "seq_lens", None))
    sm = _tensor_list(getattr(meta, "slot_mapping", None))
    bt = _tensor_list(getattr(meta, "block_table", None))
    row["query_start_loc"] = qsl
    row["seq_lens"] = sl
    row["slot_mapping"] = sm
    row["block_table"] = bt
    # num_computed_tokens is scheduler-side; approximate from seq_lens on decode.
    row["num_computed_tokens_inferred"] = (
        int(sl[0]) if sl else None
    )
    if sl and sm and bt:
        row["slot_analysis"] = _slot_analysis(int(sl[0]), sm, bt)
    return row


def _install(model: torch.nn.Module) -> int:
    global _DECODE_IDX, _SPARSE_KEY
    from vllm.forward_context import get_forward_context

    layer0 = model.model.layers[0]
    _SPARSE_KEY = layer0.self_attn.attn.layer_name
    model._kv_cap: dict[str, Any] = {"captures": {}, "engine_l0": {}}
    _DECODE_IDX = 0

    def _pre(_mod, args):
        global _DECODE_IDX
        if len(args) < 2:
            return
        h = args[1]
        if not isinstance(h, torch.Tensor) or h.shape[0] != 1:
            return
        _DECODE_IDX += 1
        step = _DECODE_IDX
        if step not in TARGET_STEPS:
            return
        ctx = get_forward_context()
        md = ctx.attn_metadata
        if not isinstance(md, dict) or _SPARSE_KEY not in md:
            return
        meta = md[_SPARSE_KEY]
        snap = _sparse_row(meta)
        snap["decode_idx"] = step
        snap["prompt_len"] = _PROMPT_LEN
        model._kv_cap["captures"][str(step)] = snap
        model._kv_cap["engine_l0"][str(step)] = h[-1].detach().float().cpu().tolist()

    model._kv_hook = model.model.layers[0].register_forward_pre_hook(_pre)
    return 0


def _read_capture(model: torch.nn.Module) -> dict[str, Any]:
    return dict(getattr(model, "_kv_cap", {}))


def _hf_greedy_prefix(max_steps: int) -> tuple[list[int], list[int]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
    prompt_ids = tok.encode(PROMPT, add_special_tokens=True)
    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    cur = prompt_ids[:]
    with torch.no_grad():
        for _ in range(max_steps):
            out = model(torch.tensor([cur], device="cuda"))
            cur.append(int(out.logits[0, -1].argmax().item()))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return prompt_ids, cur


def _hf_l0_last(prefix_ids: list[int]) -> list[float]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    pos = torch.arange(len(prefix_ids), device="cuda").unsqueeze(0)
    mask = torch.ones(1, len(prefix_ids), device="cuda")
    with torch.no_grad():
        h = model.model.embed_tokens(torch.tensor([prefix_ids], device="cuda"))
        h = h * model.config.scale_emb
        h = model.model.layers[0](h, attention_mask=mask, position_ids=pos, use_cache=False)[0]
        vec = h[0, -1].detach().float().cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return vec.tolist()


def _analyze(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"per_step": {}, "smoking_gun": False, "notes": []}
    prompt_len = int(payload.get("prompt_len", 0))
    for step_s, cap in payload.get("captures", {}).items():
        step = int(step_s)
        sa = cap.get("slot_analysis", {})
        eng = payload.get("l0_compare", {}).get(step_s, {})
        row = {
            "decode_idx": step,
            "seq_lens": cap.get("seq_lens"),
            "slot_mapping": cap.get("slot_mapping"),
            "block_table": cap.get("block_table"),
            "query_start_loc": cap.get("query_start_loc"),
            "num_computed_tokens_inferred": cap.get("num_computed_tokens_inferred"),
            "slot_analysis": sa,
            "l0_peak_abs_diff": eng.get("peak_abs_diff"),
            "l0_argmax_match": eng.get("argmax_match"),
        }
        mismatch = bool(sa.get("table_vs_slot_phys_mismatch"))
        row["block_table_slot_smoking_gun"] = mismatch
        if mismatch:
            out["smoking_gun"] = True
            out["notes"].append(
                f"step {step}: slot phys block {sa.get('slot_phys_block')} "
                f"!= block_table[0][0] {sa.get('table_phys_block0')}"
            )
        if sa.get("slot_ge_2048") and sa.get("table_phys_block0") in (0, 1):
            out["notes"].append(
                f"step {step}: slot_mapping {sa.get('slot_mapping0')} (2048+) "
                f"vs block_table={cap.get('block_table')}"
            )
        out["per_step"][step_s] = row
    out["prompt_len"] = prompt_len
    out["block_size"] = BLOCK_SIZE
    return out


def main() -> int:
    global _PROMPT_LEN
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    prompt_ids, full_ids = _hf_greedy_prefix(MAX_DECODE)
    _PROMPT_LEN = len(prompt_ids)

    llm = LLM(
        model=WEIGHTS,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        max_num_seqs=1,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        enable_chunked_prefill=False,
    )
    llm.apply_model(_install)
    out = llm.generate(
        [TokensPrompt(prompt_token_ids=prompt_ids)],
        SamplingParams(temperature=0, max_tokens=MAX_DECODE),
    )[0]
    cap = llm.apply_model(_read_capture)[0]
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    cap["prompt"] = PROMPT
    cap["prompt_len"] = _PROMPT_LEN
    cap["target_steps"] = sorted(TARGET_STEPS)
    cap["generated_token_ids"] = list(out.outputs[0].token_ids)
    cap["hf_full_prefix_len"] = len(full_ids)

    l0_compare: dict[str, Any] = {}
    for step in sorted(TARGET_STEPS):
        # Engine decode step N: seq_len = prompt_len + N; HF reference prefix before writing token N.
        prefix = full_ids[: _PROMPT_LEN + step - 1]
        ref = _hf_l0_last(prefix)
        eng = cap.get("engine_l0", {}).get(str(step))
        if eng is None:
            l0_compare[str(step)] = {"error": "missing_engine_l0"}
            continue
        ref_t = torch.tensor(ref)
        eng_t = torch.tensor(eng)
        peak = (ref_t - eng_t).abs().max().item()
        l0_compare[str(step)] = {
            "prefix_len": len(prefix),
            "peak_abs_diff": peak,
            "ref_l0_norm": ref_t.norm().item(),
            "eng_l0_norm": eng_t.norm().item(),
        }
    cap["l0_compare"] = l0_compare
    cap["analysis"] = _analyze(cap)

    TRACE_JSON.parent.mkdir(parents=True, exist_ok=True)
    TRACE_JSON.write_text(json.dumps(cap, indent=2), encoding="utf-8")
    log(json.dumps(cap["analysis"], indent=2))
    TRACE_LOG.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
