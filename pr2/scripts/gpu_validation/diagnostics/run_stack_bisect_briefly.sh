#!/usr/bin/env bash
set -euo pipefail
cd /workspace/hybridattn
export MINICPM_SALA_WEIGHTS="${MINICPM_SALA_WEIGHTS:-/workspace/models/openbmb/MiniCPM-SALA}"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "gpu_free_mib=${FREE}"
pkill -9 -f 'VLLM::EngineCore|EngineCore' 2>/dev/null || true
sleep 2
git pull origin feature/minicpm-sala-sparse
bash scripts/install_pr2_overlay.sh
mkdir -p pr2/scripts/gpu_validation/diagnostics/traces

echo "=== HF vs vLLM greedy t1 (Briefly prompt-only) ==="
python3 - <<'PY'
import os, gc, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
W = os.environ["MINICPM_SALA_WEIGHTS"]
P = "Briefly explain gravity:"
tok = AutoTokenizer.from_pretrained(W, trust_remote_code=True)
ids = tok.encode(P, add_special_tokens=True)
m = AutoModelForCausalLM.from_pretrained(W, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="flash_attention_2").eval()
with torch.no_grad():
    hf_t1 = int(m(torch.tensor([ids], device="cuda"), attention_mask=torch.ones(1,len(ids),device="cuda")).logits[0,-1].argmax())
print(f"hf_t1={hf_t1} seqlen={len(ids)}")
del m; gc.collect(); torch.cuda.empty_cache()
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
llm = LLM(model=W, trust_remote_code=True, dtype="bfloat16", max_model_len=4096, block_size=256, gpu_memory_utilization=0.5, enforce_eager=True, max_num_seqs=1, enable_prefix_caching=False, mamba_cache_mode="none")
vv_t1 = int(llm.generate([TokensPrompt(prompt_token_ids=ids)], SamplingParams(temperature=0, max_tokens=1))[0].outputs[0].token_ids[0])
print(f"vv_t1={vv_t1} match={hf_t1==vv_t1}")
del llm; gc.collect(); torch.cuda.empty_cache()
PY

echo "=== 32-layer stack bisect (prompt-only) ==="
MINICPM_SALA_PROMPT='Briefly explain gravity:' MINICPM_SALA_MODE=prompt \
  python3 pr2/scripts/gpu_validation/diagnostics/gate1_stack_bisect.py 2>&1 \
  | tee pr2/scripts/gpu_validation/diagnostics/traces/stack_bisect_briefly_prompt.log

echo "=== 32-layer stack bisect (Hello prompt-only sanity) ==="
MINICPM_SALA_PROMPT='Hello, my name is' MINICPM_SALA_MODE=prompt \
  python3 pr2/scripts/gpu_validation/diagnostics/gate1_stack_bisect.py 2>&1 \
  | tee pr2/scripts/gpu_validation/diagnostics/traces/stack_bisect_hello_prompt.log
