from pathlib import Path

REG = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/registry.py")
text = REG.read_text(encoding="utf-8")
if "MINICPM_SALA_INFLLM_V2" in text:
    print("already patched", REG)
    raise SystemExit(0)
needle = (
    '    MINIMAX_M3_SPARSE = (\n'
    '        "vllm.models.minimax_m3.common.sparse_attention.MiniMaxM3SparseBackend"\n'
    "    )\n"
)
insert = (
    needle
    + "    MINICPM_SALA_INFLLM_V2 = (\n"
    '        "vllm.v1.attention.backends.minicpm_sala_sparse.MiniCPMSALASparseAttentionBackend"\n'
    "    )\n"
)
if needle not in text:
    raise SystemExit(f"needle not found in {REG}")
REG.write_text(text.replace(needle, insert), encoding="utf-8")
print("patched", REG)
