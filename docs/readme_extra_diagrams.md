### Overall system

```mermaid
flowchart TB
    subgraph client [Client]
        API[OpenAI-compatible API]
    end
    subgraph vllm [vLLM Engine]
        SCHED[Scheduler v1]
        EXEC[Model Executor]
        KVPOOL[KV Cache Pool]
    end
    subgraph hybrid [MiniCPM-SALA]
        MODEL[MiniCPMSALAForCausalLM]
        LA[Lightning x24]
        DA[minicpm4 x8]
        SPARSE[Sparse PR2]
    end
    API --> SCHED --> EXEC --> MODEL
    MODEL --> LA
    MODEL --> DA
    DA -.->|seq greater than dense_len| SPARSE
    SCHED --> KVPOOL
    SPARSE --> KVPOOL
    LA --> KVPOOL
```

### Model hierarchy

```mermaid
classDiagram
    class MiniCPMSALAForCausalLM {
        HasInnerState
        IsHybrid
        SupportsPP
        layers 32
    }
    class MiniCPMSALADecoderLayer {
        mixer_type
        self_attn
        mlp
    }
    class MiniCPMSALALightningAttention {
        MambaBase
        PluggableLayer
        get_state_shape()
    }
    class MiniCPMSALADenseAttention {
        qkv_proj
        attn
        o_proj
    }
    class MiniCPMSALASparseAttention {
        PR2 only
        get_kv_cache_spec()
    }
    MiniCPMSALAForCausalLM --> MiniCPMSALADecoderLayer
    MiniCPMSALADecoderLayer --> MiniCPMSALALightningAttention
    MiniCPMSALADecoderLayer --> MiniCPMSALADenseAttention
    MiniCPMSALADenseAttention ..> MiniCPMSALASparseAttention
```

### Module dependency graph (PR1 / PR2 boundary)

```mermaid
flowchart LR
    subgraph PR1 [PR1 merges independently]
        M[minicpm_sala.py]
        T1[PR1 tests]
    end
    subgraph PR2 [PR2 depends on PR1]
        W[minicpm_sala_sparse_wiring]
        S[minicpm_sala_sparse]
        KV[minicpm_sala_kv_cache_spec]
        I[infllm_v2 optional]
        T2[PR2 tests]
    end
    M -.->|zero imports| W
    W --> S
    W --> KV
    S --> I
    M --> T1
    S --> T2
```

### Scheduler interaction

```mermaid
flowchart TD
    BUILD[Model build] --> GET[get_kv_cache_spec per layer]
    GET -->|lightning-attn| MAMBA[MambaSpec recurrent state]
    GET -->|minicpm4 PR2| HIER[HierarchicalCompressedAttentionSpec]
    GET -->|minicpm4 PR1| FULL[default Attention KV]
    HIER --> MGR[HierarchicalCompressedAttentionManager]
    MGR --> ALLOC[Block allocation]
    ALLOC --> SCHED[v1 Scheduler]
```

### Memory ownership

```mermaid
flowchart TB
    subgraph per_seq [Per sequence]
        LS[Lightning state O1 fixed]
        SKV[KV pages O seq_len]
    end
    subgraph engine [Engine shared]
        POOL[Paged KV block pool]
        W[Model weights]
    end
    LS --> POOL
    SKV --> POOL
    POOL --> VRAM[GPU VRAM]
    W --> VRAM
```

### Token generation flow

```mermaid
sequenceDiagram
    participant U as User
    participant E as vLLM Engine
    participant M as MiniCPMSALAForCausalLM
    participant H as LM Head
    U->>E: prompt tokens
    loop each decode step
        E->>M: forward pass 32 layers
        M->>H: logits
        H->>E: sample next token
    end
    E->>U: generated text
```

### Weight loading

```mermaid
flowchart LR
    HF[HuggingFace checkpoint] --> L[AutoWeightsLoader]
    L --> EMB[embed_tokens]
    L --> LAY[layers 0-31 weights]
    L --> HEAD[norm + lm_head]
    LAY --> QKV[qkv_proj]
    LAY --> LA_W[lightning projections]
    LAY --> MLP_W[gate_up + down]
```

### Configuration loading

```mermaid
flowchart TD
    CFG[PretrainedConfig from HF] --> MODEL[MiniCPMSALAForCausalLM]
    CFG --> MIX[mixer_types schedule]
    CFG --> SC[sparse_config PR2]
    CACHE[CacheConfig block_size] --> KV[KV page geometry]
    MIX --> DISPATCH[per-layer mixer dispatch]
```

### GPU execution paths

```mermaid
flowchart TD
    GPU{GPU compute capability}
    GPU -->|sm less than 80| LIMITED[Dense GQA + CPU unit tests]
    GPU -->|sm 80 plus| AMPERE[Ampere execution path]
    AMPERE --> LA_K[Lightning kernels]
    AMPERE --> SP_K[InfLLM-V2 sparse kernels]
    AMPERE --> FA[FlashAttention dense]
```

### Repository layout

```mermaid
flowchart TB
    ROOT[vLLM-HybridAttn repo]
    ROOT --> VLLM[vllm/ PR1 model]
    ROOT --> TESTS[tests/ PR1]
    ROOT --> PR2[pr2/ overlay]
    ROOT --> DOCS[docs/]
    ROOT --> DOCKER[docker scripts]
    PR2 --> PV[sparse vllm modules]
    PR2 --> PT[tests/v1]
    PR2 --> PG[gpu_validation]
```

### Docker workflow

```mermaid
flowchart TD
    START[docker run CUDA image] --> PIP[pip install vllm 0.24.0]
    PIP --> CHOICE{which script}
    CHOICE -->|docker_run_pr1.sh| O1[overlay PR1 model only]
    CHOICE -->|docker_run_integration.sh| O2[overlay PR2 stack]
    O1 --> RUFF[ruff check]
    O2 --> RUFF
    RUFF --> PYTEST[pytest]
    O2 --> INF[build infllm_v2]
    O2 --> GPUVAL[gpu_validation]
```

### Testing pipeline

```mermaid
flowchart LR
    subgraph verified [Verified]
        R[ruff]
        P1[22 PR1 tests]
        P2[44 PR2 tests]
    end
    subgraph pending [Pending]
        HF[check_logprobs_close]
        A40[Ampere GPU steps 2 and 4]
        TP[multi-GPU TP]
    end
    R --> P1 --> P2
    P2 -.-> HF
    P2 -.-> A40
    P2 -.-> TP
```

### CI pipeline

```mermaid
flowchart TD
    PUSH[git push] --> GHA[GitHub Actions ci.yml]
    GHA --> RUFF_JOB[ruff check and format]
    GHA --> DOCKER_JOB[docker_run_pr1.sh]
    RUFF_JOB --> OK1[pass]
    DOCKER_JOB --> OK2[22 tests pass]
```

### Upstream PR workflow

```mermaid
flowchart TD
    PR1[PR1 model to vllm-project/vllm] --> MERGE1[merge PR1]
    MERGE1 --> PR2[PR2 sparse backend]
    PR2 --> MERGE2[merge PR2]
```

### Git branching strategy

```mermaid
gitGraph
    commit id: "main monorepo"
    branch feature/minicpm-sala-model
    checkout feature/minicpm-sala-model
    commit id: "PR1 model only"
    branch feature/minicpm-sala-sparse
    checkout feature/minicpm-sala-sparse
    commit id: "PR2 sparse overlay"
```
