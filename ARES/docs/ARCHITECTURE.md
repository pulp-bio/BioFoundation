# Architecture

This document is the “mental model” for ARES: why it exists, what it generates, and where to look when changing things.

## The “Why”

- **Why INT8?** GAP9 performance + memory constraints demand integer kernels. ARES targets **bit-exact** INT8 math (not FP32 simulation) so validation is meaningful.
- **Why tiling?** L1 is fast but small, so compute is split into **tiles** that fit in L1 (inner loop), with optional **slabs** for L3→L2 streaming (outer loop).
- **Why 9 cluster cores?** GAP9 has a Fabric Controller (FC) plus a 9-core cluster. **Core 8 orchestrates**; **Cores 0–7 compute**.

## Pipeline Overview

```
PyTorch/Brevitas model
  → tools/pytorch_extractor.py        (extract weights/scales → network_info.json)
  → tools/generate_golden_outputs.py  (Python INT8 reference “golden” outputs)
  → codegen/generate_c_code.py        (plan memory + tiling, render templates)
  → GAP SDK (gvsoc / hardware)        (compile + run + validate)
```

## GAP9 Memory Hierarchy (concrete)

```
L3 (HyperRAM)      ~64–100 MB   slow, managed by FC; used for large weights/acts
  ↓ (stream slabs)
L2 (cluster SRAM)  ~1.5 MB      shared, primary activation/weight working set
  ↓ (DMA tiles)
L1 (TCDM)          128 KB total, ~104 KB usable for tiling buffers (stack/reserved space)
```

In code, the “usable L1” assumption is baked into `codegen/gap9_model.py` so tiling decisions remain conservative.

## FC vs Cluster (Core 8 vs 0–7)

- **FC (Fabric Controller)**: runs `main.c`, filesystem, L3 allocation, dispatches the cluster task.
- **Cluster Core 8 (orchestrator)**: allocates L1/L2 buffers, schedules DMA, forks work to worker cores.
- **Cluster Cores 0–7 (workers)**: run the parallel compute kernels inside `pi_cl_team_fork(8, ...)`.

Common pitfall: treating Core 8 like a worker. Most “do once” orchestration logic should run on Core 8.

## “If you want to do X, look in Y”

| Task | File/Dir |
|------|----------|
| Add a new test network | `tests/test_networks/` + `tests/generate_all_tests.py` |
| Add/modify Python INT8 reference op | `atomic_ops/` |
| Add/modify extracted layer support | `tools/pytorch_extractor.py` |
| Fix golden generation / reference pipeline | `tools/generate_golden_outputs.py`, `tools/int8_inference.py` |
| Change tiling heuristics / budgets | `codegen/gap9_model.py` |
| Change memory planning (arena) | `codegen/generate_c_code.py` (`MemoryPlanner`) |
| Change C kernels / runtime | `codegen/runtime/src/`, `codegen/runtime/inc/` |
| Change generated project structure | `codegen/templates/` |
| Run GAP9 regression suite | `tests/run_gap9_projects.py` |

## Modular Runtime Structure

The C runtime is split into per-operation modules for maintainability and to
support standalone project export (see `tools/package_standalone.py`):

```
codegen/runtime/
├── src/
│   ├── core/utils.c          # Common utilities (qround, clamp, fast_exp)
│   ├── ops/op_activation.c   # GELU, SiLU, ReLU, requantize
│   ├── ops/op_pool.c         # MaxPool, AvgPool, GlobalAvgPool, AdaptiveAvgPool
│   ├── ops/op_norm.c         # LayerNorm, GroupNorm
│   ├── ops/op_elementwise.c  # Add, Concat, Transpose
│   ├── ops/op_linear.c       # Linear tile workers
│   ├── ops/op_conv2d.c       # Conv2D tile workers
│   ├── ops/op_mhsa.c         # MHSA permute, RoPE
│   ├── network_kernels.c     # Core dispatch, SSM/Mamba ops, softmax
│   ├── network_dma_pipeline.c # DMA transfer management
│   └── network_executor.c    # Data-driven layer executor
├── inc/
│   ├── core/utils.h
│   ├── ops/*.h               # Per-op headers
│   └── *.h                   # Core runtime headers
```

**Key design principles:**

1. **Single source of truth**: Ops live in `runtime/src/ops/`, compiled via Makefile.
   Generated projects do NOT have local copies of op files.

2. **Header inclusion**: `network_kernels.h` includes all op headers, so any
   C file including it has access to all operations.

3. **Parallel compilation**: The Makefile compiles `$(RUNTIME_DIR)/src/ops/*.c`
   separately, enabling parallel builds and better cache behavior.

