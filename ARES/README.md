Copyright (C) 2026 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file at the root of the repository for details.

# ARES - Automated Runtime for Embedded Systems

ARES is a toolchain for deploying quantized neural networks on RISC-V cluster processors (GAP9, Siracusa). It takes Brevitas-quantized PyTorch models and generates optimized C code with bit-exact validation.

## Why does this exist?
I wanted to deploy large-new models to edge devices (GAP9) but found myself lacking the ability to do so. The tools available at the time either didn't support GAP9, didn't handle modern layers like attention and state space models, or required stitching together a chain of external tools just to go from a trained model to C code.

So I did it by hand. Layer by layer, model by model, building on top of Brevitas for quantization. First, a CNN. Then a transformer. Then Mamba blocks. Then cross-attention with rotary position encoding. What started as scripts grew into a codebase. Every new model made it more general. And what you see here today is the outcome of that.

The codebase mainly targets GAP9 with some very developmental support for Siracusa.

## Foundation Models

Please have a look at test_24 (FEMBA), test_26 (TinyMyo), test_31 (LUNA) and test_43 (CEReBrO).

## I want to...

- **Get running quickly** → `docs/GETTING_STARTED.md`
- **Check if my model is compatible** → `tools/check_model_compatibility.py`
- **See what operations are supported** → `docs/SUPPORTED_OPERATIONS.md`
- **Build a model using ARES blocks** → `ares/nn/` and `examples/ares_nn/`
- **Understand the system** → `docs/ARCHITECTURE.md` and `docs/README.md`
- **Run the GAP9 regression suite** → `tests/run_gap9_projects.py`
- **Browse the test networks** → `tests/test_networks/README.md`
- **Add a new operation** → `docs/ADDING_OPERATIONS.md`

## Features

- **Direct PyTorch extraction** - Extracts INT8 weights and quantization scales directly from Brevitas models
- **Automatic memory optimization** - L1 tiling, L3 staging for large weights, cross-layer fusion
- **Async DMA pipeline** - Double-buffered transfers (most of them...)
- **Bit-exact validation** - Layer-wise comparison against Python INT8 reference
- **Target abstraction** - Hardware-specific decisions routed through target objects (GAP9, Siracusa)
- **Pass-based codegen** - Modular pipeline with checkpoint export for debug/replay
- **Performance regression tooling** - Automated metrics collection, comparison, and threshold enforcement

## Validate Your Setup

```bash
python -c "import torch, brevitas, mako, numpy; print('OK', torch.__version__)"
```

Expected output looks like:

```
OK 2.8.0
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate and run a test network on GAP9 simulator (first run can take ~5–15 min due to training)
bash -lc 'python tests/generate_all_tests.py --test test_1_simplecnn'

# Run the GAP9 regression suite (~2–3 min per test, requires GAP SDK + GVSOC)
bash -lc 'python tests/run_gap9_projects.py'
```

## Bringing Your Own Model

ARES works with Brevitas-quantized PyTorch models. The recommended workflow:

### 1. Check compatibility first

Before running the full pipeline, validate your model against ARES's extraction support:

```bash
python tools/check_model_compatibility.py --model-file my_model.py --model-class MyModel
```

The checker classifies every layer as supported, unsupported, or warning, and provides actionable migration advice (e.g., replace `BatchNorm2d` with `LayerNorm`, replace `nn.MultiheadAttention` with ARES's `QuantMultiHeadAttention`).

### 2. Build models from ARES blocks (recommended)

The `ares.nn` package provides building blocks with ARES-compatible quantization defaults. Models built from these blocks are guaranteed to extract correctly:

```python
import torch.nn as nn
import ares.nn as ann

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_quant = ann.QuantIdentity(bit_width=8, return_quant_tensor=True)
        self.conv1 = ann.Conv2d(1, 16, 3, padding=1)
        self.relu = ann.ReLU()
        self.pool = ann.MaxPool2d(2)
        self.flatten = ann.Flatten()
        self.fc = ann.Linear(16 * 14 * 14, 10)

    def forward(self, x):
        x = self.input_quant(x)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.flatten(x)
        return self.fc(x)
```

Pre-built composite blocks are also available: `ann.TransformerBlock` and `ann.ResidualBlock`. See `examples/ares_nn/` for working examples.

### 3. Check the supported operations reference

`docs/SUPPORTED_OPERATIONS.md` lists every supported layer type with its exact Python class, C runtime mapping, quantization requirements, and a migration guide for common unsupported patterns.

## Pipeline Overview

```
PyTorch/Brevitas Model
        │
        ├── tools/pytorch_extractor.py    → Extract INT8 weights & scales
        │
        ├── tools/int8_inference.py       → Generate golden INT8 outputs
        │
        └── codegen/generate_c_code.py    → Pass-based codegen pipeline
                │
                ├── ExtractModelPass       → Load model metadata
                ├── BuildLayerSpecsPass    → Build layer descriptors
                ├── LegacyFusionPass       → Cross-layer fusion (conv+relu, etc.)
                ├── MemoryLevelAnnotation  → L1/L2/L3 residency decisions
                ├── MemoryPlanningPass     → Arena allocation & tiling
                └── EmitCodePass           → Generate target-specific C code
                        │
                        └── GAP9/Siracusa GVSOC execution + validation
```

## Repository Structure

| Directory | Description |
|-----------|-------------|
| `ares/nn/` | User-facing model building blocks with ARES-compatible defaults |
| `atomic_ops/` | Reference INT8 implementations (conv2d, linear, pooling, etc.) |
| `tools/` | Extraction, compatibility checker, golden output generation, perf metrics |
| `codegen/` | C code generation with pass pipeline |
| `codegen/pipeline/` | Pass-based codegen orchestration |
| `codegen/targets/` | Target abstraction (GAP9, Siracusa) |
| `codegen/memory/` | Memory-level model and planner policies |
| `codegen/optimization/fusion/` | Cross-layer fusion system |
| `codegen/checkpoints/` | Phase checkpoint export for debug/replay |
| `codegen/runtime/` | Shared C runtime (kernels, DMA, ops modules) |
| `tests/` | Test networks and automation scripts |
| `examples/` | Usage examples (`ares_nn/` for model building blocks) |
| `scripts/` | Validation orchestration (tiered gating) |
| `docs/` | Architecture docs, supported operations reference, and reports |

## Test Suite

| Category | Tests | Description |
|----------|-------|-------------|
| Core CNN | 1-10 | Basic to complex CNNs, ResNet-18, strides, padding |
| Transformer | 11-14 | LayerNorm, GELU, multi-head attention |
| Mamba/SSM | 15-20 | TinyMyo, Conv1D, SSM core, bidirectional |
| FEMBA | 21-25 | Patch embedding, full FEMBA architecture |
| Additional | 26-31 | Benchmarks, edge cases, LUNA full |
| Additional | 36-37 | Drowsiness fusion, ZeroPad2d |

Run specific tests:
```bash
bash -lc 'python tests/run_gap9_projects.py --tests test_1_simplecnn test_4_resnet_basic'
```

See `docs/PERFORMANCE_BASELINE.md` for detailed performance metrics (cycles, MACs, MACs/cycle) per test.

## Target Architectures

ARES supports RISC-V cluster processors with L3/L2/L1 memory hierarchies:

**GAP9** (production target):
- Fabric Controller (FC) + 8 worker cores + Cluster Controller (Core 8)
- Memory: L3 (HyperRAM) → L2 (1.5MB) → L1 (~128KB)
- NE16 neural engine accelerator enabled

**Siracusa** (new target):
- Similar cluster architecture with larger memories
- Memory: L3 → L2 (2MB) → L1 (~256KB)

## Environment Setup

1. **Python**: Install dependencies with `pip install -r requirements.txt`
2. **GAP SDK**: Edit `tools/gap9_env_gvsoc.sh` to point to your installation
3. **Verify**: `bash -lc "source tools/gap9_env_gvsoc.sh && which gapy"`

## Documentation

- `docs/SUPPORTED_OPERATIONS.md` - Supported operations reference and migration guide
- `docs/GETTING_STARTED.md` - Installation, first run, and compatibility checking
- `docs/ARCHITECTURE.md` - System architecture and pipeline design
- `docs/ADDING_OPERATIONS.md` - End-to-end guide for adding new operations
- `docs/PERFORMANCE_BASELINE.md` - Test suite performance metrics (cycles, MACs, MACs/cycle)
- `docs/README.md` - Documentation hub (what to read next + code map)

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full text.

See [NOTICE](NOTICE) for third-party acknowledgments.
