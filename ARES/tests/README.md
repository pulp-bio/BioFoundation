Copyright (C) 2026 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file at the root of the repository for details.
# ARES Test Suite

---

## Purpose

This test suite ensures ARES works correctly across:
- **Different network architectures** (CNNs, MLPs, ResNet, DenseNet, Transformers, TinyMyo, Mamba/SSM, FEMBA)
- **Different kernel sizes** (1x1, 3x3, 5x5; includes 1D conv variants)
- **Different operations** (Conv2D, Linear, MaxPool, AvgPool, GlobalAvgPool, Add, Concatenate, MHSA, LayerNorm, GELU, SSM)
- **Different optimizations** (L1 tiling, async DMA, L3 staging, cross-layer fusion)
- **Edge cases** (stride-2, padding, bottleneck layers, deep networks, long-sequence workloads)

---

## Test Networks

**CNN Tests (1-10):**

| Test | Architecture | Notes |
|------|--------------|-------|
| test_1_simplecnn | 2 Conv + 2 Pool + 1 Linear | Baseline CNN |
| test_2_tinycnn | 1 Conv (5x5) + MaxPool + Linear | Large kernel |
| test_3_mlp | 3 Linear (no convolutions) | MLP-only |
| test_4_resnet_basic | ResNet block + GlobalAvgPool | Add, identity skip |
| test_5_densenet_basic | DenseNet block + AvgPool | Concatenate |
| test_6_multitilecnn | Multi-tile Conv2D | Tiling stress test |
| test_7_bottleneck | 1x1 bottleneck CNN | Channel reduction/expansion |
| test_8_stride2 | Stride-2 Conv2D | Downsampling |
| test_9_padding | Padding variations | Halo handling |
| test_10_resnet18 | ResNet-18 (77 layers) | L3 staging, projection shortcuts |

**Transformer Tests (11-14):**

| Test | Architecture | Notes |
|------|--------------|-------|
| test_11_layernorm_basic | LayerNorm validation | Normalization pipeline |
| test_12_gelu_basic | GELU activation | Activation function |
| test_13_transformer_simple | 1-block transformer | LayerNorm, MHSA, GELU |
| test_14_multiblock_transformer | 4-block transformer | Scalability validation |

**Mamba/SSM Tests (15-20):**

| Test | Architecture | Notes |
|------|--------------|-------|
| test_15_tinymyo_tiny | Tiny TinyMyo | Fast GVSOC validation |
| test_16_mamba_conv1d | Depthwise Conv1D + SiLU | Mamba building block |
| test_17_mamba_ssm | SSM core | Discretization + scan |
| test_18_mamba_block | Full Mamba block | in_proj, conv1d, SSM, gating, out_proj |
| test_19_mamba_stacked | 3 stacked Mamba blocks | Cycle scaling test |
| test_20_bidirectional_mamba | Bidirectional wrapper | fwd + rev + add |

**FEMBA Tests (21-25):**

| Test | Architecture | Notes |
|------|--------------|-------|
| test_21_femba_patchembedder | PatchEmbed + 2 BiMamba | Front-end validation |
| test_22_femba_full | Full FEMBA | Positional embedding, residuals |
| test_23_femba_full_input | FEMBA with full input | 22x1280 input dimensions |
| test_24_femba_full_expand2 | FEMBA expand=2 | L3 streaming |
| test_25_femba_tiny_int8 | FEMBA production INT8 | Full architecture |

**Additional Tests (26-30):**

| Test | Architecture | Notes |
|------|--------------|-------|
| test_26_tinymyo_8ch_400tok | TinyMyo 8-channel | Stress test |
| test_27_linear3d_bench | 3D Linear benchmark | Tiling tuning |
| test_28_conv2d_remainder | Conv2D edge case | im2col remainder |
| test_29_luna_base | LUNA ops | GroupNorm, RFFT, RoPE |
| test_30_autotune_stress | Auto-tuner validation | Unusual dimensions |

**Key achievements:**
- CNN/ResNet/DenseNet/Transformer/Mamba/FEMBA coverage
- Deep networks validated: ResNet-18 (77 layers), multi-block transformers
- Mamba/SSM building blocks: Conv1D, SSM core, bidirectional

---

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

**Required packages (aligned with requirements.txt):**
- `torch==2.8.0` - PyTorch framework
- `einops==0.8.1` - Attention/sequence utilities
- `brevitas==0.12.1` - Quantization-aware training
- `numpy>=1.20.0` - Numerical operations
- `mako>=1.2.0` - Template engine for C code generation

---

## Usage

### Generate All Tests

```bash
PY=${PY:-python}
$PY tests/generate_all_tests.py
```

**This command:**
1. Loads MNIST dataset
2. Trains each network (5 epochs on 1000 samples) or loads cached models
3. Extracts Brevitas quantization parameters
4. Generates golden INT8 outputs
5. Generates GAP9 C code with templates
6. Organizes outputs in `tests/outputs/<test_name>/`

**Estimated time:** 10-15 minutes (using cached models)

---

### Generate Single Test

```bash
PY=${PY:-python}
$PY tests/generate_all_tests.py --test test_1_simplecnn --skip-gvsoc
```

**Options:**
- `--test <name>`: Generate specific test network
- `--skip-gvsoc`: Skip GAP9 compilation and execution (faster iteration)
- `--output-dir <path>`: Custom output directory (default: `tests/outputs/`)

**Estimated time:** 1-2 minutes per test (using cached model)

---

### Run Tests on GAP9

Automated testing on GAP9 GVSOC simulator:

```bash
PY=${PY:-python}
$PY tests/run_gap9_projects.py
```

**Or test specific networks:**

```bash
PY=${PY:-python}
$PY tests/run_gap9_projects.py --tests test_1_simplecnn test_4_resnet_basic
```

**This command:**
1. Sources GAP9 SDK environment
2. Compiles each test (C code → GAP9 binary)
3. Runs on GVSOC simulator
4. Validates layer-by-layer outputs
5. Reports error rates and performance metrics

**Estimated time:** 2-3 minutes per test (Maybe more if very large network!!)

---

## Output Structure

```
tests/outputs/
├── test_1_simplecnn/
│   ├── models/
│   │   └── test_1_simplecnn.pth        # Trained PyTorch model
│   ├── golden_outputs/
│   │   ├── network_info.json           # Network structure & scales
│   │   ├── weights/                    # INT8 weights, FP32 biases
│   │   │   ├── conv1_weight_int8.npy
│   │   │   ├── conv1_bias_fp32.npy
│   │   │   └── ...
│   │   └── test_cases/
│   │       ├── summary.json            # Test case summary
│   │       └── test_case_1/
│   │           ├── input_fp32.npy      # Test input
│   │           ├── output_fp32.npy     # Expected output
│   │           ├── metadata.json       # Test metadata
│   │           └── intermediate_int8/  # Layer-by-layer INT8
│   │               ├── conv1_int8.npy
│   │               ├── relu1_int8.npy
│   │               └── ...
│   └── generated/                      # C code for GAP9
│       ├── inc/
│       │   ├── network.h
│       │   ├── network_kernels.h
│       │   ├── network_data.h
│       │   ├── network_dma_pipeline.h  # Async DMA headers
│       │   ├── network_l3_prefetch.h   # L3 staging headers
│       │   └── mem.h
│       ├── src/
│       │   ├── main.c                  # Entry point
│       │   ├── network.c               # Orchestration
│       │   ├── network_kernels.c       # Operations
│       │   ├── network_dma_pipeline.c  # Async DMA (722 lines)
│       │   ├── network_l3_prefetch.c   # L3→L2 prefetch (90 lines)
│       │   └── mem.c
│       ├── bin/                        # Binary weight/golden files
│       │   ├── conv1_weight.bin
│       │   ├── conv1_bias.bin
│       │   ├── conv1_golden.bin
│       │   └── ...
│       ├── Makefile                    # GAP9 build system
│       └── gvsoc_run.log               # Execution log (after running)
│
├── test_2_tinycnn/
│   └── ... (same structure)
...
├── test_10_resnet18/
│   └── ... (same structure)
```

---

## Manual Testing on GAP9

For detailed debugging or performance analysis:

```bash
# Navigate to test directory
cd tests/outputs/test_1_simplecnn/generated

# Source GAP9 SDK environment
# Build
make clean all platform=gvsoc

# Run on GVSOC simulator
make run platform=gvsoc

# Check execution log
cat gvsoc_run.log
```

**Expected output:**
```
CL: conv1 using L1 tiling: 4 tiles (2x2)
CL: conv1 pipeline DMA overlap: 86.1%
Layer conv1 validation: 0 mismatches (max_diff=0)
Layer relu1 validation: 0 mismatches (max_diff=0)
Layer pool1 validation: 0 mismatches (max_diff=0)
...
Network validation: 0.0% error
```

---

## Validation Details

### Error Rate Calculation

```
error_rate = (num_mismatches / total_elements) * 100
```

- **0.0% error:** Bit-exact match with Python reference (core suite)
- **≤1% error:** Normal quantization tolerance target (e.g., extended tests)
- **>1% error:** Investigate; test_21_linear_l3_minimal currently at 2.88% (known tolerance issue)

### Layer-by-Layer Validation

Each test includes layer-wise golden output comparison:
- **Mismatches:** Number of elements with `|c_output - golden| > 0`
- **max_diff:** Maximum absolute difference
- **mean_diff:** Average absolute difference

**Example:**
```
Layer conv1 validation: 0 mismatches (max_diff=0)
Layer relu1 validation: 0 mismatches (max_diff=0)
Layer pool1 validation: 0 mismatches (max_diff=0)
Layer conv2 validation: 0 mismatches (max_diff=0)
...
```

---

## Expected Results

**CNN Tests (1-10):** Representative resource/cycle expectations; see each `gvsoc_run.log` for exact numbers.

| Test | Key Features Validated |
|------|------------------------|
| test_1_simplecnn | Baseline async DMA, L1 tiling |
| test_2_tinycnn | Large kernels (5x5) |
| test_3_mlp | Linear-only, no convolutions |
| test_4_resnet_basic | Add, GlobalAvgPool, fusion |
| test_5_densenet_basic | Concatenate, AvgPool, fusion |
| test_6_multitilecnn | Multi-tile Conv2D |
| test_7_bottleneck | 1x1 convolutions |
| test_8_stride2 | Stride-2 downsampling |
| test_9_padding | Padding variations |
| test_10_resnet18 | Deep network, L3 staging (~800KB L2, ~400KB L3) |

**Transformer Tests (11-14):**

| Test | Key Features Validated |
|------|------------------------|
| test_11_layernorm_basic | LayerNorm pipeline |
| test_12_gelu_basic | GELU activation |
| test_13_transformer_simple | Full transformer block |
| test_14_multiblock_transformer | 4-block transformer scalability |

**Mamba/SSM Tests (15-20):**

| Test | Key Features Validated |
|------|------------------------|
| test_15_tinymyo_tiny | TinyMyo fast validation |
| test_16_mamba_conv1d | Depthwise Conv1D + SiLU |
| test_17_mamba_ssm | SSM discretization + scan |
| test_18_mamba_block | Full Mamba block pipeline |
| test_19_mamba_stacked | Stacked blocks cycle scaling |
| test_20_bidirectional_mamba | Bidirectional wrapper |

**FEMBA Tests (21-25):**

| Test | Key Features Validated |
|------|------------------------|
| test_21_femba_patchembedder | Patch embedding + BiMamba |
| test_22_femba_full | Full FEMBA architecture |
| test_23_femba_full_input | Full input dimensions |
| test_24_femba_full_expand2 | FEMBA expand=2, L3 streaming |
| test_25_femba_tiny_int8 | FEMBA production INT8 |

**Additional Tests (26-30):**

| Test | Key Features Validated |
|------|------------------------|
| test_26_tinymyo_8ch_400tok | 8-channel TinyMyo stress test |
| test_27_linear3d_bench | Linear3D tiling benchmark |
| test_28_conv2d_remainder | Conv2D im2col remainder |
| test_29_luna_base | GroupNorm, RFFT, RoPE |
| test_30_autotune_stress | Auto-tuner unusual dimensions |

---

## Troubleshooting

### MNIST Download Fails

**Problem:** Automatic MNIST download times out or fails

**Solution:**
```bash
# Create data directory
mkdir -p tests/data/mnist

# Download manually from https://ossci-datasets.s3.amazonaws.com/mnist/
# Or use torchvision in Python:
python -c "import torchvision; torchvision.datasets.MNIST('tests/data', download=True)"
```

---

### Out of Memory on GAP9

**Problem:** L2 allocation fails during execution

**Solution:**
- Check L2 usage in gvsoc_run.log: "L2 Memory Usage Profile"
- For large networks, L3 staging is automatic (>36KB threshold)
- Consider using smaller test (e.g., test_2_tinycnn)

---

### C Code Generation Fails

**Problem:** `KeyError` or `ValueError` during code generation

**Solution:**
- Check that network_info.json is complete
- Verify all weights/*.npy files exist
- Ensure test_case directory exists
- Re-run extraction: `python tests/generate_all_tests.py --test <name> --skip-gvsoc`

---

### GAP9 Compilation Fails

**Problem:** `make clean all platform=gvsoc` fails

**Solution:**
- Source GAP9 SDK environment: `source ~/gap_sdk/configs/gap9_v2.sh`
- Check that GAP SDK is installed correctly
- Verify C compiler warnings for type mismatches
- Check that bin/*.bin files exist

---

### Runtime Errors on GAP9

**Problem:** Segmentation fault, stack overflow, or wrong outputs

**Solution:**
- **Stack overflow:** Increase stack size in templates (currently 8192 bytes, works for all tests)
- **Wrong outputs:** Check layer-by-layer validation, verify scales in network_info.json
- **Segfault:** Verify buffer sizes in network_data.h, check for null pointers

---

## Adding New Tests

To add a new test network:

### 1. Create Network Definition

Create `tests/test_networks/test_31_mynetwork.py`:

```python
import torch
import torch.nn as nn
from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantIdentity

class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant_inp = QuantIdentity(bit_width=8, return_quant_tensor=True)
        self.conv1 = QuantConv2d(1, 16, kernel_size=3, padding=1, weight_bit_width=8)
        self.relu1 = QuantReLU(bit_width=8, return_quant_tensor=True)
        # ... more layers ...

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.conv1(x)
        x = self.relu1(x)
        # ... more layers ...
        return x
```

### 2. Add to Test Registry

Edit `tests/generate_all_tests.py` and add to `NETWORKS` dict:

```python
NETWORKS = {
    # ... existing tests ...
    'test_31_mynetwork': {
        'class': MyNetwork,
        'description': 'Custom network description',
        'epochs': 5,
    }
}
```

### 3. Generate and Validate

```bash
PY=${PY:-python}
$PY tests/generate_all_tests.py --test test_31_mynetwork
```

### 4. Verify Results

Check `tests/outputs/test_31_mynetwork/generated/gvsoc_run.log`:
- Target: 0.0% error
- Acceptable: max_diff ≤ 1 for most layers

---

## Coverage Highlights

---

## Test Network Details

### CNN Tests (1-10)

**test_1_simplecnn:** Standard CNN baseline
- 2 Conv2D (3x3) → 2 MaxPool → 1 Linear
- Validates basic tiling and async DMA

**test_2_tinycnn:** Minimal architecture
- 1 Conv2D (5x5 kernel) → 1 MaxPool → 1 Linear
- Validates large kernel handling

**test_3_mlp:** Pure MLP
- 3 Linear layers (no convolutions)
- Validates Linear-only networks

**test_4_resnet_basic:** ResNet block
- Skip connection (Add operation)
- GlobalAvgPool instead of MaxPool
- Validates fusion and residual connections

**test_5_densenet_basic:** DenseNet block
- Dense connection (Concatenate operation)
- AvgPool with fusion
- Validates channel concatenation

**test_6_multitilecnn:** Multi-tile validation
- 3 Conv2D + 4 MaxPool
- Validates tile iteration and DMA batching

**test_7_bottleneck:** Channel reduction
- 1x1 convolutions for dimension reduction/expansion
- Validates efficient convolutions

**test_8_stride2:** Spatial downsampling
- Stride-2 Conv2D and MaxPool
- Validates strided operations

**test_9_padding:** Padding edge cases
- Various padding configurations
- Validates halo management

**test_10_resnet18:** Production ResNet
- Full ResNet-18 architecture (77 layers)
- ~700K parameters
- L3 staging for large weights
- Projection shortcuts (1x1 conv stride=2)

---

### Transformer Tests (11-14)

**test_11_layernorm_basic:** LayerNorm validation
- Flatten → LayerNorm → Linear
- Validates normalization pipeline

**test_12_gelu_basic:** GELU activation validation
- Linear → GELU → Linear
- Confirms GELU path in INT8 pipeline

**test_13_transformer_simple:** Simplified transformer
- 1 block with LayerNorm, MHSA, GELU
- Full transformer block coverage

**test_14_multiblock_transformer:** Multi-block transformer
- 4 blocks, 128 dim, 4 heads
- Scalability validation

---

### Mamba/SSM Tests (15-20)

**test_15_tinymyo_tiny:** TinyMyo Tiny
- 1 block, 192 dim, 3 heads, sequence length 50
- Fast GVSOC validation

**test_16_mamba_conv1d:** Mamba 1D conv building block
- Depthwise Conv1D + SiLU
- Validates Mamba front-end

**test_17_mamba_ssm:** SSM core
- State space model with discretization + scan
- Validates SSM computation

**test_18_mamba_block:** Full Mamba block
- in_proj, conv1d, SiLU, SSM, gating, out_proj
- Complete block validation

**test_19_mamba_stacked:** Stacked Mamba blocks
- 3 stacked blocks
- Cycle scaling validation

**test_20_bidirectional_mamba:** Bidirectional wrapper
- Forward + reverse + add
- Validates bidirectional processing

---

### FEMBA Tests (21-25)

**test_21_femba_patchembedder:** PatchEmbed + BiMamba
- Patch embedding + 2 bidirectional Mamba blocks
- Front-end validation

**test_22_femba_full:** Full FEMBA
- Positional embedding, residuals, LayerNorm
- Full architecture validation

**test_23_femba_full_input:** FEMBA with full input
- 22x1280 input dimensions
- Full-scale validation

---

## References

- [Main README](../README.md) - Repository overview
- [Atomic Operations](../atomic_ops/README.md) - INT8 operation references
- [Code Generation](../codegen/README.md) - Template system
- [Tools](../tools/README.md) - Extraction and validation
