Copyright (C) 2026 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file at the root of the repository for details.
# Atomic Operations

True INT8 implementations of neural network operations. Each module is self-contained with unit tests and serves as the reference implementation for C code generation.

---

## Overview

This directory contains the Python INT8 reference implementations that:
1. Define bit-exact behavior for each operation
2. Serve as golden references for GAP9 C code validation
3. Include unit tests verifiable via direct execution
4. Use INT32 accumulation to prevent overflow

All operations use **symmetric quantization** (zero_point=0) and **INT32 accumulation** for multiply-accumulate operations.

---

## Naming conventions

- `<op>_int8()` - Primary INT8 implementation
- `<op>_fp32_reference()` - FP32 reference for validation (where applicable)
- `<op>_int8_lut()` - LUT-based variant (used for bit-exact matching, where applicable)

## Operation index

| Module | Description |
|--------|-------------|
| `quantize.py` | FP32↔INT8 conversion helpers (symmetric quantization) |
| `requantize.py` | INT8→INT8 scale conversion (fusion + scale matching) |
| `conv2d.py` | INT8 Conv2D reference |
| `conv1d_depthwise.py` | INT8 depthwise Conv1D (Mamba) |
| `linear.py` | INT8 linear layers (including INT8→FP32 final logits) |
| `relu.py` | Elementwise ReLU (INT8) |
| `add.py` | Elementwise add with scale handling (ResNet) |
| `concat.py` | Channel concat (DenseNet) |
| `maxpool.py` | INT8 max pooling |
| `avgpool.py` | INT8 avg pooling (INT32 accumulate) |
| `globalavgpool.py` | Global average pooling (INT32 partial sums) |
| `layernorm.py` | Integer-only LayerNorm variants (LUT-based inverse sqrt) |
| `gelu.py` | GELU activation (LUT-based variant for bit-exact matching) |
| `softplus.py` | Softplus activation helpers (used by Mamba) |
| `silu.py` | SiLU activation (LUT-based) |
| `ssm.py` | Mamba SSM discretize/scan/gate reference |
| `mhsa.py` | Multi-head self-attention reference (hybrid precision) |
| `flatten.py` | Reshape helpers |
| `transpose.py` | Transpose helpers |
| `flip.py` | Flip helpers |
| `test_i_softmax.py` | Integer softmax reference + tests (used by MHSA) |
| `test_edge_cases.py` | Cross-op edge case suite |

## Core Operations

### quantize.py
**Purpose:** FP32 ↔ INT8 conversion with symmetric quantization

**Quantization formula:**
```python
q = clip(round(x / scale), -128, 127)  # FP32 → INT8
x = q * scale                           # INT8 → FP32
```

**Key feature:** Zero-point = 0 simplifies hardware implementation

---

### conv2d.py
**Purpose:** 2D convolution with INT8 weights and activations

**Process:**
1. INT8 x INT8 → INT32 accumulation (prevents overflow)
2. Add INT32 bias
3. Rescale: multiply by `(scale_x * scale_w) / scale_y`
4. Requantize to INT8 output

**Formula:**
```
Y = ((X ⊗ W + b) * scale_x * scale_w) / scale_y
```

**Supports:** Padding, stride, dilation

**Implementation:** `codegen/runtime/src/network_kernels.c` → `network_conv2d_int8()`

---

### linear.py
**Purpose:** Matrix multiplication (fully connected layer)

**Process:**
1. INT8 matrix multiply → INT32 accumulation
2. Add INT32 bias
3. Rescale with combined scale
4. Requantize to INT8 or return FP32 (for final layer)

**Formula:**
```
Y = ((X @ W^T + b) * scale_x * scale_w) / scale_y
```

**Variants:**
- `linear_int8()`: Intermediate layers (INT8 → INT8)
- `linear_int8_to_fp32()`: Final classifier (INT8 → FP32 logits)

**Implementation:** `codegen/runtime/src/network_kernels.c` → `network_linear_int8()`, `network_linear_int8_to_fp32()`

---

### relu.py
**Purpose:** Element-wise ReLU on INT8 values

**Formula:**
```python
y[i] = max(0, x[i])
```

**Key feature:** Preserves quantization scale (no rescaling needed)

**Implementation:** Parallelized across 8 GAP9 cluster cores with barriers

---

### maxpool.py
**Purpose:** Max pooling on INT8 values

**Process:** Finds maximum in each pooling window

**Key feature:** Max operation is order-preserving, so quantization scale is preserved

**Implementation:** L1-tiled with async DMA pipeline

---

### avgpool.py
**Purpose:** Average pooling with INT32 accumulation

**Process:**
1. Sum INT8 values in window → INT32 accumulator
2. Divide by window size
3. Rescale if needed
4. Requantize to INT8

**Fusion:** Can fuse with QuantIdentity for direct requantization

**Implementation:** L1-tiled with async DMA pipeline

---

### globalavgpool.py
**Purpose:** Global average pooling (spatial reduction HxW → 1x1)

**Process:**
1. Sum all spatial values per channel → INT32
2. Divide by spatial size (H x W)
3. Rescale if needed
4. Requantize to INT8

**Fusion:** Critical for ResNet/DenseNet accuracy - fuses with QuantIdentity to avoid double rounding

**Implementation:** L1-tiled with async DMA pipeline and partial sums

---

### add.py
**Purpose:** Element-wise INT8 addition with scale matching

**Process:**
1. Match scales between two inputs (requantize if needed)
2. Add INT8 values → INT32 accumulator
3. Clip to INT8 range

**Use case:** ResNet skip connections

**Implementation:** Currently L2-only (no L1 tiling yet)

---

### concatenate.py
**Purpose:** Channel-wise concatenation of INT8 tensors

**Process:**
1. Concatenate along channel dimension
2. Preserve quantization scales

**Use case:** DenseNet dense connections

**Implementation:** Currently L2-only (no L1 tiling yet)

---

### requantize.py
**Purpose:** Direct INT8 → INT8 scale conversion (for fusion)

**Process:**
1. Dequantize: `fp32 = int8 * scale_old`
2. Requantize: `int8_new = clip(round(fp32 / scale_new), -128, 127)`

**Use case:** Cross-layer fusion (Conv→ReLU→Quant eliminates intermediate buffers)

**Implementation:** Used in fused kernels (`conv2d_relu_quant_fused()`, `linear_relu_quant_fused()`, etc.)

---

### flatten.py
**Purpose:** Reshape tensor (no computation)

**Process:** Flattens spatial dimensions: `[B, C, H, W] → [B, C*H*W]`

**Implementation:** No-op in C code (pointer aliasing)

---

## Implementation Notes

### INT32 Accumulation
**Required for Conv2D and Linear** to prevent overflow during multiply-accumulate:
- INT8 x INT8 = INT16 (max 127 x 127 = 16,129)
- Sum of N terms needs INT32 when N > 127
- Example: 3x3 Conv with 64 input channels → 576 accumulations → requires INT32

### Rescaling Strategy
Combined scale factor accounts for quantization of inputs, weights, and outputs:
```python
scale_combined = (scale_input * scale_weight) / scale_output
output_int32 = (input_int8 @ weight_int8 + bias_int32)
output_fp32 = output_int32 * scale_combined
output_int8 = quantize(output_fp32, scale_output)
```

### Symmetric Quantization Benefits
Zero-point = 0 for all quantized values simplifies:
- No zero-point bias in accumulation
- Simpler hardware implementation
- Bias remains in INT32 representation (not quantized)

### Cross-Layer Fusion
**Detected patterns:**
1. Conv→ReLU→Quant (3-way fusion)
2. Linear→ReLU→Quant (3-way fusion)
3. GlobalAvgPool→Quant (2-way fusion)
4. AvgPool→Quant (2-way fusion)

**Benefits:**
- 66% reduction in L2 memory traffic
- Eliminates intermediate buffers
- Maintains bit-exact accuracy

---

## Testing

Run the full suite:

```bash
python atomic_ops/run_all_tests.py
```

Or run a subset:

```bash
python atomic_ops/run_all_tests.py conv2d linear
```

Each module typically includes a `test_*()` function and can be run directly:

```bash
# Test individual operations
python atomic_ops/conv2d.py
python atomic_ops/linear.py
python atomic_ops/relu.py
python atomic_ops/maxpool.py
python atomic_ops/avgpool.py
python atomic_ops/globalavgpool.py
python atomic_ops/add.py
python atomic_ops/concat.py
python atomic_ops/requantize.py
```

Tests verify against FP32 reference implementations with quantization tolerance.

---

## Usage Example

```python
import numpy as np
from atomic_ops import quantize_linear, dequantize_linear, conv2d_int8, relu_int8

# Quantize input
x_fp32 = np.random.randn(1, 3, 32, 32).astype(np.float32)
scale_x = 0.01
x_int8 = quantize_linear(x_fp32, scale_x)

# INT8 convolution
w_int8 = np.random.randint(-127, 127, (16, 3, 3, 3), dtype=np.int8)
bias_int32 = np.zeros(16, dtype=np.int32)
scale_w = 0.005
scale_y = 0.02

y_int8 = conv2d_int8(
    x_int8, w_int8, bias_int32,
    scale_x=scale_x,
    scale_w=scale_w,
    scale_y=scale_y,
    stride=(1, 1),
    padding=(1, 1)
)

# ReLU (preserves scale)
y_int8 = relu_int8(y_int8)

# Dequantize for output
y_fp32 = dequantize_linear(y_int8, scale_y)
```

---

## References

- [Main README](../README.md) - Repository overview
- [Code Generation](../codegen/README.md) - How atomic ops map to C code
- [Testing](../tests/README.md) - Test network validation
