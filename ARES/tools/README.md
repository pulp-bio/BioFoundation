Copyright (C) 2026 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file at the root of the repository for details.
# Tools
---

## Overview

This directory contains the critical tools that bridge PyTorch/Brevitas models to GAP9 deployment:

1. **pytorch_extractor.py** - Extracts quantization metadata and weights from Brevitas models
2. **int8_inference.py** - Python INT8 inference engine for validation
3. **generate_golden_outputs.py** - Creates golden reference outputs for hardware testing
4. **check_model_compatibility.py** - Validates model compatibility before extraction/codegen

Tools target bit-exact matching for all tests in the regression suite.

---

## Performance Profiling

## Repo Maintenance

### profile_suite.py

Runs gvsoc across a set of already-generated tests and writes CSV summaries:
- `tests.csv`: per-test totals + linker memory usage (L1/L2) + known-MACs summary (Conv/Linear/MHSA where parsable)
- `layers.csv`: per-layer `PERF` breakdown (total/compute/DMA/idle) + MACs (where known)
- `ssm_phases.csv`: SSM PH1/PH2/PH3 phase counters (optional)
- `subops.csv`: additional `PERF <name>: total=<cycles>` items (e.g. MambaBlock internal stage breakdowns)

Example:
```bash
python tools/profile_suite.py --enable-ssm-events
```

### merge_profile_runs.py

Merges multiple `profiling/gvsoc_*` runs into a single set of CSVs (later inputs override earlier ones):
```bash
python tools/merge_profile_runs.py \
  --out-dir profiling/gvsoc_merged \
  profiling/gvsoc_20251227_155404 profiling/gvsoc_20251227_173759
```

## pytorch_extractor.py

**Purpose:** Extracts quantization parameters and INT8 weights directly from Brevitas PyTorch models

**Why direct extraction:** ONNX's quantization representation is fragile and unreliable for Brevitas models. Direct PyTorch extraction ensures correctness.

### Usage

```bash
PY=${PY:-python}
$PY tools/pytorch_extractor.py
```

### Process

1. **Load trained model** from `models/<test_name>.pth`
2. **Detect Brevitas layers:**
   - QuantConv2d
   - QuantLinear
   - QuantReLU
   - QuantIdentity
   - QuantMaxPool
   - QuantAvgPool
   - QuantAdd (ResNet skip connections)
   - QuantConcatenate (DenseNet dense connections)
   - QuantMultiHeadAttention (Transformer layers)
3. **Extract runtime scales** via forward passes with actual data
4. **Extract INT8 weights** from Brevitas quantizers
5. **Build layer_order** (execution sequence, topological order)
6. **Serialize to JSON** with all metadata

### Outputs

**network_info.json** - Complete network structure:
```json
{
  "input_quant": {
    "type": "QuantIdentity",
    "scale": 0.0078125
  },
  "conv1": {
    "type": "QuantConv2d",
    "in_channels": 1,
    "out_channels": 16,
    "kernel_size": [3, 3],
    "stride": [1, 1],
    "padding": [1, 1],
    "scale_input": 0.0078125,
    "scale_weight": 0.003456,
    "scale_output": 0.015234
  },
  "__layer_order__": ["input_quant", "conv1", "relu1", ...]
}
```

**weights/ directory** - NumPy arrays:
```
golden_outputs/weights/
├── conv1_weight_int8.npy
├── conv1_bias_fp32.npy
├── conv2_weight_int8.npy
├── conv2_bias_fp32.npy
└── ...
```

### Key Features

- **Runtime scale extraction:** Uses actual forward passes to capture dynamic scales
- **Multi-input operation support:** Tracks tensor IDs for Add/Concatenate inputs
- **Custom layer detection:** Supports both `isinstance()` checks and `__class__.__name__` matching
- **ResNet/DenseNet support:** Handles skip connections, dense connections, projection shortcuts

## check_model_compatibility.py

**Purpose:** Standalone compatibility validator that mirrors extractor support checks and reports unsupported patterns early.

### Usage

```bash
python tools/check_model_compatibility.py --model-file my_model.py --model-class MyModel
```

Built-in test model example:

```bash
python tools/check_model_compatibility.py --test-network test_4_resnet_basic --model-class ResNetBasic
```

JSON report output:

```bash
python tools/check_model_compatibility.py \
  --test-network test_13_transformer_simple \
  --model-class SimpleTransformer \
  --json-out /tmp/compat_report.json
```

Model-loading conventions:
- `--model-file`: loads from a Python file path.
- `--test-network`: loads from `tests/test_networks/*`.
- If `--model-class` is omitted and the module exposes `create_model()`, that factory is used.
- Otherwise, the checker resolves an `nn.Module` class from the module and instantiates it.

Related docs helpers:

- `python tools/generate_supported_operations_doc.py`
- `python tools/validate_supported_operations_doc.py`

### Implementation Details

**Layer detection:** `tools/pytorch_extractor.py`
- Detects Brevitas layers (Conv, Linear, ReLU/Identity, Pooling, Add, Concatenate, MultiHeadAttention) and custom layers (LayerNorm, GELU variants, MHSA integer softmax paths).
- Maps quantizers to INT8 weights and records tensor IDs for multi-input ops.

**Scale extraction:**
- Runs forward passes with sample inputs to capture runtime scales.
- Propagates scales through the graph and preserves tensor-to-layer mappings for fusion and validation.

---

## int8_inference.py

**Purpose:** Python INT8 inference engine that chains atomic operations to verify end-to-end INT8 execution

### Usage

```bash
PY=${PY:-python}
$PY tools/int8_inference.py
```

### Key Class: INT8InferenceEngine

**Process:**
1. Loads network_info.json and weights
2. Chains atomic operations (`atomic_ops/*.py`) in layer_order
3. Executes true INT8 inference (not FP32 simulation!)
4. Captures intermediate INT8 activations
5. Returns FP32 final output for validation

### Methods

```python
class INT8InferenceEngine:
    def forward(self, input_fp32: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Execute full network in INT8.

        Returns:
            - output_fp32: Final FP32 logits
            - intermediates: Dict of INT8 activations per layer
        """
```

**Per-operation handlers:**
- `_forward_conv2d()` - INT8 convolution with rescaling
- `_forward_linear()` - INT8 linear with INT8 or FP32 output
- `_forward_relu()` - INT8 ReLU (in-place)
- `_forward_maxpool()` - INT8 max pooling
- `_forward_avgpool()` - INT8 average pooling with INT32 accumulation
- `_forward_globalavgpool()` - Spatial reduction (HxW → 1x1)
- `_forward_add()` - Element-wise addition with scale matching
- `_forward_concat()` - Channel concatenation
- `_forward_requantize()` - QuantIdentity (scale conversion)
- `_forward_flatten()` - Reshape (no computation)
- `_forward_layernorm()` - INT8 LayerNorm (FP32 or integer-only path)
- `_forward_gelu()` - INT8 GELU (LUT/integer path for bit-exact matching)
- `_forward_mhsa()` - INT8 hybrid attention with optional integer softmax
- `_forward_conv1d_depthwise()` / `_forward_silu()` - Mamba/Conv1D building blocks

### Fusion Detection

**Critical feature:** Mirrors C code fusion decisions to ensure bit-exact golden outputs

**Detected patterns:**
1. Conv→ReLU→Quant (3-way fusion)
2. Linear→ReLU→Quant (3-way fusion)
3. GlobalAvgPool→Quant (2-way fusion)
4. AvgPool→Quant (2-way fusion)

**Implementation:** `tools/int8_inference.py:200-350`
- Scans layer_order for fusion patterns
- Combines operations into single atomic call
- Uses `requantize_int8()` for direct INT8→INT8 conversion
- Eliminates intermediate buffers (matches C code behavior)

### ResNet Support

**Activation caching:** Stores intermediate activations for Add operations
```python
self.activation_cache = {}  # Stores inputs for skip connections
```

**Add operation:** Retrieves cached activations from earlier layers
```python
input1 = self.activation_cache[input1_layer]
input2 = self.activation_cache[input2_layer]
output = add_int8(input1, input2, scale1, scale2, scale_out)
```

---

## generate_golden_outputs.py

**Purpose:** Generates golden reference INT8 outputs for hardware verification on GAP9

### Usage

```bash
PY=${PY:-python}
$PY tools/generate_golden_outputs.py
```

### Process

1. **Load quantized PyTorch model** from `models/<test_name>.pth`
2. **Load MNIST test samples** (or custom test data)
3. **Run INT8InferenceEngine** for each test case
4. **Save intermediate INT8 activations** for layer-by-layer validation
5. **Save final FP32 output** for end-to-end validation
6. **Update network_info.json** with test case metadata

### Outputs

**test_cases/ directory structure:**
```
golden_outputs/test_cases/
├── summary.json                        # Test case summary
├── test_case_1/
│   ├── input_fp32.npy                  # Input image (28x28 or other)
│   ├── output_fp32.npy                 # Expected FP32 output
│   ├── metadata.json                   # Test case metadata
│   └── intermediate_int8/
│       ├── conv1_int8.npy              # Conv1 INT8 output
│       ├── relu1_int8.npy              # ReLU1 INT8 output
│       ├── pool1_int8.npy              # Pool1 INT8 output
│       └── ...
├── test_case_2/
│   └── ... (same structure)
└── test_case_3/
    └── ... (same structure)
```

### Test Case Selection

**Default:** test_case_3 (MNIST test sample #3)
**Fallback:** test_case_1 if test_case_3 doesn't exist

**Why test_case_3?** Consistent with baseline SimpleCNN validation

### Fusion-Aware Golden Generation

**Critical feature:** Mirrors C code fusion to ensure layer-wise validation works correctly

**Example:** Conv→ReLU→Quant fusion
- **C code:** Generates single output after all 3 operations
- **Golden generation:** Must also skip intermediate outputs for relu/quant
- **Validation:** Compares C output to post-quant golden value

**Implementation:** `tools/generate_golden_outputs.py:150-280`
- Detects same fusion patterns as C code
- Skips intermediate golden files for fused operations
- Saves only final fused output for comparison

### Binary File Generation

**For each golden output:**
```python
# Save .npy for Python validation
np.save(f"golden_outputs/test_cases/test_case_1/intermediate_int8/conv1_int8.npy", conv1_out)

# Generate .bin for C code validation
conv1_out.astype(np.int8).tofile(f"generated/bin/conv1_golden.bin")
```

**Format:**
- NumPy `.npy`: Python validation (preserves shape, dtype metadata)
- Binary `.bin`: C code validation (raw bytes, loaded by GAP9)

---

## Validation Workflow

Complete validation flow from extraction to GAP9 execution:

```
1. pytorch_extractor.py
   ↓ (network_info.json, weights/*.npy)

2. generate_golden_outputs.py
   ↓ (test_cases/*/intermediate_int8/*.npy)

3. codegen/generate_c_code.py
   ↓ (generated/src/*.c, generated/bin/*.bin)

4. GAP9 execution
   ↓ (layer-by-layer comparison with golden/*.bin)

5. Validation report
   - All 36 tests in the default regression suite pass with bit-exact matching.
```

---

## Advanced Features

### Multi-Input Operations

**Add operation (ResNet):**
```json
"add": {
  "type": "QuantAdd",
  "input1": "relu2_out",
  "input2": "quant1_out",
  "scale1": 0.015,
  "scale2": 0.012,
  "scale_output": 0.018
}
```

**Concatenate operation (DenseNet):**
```json
"concat": {
  "type": "QuantConcatenate",
  "inputs": ["conv1_out", "conv2_out"],
  "scales": [0.010, 0.012],
  "scale_output": 0.011
}
```

### Custom Brevitas Layers

Defined in `tests/test_networks/brevitas_custom_layers.py`:
- **QuantAdd:** Element-wise addition with output quantizer
- **QuantConcatenate:** Channel concatenation with output quantizer
- **QuantSelfAttention:** Single-head self-attention
- **QuantMultiHeadAttention:** Multi-head wrapper with output projection

### Stale Extraction Metadata Fix

**Problem:** Multi-input operations (Add, Concatenate) can show incorrect input mappings if extraction metadata is stale

**Solution:** Always regenerate when modifying network architecture:
```bash
PY=${PY:-python}
$PY tests/generate_all_tests.py --test test_4_resnet_basic --skip-gvsoc
```

This rebuilds network_info.json with fresh tensor ID tracking.

---

## Troubleshooting

### Extraction Fails

**Problem:** `AttributeError: 'QuantConv2d' object has no attribute 'weight_quant'`
**Solution:** Ensure model is trained with Brevitas quantization enabled

**Problem:** `KeyError: '__layer_order__'`
**Solution:** Re-run pytorch_extractor.py to regenerate network_info.json

### Golden Generation Fails

**Problem:** `ValueError: Tensor ID not found for Add operation`
**Solution:** Regenerate test to rebuild tensor_to_layer mappings

**Problem:** `Shape mismatch in forward pass`
**Solution:** Verify input dimensions match network expectations

### Validation Errors

**Problem:** Non-zero error on specific layer
**Solution:** Check layer-by-layer golden comparison in GAP9 output, verify scales are correct

**Problem:** Error accumulation or tolerance failures
**Solution:** All tests in the regression suite should be bit-exact. If you see tolerance failures, verify scales are correct and check for fusion mismatches between Python and C code.

---

## Capabilities Summary

- Fusion-aware golden generation (mirrors C code fusion)
- ResNet/DenseNet support (activation caching for Add, concatenate with scale preservation)
- Transformer/attention support (MHSA with optional integer softmax, GELU LUT path, LayerNorm integer path)
- Mamba/1D conv building blocks (Conv1D depthwise + SiLU LUT)
- Full regression suite (36 tests) passes with bit-exact matching
- Deep network support (ResNet-18: 77 layers)

---

## References

- [Main README](../README.md) - Repository overview
- [Atomic Operations](../atomic_ops/README.md) - INT8 operation references
- [Code Generation](../codegen/README.md) - Template system
- [Testing](../tests/README.md) - Test network validation
