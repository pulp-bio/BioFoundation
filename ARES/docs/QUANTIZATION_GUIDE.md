# Quantization Guide: Fake to True INT8 Conversion

This document explains how ARES converts Brevitas fake-quantized models to true INT8 implementations, and how operations like softmax are handled.

## 1. Fake vs True Quantization

### Brevitas Fake Quantization (Training Time)
- Values remain FP32 but are rounded/clipped to simulate INT8 behavior
- Brevitas stores the quantization `scale` as a learnable parameter
- Formula: `fake_q = round(x / scale) * scale` (still FP32!)

### ARES True Quantization (Deployment Time)
- We extract the learned `scale` by running a forward pass and accessing `output.scale` from Brevitas QuantTensor
- We convert to actual INT8: `true_q = clip(round(x / scale), -128, 127)`

## 2. Scale Extraction

The conversion happens in `tools/pytorch_extractor.py`. Scales are extracted by running forward passes:

```python
def _extract_scale_via_forward(self, module, input_shape=None) -> float:
    """Extract scale by running a forward pass and checking the QuantTensor."""
    dummy_input = torch.randn(1, *input_shape)
    with torch.no_grad():
        output = module(dummy_input)

    # Brevitas QuantTensor has .scale attribute
    if hasattr(output, 'scale'):
        return float(output.scale.detach().cpu().item())
```

## 3. Weight Quantization (FP32 → INT*)

Weights are exported as *real integer tensors* (typically INT8, sometimes lower bit-width).

```python
def _quantize_weights(self, x: np.ndarray, scale: float, bit_width: int = 8) -> np.ndarray:
    # Symmetric signed quantization range:
    #   qmax = 2^(bit_width-1) - 1
    #   qmin = -qmax
    #
    # For int8 weights this is typically [-127, 127] (symmetric around 0).
    qmax = (1 << (bit_width - 1)) - 1
    qmin = -qmax

    x_scaled = x / scale
    x_rounded = np.round(x_scaled)      # NumPy-style ties-to-even
    x_clipped = np.clip(x_rounded, qmin, qmax)
    return x_clipped.astype(np.int8)
```

## 4. Softmax Handling

**Softmax is NOT true-quantized to INT8.** It uses a hybrid FP32/INT8 approach with LUT-based integer approximation.

### Why Not INT8 Softmax?
1. `exp()` has huge dynamic range that INT8 cannot represent
2. The sum-normalization `exp(x) / Σexp(x)` requires careful precision handling
3. Numerical stability is critical for attention mechanisms

### MHSA Precision Strategy

From `atomic_ops/mhsa.py`:

```
Q/K/V projections: INT8 → INT8 (quantized linear transformations)
Attention scores:  INT8 x INT8 → INT32 → FP32
Softmax:           FP32 (for numerical stability)  ← NOT INT8
Context (AxV):     FP32 x INT8 → INT32 → FP32
Output projection: FP32 → INT8 (quantized linear)
```

### LUT-Based Integer Softmax (i-Softmax)

For fully integer execution on GAP9, we use a lookup table approach:

```python
def i_softmax_int32_to_uint8(scores_int32, scale_q, scale_k, softmax_scale, ...):
    """
    Integer-only softmax:
    1. Find max INT32 score (purely integer)
    2. Compute INT32 diff = score - max (always <= 0)
    3. Requantize diff to INT8: x_int = (diff * requant_mul + round) >> requant_shift
    4. LUT lookup: exp_val = lut[x_int + 128]  ← Pre-computed exp() table
    5. Normalize: attn = exp_val * 255 / sum (integer division)
    """
```

### The Softmax LUT

A 129-entry lookup table replaces the `exp()` function:

```python
def get_c_compatible_softmax_lut() -> np.ndarray:
    """
    129-entry UINT32 softmax LUT:
    - Entries for x in [-128, 0] (integer input)
    - Values: round(exp(x) * 2^24) as UINT32
    - Index: x + 128 (index 0 = exp(-128), index 128 = exp(0))
    """
    lut = np.zeros(129, dtype=np.uint32)
    for x in range(-128, 1):
        lut[x + 128] = int(round(np.exp(x) * (1 << 24)))
    return lut
```

## 5. Summary Table

| Component | Quantization Strategy |
|-----------|----------------------|
| Weights | True INT*: `q = clip(round(w/scale), qmin, qmax)` (typically int8 weights use `qmin=-127,qmax=127`) |
| Activations | True INT8 with per-layer scales |
| Conv/Linear | INT8 x INT8 → INT32 accumulator → requantize to INT8 |
| **Softmax** | **LUT-based integer approximation** |
| Softmax Input | INT32 attention scores (QxK^T) |
| Softmax Output | UINT8 attention weights [0-255] |
| exp() function | Pre-computed 129-entry LUT |

## 6. Other Nonlinear Operations

Similar LUT approaches are used for other complex nonlinearities:

| Operation | Implementation |
|-----------|---------------|
| SiLU | LUT-based (see `atomic_ops/silu.py`) |
| GELU | LUT-based (see `atomic_ops/gelu.py`) |
| Softplus | LUT-based (see `atomic_ops/softplus.py`) |
| ReLU | True INT8 (simple max(0, x)) |
| MaxPool | True INT8 (comparison only) |

## 7. Key Files

- `tools/pytorch_extractor.py` - Scale extraction and weight quantization
- `atomic_ops/mhsa.py` - MHSA with i-Softmax implementation
- `atomic_ops/quantize.py` - Core quantization/dequantization functions
- `tools/generate_softmax_lut.py` - Softmax LUT generation

---

## 8. Bias Quantization (Accumulator Domain)

Brevitas layers store `bias` in FP32, but **true INT8 inference accumulates in INT32**:

```
acc_int32 = Σ (x_int8 * w_int8)
```

This accumulator corresponds to the real value:

```
acc_fp32 ≈ acc_int32 * (scale_x * scale_w)
```

To add bias without losing precision, bias must be quantized into the *accumulator domain*:

```
bias_int32 = round(bias_fp32 / (scale_x * scale_w))
acc_int32 += bias_int32
```

This is implemented in:
- `tools/int8_inference.py` (Python golden path)
- `codegen/generate_c_code.py` (exports `*_bias.bin` as INT32 for C)

In this project, we add bias in the INT32 accumulator domain (before scaling/requantization). This matches the deployed kernels and keeps rounding/precision behavior consistent.
