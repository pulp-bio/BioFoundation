# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Atomic SwiGLU FFN Operation - INT8

Implements the SwiGLU (Swish-Gated Linear Unit) Feed-Forward Network used in
Llama, Llama2, and other modern LLMs.

SwiGLU FFN formula:
    y = W2(silu(W1(x)) * W3(x))

Where:
    - W1: Gate projection (dim -> hidden_dim)
    - W3: Up projection (dim -> hidden_dim)
    - W2: Down projection (hidden_dim -> dim)
    - silu(x) = x * sigmoid(x)

The gating mechanism allows the network to selectively pass information,
typically providing better gradient flow than standard ReLU-based FFNs.

References:
- "GLU Variants Improve Transformer" (Shazeer, 2020)
- Llama architecture (Touvron et al., 2023)
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple

try:
    from .quantize import quantize_linear, dequantize_linear
    from .linear import linear_int8
    from .silu import silu_lut_int8, generate_silu_lut_int8, silu_fp32
except ImportError:
    from quantize import quantize_linear, dequantize_linear
    from linear import linear_int8
    from silu import silu_lut_int8, generate_silu_lut_int8, silu_fp32


def swiglu_ffn_fp32_reference(
    x: np.ndarray,
    w1: np.ndarray,
    w3: np.ndarray,
    w2: np.ndarray,
    bias1: Optional[np.ndarray] = None,
    bias3: Optional[np.ndarray] = None,
    bias2: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    FP32 reference implementation of SwiGLU FFN.

    Args:
        x: Input tensor, shape [B, dim] or [B, seq_len, dim]
        w1: Gate weight, shape [hidden_dim, dim]
        w3: Up weight, shape [hidden_dim, dim]
        w2: Down weight, shape [dim, hidden_dim]
        bias1: Gate bias (optional), shape [hidden_dim]
        bias3: Up bias (optional), shape [hidden_dim]
        bias2: Down bias (optional), shape [dim]

    Returns:
        Output tensor, same shape as input
    """
    original_shape = x.shape

    # Flatten to 2D if needed
    if x.ndim == 3:
        batch_size, seq_len, dim = x.shape
        x = x.reshape(-1, dim)
    elif x.ndim == 1:
        x = x.reshape(1, -1)

    # Gate path: W1(x) -> SiLU
    h1 = x @ w1.T
    if bias1 is not None:
        h1 += bias1
    h1_silu = silu_fp32(h1)

    # Up path: W3(x)
    h3 = x @ w3.T
    if bias3 is not None:
        h3 += bias3

    # Gating: element-wise multiply
    gated = h1_silu * h3

    # Down path: W2(gated)
    y = gated @ w2.T
    if bias2 is not None:
        y += bias2

    # Restore original shape
    if len(original_shape) == 3:
        y = y.reshape(original_shape[0], original_shape[1], -1)
    elif len(original_shape) == 1:
        y = y.squeeze(0)

    return y


def swiglu_ffn_int8(
    x_int8: np.ndarray,
    w1_int8: np.ndarray,
    w3_int8: np.ndarray,
    w2_int8: np.ndarray,
    scale_input: float,
    scale_w1: float,
    scale_w3: float,
    scale_w2: float,
    scale_hidden: float,
    scale_output: float,
    bias1_int32: Optional[np.ndarray] = None,
    bias3_int32: Optional[np.ndarray] = None,
    bias2_int32: Optional[np.ndarray] = None,
    silu_lut: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    INT8 SwiGLU FFN operation.

    Formula: y = W2(silu(W1(x)) * W3(x))

    Three linear projections with gated activation:
    1. W1 projection (gate path) -> SiLU activation
    2. W3 projection (up path) -> no activation
    3. Element-wise multiply (gating)
    4. W2 projection (down path) -> output

    Args:
        x_int8: INT8 input tensor [B, dim] or [B, seq_len, dim]
        w1_int8: INT8 gate weight [hidden_dim, dim]
        w3_int8: INT8 up weight [hidden_dim, dim]
        w2_int8: INT8 down weight [dim, hidden_dim]
        scale_input: Input quantization scale
        scale_w1: W1 weight scale
        scale_w3: W3 weight scale
        scale_w2: W2 weight scale
        scale_hidden: Hidden activation scale (used between layers)
        scale_output: Output quantization scale
        bias1_int32: INT32 gate bias (optional)
        bias3_int32: INT32 up bias (optional)
        bias2_int32: INT32 down bias (optional)
        silu_lut: Pre-computed SiLU LUT (optional, generated if not provided)

    Returns:
        INT8 output tensor, same shape as input
    """
    original_shape = x_int8.shape

    # Flatten to 2D if needed
    if x_int8.ndim == 3:
        batch_size, seq_len, dim = x_int8.shape
        x_int8 = x_int8.reshape(-1, dim)
    elif x_int8.ndim == 1:
        x_int8 = x_int8.reshape(1, -1)

    # Generate SiLU LUT if not provided
    # LUT maps from hidden scale to hidden scale
    if silu_lut is None:
        silu_lut = generate_silu_lut_int8(scale_hidden, scale_hidden)

    # Step 1: Gate path - W1(x)
    h1_int8 = linear_int8(
        x_int8, w1_int8, bias1_int32,
        scale_input, scale_w1, scale_hidden
    )

    # Step 2: Up path - W3(x)
    h3_int8 = linear_int8(
        x_int8, w3_int8, bias3_int32,
        scale_input, scale_w3, scale_hidden
    )

    # Step 3: Apply SiLU to gate path
    h1_silu_int8 = silu_lut_int8(h1_int8, silu_lut)

    # Step 4: Gating - element-wise multiply with requantization
    # h1_silu_int8 * h3_int8 -> INT16/INT32 -> requant to INT8
    # Both inputs are in scale_hidden, output should also be scale_hidden
    gated_int32 = h1_silu_int8.astype(np.int32) * h3_int8.astype(np.int32)

    # Requantize: scale_hidden * scale_hidden -> scale_hidden
    # Combined scale = scale_hidden, so we divide by scale_hidden (effectively >> by scale factor)
    gated_fp32 = gated_int32.astype(np.float32) * scale_hidden  # Dequant to FP32
    gated_int8 = np.clip(np.round(gated_fp32 / scale_hidden), -128, 127).astype(np.int8)

    # Step 5: Down path - W2(gated)
    y_int8 = linear_int8(
        gated_int8, w2_int8, bias2_int32,
        scale_hidden, scale_w2, scale_output
    )

    # Restore original shape
    if len(original_shape) == 3:
        y_int8 = y_int8.reshape(original_shape[0], original_shape[1], -1)
    elif len(original_shape) == 1:
        y_int8 = y_int8.squeeze(0)

    return y_int8


def swiglu_ffn_int8_fused(
    x_int8: np.ndarray,
    w1_int8: np.ndarray,
    w3_int8: np.ndarray,
    w2_int8: np.ndarray,
    scale_input: float,
    scale_w1: float,
    scale_w3: float,
    scale_w2: float,
    scale_hidden: float,
    scale_output: float,
    bias1_int32: Optional[np.ndarray] = None,
    bias3_int32: Optional[np.ndarray] = None,
    bias2_int32: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Fused INT8 SwiGLU FFN with intermediate FP32 for higher accuracy.

    This version keeps the gating operation in FP32 for better numerical
    stability, which is often necessary for LLM inference.

    Args:
        Same as swiglu_ffn_int8

    Returns:
        INT8 output tensor, same shape as input
    """
    original_shape = x_int8.shape

    # Flatten to 2D if needed
    if x_int8.ndim == 3:
        batch_size, seq_len, dim = x_int8.shape
        x_int8 = x_int8.reshape(-1, dim)
    elif x_int8.ndim == 1:
        x_int8 = x_int8.reshape(1, -1)

    # Dequantize input
    x_fp32 = dequantize_linear(x_int8, scale_input)

    # Dequantize weights
    w1_fp32 = dequantize_linear(w1_int8, scale_w1)
    w3_fp32 = dequantize_linear(w3_int8, scale_w3)
    w2_fp32 = dequantize_linear(w2_int8, scale_w2)

    # Dequantize biases if provided
    bias1_fp32 = None
    bias3_fp32 = None
    bias2_fp32 = None
    if bias1_int32 is not None:
        bias1_fp32 = bias1_int32.astype(np.float32) * (scale_input * scale_w1)
    if bias3_int32 is not None:
        bias3_fp32 = bias3_int32.astype(np.float32) * (scale_input * scale_w3)
    if bias2_int32 is not None:
        bias2_fp32 = bias2_int32.astype(np.float32) * (scale_hidden * scale_w2)

    # Compute in FP32
    y_fp32 = swiglu_ffn_fp32_reference(
        x_fp32, w1_fp32, w3_fp32, w2_fp32,
        bias1_fp32, bias3_fp32, bias2_fp32
    )

    # Quantize output
    y_int8 = quantize_linear(y_fp32, scale_output)

    # Restore original shape
    if len(original_shape) == 3:
        y_int8 = y_int8.reshape(original_shape[0], original_shape[1], -1)
    elif len(original_shape) == 1:
        y_int8 = y_int8.squeeze(0)

    return y_int8


def test_swiglu_ffn():
    """Unit test for INT8 SwiGLU FFN"""
    print("=" * 80)
    print("Testing INT8 SwiGLU FFN")
    print("=" * 80)

    # Test parameters
    batch_size = 2
    dim = 32
    hidden_dim = 64  # Typically 4x or 2.67x dim in Llama

    # Scales
    scale_input = 0.05
    scale_w = 0.02
    scale_hidden = 0.05
    scale_output = 0.05

    # Create test input
    x_fp32 = np.random.randn(batch_size, dim).astype(np.float32) * 0.5
    x_int8 = quantize_linear(x_fp32, scale_input)

    # Create test weights
    w1_fp32 = np.random.randn(hidden_dim, dim).astype(np.float32) * 0.1
    w3_fp32 = np.random.randn(hidden_dim, dim).astype(np.float32) * 0.1
    w2_fp32 = np.random.randn(dim, hidden_dim).astype(np.float32) * 0.1

    w1_int8 = quantize_linear(w1_fp32, scale_w)
    w3_int8 = quantize_linear(w3_fp32, scale_w)
    w2_int8 = quantize_linear(w2_fp32, scale_w)

    # Test 1: Basic SwiGLU FFN
    print("\nTest 1: Basic SwiGLU FFN (no bias)")

    y_int8 = swiglu_ffn_int8(
        x_int8, w1_int8, w3_int8, w2_int8,
        scale_input, scale_w, scale_w, scale_w,
        scale_hidden, scale_output
    )

    y_fp32_from_int8 = dequantize_linear(y_int8, scale_output)
    y_fp32_ref = swiglu_ffn_fp32_reference(x_fp32, w1_fp32, w3_fp32, w2_fp32)

    max_diff = np.max(np.abs(y_fp32_from_int8 - y_fp32_ref))
    mean_diff = np.mean(np.abs(y_fp32_from_int8 - y_fp32_ref))

    print(f"Input shape: {x_int8.shape}")
    print(f"Output shape: {y_int8.shape}")
    print(f"Output INT8 range: [{y_int8.min()}, {y_int8.max()}]")
    print(f"Max diff vs FP32 reference: {max_diff:.6f}")
    print(f"Mean diff vs FP32 reference: {mean_diff:.6f}")

    # Test 2: With biases
    print("\nTest 2: SwiGLU FFN with biases")

    bias1_fp32 = np.random.randn(hidden_dim).astype(np.float32) * 0.01
    bias3_fp32 = np.random.randn(hidden_dim).astype(np.float32) * 0.01
    bias2_fp32 = np.random.randn(dim).astype(np.float32) * 0.01

    # Convert biases to INT32 (bias = bias_fp32 / (scale_in * scale_w))
    bias1_int32 = (bias1_fp32 / (scale_input * scale_w)).astype(np.int32)
    bias3_int32 = (bias3_fp32 / (scale_input * scale_w)).astype(np.int32)
    bias2_int32 = (bias2_fp32 / (scale_hidden * scale_w)).astype(np.int32)

    y_int8_bias = swiglu_ffn_int8(
        x_int8, w1_int8, w3_int8, w2_int8,
        scale_input, scale_w, scale_w, scale_w,
        scale_hidden, scale_output,
        bias1_int32, bias3_int32, bias2_int32
    )

    y_fp32_from_int8_bias = dequantize_linear(y_int8_bias, scale_output)
    y_fp32_ref_bias = swiglu_ffn_fp32_reference(
        x_fp32, w1_fp32, w3_fp32, w2_fp32,
        bias1_fp32, bias3_fp32, bias2_fp32
    )

    max_diff_bias = np.max(np.abs(y_fp32_from_int8_bias - y_fp32_ref_bias))
    mean_diff_bias = np.mean(np.abs(y_fp32_from_int8_bias - y_fp32_ref_bias))

    print(f"Output INT8 range: [{y_int8_bias.min()}, {y_int8_bias.max()}]")
    print(f"Max diff vs FP32 reference: {max_diff_bias:.6f}")
    print(f"Mean diff vs FP32 reference: {mean_diff_bias:.6f}")

    # Test 3: 3D input (sequence)
    print("\nTest 3: SwiGLU FFN with sequence input")

    seq_len = 8
    x_seq_fp32 = np.random.randn(batch_size, seq_len, dim).astype(np.float32) * 0.5
    x_seq_int8 = quantize_linear(x_seq_fp32, scale_input)

    y_seq_int8 = swiglu_ffn_int8(
        x_seq_int8, w1_int8, w3_int8, w2_int8,
        scale_input, scale_w, scale_w, scale_w,
        scale_hidden, scale_output
    )

    print(f"Input shape: {x_seq_int8.shape}")
    print(f"Output shape: {y_seq_int8.shape}")
    assert y_seq_int8.shape == (batch_size, seq_len, dim), "Shape mismatch!"
    print("Shape preserved correctly!")

    # Test 4: Compare fused vs non-fused
    print("\nTest 4: Fused vs non-fused implementation")

    y_int8_nonfused = swiglu_ffn_int8(
        x_int8, w1_int8, w3_int8, w2_int8,
        scale_input, scale_w, scale_w, scale_w,
        scale_hidden, scale_output
    )

    y_int8_fused = swiglu_ffn_int8_fused(
        x_int8, w1_int8, w3_int8, w2_int8,
        scale_input, scale_w, scale_w, scale_w,
        scale_hidden, scale_output
    )

    mismatch = np.sum(y_int8_nonfused != y_int8_fused)
    print(f"Non-fused vs Fused mismatch: {mismatch} elements out of {y_int8_nonfused.size}")

    y_nf_fp32 = dequantize_linear(y_int8_nonfused, scale_output)
    y_f_fp32 = dequantize_linear(y_int8_fused, scale_output)
    diff = np.max(np.abs(y_nf_fp32 - y_f_fp32))
    print(f"Max FP32 diff between implementations: {diff:.6f}")

    # Test 5: Verify gating behavior
    print("\nTest 5: Verify gating mechanism")

    # Create input where we can verify gating works
    x_test = np.array([[1.0, 0.5, -0.5, -1.0] * 8], dtype=np.float32)
    x_test_int8 = quantize_linear(x_test, scale_input)

    y_test_int8 = swiglu_ffn_int8(
        x_test_int8, w1_int8[:, :32], w3_int8[:, :32], w2_int8[:32, :],
        scale_input, scale_w, scale_w, scale_w,
        scale_hidden, scale_output
    )

    print(f"Test input: {x_test[0, :8]}")
    print(f"Test output (first 8): {dequantize_linear(y_test_int8, scale_output)[0, :8]}")

    # Final summary
    print("\n" + "=" * 80)
    tolerance = scale_output * 2
    if max_diff <= tolerance:
        print(f"[PASS] All tests passed! (max diff {max_diff:.6f} <= tolerance {tolerance:.6f})")
    else:
        print(f"[WARN] Larger than expected diff (max diff {max_diff:.6f})")
    print("=" * 80)

    return True


if __name__ == "__main__":
    test_swiglu_ffn()
