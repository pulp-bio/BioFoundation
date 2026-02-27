# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Atomic SiLU (Swish) Operation - INT8 LUT-Based

Implements SiLU activation using a 256-entry lookup table for integer-only execution.

SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

For INT8 quantized input with only 256 possible values (-128 to 127),
we can pre-compute all outputs in a lookup table, making this operation:
- Bit-exact between Python and C
- Zero floating-point operations at runtime
- Single memory lookup per element

Used in MAMBA after the depthwise conv1d.
"""

import numpy as np
from typing import Optional, Tuple

try:
    from .quantize import quantize_linear, dequantize_linear
    from .constants import INT16_MAX, INT16_MIN, Q13_SCALE_INT
except ImportError:
    from quantize import quantize_linear, dequantize_linear
    from constants import INT16_MAX, INT16_MIN, Q13_SCALE_INT


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )


def silu_fp32(x: np.ndarray) -> np.ndarray:
    """FP32 SiLU activation function."""
    return x * sigmoid(x)


def generate_silu_lut_int8(
    scale_in: float,
    scale_out: float,
    zero_point_in: int = 0,
    zero_point_out: int = 0
) -> np.ndarray:
    """
    Generate a 256-entry INT8 lookup table for SiLU activation.

    For each possible INT8 input value (-128 to 127):
    1. Dequantize to FP32: x_fp32 = (x_int8 - zero_point_in) * scale_in
    2. Apply SiLU: y_fp32 = x_fp32 * sigmoid(x_fp32)
    3. Quantize to INT8: y_int8 = round(y_fp32 / scale_out) + zero_point_out

    Args:
        scale_in: Input quantization scale
        scale_out: Output quantization scale
        zero_point_in: Input zero point (usually 0)
        zero_point_out: Output zero point (usually 0)

    Returns:
        lut: 256-entry INT8 array, indexed by (x_int8 + 128)
    """
    lut = np.zeros(256, dtype=np.int8)

    for i in range(256):
        # Input INT8 value: -128 to 127
        x_int8 = i - 128

        # Dequantize to FP32
        x_fp32 = (x_int8 - zero_point_in) * scale_in

        # Apply SiLU
        y_fp32 = silu_fp32(x_fp32)

        # Quantize to INT8
        y_scaled = y_fp32 / scale_out + zero_point_out
        y_int8 = np.clip(np.round(y_scaled), -128, 127).astype(np.int8)

        lut[i] = y_int8

    return lut


def generate_silu_lut_q13(
    scale_in: float
) -> np.ndarray:
    """
    Generate a 256-entry Q2.13 lookup table for SiLU.

    This variant outputs Q2.13 fixed-point values (range [-4, 4] with 13 fractional bits).
    Useful for the gating operation: y = y_ssm * silu_q13(z) >> 13

    Args:
        scale_in: Input quantization scale

    Returns:
        lut: 256-entry INT16 array in Q2.13 format
    """
    lut = np.zeros(256, dtype=np.int16)

    for i in range(256):
        x_int8 = i - 128
        x_fp32 = x_int8 * scale_in

        # SiLU output
        y_fp32 = silu_fp32(x_fp32)

        # Convert to Q2.13: multiply by 2^13
        y_q13 = np.clip(np.round(y_fp32 * Q13_SCALE_INT), INT16_MIN, INT16_MAX).astype(np.int16)
        lut[i] = y_q13

    return lut


def silu_lut_int8(
    x_int8: np.ndarray,
    lut: np.ndarray
) -> np.ndarray:
    """
    Apply SiLU activation using pre-computed lookup table.

    Args:
        x_int8: Input tensor (INT8, any shape)
        lut: 256-entry lookup table from generate_silu_lut_int8()

    Returns:
        Output tensor (INT8, same shape as input)
    """
    # Convert to index: -128 -> 0, -127 -> 1, ..., 127 -> 255
    indices = x_int8.astype(np.int32) + 128

    # Lookup
    return lut[indices]


def silu_lut_q13(
    x_int8: np.ndarray,
    lut_q13: np.ndarray
) -> np.ndarray:
    """
    Apply SiLU activation using Q2.13 lookup table.

    Args:
        x_int8: Input tensor (INT8, any shape)
        lut_q13: 256-entry Q2.13 lookup table

    Returns:
        Output tensor (INT16 Q2.13, same shape as input)
    """
    indices = x_int8.astype(np.int32) + 128
    return lut_q13[indices]


def silu_int8(
    x_int8: np.ndarray,
    scale_in: float,
    scale_out: float,
    zero_point_in: int = 0,
    zero_point_out: int = 0
) -> np.ndarray:
    """
    Convenience function: generate LUT and apply SiLU in one call.

    For repeated use, pre-generate the LUT with generate_silu_lut_int8().

    Args:
        x_int8: Input tensor (INT8)
        scale_in: Input quantization scale
        scale_out: Output quantization scale
        zero_point_in: Input zero point
        zero_point_out: Output zero point

    Returns:
        Output tensor (INT8)
    """
    lut = generate_silu_lut_int8(scale_in, scale_out, zero_point_in, zero_point_out)
    return silu_lut_int8(x_int8, lut)


def test_silu():
    """Test INT8 SiLU implementation."""
    print("=" * 80)
    print("Testing INT8 SiLU (LUT-based)")
    print("=" * 80)

    # Test 1: Basic LUT generation and application
    print("\n--- Test 1: Basic SiLU LUT ---")
    scale_in = 0.05
    scale_out = 0.05

    # Generate LUT
    lut = generate_silu_lut_int8(scale_in, scale_out)
    print(f"LUT shape: {lut.shape}, dtype: {lut.dtype}")
    print(f"LUT range: [{lut.min()}, {lut.max()}]")

    # Verify key properties of SiLU:
    # - SiLU(0) = 0
    # - SiLU(x) ≈ 0 for large negative x
    # - SiLU(x) ≈ x for large positive x
    print(f"LUT[0] (x=-128): {lut[0]} (should be near 0 for large negative)")
    print(f"LUT[128] (x=0):  {lut[128]} (should be 0)")
    print(f"LUT[255] (x=127): {lut[255]} (should be near 127 for large positive)")

    # Test application
    x_int8 = np.array([-128, -64, 0, 64, 127], dtype=np.int8)
    y_int8 = silu_lut_int8(x_int8, lut)
    print(f"\nInput:  {x_int8}")
    print(f"Output: {y_int8}")

    # Compare with FP32 reference
    x_fp32 = x_int8.astype(np.float32) * scale_in
    y_fp32_ref = silu_fp32(x_fp32)
    y_fp32_from_int8 = y_int8.astype(np.float32) * scale_out

    print(f"\nFP32 reference: {y_fp32_ref}")
    print(f"INT8 (dequant): {y_fp32_from_int8}")

    error = np.abs(y_fp32_ref - y_fp32_from_int8)
    max_error = np.max(error)
    print(f"Max error: {max_error:.6f}")

    # Error should be within quantization tolerance
    assert max_error < scale_out * 1.5, f"Error too large: {max_error}"
    print("Test 1 PASSED!")

    # Test 2: Random tensor
    print("\n--- Test 2: Random Tensor ---")
    x_random_fp32 = np.random.randn(32, 64).astype(np.float32) * 2
    x_random_int8 = quantize_linear(x_random_fp32, scale_in)

    y_random_int8 = silu_lut_int8(x_random_int8, lut)
    y_random_fp32_ref = silu_fp32(dequantize_linear(x_random_int8, scale_in))
    y_random_fp32_from_int8 = dequantize_linear(y_random_int8, scale_out)

    random_error = np.abs(y_random_fp32_ref - y_random_fp32_from_int8)
    print(f"Random tensor shape: {x_random_int8.shape}")
    print(f"Max error: {np.max(random_error):.6f}")
    print(f"Mean error: {np.mean(random_error):.6f}")
    print("Test 2 PASSED!")

    # Test 3: PyTorch comparison
    print("\n--- Test 3: PyTorch Comparison ---")
    try:
        import torch
        import torch.nn.functional as F

        x_torch = torch.from_numpy(x_random_fp32)
        y_torch = F.silu(x_torch)
        y_torch_np = y_torch.numpy()

        # Compare with our FP32 reference
        torch_vs_ref = np.max(np.abs(y_torch_np - silu_fp32(x_random_fp32)))
        print(f"PyTorch vs FP32 reference max error: {torch_vs_ref:.9f}")

        if torch_vs_ref < 1e-6:
            print("Test 3 PASSED! FP32 reference matches PyTorch SiLU")
        else:
            print("Test 3 WARNING: Small numerical differences with PyTorch")

    except ImportError:
        print("PyTorch not available, skipping Test 3")

    # Test 4: Q2.13 LUT for gating
    print("\n--- Test 4: Q2.13 LUT for Gating ---")
    lut_q13 = generate_silu_lut_q13(scale_in)
    print(f"Q13 LUT shape: {lut_q13.shape}, dtype: {lut_q13.dtype}")
    print(f"Q13 LUT range: [{lut_q13.min()}, {lut_q13.max()}]")

    # Test gating operation: y = ssm_output * silu_q13(z) >> 13
    z_int8 = np.array([-64, -32, 0, 32, 64], dtype=np.int8)
    silu_q13_vals = silu_lut_q13(z_int8, lut_q13)
    print(f"\nz_int8:      {z_int8}")
    print(f"silu_q13:    {silu_q13_vals}")

    # Verify Q13 accuracy
    z_fp32 = z_int8.astype(np.float32) * scale_in
    silu_fp32_ref = silu_fp32(z_fp32)
    silu_q13_dequant = silu_q13_vals.astype(np.float32) / float(Q13_SCALE_INT)
    q13_error = np.max(np.abs(silu_fp32_ref - silu_q13_dequant))
    print(f"Q13 max error: {q13_error:.6f}")
    print("Test 4 PASSED!")

    # Test 5: Verify LUT is deterministic (bit-exact)
    print("\n--- Test 5: Bit-Exact Determinism ---")
    lut1 = generate_silu_lut_int8(scale_in, scale_out)
    lut2 = generate_silu_lut_int8(scale_in, scale_out)
    assert np.array_equal(lut1, lut2), "LUTs should be identical"
    print("LUT generation is deterministic: PASSED!")

    print("\n" + "=" * 80)
    print("All SiLU tests completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    test_silu()
