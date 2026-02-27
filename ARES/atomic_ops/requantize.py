# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Atomic Requantize Operation - INT8

Direct INT8-to-INT8 requantization without FP32 intermediate.
This matches the C code behavior for fused operations.
"""

import numpy as np


def requantize_int8(
    x_int8: np.ndarray,
    scale_in: float,
    scale_out: float
) -> np.ndarray:
    """
    Direct INT8→INT8 requantization.

    This is mathematically equivalent to:
        dequantize(x_int8, scale_in) → quantize(_, scale_out)

    But uses the same single-step computation as the C code to ensure
    bit-exact matching when fusion is applied.

    Formula:
        output = clip(floor((x_int8 * scale_in / scale_out) + 0.5), -128, 127)

    Args:
        x_int8: Input INT8 tensor
        scale_in: Input quantization scale
        scale_out: Output quantization scale

    Returns:
        Output INT8 tensor with new scale
    """
    # Compute scaling factor
    scale_factor = scale_in / scale_out

    # Convert to float, rescale, and round
    x_float = x_int8.astype(np.float32) * np.float32(scale_factor)

    # Round using same method as C: (int)(x + 0.5)
    x_rounded = np.floor(x_float + 0.5).astype(np.int32)

    # Clip to INT8 range
    x_clipped = np.clip(x_rounded, -128, 127)

    return x_clipped.astype(np.int8)


def test_requantize_int8():
    """Test requantize_int8 against dequantize→quantize path."""
    print("Testing requantize_int8...")

    from .quantize import quantize_linear, dequantize_linear

    # Test case 1: Basic requantization
    print("\n  Test 1: Basic requantization")
    x_int8 = np.array([10, 20, 30, -10, -20, -30], dtype=np.int8)
    scale_in = 0.1
    scale_out = 0.05

    # Direct requantize
    y_direct = requantize_int8(x_int8, scale_in, scale_out)

    # Two-step (dequant→quant)
    x_fp32 = dequantize_linear(x_int8, scale=scale_in, zero_point=0)
    y_twostep = quantize_linear(x_fp32, scale=scale_out)

    print(f"    Input: {x_int8}")
    print(f"    Direct: {y_direct}")
    print(f"    Two-step: {y_twostep}")
    print(f"    Max diff: {np.max(np.abs(y_direct.astype(np.int32) - y_twostep.astype(np.int32)))}")

    # Test case 2: Edge cases
    print("\n  Test 2: Edge cases (saturation)")
    x_int8 = np.array([127, -128, 0], dtype=np.int8)
    scale_in = 0.1
    scale_out = 0.01  # 10x smaller scale → larger INT8 values

    y_direct = requantize_int8(x_int8, scale_in, scale_out)
    print(f"    Input: {x_int8} (scale={scale_in})")
    print(f"    Output: {y_direct} (scale={scale_out})")
    print(f"    Expected saturation at ±127")

    print("\n  [OK] All tests complete")
    return True


if __name__ == "__main__":
    test_requantize_int8()
