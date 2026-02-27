# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Atomic Add Operation - INT8

Implements element-wise addition with TRUE INT8 arithmetic.

Key insight: Adding quantized values with different scales requires rescaling!
Q1(x) + Q2(y) requires converting to common scale.
"""

import numpy as np


def add_int8(
    x1_int8: np.ndarray,
    x2_int8: np.ndarray,
    scale_x1: float,
    scale_x2: float,
    scale_output: float
) -> np.ndarray:
    """
    INT8 element-wise addition with rescaling.

    Adds two INT8 tensors that may have different quantization scales.

    Process:
        1. Dequantize both inputs to FP32
        2. Add in FP32
        3. Requantize to output scale

    Formula:
        out = Quantize(Dequantize(x1) + Dequantize(x2), scale_output)
            = Quantize(x1 * scale_x1 + x2 * scale_x2, scale_output)

    Args:
        x1_int8: First input INT8 tensor [B, C, H, W] or any shape
        x2_int8: Second input INT8 tensor (same shape as x1)
        scale_x1: Quantization scale for x1
        scale_x2: Quantization scale for x2
        scale_output: Output quantization scale

    Returns:
        Output INT8 tensor (same shape as inputs)

    Notes:
        - Critical for ResNet skip connections
        - Requires scale matching - inputs may have different scales
        - Uses FP32 intermediate to maintain accuracy
    """
    # Validate shapes match
    assert x1_int8.shape == x2_int8.shape, \
        f"Input shapes must match: {x1_int8.shape} vs {x2_int8.shape}"

    # Dequantize to FP32
    x1_fp32 = x1_int8.astype(np.float32) * scale_x1
    x2_fp32 = x2_int8.astype(np.float32) * scale_x2

    # Add in FP32
    sum_fp32 = x1_fp32 + x2_fp32

    # Requantize to output scale
    output_val = sum_fp32 / scale_output
    output_int32 = np.round(output_val).astype(np.int32)

    # Clip to INT8 range
    output_int32 = np.clip(output_int32, -128, 127)

    return output_int32.astype(np.int8)


def add_int8_optimized(
    x1_int8: np.ndarray,
    x2_int8: np.ndarray,
    scale_x1: float,
    scale_x2: float,
    scale_output: float
) -> np.ndarray:
    """
    Optimized INT8 addition using integer math where possible.

    When scales are related by simple ratios, we can avoid FP32 conversion.

    Special case: If scale_x1 == scale_x2 == scale_output:
        out = clip(x1 + x2, -128, 127)  # Direct INT8 addition!

    Args:
        x1_int8: First input INT8 tensor
        x2_int8: Second input INT8 tensor
        scale_x1: Quantization scale for x1
        scale_x2: Quantization scale for x2
        scale_output: Output quantization scale

    Returns:
        Output INT8 tensor
    """
    # Check for special case: all scales equal
    if np.isclose(scale_x1, scale_x2) and np.isclose(scale_x1, scale_output):
        # Direct INT8 addition (with overflow handling)
        sum_int32 = x1_int8.astype(np.int32) + x2_int8.astype(np.int32)
        sum_int32 = np.clip(sum_int32, -128, 127)
        return sum_int32.astype(np.int8)
    else:
        # Fall back to general case
        return add_int8(x1_int8, x2_int8, scale_x1, scale_x2, scale_output)


def test_add_int8():
    """Test INT8 addition against FP32 reference."""
    print("Testing add_int8...")

    # Test case 1: Same scales
    print("\n  Test 1: Same scales")
    scale = 0.1
    x1_fp32 = np.array([[1.5, -2.3], [0.8, -1.1]], dtype=np.float32)
    x2_fp32 = np.array([[0.5, 1.2], [-0.3, 2.0]], dtype=np.float32)

    x1_int8 = np.clip(np.round(x1_fp32 / scale), -128, 127).astype(np.int8)
    x2_int8 = np.clip(np.round(x2_fp32 / scale), -128, 127).astype(np.int8)

    result_int8 = add_int8(x1_int8, x2_int8, scale, scale, scale)

    # FP32 reference
    sum_fp32 = x1_fp32 + x2_fp32
    result_ref = np.clip(np.round(sum_fp32 / scale), -128, 127).astype(np.int8)

    diff = np.max(np.abs(result_int8.astype(np.int32) - result_ref.astype(np.int32)))
    print(f"    Max difference: {diff}")
    assert diff <= 1, f"Test 1 failed: diff={diff}"
    print("    [OK] Test 1 passed")

    # Test case 2: Different scales
    print("\n  Test 2: Different scales")
    scale_x1 = 0.1
    scale_x2 = 0.2
    scale_out = 0.15

    x1_int8 = np.array([[10, -20], [30, -40]], dtype=np.int8)
    x2_int8 = np.array([[5, 10], [-5, 15]], dtype=np.int8)

    result_int8 = add_int8(x1_int8, x2_int8, scale_x1, scale_x2, scale_out)

    # FP32 reference
    x1_fp32 = x1_int8.astype(np.float32) * scale_x1
    x2_fp32 = x2_int8.astype(np.float32) * scale_x2
    sum_fp32 = x1_fp32 + x2_fp32
    result_ref = np.clip(np.round(sum_fp32 / scale_out), -128, 127).astype(np.int8)

    diff = np.max(np.abs(result_int8.astype(np.int32) - result_ref.astype(np.int32)))
    print(f"    Max difference: {diff}")
    assert diff <= 1, f"Test 2 failed: diff={diff}"
    print("    [OK] Test 2 passed")

    # Test case 3: Optimized path (same scales)
    print("\n  Test 3: Optimized path (same scales)")
    result_opt = add_int8_optimized(x1_int8, x2_int8, scale, scale, scale)
    result_gen = add_int8(x1_int8, x2_int8, scale, scale, scale)

    diff = np.max(np.abs(result_opt.astype(np.int32) - result_gen.astype(np.int32)))
    print(f"    Max difference: {diff}")
    assert diff == 0, f"Optimized version differs from general: diff={diff}"
    print("    [OK] Test 3 passed")

    print("\n  [OK] All tests passed!")
    return True


if __name__ == "__main__":
    test_add_int8()
