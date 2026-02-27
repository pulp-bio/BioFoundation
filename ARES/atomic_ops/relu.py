# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Atomic ReLU Operation - INT8

Implements ReLU activation with TRUE INT8 arithmetic.

ReLU is simple: max(0, x), but in quantized form we need to consider
the zero point (though for symmetric quantization, zero_point=0).
"""

import numpy as np


def relu_int8(x_int8: np.ndarray, zero_point: int = 0) -> np.ndarray:
    """
    INT8 ReLU activation.

    Formula:
        y = max(zero_point, x)

    For symmetric quantization (zero_point=0):
        y = max(0, x)

    Args:
        x_int8: Input INT8 tensor
        zero_point: Quantization zero point (default 0 for symmetric)

    Returns:
        Output INT8 tensor with ReLU applied

    Notes:
        - ReLU preserves quantization (no rescaling needed)
        - Simply clips negative values to zero_point
        - For symmetric quantization: zero_point=0, so max(0, x)
    """
    return np.maximum(zero_point, x_int8).astype(np.int8)


def relu_fp32_reference(x: np.ndarray) -> np.ndarray:
    """
    FP32 ReLU reference implementation.

    Used for testing and verification.
    """
    return np.maximum(0.0, x)


def test_relu():
    """Test INT8 ReLU implementation."""
    print("="*80)
    print("Testing INT8 ReLU")
    print("="*80)

    # Test 1: Basic symmetric quantization (zero_point=0)
    print("\nTest 1: Symmetric Quantization (zero_point=0)")
    x_int8 = np.array([-50, -10, 0, 10, 50], dtype=np.int8)
    expected = np.array([0, 0, 0, 10, 50], dtype=np.int8)

    result = relu_int8(x_int8, zero_point=0)

    print(f"Input:    {x_int8}")
    print(f"Output:   {result}")
    print(f"Expected: {expected}")

    assert np.array_equal(result, expected), "Test 1 failed!"
    print("[PASS] Test 1 passed!")

    # Test 2: 2D tensor
    print("\n" + "="*80)
    print("Test 2: 2D Tensor")
    print("="*80)

    x_int8_2d = np.array([
        [-30, -20, -10],
        [0, 10, 20],
        [30, 40, 50]
    ], dtype=np.int8)

    expected_2d = np.array([
        [0, 0, 0],
        [0, 10, 20],
        [30, 40, 50]
    ], dtype=np.int8)

    result_2d = relu_int8(x_int8_2d)

    print(f"Input:\n{x_int8_2d}")
    print(f"\nOutput:\n{result_2d}")
    print(f"\nExpected:\n{expected_2d}")

    assert np.array_equal(result_2d, expected_2d), "Test 2 failed!"
    print("[PASS] Test 2 passed!")

    # Test 3: Compare with FP32 after quantization
    print("\n" + "="*80)
    print("Test 3: Compare INT8 vs FP32 ReLU")
    print("="*80)

    # Create FP32 input
    x_fp32 = np.array([-5.0, -2.5, 0.0, 2.5, 5.0], dtype=np.float32)
    scale = 0.1

    # Quantize
    x_int8 = np.round(x_fp32 / scale).clip(-128, 127).astype(np.int8)

    # Apply ReLU in INT8
    result_int8 = relu_int8(x_int8)

    # Apply ReLU in FP32 then quantize
    x_fp32_relu = relu_fp32_reference(x_fp32)
    expected_int8 = np.round(x_fp32_relu / scale).clip(-128, 127).astype(np.int8)

    print(f"Original FP32:  {x_fp32}")
    print(f"Quantized INT8: {x_int8}")
    print(f"After ReLU:     {result_int8}")
    print(f"Expected:       {expected_int8}")

    assert np.array_equal(result_int8, expected_int8), "Test 3 failed!"
    print("[PASS] Test 3 passed!")

    # Test 4: 4D tensor (typical CNN activation shape)
    print("\n" + "="*80)
    print("Test 4: 4D Tensor (CNN Activation Shape)")
    print("="*80)

    batch, channels, height, width = 2, 3, 4, 4
    x_int8_4d = np.random.randint(-50, 50, (batch, channels, height, width), dtype=np.int8)

    result_4d = relu_int8(x_int8_4d)

    # Verify shape preserved
    assert result_4d.shape == x_int8_4d.shape, "Shape mismatch!"

    # Verify no negative values
    assert np.all(result_4d >= 0), "Negative values found after ReLU!"

    # Verify positive values unchanged
    positive_mask = x_int8_4d > 0
    assert np.array_equal(result_4d[positive_mask], x_int8_4d[positive_mask]), \
        "Positive values changed!"

    print(f"Input shape:  {x_int8_4d.shape}")
    print(f"Output shape: {result_4d.shape}")
    print(f"Min input:    {x_int8_4d.min()}")
    print(f"Min output:   {result_4d.min()}")
    print(f"Max input:    {x_int8_4d.max()}")
    print(f"Max output:   {result_4d.max()}")
    print("[PASS] Test 4 passed!")

    # Test 5: Edge cases
    print("\n" + "="*80)
    print("Test 5: Edge Cases")
    print("="*80)

    # All negative
    x_all_neg = np.array([-128, -100, -50, -1], dtype=np.int8)
    result_all_neg = relu_int8(x_all_neg)
    expected_all_neg = np.array([0, 0, 0, 0], dtype=np.int8)
    assert np.array_equal(result_all_neg, expected_all_neg), "All negative test failed!"
    print("  All negative: [PASS]")

    # All positive
    x_all_pos = np.array([1, 50, 100, 127], dtype=np.int8)
    result_all_pos = relu_int8(x_all_pos)
    assert np.array_equal(result_all_pos, x_all_pos), "All positive test failed!"
    print("  All positive: [PASS]")

    # All zeros
    x_all_zero = np.zeros(10, dtype=np.int8)
    result_all_zero = relu_int8(x_all_zero)
    assert np.array_equal(result_all_zero, x_all_zero), "All zeros test failed!"
    print("  All zeros: [PASS]")

    print("\n" + "="*80)
    print("[PASS] All ReLU tests passed!")
    print("="*80)


if __name__ == "__main__":
    test_relu()
