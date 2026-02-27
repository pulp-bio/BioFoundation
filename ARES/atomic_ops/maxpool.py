# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Atomic MaxPool2D Operation - INT8

Implements 2D max pooling with TRUE INT8 arithmetic.

Key insight: MaxPool is order-preserving, so quantization is maintained!
We can operate directly on INT8 values without any rescaling.
"""

import numpy as np


def maxpool2d_int8(
    x_int8: np.ndarray,
    kernel_size: tuple = (2, 2),
    stride: tuple = None,
    padding: tuple = (0, 0)
) -> np.ndarray:
    """
    INT8 2D max pooling.

    MaxPool is order-preserving, so quantization is maintained.
    Simply take max of INT8 values directly - no rescaling needed!

    Formula:
        out[b, c, oh, ow] = max over (kh, kw) of:
            input[b, c, oh*stride+kh, ow*stride+kw]

    Args:
        x_int8: Input INT8 tensor [B, C, H, W]
        kernel_size: Pooling window size (kh, kw)
        stride: Stride (sh, sw). If None, defaults to kernel_size
        padding: Zero padding (ph, pw)

    Returns:
        Output INT8 tensor [B, C, H_out, W_out]

    Notes:
        - MaxPool preserves quantization (no scale changes)
        - Order-preserving: max(Q(a), Q(b)) = Q(max(a, b))
        - No dequantization or requantization needed!
    """
    if stride is None:
        stride = kernel_size

    batch, channels, h_in, w_in = x_int8.shape
    k_h, k_w = kernel_size
    s_h, s_w = stride
    p_h, p_w = padding

    # Calculate output dimensions
    h_out = (h_in + 2 * p_h - k_h) // s_h + 1
    w_out = (w_in + 2 * p_w - k_w) // s_w + 1

    # Apply padding if needed
    if p_h > 0 or p_w > 0:
        # Pad with -128 (minimum INT8 value) so it doesn't affect max
        x_padded = np.pad(
            x_int8,
            ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)),
            mode='constant',
            constant_values=-128
        )
    else:
        x_padded = x_int8

    # Initialize output
    output = np.zeros((batch, channels, h_out, w_out), dtype=np.int8)

    # Perform max pooling
    for b in range(batch):
        for c in range(channels):
            for oh in range(h_out):
                for ow in range(w_out):
                    # Compute starting position
                    h_start = oh * s_h
                    w_start = ow * s_w

                    # Extract pooling window
                    window = x_padded[
                        b, c,
                        h_start:h_start + k_h,
                        w_start:w_start + k_w
                    ]

                    # Take maximum
                    output[b, c, oh, ow] = np.max(window)

    return output


def maxpool2d_fp32_reference(
    x: np.ndarray,
    kernel_size: tuple = (2, 2),
    stride: tuple = None,
    padding: tuple = (0, 0)
) -> np.ndarray:
    """
    FP32 MaxPool2D reference implementation.

    Used for testing and verification.
    """
    if stride is None:
        stride = kernel_size

    batch, channels, h_in, w_in = x.shape
    k_h, k_w = kernel_size
    s_h, s_w = stride
    p_h, p_w = padding

    # Calculate output dimensions
    h_out = (h_in + 2 * p_h - k_h) // s_h + 1
    w_out = (w_in + 2 * p_w - k_w) // s_w + 1

    # Apply padding if needed
    if p_h > 0 or p_w > 0:
        x_padded = np.pad(
            x,
            ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)),
            mode='constant',
            constant_values=-np.inf
        )
    else:
        x_padded = x

    # Initialize output
    output = np.zeros((batch, channels, h_out, w_out), dtype=np.float32)

    # Perform max pooling
    for b in range(batch):
        for c in range(channels):
            for oh in range(h_out):
                for ow in range(w_out):
                    h_start = oh * s_h
                    w_start = ow * s_w

                    window = x_padded[
                        b, c,
                        h_start:h_start + k_h,
                        w_start:w_start + k_w
                    ]

                    output[b, c, oh, ow] = np.max(window)

    return output


def test_maxpool():
    """Test INT8 MaxPool2D implementation."""
    print("="*80)
    print("Testing INT8 MaxPool2D")
    print("="*80)

    # Test 1: Basic 2x2 maxpool
    print("\nTest 1: Basic 2x2 MaxPool")
    x_int8 = np.array([[
        [[10, 20, 30, 40],
         [50, 60, 70, 80],
         [15, 25, 35, 45],
         [55, 65, 75, 85]]
    ]], dtype=np.int8)  # [1, 1, 4, 4]

    result = maxpool2d_int8(x_int8, kernel_size=(2, 2), stride=(2, 2))

    expected = np.array([[
        [[60, 80],
         [65, 85]]
    ]], dtype=np.int8)  # [1, 1, 2, 2]

    print(f"Input shape:  {x_int8.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Input:\n{x_int8[0, 0]}")
    print(f"\nOutput:\n{result[0, 0]}")
    print(f"\nExpected:\n{expected[0, 0]}")

    assert np.array_equal(result, expected), "Test 1 failed!"
    print("[PASS] Test 1 passed!")

    # Test 2: Compare with FP32 after quantization
    print("\n" + "="*80)
    print("Test 2: Compare INT8 vs FP32 MaxPool")
    print("="*80)

    # Create FP32 input
    x_fp32 = np.random.randn(2, 3, 8, 8).astype(np.float32)
    scale = 0.1

    # Quantize to INT8
    x_int8 = np.round(x_fp32 / scale).clip(-128, 127).astype(np.int8)

    # Apply MaxPool in INT8
    result_int8 = maxpool2d_int8(x_int8, kernel_size=(2, 2), stride=(2, 2))

    # Apply MaxPool in FP32 then quantize
    result_fp32 = maxpool2d_fp32_reference(x_fp32, kernel_size=(2, 2), stride=(2, 2))
    expected_int8 = np.round(result_fp32 / scale).clip(-128, 127).astype(np.int8)

    print(f"Input shape:     {x_int8.shape}")
    print(f"Output shape:    {result_int8.shape}")
    print(f"Expected shape:  {expected_int8.shape}")

    # Check if results match
    match = np.array_equal(result_int8, expected_int8)
    if match:
        print("[PASS] INT8 MaxPool matches FP32 → quantize perfectly!")
    else:
        # Check how close they are
        diff = np.abs(result_int8.astype(np.int32) - expected_int8.astype(np.int32))
        max_diff = np.max(diff)
        print(f"[WARN]  Max difference: {max_diff} (due to rounding)")
        # Allow small difference due to rounding
        assert max_diff <= 1, f"Difference too large: {max_diff}"
        print("[PASS] Test 2 passed (within rounding tolerance)!")

    # Test 3: Different kernel sizes and strides
    print("\n" + "="*80)
    print("Test 3: Different Kernel Sizes and Strides")
    print("="*80)

    x_int8 = np.random.randint(-50, 50, (1, 2, 8, 8), dtype=np.int8)

    # 3x3 kernel, stride 1
    result_3x3_s1 = maxpool2d_int8(x_int8, kernel_size=(3, 3), stride=(1, 1))
    expected_shape_1 = (1, 2, 6, 6)
    assert result_3x3_s1.shape == expected_shape_1, f"Shape mismatch: {result_3x3_s1.shape}"
    print(f"  3x3 kernel, stride 1: {x_int8.shape} → {result_3x3_s1.shape} [PASS]")

    # 2x2 kernel, stride 1 (overlapping)
    result_2x2_s1 = maxpool2d_int8(x_int8, kernel_size=(2, 2), stride=(1, 1))
    expected_shape_2 = (1, 2, 7, 7)
    assert result_2x2_s1.shape == expected_shape_2, f"Shape mismatch: {result_2x2_s1.shape}"
    print(f"  2x2 kernel, stride 1: {x_int8.shape} → {result_2x2_s1.shape} [PASS]")

    # 4x4 kernel, stride 2
    result_4x4_s2 = maxpool2d_int8(x_int8, kernel_size=(4, 4), stride=(2, 2))
    expected_shape_3 = (1, 2, 3, 3)
    assert result_4x4_s2.shape == expected_shape_3, f"Shape mismatch: {result_4x4_s2.shape}"
    print(f"  4x4 kernel, stride 2: {x_int8.shape} → {result_4x4_s2.shape} [PASS]")

    # Test 4: Verify max property
    print("\n" + "="*80)
    print("Test 4: Verify Max Property")
    print("="*80)

    x_int8 = np.array([[
        [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12],
         [13, 14, 15, 16]]
    ]], dtype=np.int8)

    result = maxpool2d_int8(x_int8, kernel_size=(2, 2), stride=(2, 2))

    # Each pooled value should be max of its window
    assert result[0, 0, 0, 0] == 6, "Top-left pool incorrect"
    assert result[0, 0, 0, 1] == 8, "Top-right pool incorrect"
    assert result[0, 0, 1, 0] == 14, "Bottom-left pool incorrect"
    assert result[0, 0, 1, 1] == 16, "Bottom-right pool incorrect"

    print(f"Input:\n{x_int8[0, 0]}")
    print(f"\nPooled output:\n{result[0, 0]}")
    print("[PASS] All pooled values are correct maximums!")

    # Test 5: Negative values
    print("\n" + "="*80)
    print("Test 5: Negative Values")
    print("="*80)

    x_int8 = np.array([[
        [[-50, -40, -30, -20],
         [-10, 0, 10, 20],
         [-60, -50, -40, -30],
         [-20, -10, 0, 10]]
    ]], dtype=np.int8)

    result = maxpool2d_int8(x_int8, kernel_size=(2, 2), stride=(2, 2))

    print(f"Input:\n{x_int8[0, 0]}")
    print(f"\nPooled output:\n{result[0, 0]}")

    # Verify correct maxes (including negative values)
    assert result[0, 0, 0, 0] == 0, "Max of negatives + 0 should be 0"
    assert result[0, 0, 0, 1] == 20, "Max should be 20"
    assert result[0, 0, 1, 0] == -10, "Max of all negatives"
    assert result[0, 0, 1, 1] == 10, "Max should be 10"

    print("[PASS] Test 5 passed!")

    # Test 6: Multi-channel
    print("\n" + "="*80)
    print("Test 6: Multi-Channel")
    print("="*80)

    batch, channels = 2, 16
    x_int8 = np.random.randint(-50, 50, (batch, channels, 14, 14), dtype=np.int8)

    result = maxpool2d_int8(x_int8, kernel_size=(2, 2), stride=(2, 2))

    expected_shape = (2, 16, 7, 7)
    assert result.shape == expected_shape, f"Shape mismatch: {result.shape}"

    print(f"Input:  {x_int8.shape}")
    print(f"Output: {result.shape}")
    print("[PASS] Multi-channel test passed!")

    print("\n" + "="*80)
    print("[PASS] All MaxPool2D tests passed!")
    print("="*80)


if __name__ == "__main__":
    test_maxpool()
