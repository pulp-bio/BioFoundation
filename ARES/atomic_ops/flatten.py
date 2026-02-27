# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Atomic Flatten Operation - INT8

Implements tensor flattening (reshape) for INT8 tensors.

This is NOT a computational operation - just a memory layout transformation!
Quantization is preserved since we're not modifying any values.
"""

import numpy as np


def flatten_int8(x_int8: np.ndarray, start_dim: int = 1) -> np.ndarray:
    """
    Flatten INT8 tensor starting from a specific dimension.

    This is just a reshape operation - no computation needed!
    Quantization is preserved because we don't modify any values.

    Args:
        x_int8: Input INT8 tensor, typically [B, C, H, W]
        start_dim: Dimension to start flattening from (default 1, preserves batch)

    Returns:
        Flattened INT8 tensor, typically [B, C*H*W]

    Example:
        Input: [2, 32, 7, 7] (batch=2, channels=32, h=7, w=7)
        Output: [2, 1568] (batch=2, features=32*7*7=1568)

    Notes:
        - No quantization or dequantization needed
        - No scale changes
        - Just memory layout transformation
        - Equivalent to PyTorch's torch.flatten(x, start_dim=1)
    """
    if start_dim < 0:
        start_dim = x_int8.ndim + start_dim

    # Get shape before and after start_dim
    shape_before = x_int8.shape[:start_dim]
    shape_after = x_int8.shape[start_dim:]

    # Calculate total elements after start_dim
    flattened_size = 1
    for dim in shape_after:
        flattened_size *= dim

    # Reshape: keep dimensions before start_dim, flatten the rest
    new_shape = shape_before + (flattened_size,)

    return x_int8.reshape(new_shape)


def test_flatten():
    """Test INT8 Flatten implementation."""
    print("="*80)
    print("Testing INT8 Flatten")
    print("="*80)

    # Test 1: Basic 4D → 2D flatten (CNN use case)
    print("\nTest 1: 4D → 2D Flatten (Typical CNN)")

    # Simulate Conv2D output: [B, C, H, W]
    x_int8 = np.arange(2 * 32 * 7 * 7, dtype=np.int8).reshape(2, 32, 7, 7)

    result = flatten_int8(x_int8, start_dim=1)

    expected_shape = (2, 32 * 7 * 7)  # [2, 1568]
    assert result.shape == expected_shape, f"Shape mismatch: {result.shape} != {expected_shape}"

    print(f"Input shape:  {x_int8.shape} [B, C, H, W]")
    print(f"Output shape: {result.shape} [B, C*H*W]")
    print(f"Flattened size: 32 x 7 x 7 = {32*7*7}")

    # Verify data is preserved (just reshaped)
    assert np.array_equal(x_int8.flatten(), result.flatten()), "Data was modified!"
    print("[PASS] Test 1 passed! Data preserved, only shape changed.")

    # Test 2: Verify no data modification
    print("\n" + "="*80)
    print("Test 2: Verify No Data Modification")
    print("="*80)

    x_int8 = np.array([
        [[[1, 2], [3, 4]],
         [[5, 6], [7, 8]]],
        [[[9, 10], [11, 12]],
         [[13, 14], [15, 16]]]
    ], dtype=np.int8)  # [2, 2, 2, 2]

    result = flatten_int8(x_int8, start_dim=1)

    print(f"Input shape:  {x_int8.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Input[0]:  {x_int8[0].flatten()}")
    print(f"Output[0]: {result[0]}")

    # First batch should be [1,2,3,4,5,6,7,8]
    expected_batch0 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int8)
    assert np.array_equal(result[0], expected_batch0), "Data order changed!"

    # Second batch should be [9,10,11,12,13,14,15,16]
    expected_batch1 = np.array([9, 10, 11, 12, 13, 14, 15, 16], dtype=np.int8)
    assert np.array_equal(result[1], expected_batch1), "Data order changed!"

    print("[PASS] Test 2 passed! Data order preserved correctly.")

    # Test 3: Different start_dim
    print("\n" + "="*80)
    print("Test 3: Different start_dim Values")
    print("="*80)

    x_int8 = np.random.randint(-50, 50, (2, 3, 4, 5), dtype=np.int8)

    # start_dim=0: Flatten everything
    result_0 = flatten_int8(x_int8, start_dim=0)
    assert result_0.shape == (2*3*4*5,), f"Shape mismatch: {result_0.shape}"
    print(f"start_dim=0: {x_int8.shape} → {result_0.shape} (flatten all)")

    # start_dim=1: Preserve batch dimension
    result_1 = flatten_int8(x_int8, start_dim=1)
    assert result_1.shape == (2, 3*4*5), f"Shape mismatch: {result_1.shape}"
    print(f"start_dim=1: {x_int8.shape} → {result_1.shape} (preserve batch)")

    # start_dim=2: Preserve batch and channels
    result_2 = flatten_int8(x_int8, start_dim=2)
    assert result_2.shape == (2, 3, 4*5), f"Shape mismatch: {result_2.shape}"
    print(f"start_dim=2: {x_int8.shape} → {result_2.shape} (preserve B,C)")

    # start_dim=3: Flatten only last dimension (no change in this case)
    result_3 = flatten_int8(x_int8, start_dim=3)
    assert result_3.shape == (2, 3, 4, 5), f"Shape mismatch: {result_3.shape}"
    print(f"start_dim=3: {x_int8.shape} → {result_3.shape} (no change)")

    print("[PASS] Test 3 passed! Different start_dim values work correctly.")

    # Test 4: SimpleCNN use case
    print("\n" + "="*80)
    print("Test 4: SimpleCNN Use Case")
    print("="*80)

    # After pool2: [batch=1, channels=32, h=7, w=7]
    batch = 1
    channels = 32
    h, w = 7, 7

    x_int8 = np.random.randint(-100, 100, (batch, channels, h, w), dtype=np.int8)

    result = flatten_int8(x_int8, start_dim=1)

    expected_shape = (batch, channels * h * w)  # [1, 1568]
    assert result.shape == expected_shape, f"Shape mismatch: {result.shape}"

    print(f"Input:  {x_int8.shape} (after pool2)")
    print(f"Output: {result.shape} (ready for classifier)")
    print(f"Features: {channels} x {h} x {w} = {channels*h*w}")
    print("[PASS] Test 4 passed! Ready for Linear(1568 → 10).")

    # Test 5: Quantization scale preservation
    print("\n" + "="*80)
    print("Test 5: Quantization Scale Preservation")
    print("="*80)

    # Create some quantized values
    x_int8 = np.array([
        [[[-50, -25], [0, 25]],
         [[50, 75], [100, 127]]]
    ], dtype=np.int8)  # [1, 2, 2, 2]

    result = flatten_int8(x_int8, start_dim=1)

    print(f"Input INT8 values:  {x_int8[0].flatten()}")
    print(f"Output INT8 values: {result[0]}")

    # All values should be identical
    assert np.array_equal(x_int8.flatten(), result.flatten()), "Values changed!"

    # If input scale was 0.1, output scale is still 0.1
    input_scale = 0.1
    output_scale = input_scale  # PRESERVED!

    print(f"Input scale:  {input_scale}")
    print(f"Output scale: {output_scale} (unchanged)")
    print("[PASS] Test 5 passed! Quantization scale preserved.")

    # Test 6: Edge case - single element
    print("\n" + "="*80)
    print("Test 6: Edge Cases")
    print("="*80)

    # Single element
    x_int8 = np.array([[[42]]], dtype=np.int8)  # [1, 1, 1]
    result = flatten_int8(x_int8, start_dim=1)
    assert result.shape == (1, 1), f"Shape mismatch: {result.shape}"
    assert result[0, 0] == 42, "Value changed!"
    print(f"Single element: {x_int8.shape} → {result.shape} [PASS]")

    # Large tensor
    x_int8 = np.random.randint(-128, 127, (8, 64, 56, 56), dtype=np.int8)
    result = flatten_int8(x_int8, start_dim=1)
    assert result.shape == (8, 64*56*56), f"Shape mismatch: {result.shape}"
    print(f"Large tensor: {x_int8.shape} → {result.shape} [PASS]")

    print("[PASS] Test 6 passed!")

    print("\n" + "="*80)
    print("[PASS] All Flatten tests passed!")
    print("="*80)
    print("\nKey insight: Flatten is JUST a reshape - no computation!")
    print("- No quantization/dequantization needed")
    print("- No scale changes")
    print("- No data modification")
    print("- Zero computational cost")


if __name__ == "__main__":
    test_flatten()
