# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Atomic Concatenate Operation - INT8

Implements channel concatenation with TRUE INT8 arithmetic.

Key insight: Concatenating tensors with different scales requires rescaling!
We need to bring all inputs to the same scale before concatenating.
"""

import numpy as np
from typing import List


def concat_int8(
    tensors: List[np.ndarray],
    scales: List[float],
    scale_output: float,
    axis: int = 1
) -> np.ndarray:
    """
    INT8 concatenation with rescaling.

    Concatenates multiple INT8 tensors along a specified axis (typically channel axis).
    Each input may have a different quantization scale.

    Process:
        1. Dequantize each input to FP32
        2. Requantize to output scale
        3. Concatenate along specified axis

    Formula:
        For each tensor i:
            tensor_i_rescaled = Quantize(Dequantize(tensor_i, scale_i), scale_output)
        out = Concat(tensor_0_rescaled, tensor_1_rescaled, ...)

    Args:
        tensors: List of INT8 tensors to concatenate
        scales: List of quantization scales (one per tensor)
        scale_output: Output quantization scale
        axis: Axis along which to concatenate (default=1 for channels)

    Returns:
        Concatenated INT8 tensor

    Notes:
        - Critical for DenseNet, U-Net, and similar architectures
        - Typically used on channel axis (axis=1 for [B, C, H, W])
        - All tensors must have same shape except along concat axis
    """
    assert len(tensors) == len(scales), \
        f"Number of tensors ({len(tensors)}) must match number of scales ({len(scales)})"

    assert len(tensors) > 0, "Must provide at least one tensor to concatenate"

    # Rescale each tensor to output scale
    rescaled_tensors = []
    for tensor, scale_in in zip(tensors, scales):
        if np.isclose(scale_in, scale_output):
            # No rescaling needed
            rescaled_tensors.append(tensor)
        else:
            # Dequantize and requantize
            fp32 = tensor.astype(np.float32) * scale_in
            output_val = fp32 / scale_output
            output_int32 = np.round(output_val).astype(np.int32)
            output_int32 = np.clip(output_int32, -128, 127)
            rescaled_tensors.append(output_int32.astype(np.int8))

    # Concatenate along specified axis
    return np.concatenate(rescaled_tensors, axis=axis)


def concat_int8_channel(
    tensors: List[np.ndarray],
    scales: List[float],
    scale_output: float
) -> np.ndarray:
    """
    Convenience function for channel concatenation (axis=1).

    Args:
        tensors: List of INT8 tensors [B, C_i, H, W]
        scales: List of quantization scales
        scale_output: Output quantization scale

    Returns:
        Concatenated INT8 tensor [B, sum(C_i), H, W]
    """
    return concat_int8(tensors, scales, scale_output, axis=1)


def test_concat_int8():
    """Test INT8 concatenation against FP32 reference."""
    print("Testing concat_int8...")

    # Test case 1: Two tensors, same scale
    print("\n  Test 1: Two tensors, same scale")
    scale = 0.1
    batch, h, w = 1, 4, 4

    x1_fp32 = np.random.randn(batch, 2, h, w).astype(np.float32)
    x2_fp32 = np.random.randn(batch, 3, h, w).astype(np.float32)

    x1_int8 = np.clip(np.round(x1_fp32 / scale), -128, 127).astype(np.int8)
    x2_int8 = np.clip(np.round(x2_fp32 / scale), -128, 127).astype(np.int8)

    result_int8 = concat_int8_channel([x1_int8, x2_int8], [scale, scale], scale)

    # FP32 reference
    concat_fp32 = np.concatenate([x1_fp32, x2_fp32], axis=1)
    result_ref = np.clip(np.round(concat_fp32 / scale), -128, 127).astype(np.int8)

    diff = np.max(np.abs(result_int8.astype(np.int32) - result_ref.astype(np.int32)))
    print(f"    Output shape: {result_int8.shape}")
    print(f"    Expected: (1, 5, 4, 4), Got: {result_int8.shape}")
    print(f"    Max difference: {diff}")
    assert result_int8.shape == (1, 5, 4, 4), f"Wrong output shape: {result_int8.shape}"
    assert diff <= 1, f"Test 1 failed: diff={diff}"
    print("    [OK] Test 1 passed")

    # Test case 2: Three tensors, different scales
    print("\n  Test 2: Three tensors, different scales")
    scale_x1 = 0.1
    scale_x2 = 0.2
    scale_x3 = 0.15
    scale_out = 0.12

    x1_int8 = np.random.randint(-100, 100, (batch, 2, h, w), dtype=np.int8)
    x2_int8 = np.random.randint(-100, 100, (batch, 3, h, w), dtype=np.int8)
    x3_int8 = np.random.randint(-100, 100, (batch, 1, h, w), dtype=np.int8)

    result_int8 = concat_int8_channel(
        [x1_int8, x2_int8, x3_int8],
        [scale_x1, scale_x2, scale_x3],
        scale_out
    )

    # FP32 reference
    x1_fp32 = x1_int8.astype(np.float32) * scale_x1
    x2_fp32 = x2_int8.astype(np.float32) * scale_x2
    x3_fp32 = x3_int8.astype(np.float32) * scale_x3
    concat_fp32 = np.concatenate([x1_fp32, x2_fp32, x3_fp32], axis=1)
    result_ref = np.clip(np.round(concat_fp32 / scale_out), -128, 127).astype(np.int8)

    diff = np.max(np.abs(result_int8.astype(np.int32) - result_ref.astype(np.int32)))
    print(f"    Output shape: {result_int8.shape}")
    print(f"    Expected: (1, 6, 4, 4), Got: {result_int8.shape}")
    print(f"    Max difference: {diff}")
    assert result_int8.shape == (1, 6, 4, 4), f"Wrong output shape: {result_int8.shape}"
    assert diff <= 1, f"Test 2 failed: diff={diff}"
    print("    [OK] Test 2 passed")

    # Test case 3: Different axis
    print("\n  Test 3: Concatenate along spatial axis")
    x1_int8 = np.random.randint(-100, 100, (batch, 2, 2, w), dtype=np.int8)
    x2_int8 = np.random.randint(-100, 100, (batch, 2, 2, w), dtype=np.int8)

    result_int8 = concat_int8([x1_int8, x2_int8], [scale, scale], scale, axis=2)

    assert result_int8.shape == (batch, 2, 4, w), f"Wrong output shape: {result_int8.shape}"
    print(f"    Output shape: {result_int8.shape}")
    print("    [OK] Test 3 passed")

    print("\n  [OK] All tests passed!")
    return True


if __name__ == "__main__":
    test_concat_int8()
