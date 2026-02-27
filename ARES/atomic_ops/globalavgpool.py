# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Atomic GlobalAvgPool Operation - INT8

Implements global average pooling with TRUE INT8 arithmetic.

Key insight: Global average pooling averages over entire spatial dimensions (HxW),
reducing [B, C, H, W] → [B, C, 1, 1]. Requires rescaling like regular avgpool.
"""

import numpy as np


def global_avgpool_int8(
    x_int8: np.ndarray,
    scale_input: float = 1.0,
    scale_output: float = 1.0,
    keepdims: bool = True
) -> np.ndarray:
    """
    INT8 global average pooling with rescaling.

    Averages over entire spatial dimensions (H, W), producing one value per channel.

    Process:
        1. Accumulate INT8 values → INT32 (prevent overflow)
        2. Average over HxW and rescale
        3. Requantize to INT8

    Formula:
        out[b, c] = Quantize(
            avg over (h, w) of: Dequantize(input[b, c, h, w])
        )

    Args:
        x_int8: Input INT8 tensor [B, C, H, W]
        scale_input: Input quantization scale
        scale_output: Output quantization scale
        keepdims: If True, output is [B, C, 1, 1]; if False, output is [B, C]

    Returns:
        Output INT8 tensor [B, C, 1, 1] or [B, C]

    Notes:
        - Very common in modern CNNs (ResNet, MobileNet, etc.)
        - Reduces spatial dimensions to 1x1 before final classification
        - Equivalent to avgpool with kernel_size=(H, W)
    """
    batch, channels, h, w = x_int8.shape

    # Number of spatial elements
    spatial_size = h * w

    if keepdims:
        output = np.zeros((batch, channels, 1, 1), dtype=np.int8)
    else:
        output = np.zeros((batch, channels), dtype=np.int8)

    # Perform global average pooling
    for b in range(batch):
        for c in range(channels):
            # Sum all spatial elements (INT32 to prevent overflow)
            sum_val = np.sum(x_int8[b, c, :, :].astype(np.int32))

            # Convert to FP32, compute average, and rescale
            avg_fp32 = (sum_val / spatial_size) * scale_input

            # Requantize to INT8
            # Use same rounding as C: (int)(x + 0.5) for positive, (int)(x - 0.5) for negative
            output_val = avg_fp32 / scale_output
            output_int32 = np.int32(np.floor(output_val + 0.5))

            # Clip to INT8 range
            output_int32 = np.clip(output_int32, -128, 127)

            if keepdims:
                output[b, c, 0, 0] = np.int8(output_int32)
            else:
                output[b, c] = np.int8(output_int32)

    return output


def global_avgpool_int8_fast(
    x_int8: np.ndarray,
    scale_input: float = 1.0,
    scale_output: float = 1.0,
    keepdims: bool = True
) -> np.ndarray:
    """
    Vectorized version of global average pooling for better performance.

    Args:
        x_int8: Input INT8 tensor [B, C, H, W]
        scale_input: Input quantization scale
        scale_output: Output quantization scale
        keepdims: If True, output is [B, C, 1, 1]; if False, output is [B, C]

    Returns:
        Output INT8 tensor
    """
    batch, channels, h, w = x_int8.shape
    spatial_size = h * w

    # Sum over spatial dimensions (keepdims for broadcasting)
    sum_val = np.sum(x_int8.astype(np.int32), axis=(2, 3), keepdims=True)

    # Compute average and rescale
    avg_fp32 = (sum_val / spatial_size) * scale_input
    output_val = avg_fp32 / scale_output
    # Use same rounding as C: (int)(x + 0.5) for positive, (int)(x - 0.5) for negative
    output_int32 = np.floor(output_val + 0.5).astype(np.int32)

    # Clip to INT8 range
    output_int32 = np.clip(output_int32, -128, 127)
    output = output_int32.astype(np.int8)

    if not keepdims:
        output = output.squeeze(axis=(2, 3))

    return output


def test_global_avgpool_int8():
    """Test INT8 global average pooling against FP32 reference."""
    print("Testing global_avgpool_int8...")

    # Test case 1: Basic global pooling
    print("\n  Test 1: Basic global pooling")
    batch, channels, h, w = 2, 3, 4, 4
    scale = 0.1

    x_fp32 = np.random.randn(batch, channels, h, w).astype(np.float32)
    x_int8 = np.clip(np.round(x_fp32 / scale), -128, 127).astype(np.int8)

    # INT8 global avgpool
    y_int8 = global_avgpool_int8(
        x_int8,
        scale_input=scale,
        scale_output=scale,
        keepdims=True
    )

    # FP32 reference
    y_fp32_ref = np.mean(x_fp32, axis=(2, 3), keepdims=True)
    y_int8_ref = np.clip(np.round(y_fp32_ref / scale), -128, 127).astype(np.int8)

    diff = np.max(np.abs(y_int8.astype(np.int32) - y_int8_ref.astype(np.int32)))
    print(f"    Output shape: {y_int8.shape}")
    print(f"    Expected: (2, 3, 1, 1), Got: {y_int8.shape}")
    print(f"    Max difference: {diff}")
    assert y_int8.shape == (2, 3, 1, 1), f"Wrong output shape: {y_int8.shape}"
    assert diff <= 1, f"Test 1 failed: diff={diff}"
    print("    [OK] Test 1 passed")

    # Test case 2: Without keepdims
    print("\n  Test 2: Without keepdims")
    y_int8 = global_avgpool_int8(
        x_int8,
        scale_input=scale,
        scale_output=scale,
        keepdims=False
    )

    print(f"    Output shape: {y_int8.shape}")
    print(f"    Expected: (2, 3), Got: {y_int8.shape}")
    assert y_int8.shape == (2, 3), f"Wrong output shape: {y_int8.shape}"
    print("    [OK] Test 2 passed")

    # Test case 3: Fast version
    print("\n  Test 3: Fast vectorized version")
    y_int8_slow = global_avgpool_int8(x_int8, scale, scale, keepdims=True)
    y_int8_fast = global_avgpool_int8_fast(x_int8, scale, scale, keepdims=True)

    diff = np.max(np.abs(y_int8_slow.astype(np.int32) - y_int8_fast.astype(np.int32)))
    print(f"    Max difference: {diff}")
    assert diff == 0, f"Fast version differs from slow: diff={diff}"
    print("    [OK] Test 3 passed")

    # Test case 4: Different scales
    print("\n  Test 4: Different input/output scales")
    scale_in = 0.1
    scale_out = 0.05

    y_int8 = global_avgpool_int8(x_int8, scale_in, scale_out, keepdims=True)

    # FP32 reference
    y_fp32 = x_int8.astype(np.float32) * scale_in
    y_fp32_avg = np.mean(y_fp32, axis=(2, 3), keepdims=True)
    y_int8_ref = np.clip(np.round(y_fp32_avg / scale_out), -128, 127).astype(np.int8)

    diff = np.max(np.abs(y_int8.astype(np.int32) - y_int8_ref.astype(np.int32)))
    print(f"    Max difference: {diff}")
    assert diff <= 1, f"Test 4 failed: diff={diff}"
    print("    [OK] Test 4 passed")

    print("\n  [OK] All tests passed!")
    return True


if __name__ == "__main__":
    test_global_avgpool_int8()
