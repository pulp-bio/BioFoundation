# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Atomic Conv2D Operation - INT8

Implements 2D convolution with TRUE INT8 arithmetic.

This is a reference implementation that shows exactly how INT8 convolution
should work on an MCU. The actual computation uses INT8 inputs/weights with
INT32 accumulation.
"""

import numpy as np
from typing import Tuple, Optional

# Handle both module import and standalone execution
try:
    from .quantize import quantize_linear, dequantize_linear
except ImportError:
    from quantize import quantize_linear, dequantize_linear


def conv2d_int8(
    x_int8: np.ndarray,
    w_int8: np.ndarray,
    bias_int32: Optional[np.ndarray],
    scale_x: float,
    scale_w: float,
    scale_y: float,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    zero_point_x: int = 0,
    zero_point_w: int = 0,
    zero_point_y: int = 0,
    groups: int = 1
) -> np.ndarray:
    """
    INT8 2D Convolution with INT32 accumulation.

    This implements TRUE INT8 convolution as it would run on an MCU.

    Args:
        x_int8: Input tensor [B, C_in, H, W] (INT8)
        w_int8: Weight tensor [C_out, C_in/groups, K_h, K_w] (INT8)
        bias_int32: Bias tensor [C_out] (INT32), optional
        scale_x: Input quantization scale
        scale_w: Weight quantization scale
        scale_y: Output quantization scale
        stride: Convolution stride (h, w)
        padding: Zero padding (h, w)
        zero_point_x: Input zero point (usually 0 for symmetric)
        zero_point_w: Weight zero point (usually 0)
        zero_point_y: Output zero point (usually 0)
        groups: Number of groups (1=standard conv, C_in=depthwise)

    Returns:
        Output tensor [B, C_out, H_out, W_out] (INT8)

    Mathematical Details:
        1. Compute INT8 x INT8 → INT32 convolution
        2. Add bias (INT32)
        3. Rescale: scale_combined = (scale_x * scale_w) / scale_y
        4. Requantize to INT8

    Notes:
        - All intermediate accumulation is done in INT32 to prevent overflow
        - This matches hardware INT8 MAC (multiply-accumulate) behavior
        - On MCU, this would use optimized INT8 SIMD instructions
        - groups parameter enables depthwise convolution (groups=C_in)
    """
    batch, c_in, h_in, w_in = x_int8.shape
    c_out, c_per_group, k_h, k_w = w_int8.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding

    # Validate groups parameter
    assert c_in % groups == 0, f"Input channels {c_in} must be divisible by groups {groups}"
    assert c_out % groups == 0, f"Output channels {c_out} must be divisible by groups {groups}"
    in_channels_per_group = c_in // groups
    out_channels_per_group = c_out // groups
    assert c_per_group == in_channels_per_group, \
        f"Weight shape mismatch: expected {in_channels_per_group} in_channels/group, got {c_per_group}"

    # Calculate output dimensions
    h_out = (h_in + 2 * pad_h - k_h) // stride_h + 1
    w_out = (w_in + 2 * pad_w - k_w) // stride_w + 1

    # Apply padding if needed
    if pad_h > 0 or pad_w > 0:
        x_padded = np.pad(
            x_int8,
            ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
            mode='constant',
            constant_values=zero_point_x  # Pad with zero point
        )
    else:
        x_padded = x_int8

    # Initialize output accumulator (INT32)
    output_int32 = np.zeros((batch, c_out, h_out, w_out), dtype=np.int32)

    # Perform INT8 convolution with INT32 accumulation
    for b in range(batch):
        for g in range(groups):
            # Input channels for this group
            ic_start = g * in_channels_per_group
            ic_end = ic_start + in_channels_per_group
            # Output channels for this group
            oc_start = g * out_channels_per_group
            oc_end = oc_start + out_channels_per_group

            for oc_idx, oc in enumerate(range(oc_start, oc_end)):
                for oh in range(h_out):
                    for ow in range(w_out):
                        # Compute starting position
                        h_start = oh * stride_h
                        w_start = ow * stride_w

                        # Accumulator for this output position (INT32)
                        acc = np.int32(0)

                        # Convolve over input channels within this group
                        for ic_local, ic in enumerate(range(ic_start, ic_end)):
                            for kh in range(k_h):
                                for kw in range(k_w):
                                    h_idx = h_start + kh
                                    w_idx = w_start + kw

                                    # INT8 x INT8 → INT32
                                    x_val = np.int32(x_padded[b, ic, h_idx, w_idx])
                                    w_val = np.int32(w_int8[oc, ic_local, kh, kw])

                                    # Accumulate (MAC operation)
                                    acc += x_val * w_val

                        output_int32[b, oc, oh, ow] = acc

    # Add bias if provided (INT32 + INT32)
    if bias_int32 is not None:
        output_int32 += bias_int32.reshape(1, -1, 1, 1)

    # Requantization: Convert INT32 → FP32 → INT8
    # This is necessary because we need to rescale the output
    scale_combined = (scale_x * scale_w) / scale_y

    # Convert to float, apply scale, requantize
    output_fp32 = output_int32.astype(np.float32) * scale_combined

    # Quantize back to INT8
    output_int8 = quantize_linear(
        output_fp32,
        scale=1.0,  # Already scaled
        zero_point=zero_point_y,
        qmin=-128,
        qmax=127
    )

    return output_int8


def conv2d_fp32_reference(
    x: np.ndarray,
    w: np.ndarray,
    bias: Optional[np.ndarray],
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0)
) -> np.ndarray:
    """
    FP32 convolution reference implementation.

    Used for testing and verification against INT8 version.
    """
    batch, c_in, h_in, w_in = x.shape
    c_out, _, k_h, k_w = w.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding

    # Calculate output dimensions
    h_out = (h_in + 2 * pad_h - k_h) // stride_h + 1
    w_out = (w_in + 2 * pad_w - k_w) // stride_w + 1

    # Apply padding
    if pad_h > 0 or pad_w > 0:
        x_padded = np.pad(
            x,
            ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
            mode='constant',
            constant_values=0.0
        )
    else:
        x_padded = x

    # Initialize output
    output = np.zeros((batch, c_out, h_out, w_out), dtype=np.float32)

    # Perform convolution
    for b in range(batch):
        for oc in range(c_out):
            for oh in range(h_out):
                for ow in range(w_out):
                    h_start = oh * stride_h
                    w_start = ow * stride_w

                    acc = 0.0
                    for ic in range(c_in):
                        for kh in range(k_h):
                            for kw in range(k_w):
                                h_idx = h_start + kh
                                w_idx = w_start + kw
                                acc += x_padded[b, ic, h_idx, w_idx] * w[oc, ic, kh, kw]

                    output[b, oc, oh, ow] = acc

    # Add bias
    if bias is not None:
        output += bias.reshape(1, -1, 1, 1)

    return output


def test_conv2d():
    """Test INT8 Conv2D implementation."""
    print("="*80)
    print("Testing INT8 Conv2D")
    print("="*80)

    # Create simple test case
    batch, c_in, h_in, w_in = 1, 1, 5, 5
    c_out, k_h, k_w = 1, 3, 3

    # FP32 inputs
    x_fp32 = np.random.randn(batch, c_in, h_in, w_in).astype(np.float32)
    w_fp32 = np.random.randn(c_out, c_in, k_h, k_w).astype(np.float32)
    bias_fp32 = np.random.randn(c_out).astype(np.float32)

    # Quantization scales
    scale_x = 0.1
    scale_w = 0.05
    scale_y = 0.1

    # Quantize inputs
    x_int8 = quantize_linear(x_fp32, scale_x)
    w_int8 = quantize_linear(w_fp32, scale_w)
    bias_int32 = (bias_fp32 / (scale_x * scale_w)).astype(np.int32)

    print(f"\nInput shape: {x_int8.shape}")
    print(f"Weight shape: {w_int8.shape}")
    print(f"Bias shape: {bias_int32.shape}")

    # Run INT8 convolution
    output_int8 = conv2d_int8(
        x_int8, w_int8, bias_int32,
        scale_x, scale_w, scale_y,
        stride=(1, 1),
        padding=(1, 1)
    )

    print(f"Output shape: {output_int8.shape}")

    # Run FP32 reference
    output_fp32_ref = conv2d_fp32_reference(
        x_fp32, w_fp32, bias_fp32,
        stride=(1, 1),
        padding=(1, 1)
    )

    # Dequantize INT8 output for comparison
    output_fp32_from_int8 = dequantize_linear(output_int8, scale_y)

    # Compare
    error = np.abs(output_fp32_ref - output_fp32_from_int8)
    max_error = np.max(error)
    mean_error = np.mean(error)

    print(f"\nFP32 reference output (first few values):")
    print(output_fp32_ref[0, 0, :3, :3])

    print(f"\nINT8 output (dequantized, first few values):")
    print(output_fp32_from_int8[0, 0, :3, :3])

    print(f"\nError statistics:")
    print(f"  Max error:  {max_error:.6f}")
    print(f"  Mean error: {mean_error:.6f}")

    # Check if error is within acceptable range (quantization error)
    acceptable_error = max(scale_x, scale_w, scale_y) * 2  # Rough estimate
    if max_error < acceptable_error:
        print(f"\n[PASS] Test passed! Error {max_error:.6f} < {acceptable_error:.6f}")
    else:
        print(f"\n[WARN]  Warning: Error {max_error:.6f} >= {acceptable_error:.6f}")

    print("="*80)


if __name__ == "__main__":
    test_conv2d()
