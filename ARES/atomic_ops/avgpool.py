# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Atomic AvgPool2D Operation - INT8

Implements 2D average pooling with TRUE INT8 arithmetic.

Key insight: AvgPool changes the scale! We need to rescale after averaging.
Average of quantized values: avg(Q(x)) ≠ Q(avg(x)) without scale adjustment.
"""

import numpy as np


def avgpool2d_int8(
    x_int8: np.ndarray,
    kernel_size: tuple = (2, 2),
    stride: tuple = None,
    padding: tuple = (0, 0),
    scale_input: float = 1.0,
    scale_output: float = 1.0
) -> np.ndarray:
    """
    INT8 2D average pooling with rescaling.

    Average pooling changes the effective scale, so we need to:
    1. Accumulate INT8 values → INT32 (prevent overflow)
    2. Average and rescale
    3. Requantize to INT8

    Formula:
        out[b, c, oh, ow] = Quantize(
            avg over (kh, kw) of: Dequantize(input[...])
        )

    Args:
        x_int8: Input INT8 tensor [B, C, H, W]
        kernel_size: Pooling window size (kh, kw)
        stride: Stride (sh, sw). If None, defaults to kernel_size
        padding: Zero padding (ph, pw)
        scale_input: Input quantization scale
        scale_output: Output quantization scale

    Returns:
        Output INT8 tensor [B, C, H_out, W_out]

    Notes:
        - Unlike MaxPool, AvgPool requires rescaling
        - We use INT32 accumulation to prevent overflow
        - Scale factor: scale_output / scale_input (typically ~1.0 if scales match)
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
        # Pad with 0 for average pooling
        x_padded = np.pad(
            x_int8,
            ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)),
            mode='constant',
            constant_values=0
        )
    else:
        x_padded = x_int8

    # Initialize output
    output = np.zeros((batch, channels, h_out, w_out), dtype=np.int8)

    # Kernel area for averaging
    kernel_area = k_h * k_w

    # Perform average pooling
    for b in range(batch):
        for c in range(channels):
            for oh in range(h_out):
                for ow in range(w_out):
                    # Compute starting position
                    h_start = oh * s_h
                    w_start = ow * s_w

                    # Extract pooling window and cast to INT32
                    window = x_padded[
                        b, c,
                        h_start:h_start + k_h,
                        w_start:w_start + k_w
                    ].astype(np.int32)

                    # Sum (INT32 to prevent overflow)
                    sum_val = np.sum(window)

                    # Convert to FP32, compute average, and rescale
                    avg_fp32 = (sum_val / kernel_area) * scale_input

                    # Requantize to INT8
                    output_val = avg_fp32 / scale_output
                    output_int32 = np.int32(np.round(output_val))

                    # Clip to INT8 range
                    output_int32 = np.clip(output_int32, -128, 127)
                    output[b, c, oh, ow] = np.int8(output_int32)

    return output


def test_avgpool2d_int8():
    """Test INT8 average pooling against FP32 reference."""
    print("Testing avgpool2d_int8...")

    # Test case 1: Simple 2x2 pooling
    batch, channels, h, w = 1, 2, 4, 4

    # Create test input
    x_fp32 = np.random.randn(batch, channels, h, w).astype(np.float32)

    # Quantize
    scale = 0.1
    x_int8 = np.clip(np.round(x_fp32 / scale), -128, 127).astype(np.int8)

    # INT8 avgpool
    y_int8 = avgpool2d_int8(
        x_int8,
        kernel_size=(2, 2),
        stride=(2, 2),
        padding=(0, 0),
        scale_input=scale,
        scale_output=scale
    )

    # FP32 reference (manual implementation)
    y_fp32_ref = np.zeros((batch, channels, 2, 2), dtype=np.float32)
    for b in range(batch):
        for c in range(channels):
            for oh in range(2):
                for ow in range(2):
                    h_start = oh * 2
                    w_start = ow * 2
                    window = x_fp32[b, c, h_start:h_start+2, w_start:w_start+2]
                    y_fp32_ref[b, c, oh, ow] = np.mean(window)

    # Quantize reference
    y_int8_ref = np.clip(np.round(y_fp32_ref / scale), -128, 127).astype(np.int8)

    # Compare
    max_diff = np.max(np.abs(y_int8.astype(np.int32) - y_int8_ref.astype(np.int32)))
    print(f"  Max difference: {max_diff}")
    print(f"  Output shape: {y_int8.shape}")
    print(f"  Output range: [{y_int8.min()}, {y_int8.max()}]")

    assert max_diff <= 1, f"INT8 avgpool differs too much from reference (max_diff={max_diff})"
    print("  [OK] Test passed!")

    return True


if __name__ == "__main__":
    test_avgpool2d_int8()
