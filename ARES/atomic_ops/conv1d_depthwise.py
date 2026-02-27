# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Atomic Conv1D Depthwise Operation - INT8

Implements 1D depthwise convolution with TRUE INT8 arithmetic.
Used in MAMBA for the conv1d layer after in_proj.

Key Differences from Standard Conv2D:
1. 1D: operates on time/sequence dimension only
2. Depthwise: each channel has its own filter (groups=C)
3. Causal: left-only padding for autoregressive models

Data Layout:
- Input: [B, C, L] or [C, L] (channel-first)
- Weight: [C, K] (one K-tap filter per channel)
- Output: [B, C, L] or [C, L] (same shape as input with causal padding)
"""

import numpy as np
from typing import Optional, Tuple

try:
    from .quantize import quantize_linear, dequantize_linear
except ImportError:
    from quantize import quantize_linear, dequantize_linear


def conv1d_depthwise_int8(
    x_int8: np.ndarray,
    w_int8: np.ndarray,
    bias_int32: Optional[np.ndarray],
    scale_x: float,
    scale_w: float,
    scale_y: float,
    padding: int = 0,
    causal: bool = True,
    zero_point_x: int = 0,
    zero_point_w: int = 0,
    zero_point_y: int = 0
) -> np.ndarray:
    """
    INT8 1D Depthwise Convolution with INT32 accumulation.

    This implements TRUE INT8 depthwise convolution as it would run on an MCU.
    Each input channel is convolved with its own filter (no cross-channel mixing).

    Args:
        x_int8: Input tensor [B, C, L] or [C, L] (INT8)
        w_int8: Weight tensor [C, K] (INT8), one K-tap filter per channel
        bias_int32: Bias tensor [C] (INT32), optional
        scale_x: Input quantization scale
        scale_w: Weight quantization scale
        scale_y: Output quantization scale
        padding: Left padding amount (for causal mode) or total padding
        causal: If True, apply left-only padding (default for Mamba)
        zero_point_x: Input zero point (usually 0 for symmetric)
        zero_point_w: Weight zero point (usually 0)
        zero_point_y: Output zero point (usually 0)

    Returns:
        Output tensor [B, C, L_out] or [C, L_out] (INT8)
        With causal padding, L_out = L

    Mathematical Details:
        For each channel c and position l:
        output[c, l] = sum_{k=0}^{K-1} x[c, l-K+1+k+pad] * w[c, k] + bias[c]

        Then rescale: scale_combined = (scale_x * scale_w) / scale_y
        And requantize to INT8.
    """
    # Handle both 2D [C, L] and 3D [B, C, L] inputs
    squeeze_batch = False
    if x_int8.ndim == 2:
        x_int8 = x_int8[np.newaxis, :, :]  # Add batch dimension
        squeeze_batch = True

    batch, c_in, l_in = x_int8.shape
    c_w, k = w_int8.shape

    assert c_in == c_w, f"Input channels {c_in} must match weight channels {c_w}"

    # For causal convolution: pad left only
    if causal:
        pad_left = k - 1 if padding == 0 else padding
        pad_right = 0
    else:
        # Standard same-padding: split evenly
        pad_left = padding // 2
        pad_right = padding - pad_left

    # Calculate output length
    l_out = l_in + pad_left + pad_right - k + 1

    # Apply padding
    if pad_left > 0 or pad_right > 0:
        x_padded = np.pad(
            x_int8,
            ((0, 0), (0, 0), (pad_left, pad_right)),
            mode='constant',
            constant_values=zero_point_x
        )
    else:
        x_padded = x_int8

    # Initialize output accumulator (INT32)
    output_int32 = np.zeros((batch, c_in, l_out), dtype=np.int32)

    # Perform INT8 depthwise convolution with INT32 accumulation
    for b in range(batch):
        for c in range(c_in):
            for ol in range(l_out):
                # Accumulator for this output position (INT32)
                acc = np.int32(0)

                # Convolve this channel with its filter
                for kk in range(k):
                    l_idx = ol + kk

                    # INT8 x INT8 -> INT32
                    x_val = np.int32(x_padded[b, c, l_idx])
                    w_val = np.int32(w_int8[c, kk])

                    # Accumulate (MAC operation)
                    acc += x_val * w_val

                output_int32[b, c, ol] = acc

    # Add bias if provided (INT32 + INT32)
    if bias_int32 is not None:
        output_int32 += bias_int32.reshape(1, -1, 1)

    # Requantization: Convert INT32 -> FP32 -> INT8
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

    # Remove batch dimension if input was 2D
    if squeeze_batch:
        output_int8 = output_int8[0]

    return output_int8


def _div_round_nearest_even_s64(num: int, den: int) -> int:
    """Divide with round-to-nearest-even (ties to even). Denominator must be > 0."""
    if den == 0:
        return 0
    if num >= 0:
        q = num // den
    else:
        q = -((-num) // den)
    r = num - q * den
    if r == 0:
        return int(q)
    abs_r = r if r >= 0 else -r
    twice_r = abs_r << 1
    if twice_r > den or (twice_r == den and (q & 1)):
        q += 1 if num >= 0 else -1
    return int(q)


def _mul_shift_round_nearest_even(val: int, mul: int, shift: int) -> int:
    """Fixed-point multiply + shift with round-to-nearest-even."""
    prod = int(val) * int(mul)
    if shift <= 0:
        return int(prod)
    return _div_round_nearest_even_s64(prod, 1 << shift)


def conv1d_depthwise_int8_fixedpoint(
    x_int8: np.ndarray,
    w_int8: np.ndarray,
    bias_int32: Optional[np.ndarray],
    scale_x: float,
    scale_w: float,
    scale_y: float,
    padding: int = 0,
    causal: bool = True,
    zero_point_x: int = 0,
    zero_point_y: int = 0,
    requant_shift: int = 24,
) -> np.ndarray:
    """
    INT8 depthwise Conv1D with fixed-point requantization (matches GAP9 kernel).

    Uses:
      - requant_mul = round((scale_x * scale_w / scale_y) * 2^requant_shift)
      - out = mul_shift_round_nearest_even(acc, requant_mul, requant_shift)
    """
    squeeze_batch = False
    if x_int8.ndim == 2:
        x_int8 = x_int8[np.newaxis, :, :]
        squeeze_batch = True

    batch, c_in, l_in = x_int8.shape
    c_w, k = w_int8.shape
    assert c_in == c_w, f"Input channels {c_in} must match weight channels {c_w}"

    if causal:
        pad_left = k - 1 if padding == 0 else padding
        pad_right = 0
    else:
        pad_left = padding // 2
        pad_right = padding - pad_left

    l_out = l_in + pad_left + pad_right - k + 1

    if pad_left > 0 or pad_right > 0:
        x_padded = np.pad(
            x_int8,
            ((0, 0), (0, 0), (pad_left, pad_right)),
            mode='constant',
            constant_values=zero_point_x
        )
    else:
        x_padded = x_int8

    output_int8 = np.zeros((batch, c_in, l_out), dtype=np.int8)

    scale_combined = (scale_x * scale_w) / scale_y
    requant_mul = int(np.round(scale_combined * (1 << requant_shift)))

    for b in range(batch):
        for c in range(c_in):
            bias_val = int(bias_int32[c]) if bias_int32 is not None else 0
            in_ch = x_padded[b, c]
            w_ch = w_int8[c]

            for l in range(l_out):
                acc = bias_val
                for kk in range(k):
                    il = l + kk - pad_left
                    if 0 <= il < l_in:
                        acc += int(in_ch[il + pad_left]) * int(w_ch[kk])

                out_int = _mul_shift_round_nearest_even(acc, requant_mul, requant_shift)
                if out_int < -128:
                    out_int = -128
                elif out_int > 127:
                    out_int = 127
                output_int8[b, c, l] = np.int8(out_int)

    if squeeze_batch:
        output_int8 = output_int8[0]

    return output_int8

def conv1d_depthwise_fp32_reference(
    x: np.ndarray,
    w: np.ndarray,
    bias: Optional[np.ndarray],
    padding: int = 0,
    causal: bool = True
) -> np.ndarray:
    """
    FP32 depthwise 1D convolution reference implementation.

    Used for testing and verification against INT8 version.
    """
    squeeze_batch = False
    if x.ndim == 2:
        x = x[np.newaxis, :, :]
        squeeze_batch = True

    batch, c_in, l_in = x.shape
    c_w, k = w.shape

    assert c_in == c_w, f"Input channels {c_in} must match weight channels {c_w}"

    if causal:
        pad_left = k - 1 if padding == 0 else padding
        pad_right = 0
    else:
        pad_left = padding // 2
        pad_right = padding - pad_left

    l_out = l_in + pad_left + pad_right - k + 1

    if pad_left > 0 or pad_right > 0:
        x_padded = np.pad(
            x,
            ((0, 0), (0, 0), (pad_left, pad_right)),
            mode='constant',
            constant_values=0.0
        )
    else:
        x_padded = x

    output = np.zeros((batch, c_in, l_out), dtype=np.float32)

    for b in range(batch):
        for c in range(c_in):
            for ol in range(l_out):
                acc = 0.0
                for kk in range(k):
                    l_idx = ol + kk
                    acc += x_padded[b, c, l_idx] * w[c, kk]
                output[b, c, ol] = acc

    if bias is not None:
        output += bias.reshape(1, -1, 1)

    if squeeze_batch:
        output = output[0]

    return output


def test_conv1d_depthwise():
    """Test INT8 Conv1D Depthwise implementation."""
    print("=" * 80)
    print("Testing INT8 Conv1D Depthwise")
    print("=" * 80)

    # Test 1: Basic causal convolution (Mamba-style)
    print("\n--- Test 1: Basic Causal Conv1D ---")
    batch, channels, length = 1, 4, 8
    kernel_size = 4  # Typical for Mamba

    # FP32 inputs
    x_fp32 = np.random.randn(batch, channels, length).astype(np.float32) * 0.5
    w_fp32 = np.random.randn(channels, kernel_size).astype(np.float32) * 0.1
    bias_fp32 = np.random.randn(channels).astype(np.float32) * 0.1

    # Quantization scales
    scale_x = 0.05
    scale_w = 0.02
    scale_y = 0.05

    # Quantize inputs
    x_int8 = quantize_linear(x_fp32, scale_x)
    w_int8 = quantize_linear(w_fp32, scale_w)
    bias_int32 = np.round(bias_fp32 / (scale_x * scale_w)).astype(np.int32)

    print(f"Input shape: {x_int8.shape}")
    print(f"Weight shape: {w_int8.shape}")
    print(f"Kernel size: {kernel_size}")

    # Run INT8 convolution (causal)
    output_int8 = conv1d_depthwise_int8(
        x_int8, w_int8, bias_int32,
        scale_x, scale_w, scale_y,
        causal=True
    )

    print(f"Output shape: {output_int8.shape}")
    assert output_int8.shape == x_int8.shape, "Causal conv should preserve length"

    # Run FP32 reference
    output_fp32_ref = conv1d_depthwise_fp32_reference(
        x_fp32, w_fp32, bias_fp32, causal=True
    )

    # Dequantize INT8 output for comparison
    output_fp32_from_int8 = dequantize_linear(output_int8, scale_y)

    # Compare
    error = np.abs(output_fp32_ref - output_fp32_from_int8)
    max_error = np.max(error)
    mean_error = np.mean(error)

    print(f"\nFP32 reference output (channel 0): {output_fp32_ref[0, 0, :4]}")
    print(f"INT8 output (dequantized, ch 0):   {output_fp32_from_int8[0, 0, :4]}")
    print(f"\nMax error:  {max_error:.6f}")
    print(f"Mean error: {mean_error:.6f}")

    acceptable_error = max(scale_x, scale_w, scale_y) * 2
    if max_error < acceptable_error:
        print(f"Test 1 PASSED! Error {max_error:.6f} < {acceptable_error:.6f}")
    else:
        print(f"Test 1 WARNING: Error {max_error:.6f} >= {acceptable_error:.6f}")

    # Test 2: Verify against PyTorch F.conv1d with groups=C
    print("\n--- Test 2: Verify Against PyTorch (if available) ---")
    try:
        import torch
        import torch.nn.functional as F

        # Create PyTorch tensors
        x_torch = torch.from_numpy(x_fp32)
        # PyTorch depthwise conv1d expects weight shape [C_out, 1, K]
        w_torch = torch.from_numpy(w_fp32[:, np.newaxis, :])
        bias_torch = torch.from_numpy(bias_fp32)

        # PyTorch causal conv: pad left only
        x_padded = F.pad(x_torch, (kernel_size - 1, 0))
        output_torch = F.conv1d(x_padded, w_torch, bias_torch, groups=channels)
        output_torch_np = output_torch.numpy()

        # Compare PyTorch with our FP32 reference
        pytorch_vs_ref_error = np.max(np.abs(output_torch_np - output_fp32_ref))
        print(f"PyTorch vs FP32 reference max error: {pytorch_vs_ref_error:.9f}")

        if pytorch_vs_ref_error < 1e-5:
            print("Test 2 PASSED! FP32 reference matches PyTorch")
        else:
            print("Test 2 WARNING: FP32 reference differs from PyTorch")

    except ImportError:
        print("PyTorch not available, skipping Test 2")

    # Test 3: Multi-batch processing
    print("\n--- Test 3: Multi-Batch Processing ---")
    batch_sizes = [1, 4, 8]
    for bs in batch_sizes:
        x_batch = np.random.randn(bs, channels, length).astype(np.float32) * 0.5
        x_batch_int8 = quantize_linear(x_batch, scale_x)

        output_batch = conv1d_depthwise_int8(
            x_batch_int8, w_int8, bias_int32,
            scale_x, scale_w, scale_y, causal=True
        )

        assert output_batch.shape == (bs, channels, length), f"Batch {bs} shape mismatch"
        print(f"  Batch size {bs}: output shape {output_batch.shape} - OK")

    print("Test 3 PASSED!")

    # Test 4: 2D input (no batch dimension)
    print("\n--- Test 4: 2D Input [C, L] ---")
    x_2d = np.random.randn(channels, length).astype(np.float32) * 0.5
    x_2d_int8 = quantize_linear(x_2d, scale_x)

    output_2d = conv1d_depthwise_int8(
        x_2d_int8, w_int8, bias_int32,
        scale_x, scale_w, scale_y, causal=True
    )

    assert output_2d.shape == (channels, length), f"2D output shape mismatch"
    print(f"  2D input shape: {x_2d_int8.shape} -> output shape: {output_2d.shape}")
    print("Test 4 PASSED!")

    # Test 5: No bias
    print("\n--- Test 5: No Bias ---")
    output_no_bias = conv1d_depthwise_int8(
        x_int8, w_int8, None,
        scale_x, scale_w, scale_y, causal=True
    )
    output_ref_no_bias = conv1d_depthwise_fp32_reference(
        x_fp32, w_fp32, None, causal=True
    )
    output_from_int8_no_bias = dequantize_linear(output_no_bias, scale_y)

    error_no_bias = np.max(np.abs(output_ref_no_bias - output_from_int8_no_bias))
    print(f"  Max error (no bias): {error_no_bias:.6f}")
    if error_no_bias < acceptable_error:
        print("Test 5 PASSED!")
    else:
        print("Test 5 WARNING: Error higher than expected")

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    test_conv1d_depthwise()
