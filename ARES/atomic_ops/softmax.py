# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Atomic Softmax Operation - Multiple INT8 Implementations

Provides several softmax implementations:
1. softmax_int8: Simple dequantize-compute-requantize
2. softmax_int8_lut: Lookup table for exact exp() computation
3. softmax_int8_ibert: I-BERT polynomial approximation

The LUT approach is more accurate (no polynomial error) and uses only 256 entries.
The I-BERT approach follows the polynomial exp approximation from the I-BERT paper.

Reference:
    "I-BERT: Integer-only BERT Quantization" (Kim et al., 2021)
"""

import numpy as np
from typing import Dict, Tuple

# ---
# LUT-based Softmax (More Accurate)
# ---
# For INT8, only 256 entries are needed per scale.
# Advantages: no polynomial error, no z-boundary jitter

_exp_lut_cache: Dict[Tuple[float, int], np.ndarray] = {}


def build_exp_lut(scale_in: float, precision: int = 32) -> np.ndarray:
    """
    Build a lookup table for exp(x * scale_in) for all INT8 values.

    Args:
        scale_in: Input quantization scale
        precision: Output precision (16 or 32 bits)

    Returns:
        LUT array of shape (256,) mapping INT8 index to exp value
    """
    int8_values = np.arange(-128, 128, dtype=np.int32)
    x_fp32 = int8_values.astype(np.float32) * scale_in

    # Shift for numerical stability
    x_shifted = x_fp32 - (127 * scale_in)
    exp_values = np.exp(x_shifted)

    if precision == 32:
        max_exp = exp_values.max()
        int_scale = (2**30) / max_exp if max_exp > 0 else 1.0
        lut = (exp_values * int_scale).astype(np.int64)
    else:
        max_exp = exp_values.max()
        int_scale = (2**14) / max_exp if max_exp > 0 else 1.0
        lut = (exp_values * int_scale).astype(np.int32)

    return lut


def get_exp_lut(scale_in: float, precision: int = 32) -> np.ndarray:
    """Get or create cached LUT for given scale."""
    cache_key = (scale_in, precision)
    if cache_key not in _exp_lut_cache:
        _exp_lut_cache[cache_key] = build_exp_lut(scale_in, precision)
    return _exp_lut_cache[cache_key]


def softmax_int8_lut(
    x_int8: np.ndarray,
    scale_x: float,
    scale_y: float,
    axis: int = -1,
    zero_point_x: int = 0,
    zero_point_y: int = 0
) -> np.ndarray:
    """
    INT8 Softmax using Lookup Table for exact exp() computation.

    Args:
        x_int8: Input tensor (INT8) - typically attention scores
        scale_x: Input quantization scale
        scale_y: Output quantization scale (typically 1/127 for [0,1] range)
        axis: Dimension to compute softmax over (default: -1)
        zero_point_x: Input zero point (usually 0 for symmetric)
        zero_point_y: Output zero point (usually 0 for symmetric)

    Returns:
        Output tensor (INT8) representing softmax probabilities
    """
    x_adj = x_int8.astype(np.int32) - zero_point_x

    # Subtract max for numerical stability
    x_max = np.max(x_adj, axis=axis, keepdims=True)
    x_shifted = x_adj - x_max
    x_shifted = np.clip(x_shifted, -128, 127).astype(np.int8)

    # Get LUT
    lut = get_exp_lut(scale_x, precision=32)

    # Lookup exp values
    indices = (x_shifted.astype(np.int32) + 128).astype(np.int32)
    exp_values = lut[indices]

    # Sum along axis
    sum_exp = np.sum(exp_values, axis=axis, keepdims=True)
    sum_exp = np.maximum(sum_exp, 1)

    # Normalize and requantize
    inv_scale_y = 1.0 / scale_y
    numerator = exp_values.astype(np.float64) * inv_scale_y
    y_fp = numerator / sum_exp.astype(np.float64)

    # Quantize to INT8
    y_int32 = np.round(y_fp).astype(np.int32) + zero_point_y
    return np.clip(y_int32, -128, 127).astype(np.int8)


def softmax_int8_lut_pure_integer(
    x_int8: np.ndarray,
    scale_x: float,
    scale_y: float,
    axis: int = -1,
) -> np.ndarray:
    """
    Pure integer LUT softmax (no floating point in main loop).

    More hardware-friendly as it avoids FP division.
    """
    x_adj = x_int8.astype(np.int32)
    x_max = np.max(x_adj, axis=axis, keepdims=True)
    x_shifted = np.clip(x_adj - x_max, -128, 127).astype(np.int8)

    lut = get_exp_lut(scale_x, precision=32)
    indices = (x_shifted.astype(np.int32) + 128)
    exp_values = lut[indices].astype(np.int64)

    sum_exp = np.sum(exp_values, axis=axis, keepdims=True)
    sum_exp = np.maximum(sum_exp, 1)

    output_scale = int(round(1.0 / scale_y))
    numerator = exp_values * output_scale
    y_int64 = numerator // sum_exp

    return np.clip(y_int64, -128, 127).astype(np.int8)


# --- I-BERT Softmax (Polynomial Approximation) ---

def _ibert_exp_poly(q: np.ndarray, S: float):
    """I-BERT polynomial for exp approximation."""
    a, b, c = 0.3585, 1.353, 0.344

    q_b = np.round(b / S).astype(np.int32)
    S_out = a * (S ** 2)
    if S_out < 1e-12:
        S_out = 1e-12
    q_c = np.round(c / S_out).astype(np.int32)

    q_int32 = q.astype(np.int32)
    term = q_int32 + q_b
    poly_res = (term * term) + q_c

    return poly_res, S_out


def _ibert_exp(q: np.ndarray, S: float):
    """I-BERT integer exponential using polynomial and bit-shift."""
    ln2 = 0.69314718
    q_ln2 = np.round(ln2 / S).astype(np.int32)
    if q_ln2 == 0:
        q_ln2 = 1

    z = np.floor(-q / q_ln2).astype(np.int32)
    z = np.clip(z, 0, 30)

    q_p = q + (z * q_ln2)
    q_poly, S_poly = _ibert_exp_poly(q_p, S)
    q_exp = np.right_shift(q_poly, z)

    return q_exp, S_poly


def softmax_int8_ibert(
    x_int8: np.ndarray,
    scale_x: float,
    scale_y: float,
    axis: int = -1,
    zero_point_x: int = 0,
    zero_point_y: int = 0
) -> np.ndarray:
    """
    I-BERT Softmax using polynomial exp approximation.

    Args:
        x_int8: Input tensor (INT8)
        scale_x: Input quantization scale
        scale_y: Output quantization scale
        axis: Softmax axis
        zero_point_x, zero_point_y: Zero points (usually 0)

    Returns:
        INT8 softmax output
    """
    q = x_int8.astype(np.int32) - zero_point_x
    max_q = np.max(q, axis=axis, keepdims=True)
    q_shifted = q - max_q

    q_exp, S_exp = _ibert_exp(q_shifted, scale_x)
    sum_exp = np.sum(q_exp, axis=axis, keepdims=True)
    sum_exp = np.maximum(sum_exp, 1)

    inv_scale_y = 1.0 / scale_y
    numerator = q_exp.astype(np.float64) * inv_scale_y
    y_int32 = np.round(numerator / sum_exp).astype(np.int32)
    y_int32 += zero_point_y

    return np.clip(y_int32, -128, 127).astype(np.int8)


# --- Simple FP32-based Softmax ---

def softmax_int8(
    x_int8: np.ndarray,
    scale_x: float,
    scale_y: float,
    axis: int = -1,
    zero_point_x: int = 0,
    zero_point_y: int = 0
) -> np.ndarray:
    """
    Simple INT8 softmax using dequantize-compute-requantize.

    Args:
        x_int8: Input tensor (INT8)
        scale_x: Input quantization scale
        scale_y: Output quantization scale
        axis: Softmax axis
        zero_point_x, zero_point_y: Zero points

    Returns:
        INT8 softmax output
    """
    x_fp32 = (x_int8.astype(np.float32) - zero_point_x) * scale_x
    e_x = np.exp(x_fp32 - np.max(x_fp32, axis=axis, keepdims=True))
    y_fp32 = e_x / e_x.sum(axis=axis, keepdims=True)
    y_int32 = np.round(y_fp32 / scale_y).astype(np.int32) + zero_point_y
    return np.clip(y_int32, -128, 127).astype(np.int8)


def softmax_fp32_reference(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """FP32 reference softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


# --- Tests ---

def test_softmax_int8():
    """Unit tests for INT8 softmax implementations."""
    print("=" * 80)
    print("Testing INT8 Softmax")
    print("=" * 80)

    np.random.seed(42)

    scale_x = 0.0625
    scale_y = 1.0 / 127.0

    # Test 1: Accuracy comparison
    print("\nTest 1: Accuracy Comparison")
    print("-" * 60)

    # Simulate attention scores
    x_fp32 = np.random.randn(2, 5, 14, 14).astype(np.float32) * 2.0
    x_int8 = np.clip(np.round(x_fp32 / scale_x), -128, 127).astype(np.int8)

    # FP32 reference
    x_dequant = x_int8.astype(np.float32) * scale_x
    probs_ref = softmax_fp32_reference(x_dequant, axis=-1)
    y_ref = np.clip(np.round(probs_ref / scale_y), -128, 127).astype(np.int8)

    # Implementations
    y_lut = softmax_int8_lut(x_int8, scale_x, scale_y, axis=-1)
    y_ibert = softmax_int8_ibert(x_int8, scale_x, scale_y, axis=-1)
    y_simple = softmax_int8(x_int8, scale_x, scale_y, axis=-1)

    # Compare errors
    diff_lut = np.abs(y_lut.astype(np.int32) - y_ref.astype(np.int32))
    diff_ibert = np.abs(y_ibert.astype(np.int32) - y_ref.astype(np.int32))
    diff_simple = np.abs(y_simple.astype(np.int32) - y_ref.astype(np.int32))

    print(f"Scale: input={scale_x}, output={scale_y}")
    print(f"\nLUT vs Reference:")
    print(f"  Max error:  {np.max(diff_lut)} bits")
    print(f"  Mean error: {np.mean(diff_lut):.4f} bits")

    print(f"\nI-BERT vs Reference:")
    print(f"  Max error:  {np.max(diff_ibert)} bits")
    print(f"  Mean error: {np.mean(diff_ibert):.4f} bits")

    print(f"\nSimple vs Reference:")
    print(f"  Max error:  {np.max(diff_simple)} bits")
    print(f"  Mean error: {np.mean(diff_simple):.4f} bits")

    if np.mean(diff_lut) < np.mean(diff_ibert):
        improvement = np.mean(diff_ibert) / np.mean(diff_lut) if np.mean(diff_lut) > 0 else float('inf')
        print(f"\n→ LUT is {improvement:.2f}x more accurate than I-BERT")

    # Test 2: Probability sum check
    print("\n" + "-" * 60)
    print("Test 2: Probability Sum Consistency")
    print("-" * 60)

    expected_sum = int(1.0 / scale_y)
    lut_sums = y_lut.sum(axis=-1)
    ibert_sums = y_ibert.sum(axis=-1)

    lut_sum_error = np.abs(lut_sums - expected_sum)
    ibert_sum_error = np.abs(ibert_sums - expected_sum)

    print(f"Expected sum: {expected_sum}")
    print(f"LUT sum error:   max={np.max(lut_sum_error)}, mean={np.mean(lut_sum_error):.2f}")
    print(f"I-BERT sum error: max={np.max(ibert_sum_error)}, mean={np.mean(ibert_sum_error):.2f}")

    # Test 3: Small input (single row)
    print("\n" + "-" * 60)
    print("Test 3: Single Row Softmax")
    print("-" * 60)

    x_small = np.array([[10, 20, 30, 40]], dtype=np.int8)
    y_lut_small = softmax_int8_lut(x_small, scale_x, scale_y, axis=-1)
    y_ibert_small = softmax_int8_ibert(x_small, scale_x, scale_y, axis=-1)
    y_simple_small = softmax_int8(x_small, scale_x, scale_y, axis=-1)

    print(f"Input:   {x_small[0]}")
    print(f"LUT:     {y_lut_small[0]} (sum={y_lut_small.sum()})")
    print(f"I-BERT:  {y_ibert_small[0]} (sum={y_ibert_small.sum()})")
    print(f"Simple:  {y_simple_small[0]} (sum={y_simple_small.sum()})")

    # Test 4: Pure integer LUT
    print("\n" + "-" * 60)
    print("Test 4: Pure Integer LUT (Hardware-Friendly)")
    print("-" * 60)

    y_pure_int = softmax_int8_lut_pure_integer(x_int8, scale_x, scale_y, axis=-1)
    diff_pure = np.abs(y_pure_int.astype(np.int32) - y_ref.astype(np.int32))

    print(f"Pure integer LUT vs Reference:")
    print(f"  Max error:  {np.max(diff_pure)} bits")
    print(f"  Mean error: {np.mean(diff_pure):.4f} bits")

    print("=" * 80)


if __name__ == "__main__":
    test_softmax_int8()
