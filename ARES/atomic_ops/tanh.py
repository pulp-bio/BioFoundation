# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Atomic Tanh Operation - I-BERT Integer-only Implementation

Implements integer-only tanh approximation using I-BERT style polynomial.
Uses the "Sign Absorption" technique for positive scales (hardware compatibility).

Formula:
    tanh(x) ≈ sign(x) * [a * (clip(|x|, max=b) - b)² + c]

    where: a = -0.245975, b = 1.977486, c = 0.985300

For |x| > b (1.977486): tanh saturates to ±1

Reference:
    "I-BERT: Integer-only BERT Quantization" (Kim et al., 2021)
"""

import numpy as np

# Minimum scale for safe I-BERT polynomial computation
IBERT_MIN_SAFE_SCALE = 1e-3


def _ibert_tanh_poly(q: np.ndarray, S: float):
    """
    Integer-only tanh approximation using polynomial.

    Approximates tanh(x) using:
        tanh(x) ≈ sign(x) * [a * (clip(|x|, max=b) - b)² + c]

    Where a = -0.245975, b = 1.977486, c = 0.985300

    Special handling:
    - x = 0: Return 0 exactly (tanh(0) = 0)
    - |x| >= b: Return ±1 (saturation)
    """
    a = -0.245975
    b = 1.977486
    c = 0.985300

    is_zero = (q == 0)

    # Extract sign and absolute value
    sign = np.sign(q).astype(np.int32)
    sign[sign == 0] = 1
    q_abs = np.abs(q)

    # Compute clipping threshold in quantized domain
    q_b = int(np.round(b / S))
    q_b = max(q_b, 1)

    # Track saturation
    saturation_margin = max(2, int(0.1 / S))
    is_saturated = q_abs > (q_b + saturation_margin)

    # Clip to valid polynomial range
    q_clipped = np.clip(q_abs, 0, q_b).astype(np.int32)

    # Compute (|x| - b) in quantized domain
    q_shifted = q_clipped - q_b

    # Compute polynomial: a * (|x| - b)² + c
    S_out_raw = a * (S ** 2)

    if abs(S_out_raw) < 1e-15:
        S_out_raw = -1e-15

    q_c = int(np.round(c / S_out_raw))

    q_sq = q_shifted * q_shifted
    poly_res = q_sq + q_c

    # Sign absorption: since a < 0, S_out_raw < 0
    S_poly = -S_out_raw
    q_poly = -poly_res

    # Apply sign
    q_out = sign * q_poly

    # Handle special cases
    q_out = np.where(is_zero, 0, q_out)

    # For saturation: use exactly 1.0
    q_one = int(np.round(1.0 / S_poly))
    q_out = np.where(is_saturated & (sign > 0), q_one, q_out)
    q_out = np.where(is_saturated & (sign < 0), -q_one, q_out)

    return q_out, S_poly


def tanh_int8_ibert(
    x_int8: np.ndarray,
    scale_x: float,
    scale_y: float,
    zero_point_x: int = 0,
    zero_point_y: int = 0
) -> np.ndarray:
    """
    I-BERT Integer-only tanh with automatic scale fallback.

    Args:
        x_int8: Input tensor (INT8)
        scale_x: Input quantization scale
        scale_y: Output quantization scale (typically ~1/127 for [-1,1] range)
        zero_point_x: Input zero point (usually 0 for symmetric)
        zero_point_y: Output zero point (usually 0 for symmetric)

    Returns:
        Output tensor (INT8) representing tanh(x)

    Notes:
        - For scale_x >= 1e-3: Uses I-BERT polynomial
        - For scale_x < 1e-3: Uses linear approximation (tanh(x) ≈ x)
        - Output range is [-1, 1]
    """
    if scale_x < IBERT_MIN_SAFE_SCALE:
        # Linear approximation for tiny scales
        x_fp32 = (x_int8.astype(np.float32) - zero_point_x) * scale_x
        y_fp32 = np.clip(x_fp32, -1.0, 1.0)
        y_int32 = np.round(y_fp32 / scale_y).astype(np.int32) + zero_point_y
        return np.clip(y_int32, -128, 127).astype(np.int8)

    # Full I-BERT polynomial path
    q = x_int8.astype(np.int32) - zero_point_x
    q_tanh, S_tanh = _ibert_tanh_poly(q, scale_x)

    # Requantize to output scale
    requant_factor = S_tanh / scale_y
    y_int32 = np.round(q_tanh * requant_factor).astype(np.int32)
    y_int32 += zero_point_y

    return np.clip(y_int32, -128, 127).astype(np.int8)


def tanh_int8(
    x_int8: np.ndarray,
    scale_x: float,
    scale_y: float,
    zero_point_x: int = 0,
    zero_point_y: int = 0
) -> np.ndarray:
    """
    Simple INT8 tanh using dequantize-compute-requantize.

    Args:
        x_int8: Input tensor (INT8)
        scale_x: Input quantization scale
        scale_y: Output quantization scale
        zero_point_x: Input zero point
        zero_point_y: Output zero point

    Returns:
        Output tensor (INT8)
    """
    x_fp32 = (x_int8.astype(np.float32) - zero_point_x) * scale_x
    y_fp32 = np.tanh(x_fp32)
    y_int32 = np.round(y_fp32 / scale_y).astype(np.int32) + zero_point_y
    return np.clip(y_int32, -128, 127).astype(np.int8)


def tanh_fp32_reference(x: np.ndarray) -> np.ndarray:
    """FP32 tanh reference."""
    return np.tanh(x)


# --- Tests ---

def test_tanh_int8():
    """Unit tests for INT8 tanh implementations."""
    print("=" * 80)
    print("Testing INT8 Tanh")
    print("=" * 80)

    # Test 1: Normal scale (I-BERT polynomial path)
    print("\nTest 1: Normal Scale (I-BERT Polynomial)")
    scale_x = 4.0 / 127.0  # ~0.0315, range ±4.0
    scale_y = 1.0 / 127.0  # ~0.00787, range ±1.0

    print(f"Input Scale:  {scale_x:.6f}")
    print(f"Output Scale: {scale_y:.6f}")

    x_int8 = np.arange(-128, 128, dtype=np.int8)
    y_ibert = tanh_int8_ibert(x_int8, scale_x, scale_y)
    y_simple = tanh_int8(x_int8, scale_x, scale_y)

    # FP32 reference
    x_fp32 = x_int8.astype(np.float32) * scale_x
    y_fp32_ref = tanh_fp32_reference(x_fp32)
    y_int8_expected = np.clip(np.round(y_fp32_ref / scale_y), -128, 127).astype(np.int8)

    diff_ibert = np.abs(y_ibert.astype(np.int32) - y_int8_expected.astype(np.int32))
    diff_simple = np.abs(y_simple.astype(np.int32) - y_int8_expected.astype(np.int32))

    print(f"I-BERT Max Error:  {np.max(diff_ibert)} bits")
    print(f"I-BERT Mean Error: {np.mean(diff_ibert):.4f} bits")
    print(f"Simple Max Error:  {np.max(diff_simple)} bits")

    if np.max(diff_ibert) <= 3:
        print("[OK] Normal scale test PASSED")
    else:
        print(f"[WARN] Normal scale test: I-BERT error = {np.max(diff_ibert)} bits")

    # Test 2: Tiny scale (linear fallback)
    print("\nTest 2: Tiny Scale (Linear Fallback)")
    scale_x_tiny = 3e-5
    scale_y_tiny = 1.0 / 127.0

    print(f"Input Scale: {scale_x_tiny:.2e}")

    try:
        y_ibert_tiny = tanh_int8_ibert(x_int8, scale_x_tiny, scale_y_tiny)
        print("[OK] No overflow!")
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return

    x_fp32_tiny = x_int8.astype(np.float32) * scale_x_tiny
    y_fp32_ref_tiny = tanh_fp32_reference(x_fp32_tiny)
    y_int8_expected_tiny = np.clip(np.round(y_fp32_ref_tiny / scale_y_tiny), -128, 127).astype(np.int8)

    diff_tiny = np.abs(y_ibert_tiny.astype(np.int32) - y_int8_expected_tiny.astype(np.int32))
    print(f"Max Error: {np.max(diff_tiny)} bits")

    if np.max(diff_tiny) <= 1:
        print("[OK] Tiny scale test PASSED")
    else:
        print(f"[WARN] Tiny scale test: Error = {np.max(diff_tiny)} bits")

    # Test 3: Key values spot check
    print("\nTest 3: Key Values Spot Check")
    test_inputs = np.array([-120, -64, -32, 0, 32, 64, 120], dtype=np.int8)
    y_test = tanh_int8_ibert(test_inputs, scale_x, scale_y)
    y_expected = tanh_int8(test_inputs, scale_x, scale_y)

    print(f"Input:    {test_inputs}")
    print(f"I-BERT:   {y_test}")
    print(f"Simple:   {y_expected}")

    # Test 4: Saturation behavior
    print("\nTest 4: Saturation Check")
    large_pos = y_ibert[-1]
    large_neg = y_ibert[0]

    print(f"tanh(+4.0) in INT8: {large_pos} (expected: ~127)")
    print(f"tanh(-4.0) in INT8: {large_neg} (expected: ~-127)")

    if large_pos > 120 and large_neg < -120:
        print("[OK] Saturation behavior correct")
    else:
        print(f"[WARN] Saturation values: pos={large_pos}, neg={large_neg}")

    # Test 5: Odd function property
    print("\nTest 5: Odd Function Property tanh(-x) = -tanh(x)")
    symmetry_errors = []
    for i in range(1, 128):
        pos_idx = 128 + i - 1
        neg_idx = 128 - i
        if pos_idx < 256 and neg_idx >= 0:
            err = abs(int(y_ibert[neg_idx]) + int(y_ibert[pos_idx]))
            symmetry_errors.append(err)

    max_sym_err = max(symmetry_errors) if symmetry_errors else 0
    print(f"Max symmetry error: {max_sym_err}")

    if max_sym_err <= 1:
        print("[OK] Odd function property preserved")
    else:
        print(f"[WARN] Symmetry error: {max_sym_err}")

    print("=" * 80)


if __name__ == "__main__":
    test_tanh_int8()
