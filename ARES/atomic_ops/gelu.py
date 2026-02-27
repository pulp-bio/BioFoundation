# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Atomic GELU Operation - Multiple INT8 Implementations

Provides several GELU implementations for different use cases:
1. gelu_int8: Simple dequantize-compute-requantize
2. gelu_int8_lut: Lookup table for bit-exact C matching
3. gelu_int8_ibert: I-BERT polynomial approximation (no LUT memory needed)

The I-BERT implementation follows Algorithm 2 from:
"I-BERT: Integer-only BERT Quantization" (Kim et al., 2021)
"""

import numpy as np
import math

try:
    from .constants import INT16_MAX_Q15
except ImportError:
    from constants import INT16_MAX_Q15

# ---
# I-BERT GELU: Integer-only Polynomial Approximation
# ---
# Implements GELU using second-order polynomial approximation of erf(x).
# No lookup table needed - useful when memory is constrained.
#
# GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
#
# The erf is approximated with: erf(x) ≈ sign(x) * [a(|x|+b)² + c]
# where a=-0.2888, b=-1.769, c=1.0
# ---

# Minimum scale for safe I-BERT polynomial computation
# Below this threshold, use linear approximation to avoid INT32 overflow
IBERT_MIN_SAFE_SCALE = 1e-3


def _ibert_poly(q: np.ndarray, S: float, a: float, b: float, c: float):
    """
    I-BERT Algorithm 1: Integer-only second-order polynomial a(x+b)² + c.

    Includes sign absorption for positive output scales (hardware compatibility).
    """
    S_out_raw = a * (S ** 2)
    q_b = np.floor(b / S).astype(np.int32)

    if abs(S_out_raw) < 1e-15:
        S_out_raw = np.sign(S_out_raw) * 1e-15 if S_out_raw != 0 else 1e-15

    q_c = np.floor(c / S_out_raw).astype(np.int32)
    q_int32 = q.astype(np.int32)
    term = q_int32 + q_b
    poly_res = (term * term) + q_c

    # Sign absorption: invert both if scale is negative
    if S_out_raw < 0:
        return -poly_res, -S_out_raw
    return poly_res, S_out_raw


def _ibert_erf(q: np.ndarray, S: float):
    """
    I-BERT integer-only error function approximation.
    Uses polynomial: erf(x) ≈ sign(x) * [a(|x|+b)² + c]
    """
    a, b, c = -0.2888, -1.769, 1.0

    sign = np.sign(q).astype(np.int32)
    sign[sign == 0] = 1
    q_abs = np.abs(q)

    # Clip to polynomial validity range
    clip_max = np.floor(-b / S).astype(np.int32)
    q_clipped = np.clip(q_abs, 0, clip_max).astype(np.int32)

    q_poly, S_poly = _ibert_poly(q_clipped, S, a, b, c)
    q_out = sign * q_poly

    return q_out, S_poly


def gelu_int8_ibert(
    x_int8: np.ndarray,
    scale_x: float,
    scale_y: float,
    zero_point_x: int = 0,
    zero_point_y: int = 0
) -> np.ndarray:
    """
    I-BERT Integer-only GELU (Algorithm 2 from I-BERT paper).

    Uses polynomial approximation of erf - no lookup table needed.
    Automatically falls back to linear approximation for tiny scales.

    Args:
        x_int8: Input tensor (INT8)
        scale_x: Input quantization scale
        scale_y: Output quantization scale
        zero_point_x: Input zero point (usually 0 for symmetric)
        zero_point_y: Output zero point (usually 0 for symmetric)

    Returns:
        Output tensor (INT8)

    Notes:
        - For scale_x >= 1e-3: Uses I-BERT polynomial
        - For scale_x < 1e-3: Uses linear approximation GELU(x) ≈ 0.5*x
    """
    if scale_x < IBERT_MIN_SAFE_SCALE:
        # Linear approximation for tiny scales
        x_fp32 = (x_int8.astype(np.float32) - zero_point_x) * scale_x
        y_fp32 = 0.5 * x_fp32
        y_int32 = np.round(y_fp32 / scale_y).astype(np.int32) + zero_point_y
        return np.clip(y_int32, -128, 127).astype(np.int8)

    # Full I-BERT polynomial path
    q = x_int8.astype(np.int32) - zero_point_x

    # I-ERF(x / sqrt(2))
    S_erf_in = scale_x / 1.41421356
    q_erf, S_erf = _ibert_erf(q, S_erf_in)

    if S_erf == 0:
        S_erf = 1e-9
    q_1 = np.floor(1.0 / S_erf).astype(np.int32)

    # GELU = x * (erf + 1) / 2
    q_gelu_unscaled = q * (q_erf + q_1)
    S_gelu = (scale_x * S_erf) / 2.0

    # Requantize to output scale
    requant_factor = S_gelu / scale_y
    y_int32 = np.round(q_gelu_unscaled * requant_factor).astype(np.int32)
    y_int32 += zero_point_y

    return np.clip(y_int32, -128, 127).astype(np.int8)


# ---
# i-GELU: Integer-only GELU with Lookup Table for Bit-Exact Matching
# ---
# Similar to i-Softmax, we use a LUT to achieve bit-exact reproducibility
# between Python and C implementations.
#
# LUT Parameters:
# - Input range: [-4.0, 4.0] (covers ~99.99% of GELU's active range)
# - NUM_ENTRIES: 256 (8-bit index, sufficient for INT8 precision)
# - Output scale: 32767 (INT16 for precision)
#
# GELU(x) = x * Φ(x) where Φ(x) is the CDF of standard normal
# Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
# ---

I_GELU_INPUT_MIN = -4.0
I_GELU_INPUT_MAX = 4.0
I_GELU_NUM_ENTRIES = 256
I_GELU_OUTPUT_SCALE = INT16_MAX_Q15  # INT16 scale for output LUT values


def get_builtin_gelu_lut() -> tuple:
    """
    Generate the builtin GELU lookup table for bit-exact matching with C code.

    The LUT stores GELU(x) / x (the "gate" value) scaled to INT16.
    For x in [-4.0, 4.0], GELU(x)/x ranges from ~0 to ~1.

    Returns:
        tuple: (lut, metadata) where:
            - lut: INT16 numpy array of LUT values
            - metadata: dict with LUT parameters
    """
    x = np.linspace(I_GELU_INPUT_MIN, I_GELU_INPUT_MAX, I_GELU_NUM_ENTRIES)

    # Compute GELU gate: cdf = 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x ** 3)
    gate = 0.5 * (1.0 + np.tanh(tanh_arg))

    # Scale to INT16 and store
    lut = np.round(gate * I_GELU_OUTPUT_SCALE).astype(np.int16)

    metadata = {
        'input_min': I_GELU_INPUT_MIN,
        'input_max': I_GELU_INPUT_MAX,
        'input_step': (I_GELU_INPUT_MAX - I_GELU_INPUT_MIN) / I_GELU_NUM_ENTRIES,
        'output_scale': float(I_GELU_OUTPUT_SCALE),
        'num_entries': I_GELU_NUM_ENTRIES,
    }

    return lut, metadata


def gelu_int8_lut(input_int8, scale_input, scale_output, gelu_lut=None, lut_metadata=None):
    """
    INT8 GELU using lookup table for bit-exact matching with C implementation.

    This version uses a precomputed LUT to avoid FP32 tanh variations between
    Python and C, achieving bit-exact results.

    Args:
        input_int8: INT8 input tensor
        scale_input: Input quantization scale
        scale_output: Output quantization scale
        gelu_lut: Optional INT16 LUT (uses builtin if None)
        lut_metadata: Optional metadata dict (uses builtin if None)

    Returns:
        output_int8: INT8 output tensor, same shape as input
    """
    if gelu_lut is None or lut_metadata is None:
        gelu_lut, lut_metadata = get_builtin_gelu_lut()

    input_min = lut_metadata['input_min']
    input_step = lut_metadata['input_step']
    output_scale = lut_metadata['output_scale']
    num_entries = lut_metadata['num_entries']

    # Flatten for processing
    original_shape = input_int8.shape
    input_flat = input_int8.flatten()
    output_flat = np.zeros_like(input_flat, dtype=np.int8)

    for i in range(len(input_flat)):
        # Dequantize to FP32
        x = float(input_flat[i]) * scale_input

        # Compute LUT index (linear interpolation between min and max)
        idx = int(round((x - input_min) / input_step))
        if idx < 0:
            idx = 0
        if idx >= num_entries:
            idx = num_entries - 1

        # Get gate value from LUT
        gate_int16 = int(gelu_lut[idx])

        # GELU(x) = x * gate
        # Using integer arithmetic: (x_scaled * gate_int16) / output_scale
        # First compute x in a scaled integer form
        # x_int32 = x / scale_output * 32767 (to match LUT scale)
        # result_int32 = (x_int32 * gate_int16) / 32767

        # Simpler: compute in FP32 but use LUT gate
        gate_fp32 = float(gate_int16) / output_scale
        result_fp32 = x * gate_fp32

        # Quantize to INT8
        q = int(round(result_fp32 / scale_output))
        if q > 127:
            q = 127
        if q < -128:
            q = -128
        output_flat[i] = np.int8(q)

    return output_flat.reshape(original_shape)


def gelu_int8(input_int8, scale_input, scale_output):
    """
    INT8 GELU (Gaussian Error Linear Unit) activation.

    GELU is defined as: GELU(x) = x * Φ(x)
    where Φ(x) is the cumulative distribution function of standard normal.

    Common approximation:
    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

    Alternative (faster) approximation:
    GELU(x) ≈ x * sigmoid(1.702 * x)

    Args:
        input_int8: INT8 input tensor
        scale_input: Input quantization scale
        scale_output: Output quantization scale

    Returns:
        output_int8: INT8 output tensor, same shape as input

    Implementation:
        1. Dequantize input to FP32
        2. Apply GELU activation
        3. Quantize back to INT8
    """
    # Step 1: Dequantize INT8 input to FP32
    input_fp32 = input_int8.astype(np.float32) * scale_input

    # Step 2: Apply GELU
    # Using the tanh approximation for better accuracy
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    cube_term = 0.044715 * input_fp32 ** 3
    tanh_arg = sqrt_2_over_pi * (input_fp32 + cube_term)
    output_fp32 = 0.5 * input_fp32 * (1.0 + np.tanh(tanh_arg))

    # Step 3: Quantize to INT8
    output_int8 = np.clip(np.round(output_fp32 / scale_output), -128, 127).astype(np.int8)

    return output_int8


def gelu_fp32_reference(x):
    """Reference FP32 GELU for testing."""
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    cube_term = 0.044715 * x ** 3
    tanh_arg = sqrt_2_over_pi * (x + cube_term)
    return 0.5 * x * (1.0 + np.tanh(tanh_arg))


gelu_fp32 = gelu_fp32_reference


def test_gelu_int8():
    """Unit test for INT8 GELU"""
    print("="*80)
    print("Testing INT8 GELU")
    print("="*80)

    # Test case 1: Simple 1D array
    print("\nTest 1: 1D GELU (8 elements)")

    # Create FP32 input with range around 0
    input_fp32 = np.array([-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)

    # Quantize to INT8
    scale_input = 0.05  # Scale to cover range [-6.4, 6.35]
    input_int8 = np.clip(np.round(input_fp32 / scale_input), -128, 127).astype(np.int8)

    # Output scale (similar to input)
    scale_output = 0.05

    # Apply INT8 GELU
    output_int8 = gelu_int8(input_int8, scale_input, scale_output)

    # Dequantize for comparison
    output_fp32 = output_int8.astype(np.float32) * scale_output

    # Compute reference FP32 GELU
    reference_fp32 = gelu_fp32_reference(input_fp32)

    # Compare
    print(f"Input (FP32):     {input_fp32}")
    print(f"Input (INT8):     {input_int8}")
    print(f"Output (INT8):    {output_int8}")
    print(f"Output (FP32):    {output_fp32}")
    print(f"Reference (FP32): {reference_fp32}")
    print(f"Max diff:         {np.max(np.abs(output_fp32 - reference_fp32)):.6f}")

    # Test case 2: 2D array
    print("\nTest 2: 2D GELU (4x8 matrix)")
    np.random.seed(42)
    input_fp32 = np.random.randn(4, 8).astype(np.float32) * 2.0  # Range approx [-6, 6]

    scale_input = 0.05
    input_int8 = np.clip(np.round(input_fp32 / scale_input), -128, 127).astype(np.int8)

    # Apply INT8 GELU
    output_int8 = gelu_int8(input_int8, scale_input, scale_output)

    # Dequantize
    output_fp32 = output_int8.astype(np.float32) * scale_output

    # Reference FP32 GELU
    reference_fp32 = gelu_fp32_reference(input_fp32)

    print(f"Input shape:      {input_fp32.shape}")
    print(f"Output shape:     {output_fp32.shape}")
    print(f"Input range:      [{np.min(input_fp32):.2f}, {np.max(input_fp32):.2f}]")
    print(f"Output range:     [{np.min(output_fp32):.2f}, {np.max(output_fp32):.2f}]")
    print(f"Reference range:  [{np.min(reference_fp32):.2f}, {np.max(reference_fp32):.2f}]")
    print(f"Max diff:         {np.max(np.abs(output_fp32 - reference_fp32)):.6f}")
    print(f"Mean diff:        {np.mean(np.abs(output_fp32 - reference_fp32)):.6f}")

    # Test case 3: Edge cases
    print("\nTest 3: Edge cases")
    edge_cases_fp32 = np.array([-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0], dtype=np.float32)
    edge_cases_int8 = np.clip(np.round(edge_cases_fp32 / scale_input), -128, 127).astype(np.int8)

    output_int8 = gelu_int8(edge_cases_int8, scale_input, scale_output)
    output_fp32 = output_int8.astype(np.float32) * scale_output
    reference_fp32 = gelu_fp32_reference(edge_cases_fp32)

    for i in range(len(edge_cases_fp32)):
        print(f"  x={edge_cases_fp32[i]:6.2f}: GELU={output_fp32[i]:7.3f}, "
              f"ref={reference_fp32[i]:7.3f}, diff={abs(output_fp32[i] - reference_fp32[i]):.4f}")

    # Check if errors are within quantization tolerance
    max_diff = np.max(np.abs(output_fp32 - reference_fp32))
    tolerance = max(scale_input, scale_output) * 2  # 2x quantization step

    if max_diff <= tolerance:
        print(f"\n[OK] All tests passed! (max diff {max_diff:.6f} <= tolerance {tolerance:.6f})")
    else:
        print(f"\n[FAIL] Test failed! (max diff {max_diff:.6f} > tolerance {tolerance:.6f})")

    # Test case 4: Verify GELU properties
    print("\nTest 4: GELU properties")
    x = np.linspace(-4, 4, 100).astype(np.float32)
    y = gelu_fp32_reference(x)
    print(f"  GELU(0) ≈ 0: {abs(gelu_fp32_reference(np.array([0.0]))[0]) < 0.01}")
    print(f"  GELU is monotonic increasing: {np.all(np.diff(y) >= 0)}")
    print(f"  For large positive x, GELU(x) ≈ x: GELU(4)={y[-1]:.3f}, 4.0 (diff={abs(y[-1]-4.0):.3f})")
    print(f"  For large negative x, GELU(x) ≈ 0: GELU(-4)={y[0]:.3f}, 0.0 (diff={abs(y[0]):.3f})")

    print("="*80)


def test_gelu_int8_lut():
    """Unit test for LUT-based INT8 GELU"""
    print("="*80)
    print("Testing LUT-based INT8 GELU (i-GELU)")
    print("="*80)

    # Generate and display LUT info
    lut, metadata = get_builtin_gelu_lut()
    print(f"\nLUT Parameters:")
    print(f"  Input range: [{metadata['input_min']}, {metadata['input_max']}]")
    print(f"  Step size: {metadata['input_step']:.6f}")
    print(f"  Output scale: {metadata['output_scale']}")
    print(f"  Num entries: {metadata['num_entries']}")
    print(f"  LUT dtype: {lut.dtype}")
    print(f"  LUT range: [{lut.min()}, {lut.max()}]")
    print(f"  First 10 entries: {lut[:10]}")
    print(f"  Last 10 entries: {lut[-10:]}")

    # Test case 1: Simple 1D array
    print("\nTest 1: 1D i-GELU (8 elements)")
    input_fp32 = np.array([-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    scale_input = 0.05
    input_int8 = np.clip(np.round(input_fp32 / scale_input), -128, 127).astype(np.int8)
    scale_output = 0.05

    # Apply LUT-based GELU
    output_lut = gelu_int8_lut(input_int8, scale_input, scale_output)

    # Apply original GELU for comparison
    output_orig = gelu_int8(input_int8, scale_input, scale_output)

    # Dequantize
    output_lut_fp32 = output_lut.astype(np.float32) * scale_output
    output_orig_fp32 = output_orig.astype(np.float32) * scale_output
    reference_fp32 = gelu_fp32_reference(input_fp32)

    print(f"Input (FP32):     {input_fp32}")
    print(f"Input (INT8):     {input_int8}")
    print(f"Output LUT (INT8): {output_lut}")
    print(f"Output orig (INT8): {output_orig}")
    print(f"LUT vs orig INT8 diff: {np.max(np.abs(output_lut.astype(np.int32) - output_orig.astype(np.int32)))}")
    print(f"LUT FP32 vs ref max diff: {np.max(np.abs(output_lut_fp32 - reference_fp32)):.6f}")

    # Test case 2: Verify LUT is bit-exact across calls
    print("\nTest 2: Verify bit-exact reproducibility")
    np.random.seed(42)
    input_fp32 = np.random.randn(100).astype(np.float32) * 2.0
    input_int8 = np.clip(np.round(input_fp32 / scale_input), -128, 127).astype(np.int8)

    output1 = gelu_int8_lut(input_int8, scale_input, scale_output)
    output2 = gelu_int8_lut(input_int8, scale_input, scale_output)

    if np.array_equal(output1, output2):
        print("  [OK] LUT-based GELU is bit-exact reproducible")
    else:
        print("  [FAIL] LUT-based GELU is NOT reproducible!")

    print("="*80)


def test_gelu_int8_ibert():
    """Unit test for I-BERT polynomial GELU"""
    print("=" * 80)
    print("Testing I-BERT Polynomial GELU")
    print("=" * 80)

    # Test 1: Normal scale (I-BERT polynomial path)
    print("\nTest 1: Normal Scale (I-BERT Polynomial)")
    scale_x = 4.0 / 127.0  # ~0.0315
    scale_y = scale_x

    print(f"Input Scale: {scale_x:.6f}")

    x_int8 = np.arange(-128, 128, dtype=np.int8)
    y_int8 = gelu_int8_ibert(x_int8, scale_x, scale_y)

    # FP32 reference
    x_fp32 = x_int8.astype(np.float32) * scale_x
    y_fp32_ref = gelu_fp32_reference(x_fp32)
    y_int8_expected = np.clip(np.round(y_fp32_ref / scale_y), -128, 127).astype(np.int8)

    diff = np.abs(y_int8.astype(np.int32) - y_int8_expected.astype(np.int32))
    print(f"Max Error: {np.max(diff)} bits")
    print(f"Mean Error: {np.mean(diff):.4f} bits")

    if np.max(diff) <= 3:
        print("[OK] Normal scale test PASSED")
    else:
        print(f"[FAIL] Normal scale test: Error = {np.max(diff)} bits")

    # Test 2: Tiny scale (linear fallback)
    print("\nTest 2: Tiny Scale (Linear Fallback)")
    scale_x_tiny = 3.05e-5
    scale_y_tiny = 0.015625

    print(f"Input Scale: {scale_x_tiny:.2e}")

    x_int8_tiny = np.arange(-128, 128, dtype=np.int8)
    try:
        y_int8_tiny = gelu_int8_ibert(x_int8_tiny, scale_x_tiny, scale_y_tiny)
        print("[OK] No overflow!")
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return

    x_fp32_tiny = x_int8_tiny.astype(np.float32) * scale_x_tiny
    y_fp32_ref_tiny = gelu_fp32_reference(x_fp32_tiny)
    y_int8_expected_tiny = np.clip(np.round(y_fp32_ref_tiny / scale_y_tiny), -128, 127).astype(np.int8)

    diff_tiny = np.abs(y_int8_tiny.astype(np.int32) - y_int8_expected_tiny.astype(np.int32))
    print(f"Max Error: {np.max(diff_tiny)} bits")
    print(f"Mean Error: {np.mean(diff_tiny):.4f} bits")

    if np.max(diff_tiny) <= 2:
        print("[OK] Tiny scale test PASSED")
    else:
        print(f"[WARN] Tiny scale test: Error = {np.max(diff_tiny)} bits")

    # Test 3: Compare I-BERT vs LUT
    print("\nTest 3: I-BERT vs LUT Comparison")
    scale = 0.05
    x_int8 = np.array([-128, -64, -32, 0, 32, 64, 127], dtype=np.int8)

    y_ibert = gelu_int8_ibert(x_int8, scale, scale)
    y_lut = gelu_int8_lut(x_int8, scale, scale)
    y_simple = gelu_int8(x_int8, scale, scale)

    print(f"Input:    {x_int8}")
    print(f"I-BERT:   {y_ibert}")
    print(f"LUT:      {y_lut}")
    print(f"Simple:   {y_simple}")
    print(f"I-BERT vs LUT max diff: {np.max(np.abs(y_ibert.astype(np.int32) - y_lut.astype(np.int32)))}")

    print("=" * 80)


if __name__ == "__main__":
    test_gelu_int8()
    test_gelu_int8_lut()
    test_gelu_int8_ibert()
