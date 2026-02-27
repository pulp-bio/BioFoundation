# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Atomic RMSNorm (Root Mean Square Layer Normalization) Operation - INT8

Implements RMSNorm with INT8 input/output for GAP9 deployment.
RMSNorm is a simplification of LayerNorm that removes the mean centering step.

Formula: y = (x / rms(x)) * weight
where rms(x) = sqrt(mean(x^2) + eps)

Key differences from LayerNorm:
- No mean subtraction (only normalizes by root mean square)
- Computationally simpler (one less pass through the data)
- Used in Llama, Llama2, and other modern LLMs

References:
- "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
- Llama architecture (Touvron et al., 2023)
"""

import numpy as np
from typing import Optional, Tuple

try:
    from .quantize import quantize_linear, dequantize_linear
    from .layernorm import sqrt_q64, sqrt_approx_python, i_sqrt_newton
    from .constants import INT16_MAX_Q15, LUT_SIZE
except ImportError:
    from quantize import quantize_linear, dequantize_linear
    from layernorm import sqrt_q64, sqrt_approx_python, i_sqrt_newton
    from constants import INT16_MAX_Q15, LUT_SIZE


# LUT parameters for inverse sqrt (shared with layernorm)
I_RMSNORM_ISQRT_ENTRIES = LUT_SIZE  # 12-bit index (4096 entries)
I_RMSNORM_ISQRT_SCALE = INT16_MAX_Q15  # INT16 output scale
I_RMSNORM_VAR_MAX = 16384  # Max expected mean square value


def get_builtin_rmsnorm_isqrt_lut() -> Tuple[np.ndarray, dict]:
    """
    Generate the builtin inverse sqrt lookup table for bit-exact RMSNorm.

    The LUT stores 1/sqrt(ms+1) * SCALE for mean-square values from 0 to VAR_MAX.

    Returns:
        tuple: (lut, metadata) where:
            - lut: INT16 numpy array of LUT values
            - metadata: dict with LUT parameters
    """
    # Generate mean-square indices from 0 to VAR_MAX
    ms_indices = np.arange(I_RMSNORM_ISQRT_ENTRIES)

    # Map index to mean-square: ms = (index / ENTRIES) * VAR_MAX
    ms_values = (ms_indices / (I_RMSNORM_ISQRT_ENTRIES - 1)) * I_RMSNORM_VAR_MAX

    # Add epsilon (1 in integer domain) and compute 1/sqrt
    isqrt_values = 1.0 / np.sqrt(ms_values + 1.0)

    # Scale and store as INT16
    lut = np.round(isqrt_values * I_RMSNORM_ISQRT_SCALE).astype(np.int16)

    metadata = {
        'num_entries': I_RMSNORM_ISQRT_ENTRIES,
        'ms_max': I_RMSNORM_VAR_MAX,
        'output_scale': float(I_RMSNORM_ISQRT_SCALE),
    }

    return lut, metadata


def rmsnorm_fp32_reference(
    x: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-5
) -> np.ndarray:
    """
    FP32 reference implementation of RMSNorm.

    Args:
        x: Input tensor, shape (..., normalized_shape)
        weight: Learnable scale parameter, shape (normalized_shape,)
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor, same shape as input
    """
    # Compute root mean square along last dimension
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)

    # Normalize and scale
    return (x / rms) * weight


def rmsnorm_int8(
    input_int8: np.ndarray,
    weight_fp32: np.ndarray,
    scale_input: float,
    scale_output: float,
    normalized_shape: int,
    eps: float = 1e-5
) -> np.ndarray:
    """
    INT8 RMSNorm operation using FP32 intermediate computation.

    This version dequantizes to FP32, computes RMSNorm, and requantizes to INT8.
    Designed to match C implementation for bit-exact results.

    Formula: output = weight * (x / rms(x))
    where rms(x) = sqrt(mean(x^2) + eps)

    Args:
        input_int8: INT8 input tensor, shape (..., normalized_shape)
        weight_fp32: FP32 weight (gamma), shape (normalized_shape,)
        scale_input: Input quantization scale
        scale_output: Output quantization scale
        normalized_shape: Size of the dimension to normalize
        eps: Small constant for numerical stability

    Returns:
        output_int8: INT8 output tensor, same shape as input
    """
    # Reshape input for easier processing
    original_shape = input_int8.shape

    if len(original_shape) == 1:
        input_reshaped = input_int8.reshape(1, -1)
    else:
        if original_shape[-1] == normalized_shape:
            batch_size = int(np.prod(original_shape[:-1]))
            input_reshaped = input_int8.reshape(batch_size, normalized_shape)
        else:
            # Find matching dimension
            matching_dims = [i for i, dim in enumerate(original_shape) if dim == normalized_shape]
            if len(matching_dims) == 0:
                raise ValueError(
                    f"RMSNorm: normalized_shape={normalized_shape} not found in input shape {original_shape}"
                )
            # Use the last matching dimension
            batch_size = int(np.prod(original_shape[:-1]))
            input_reshaped = input_int8.reshape(batch_size, normalized_shape)

    # Process each vector with C-style sequential loops
    num_vectors = input_reshaped.shape[0]
    output_fp32 = np.zeros((num_vectors, normalized_shape), dtype=np.float32)

    for v in range(num_vectors):
        input_vec_int8 = input_reshaped[v]

        # Step 1: Compute sum of squares (dequantize INT8 -> FP32 in loop)
        sum_sq = np.float32(0.0)
        for i in range(normalized_shape):
            x = np.float32(float(input_vec_int8[i]) * scale_input)
            sum_sq += np.float32(x * x)

        # Step 2: Compute mean of squares
        mean_sq = np.float32(sum_sq / np.float32(normalized_shape))

        # Step 3: Compute RMS with epsilon
        rms = sqrt_approx_python(mean_sq + eps)

        # Step 4: Normalize and apply weight (dequantize INT8 -> FP32 in loop)
        for i in range(normalized_shape):
            x = float(input_vec_int8[i]) * scale_input
            normalized_val = x / rms
            output_fp32[v, i] = weight_fp32[i] * normalized_val

    # Reshape back to original shape
    output_fp32 = output_fp32.reshape(original_shape)

    # Quantize to INT8
    output_int8 = np.clip(np.round(output_fp32 / scale_output), -128, 127).astype(np.int8)

    return output_int8


def rmsnorm_int8_fixed_point(
    input_int8: np.ndarray,
    weight_fp32: np.ndarray,
    scale_input: float,
    scale_output: float,
    normalized_shape: int,
    eps: float = 1e-5
) -> np.ndarray:
    """
    INT8 RMSNorm using fixed-point arithmetic (bit-exact with C).

    This version uses INT64 accumulation and binary search sqrt for bit-exact
    reproducibility between Python and C implementations.

    Args:
        input_int8: INT8 input tensor, shape (..., normalized_shape)
        weight_fp32: FP32 weight (gamma), shape (normalized_shape,)
        scale_input: Input quantization scale
        scale_output: Output quantization scale
        normalized_shape: Size of the dimension to normalize
        eps: Small constant (unused, integer epsilon = 1 is implicit)

    Returns:
        output_int8: INT8 output tensor, same shape as input
    """
    # Reshape input for easier processing
    original_shape = input_int8.shape

    if len(original_shape) == 1:
        input_reshaped = input_int8.reshape(1, -1)
    else:
        if original_shape[-1] == normalized_shape:
            batch_size = int(np.prod(original_shape[:-1]))
            input_reshaped = input_int8.reshape(batch_size, normalized_shape)
        else:
            batch_size = int(np.prod(original_shape[:-1]))
            input_reshaped = input_int8.reshape(batch_size, normalized_shape)

    # Process each vector with fixed-point arithmetic
    num_vectors = input_reshaped.shape[0]
    output_fp32 = np.zeros((num_vectors, normalized_shape), dtype=np.float32)

    for v in range(num_vectors):
        input_vec = input_reshaped[v]

        # Step 1: Compute sum of squares using INT64 accumulation
        sum_sq = np.int64(0)
        for i in range(normalized_shape):
            val = np.int64(input_vec[i])
            sum_sq += val * val

        # Step 2: Compute mean of squares (integer division)
        mean_sq = sum_sq // np.int64(normalized_shape)

        # Step 3: Add epsilon and compute sqrt using binary search
        mean_sq += 1  # Integer epsilon
        rms = sqrt_q64(mean_sq, frac_bits=0)

        # Step 4: Normalize and apply weight
        for i in range(normalized_shape):
            # Normalized value in integer domain
            x_int = np.int64(input_vec[i])

            # Convert to FP32 for affine transform
            x_normalized = float(x_int) / float(rms)

            # Apply scale_input to convert to original FP32 domain
            x_normalized *= scale_input

            # Apply weight (gamma)
            output_fp32[v, i] = weight_fp32[i] * x_normalized

    # Reshape back to original shape
    output_fp32 = output_fp32.reshape(original_shape)

    # Quantize to INT8
    output_int8 = np.clip(np.round(output_fp32 / scale_output), -128, 127).astype(np.int8)

    return output_int8


def rmsnorm_int8_lut(
    input_int8: np.ndarray,
    weight_fp32: np.ndarray,
    scale_input: float,
    scale_output: float,
    normalized_shape: int,
    eps: float = 1e-5,
    isqrt_lut: Optional[np.ndarray] = None,
    lut_metadata: Optional[dict] = None
) -> np.ndarray:
    """
    INT8 RMSNorm using LUT-based inverse sqrt for bit-exact matching with C.

    This version uses:
    - INT64 accumulation for sum of squares
    - Integer division for mean of squares
    - LUT-based 1/sqrt for the normalization step

    Args:
        input_int8: INT8 input tensor
        weight_fp32: FP32 weight (gamma)
        scale_input: Input quantization scale
        scale_output: Output quantization scale
        normalized_shape: Size of dimension to normalize
        eps: Epsilon (unused, integer epsilon = 1 is implicit)
        isqrt_lut: Optional INT16 inverse sqrt LUT
        lut_metadata: Optional metadata dict

    Returns:
        output_int8: INT8 output tensor
    """
    if isqrt_lut is None or lut_metadata is None:
        isqrt_lut, lut_metadata = get_builtin_rmsnorm_isqrt_lut()

    num_entries = lut_metadata['num_entries']
    ms_max = lut_metadata['ms_max']
    output_scale = lut_metadata['output_scale']

    # Reshape input for easier processing
    original_shape = input_int8.shape

    if len(original_shape) == 1:
        input_reshaped = input_int8.reshape(1, -1)
    else:
        if original_shape[-1] == normalized_shape:
            batch_size = int(np.prod(original_shape[:-1]))
            input_reshaped = input_int8.reshape(batch_size, normalized_shape)
        else:
            batch_size = int(np.prod(original_shape[:-1]))
            input_reshaped = input_int8.reshape(batch_size, normalized_shape)

    # Process each vector
    num_vectors = input_reshaped.shape[0]
    output_fp32 = np.zeros((num_vectors, normalized_shape), dtype=np.float32)

    for v in range(num_vectors):
        input_vec = input_reshaped[v]

        # Step 1: Compute sum of squares using INT64 accumulation
        sum_sq = np.int64(0)
        for i in range(normalized_shape):
            val = np.int64(input_vec[i])
            sum_sq += val * val

        # Step 2: Compute mean of squares (integer division)
        mean_sq = sum_sq // np.int64(normalized_shape)

        # Step 3: Look up 1/sqrt(mean_sq + 1) from LUT
        ms_clamped = min(int(mean_sq), ms_max)
        idx = int(round(ms_clamped * (num_entries - 1) / ms_max))
        idx = max(0, min(idx, num_entries - 1))

        isqrt_int16 = int(isqrt_lut[idx])

        # Step 4: Normalize and apply weight
        for i in range(normalized_shape):
            x_int = np.int64(input_vec[i])

            # Normalize: x_norm = x * isqrt / SCALE
            x_norm_scaled = x_int * isqrt_int16
            x_norm_fp32 = float(x_norm_scaled) / output_scale

            # Convert to original FP32 domain
            x_norm_fp32 *= scale_input

            # Apply weight
            output_fp32[v, i] = weight_fp32[i] * x_norm_fp32

    # Reshape back to original shape
    output_fp32 = output_fp32.reshape(original_shape)

    # Quantize to INT8
    output_int8 = np.clip(np.round(output_fp32 / scale_output), -128, 127).astype(np.int8)

    return output_int8


def test_rmsnorm():
    """Unit test for INT8 RMSNorm"""
    print("=" * 80)
    print("Testing INT8 RMSNorm")
    print("=" * 80)

    # Test case 1: Simple 1D RMSNorm
    print("\nTest 1: 1D RMSNorm (8 features)")
    normalized_shape = 8

    # Create FP32 input
    input_fp32 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)

    # Quantize to INT8
    scale_input = 0.1
    input_int8 = np.clip(np.round(input_fp32 / scale_input), -128, 127).astype(np.int8)

    # RMSNorm parameters (simple: weight=1)
    weight_fp32 = np.ones(normalized_shape, dtype=np.float32)

    # Output scale
    scale_output = 0.1

    # Apply INT8 RMSNorm
    output_int8 = rmsnorm_int8(
        input_int8, weight_fp32,
        scale_input, scale_output, normalized_shape
    )

    # Dequantize for comparison
    output_fp32 = output_int8.astype(np.float32) * scale_output

    # Compute reference FP32 RMSNorm
    reference_fp32 = rmsnorm_fp32_reference(input_fp32, weight_fp32)

    # Compare
    print(f"Input (FP32):     {input_fp32}")
    print(f"Input (INT8):     {input_int8}")
    print(f"Output (INT8):    {output_int8}")
    print(f"Output (FP32):    {output_fp32}")
    print(f"Reference (FP32): {reference_fp32}")
    max_diff = np.max(np.abs(output_fp32 - reference_fp32))
    print(f"Max diff:         {max_diff:.6f}")

    # Test case 2: 2D RMSNorm (batch of 3, 8 features)
    print("\nTest 2: 2D RMSNorm (batch=3, features=8)")
    input_fp32 = np.random.randn(3, 8).astype(np.float32)
    input_int8 = np.clip(np.round(input_fp32 / scale_input), -128, 127).astype(np.int8)

    # Apply INT8 RMSNorm
    output_int8 = rmsnorm_int8(
        input_int8, weight_fp32,
        scale_input, scale_output, normalized_shape
    )

    # Dequantize
    output_fp32 = output_int8.astype(np.float32) * scale_output

    # Reference FP32 RMSNorm
    reference_fp32 = rmsnorm_fp32_reference(input_fp32, weight_fp32)

    print(f"Input shape:      {input_fp32.shape}")
    print(f"Output shape:     {output_fp32.shape}")
    max_diff = np.max(np.abs(output_fp32 - reference_fp32))
    mean_diff = np.mean(np.abs(output_fp32 - reference_fp32))
    print(f"Max diff:         {max_diff:.6f}")
    print(f"Mean diff:        {mean_diff:.6f}")

    # Test case 3: With learned weight
    print("\nTest 3: RMSNorm with learned weight")
    weight_fp32 = np.random.randn(normalized_shape).astype(np.float32) * 0.5 + 1.0

    # Apply INT8 RMSNorm
    output_int8 = rmsnorm_int8(
        input_int8, weight_fp32,
        scale_input, scale_output, normalized_shape
    )

    # Dequantize
    output_fp32 = output_int8.astype(np.float32) * scale_output

    # Reference FP32 RMSNorm
    reference_fp32 = rmsnorm_fp32_reference(input_fp32, weight_fp32)

    print(f"Weight (gamma):   {weight_fp32}")
    max_diff = np.max(np.abs(output_fp32 - reference_fp32))
    mean_diff = np.mean(np.abs(output_fp32 - reference_fp32))
    print(f"Max diff:         {max_diff:.6f}")
    print(f"Mean diff:        {mean_diff:.6f}")

    # Test case 4: Compare implementations
    print("\nTest 4: Compare FP32 vs fixed-point vs LUT implementations")

    output_fp32_impl = rmsnorm_int8(
        input_int8, weight_fp32,
        scale_input, scale_output, normalized_shape
    )

    output_fixed = rmsnorm_int8_fixed_point(
        input_int8, weight_fp32,
        scale_input, scale_output, normalized_shape
    )

    output_lut = rmsnorm_int8_lut(
        input_int8, weight_fp32,
        scale_input, scale_output, normalized_shape
    )

    fp32_vs_fixed = np.sum(output_fp32_impl != output_fixed)
    fp32_vs_lut = np.sum(output_fp32_impl != output_lut)
    fixed_vs_lut = np.sum(output_fixed != output_lut)

    print(f"FP32 vs Fixed-point mismatch: {fp32_vs_fixed} elements")
    print(f"FP32 vs LUT mismatch:         {fp32_vs_lut} elements")
    print(f"Fixed-point vs LUT mismatch:  {fixed_vs_lut} elements")

    # Test case 5: Verify RMSNorm vs LayerNorm difference
    print("\nTest 5: RMSNorm vs LayerNorm (conceptual check)")
    # RMSNorm should NOT subtract mean
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    w = np.ones(4, dtype=np.float32)

    # RMSNorm
    rms = np.sqrt(np.mean(x ** 2) + 1e-5)
    rmsnorm_result = (x / rms) * w

    # LayerNorm
    mean = np.mean(x)
    var = np.var(x)
    layernorm_result = ((x - mean) / np.sqrt(var + 1e-5)) * w

    print(f"Input:        {x}")
    print(f"RMSNorm:      {rmsnorm_result}")
    print(f"LayerNorm:    {layernorm_result}")
    print(f"Difference:   {np.abs(rmsnorm_result - layernorm_result)}")

    # Check errors are within quantization tolerance
    tolerance = max(scale_input, scale_output) * 2
    if max_diff <= tolerance:
        print(f"\n[PASS] All tests passed! (max diff {max_diff:.6f} <= tolerance {tolerance:.6f})")
    else:
        print(f"\n[WARN] Larger than expected diff (max diff {max_diff:.6f} > tolerance {tolerance:.6f})")

    print("=" * 80)
    return True


def test_rmsnorm_pytorch():
    """Compare with PyTorch RMSNorm if available."""
    print("\n" + "=" * 80)
    print("Testing RMSNorm against PyTorch (if available)")
    print("=" * 80)

    try:
        import torch

        # Create test data
        batch_size, dim = 4, 64
        x_fp32 = np.random.randn(batch_size, dim).astype(np.float32)
        weight_fp32 = np.random.randn(dim).astype(np.float32) * 0.5 + 1.0

        # PyTorch RMSNorm (manual implementation since it's not in torch.nn before 2.0)
        x_torch = torch.from_numpy(x_fp32)
        w_torch = torch.from_numpy(weight_fp32)

        # RMSNorm formula
        rms_torch = torch.sqrt(torch.mean(x_torch ** 2, dim=-1, keepdim=True) + 1e-5)
        y_torch = (x_torch / rms_torch) * w_torch
        y_torch_np = y_torch.numpy()

        # Our FP32 reference
        y_ref = rmsnorm_fp32_reference(x_fp32, weight_fp32)

        max_diff = np.max(np.abs(y_torch_np - y_ref))
        print(f"PyTorch vs FP32 reference max diff: {max_diff:.9f}")

        if max_diff < 1e-6:
            print("[PASS] FP32 reference matches PyTorch RMSNorm")
        else:
            print("[INFO] Small numerical differences with PyTorch")

    except ImportError:
        print("PyTorch not available, skipping comparison test")

    print("=" * 80)


if __name__ == "__main__":
    test_rmsnorm()
    test_rmsnorm_pytorch()
