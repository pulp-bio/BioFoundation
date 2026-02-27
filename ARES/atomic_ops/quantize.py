# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Atomic Quantization Operations

Implements QuantizeLinear and DequantizeLinear operations for TRUE INT8 computation.

These are the fundamental building blocks for quantized neural networks.
"""

import numpy as np
from typing import Union, Tuple


def quantize_linear(
    x: np.ndarray,
    scale: Union[float, np.ndarray],
    zero_point: Union[int, np.ndarray] = 0,
    qmin: int = -128,
    qmax: int = 127,
    dtype: np.dtype = np.int8
) -> np.ndarray:
    """
    Quantize FP32 tensor to INT8.

    Formula:
        q = clip(round(x / scale) + zero_point, qmin, qmax)

    Args:
        x: Input FP32 tensor
        scale: Quantization scale (FP32)
        zero_point: Quantization zero point (INT8), default 0 (symmetric)
        qmin: Minimum quantized value (default -128 for INT8)
        qmax: Maximum quantized value (default 127 for INT8)
        dtype: Output data type (default np.int8)

    Returns:
        Quantized INT8 tensor

    Notes:
        - Symmetric quantization uses zero_point=0
        - Asymmetric quantization uses zero_point!=0
        - Brevitas typically uses symmetric quantization for activations
    """
    # Divide by scale
    x_scaled = x / scale

    # Add zero point
    x_shifted = x_scaled + zero_point

    # Round to nearest integer
    x_rounded = np.round(x_shifted)

    # Clip to valid range
    x_clipped = np.clip(x_rounded, qmin, qmax)

    # Cast to target dtype
    return x_clipped.astype(dtype)


def dequantize_linear(
    q: np.ndarray,
    scale: Union[float, np.ndarray],
    zero_point: Union[int, np.ndarray] = 0
) -> np.ndarray:
    """
    Dequantize INT8 tensor to FP32.

    Formula:
        x = (q - zero_point) * scale

    Args:
        q: Quantized INT8 tensor
        scale: Quantization scale (FP32)
        zero_point: Quantization zero point (INT8), default 0

    Returns:
        Dequantized FP32 tensor
    """
    # Convert to float first to avoid overflow
    q_float = q.astype(np.float32)

    # Subtract zero point
    q_shifted = q_float - zero_point

    # Multiply by scale
    x = q_shifted * scale

    return x


def compute_quantization_params(
    x: np.ndarray,
    qmin: int = -128,
    qmax: int = 127,
    symmetric: bool = True
) -> Tuple[float, int]:
    """
    Compute quantization parameters (scale and zero_point) for a given tensor.

    Args:
        x: Input FP32 tensor
        qmin: Minimum quantized value
        qmax: Maximum quantized value
        symmetric: Use symmetric quantization (zero_point=0)

    Returns:
        Tuple of (scale, zero_point)

    Notes:
        Symmetric quantization (Brevitas default for activations):
            scale = max(|x_min|, |x_max|) / 127
            zero_point = 0

        Asymmetric quantization:
            scale = (x_max - x_min) / (qmax - qmin)
            zero_point = round(qmin - x_min / scale)
    """
    x_min = float(np.min(x))
    x_max = float(np.max(x))

    if symmetric:
        # Symmetric quantization
        max_abs = max(abs(x_min), abs(x_max))
        scale = max_abs / 127.0
        zero_point = 0
    else:
        # Asymmetric quantization
        scale = (x_max - x_min) / (qmax - qmin)
        zero_point = int(np.round(qmin - x_min / scale))

    # Avoid division by zero
    if scale == 0:
        scale = 1.0

    return scale, zero_point


def test_quantization():
    """Test quantization operations."""
    print("="*80)
    print("Testing Quantization Operations")
    print("="*80)

    # Test 1: Simple symmetric quantization
    print("\nTest 1: Symmetric Quantization")
    x = np.array([-12.7, -6.35, 0.0, 6.35, 12.7], dtype=np.float32)
    scale = 0.1
    zero_point = 0

    q = quantize_linear(x, scale, zero_point)
    x_reconstructed = dequantize_linear(q, scale, zero_point)

    print(f"Original:      {x}")
    print(f"Quantized:     {q}")
    print(f"Reconstructed: {x_reconstructed}")
    print(f"Error:         {x - x_reconstructed}")
    print(f"Max error:     {np.max(np.abs(x - x_reconstructed))}")

    # Test 2: Compute quantization params
    print("\n" + "="*80)
    print("Test 2: Compute Quantization Parameters")
    print("="*80)
    x = np.random.randn(10, 10).astype(np.float32)

    scale_sym, zp_sym = compute_quantization_params(x, symmetric=True)
    scale_asym, zp_asym = compute_quantization_params(x, symmetric=False)

    print(f"\nSymmetric:  scale={scale_sym:.6f}, zero_point={zp_sym}")
    print(f"Asymmetric: scale={scale_asym:.6f}, zero_point={zp_asym}")

    q_sym = quantize_linear(x, scale_sym, zp_sym)
    q_asym = quantize_linear(x, scale_asym, zp_asym)

    x_sym_rec = dequantize_linear(q_sym, scale_sym, zp_sym)
    x_asym_rec = dequantize_linear(q_asym, scale_asym, zp_asym)

    error_sym = np.max(np.abs(x - x_sym_rec))
    error_asym = np.max(np.abs(x - x_asym_rec))

    print(f"\nSymmetric max error:  {error_sym:.6f}")
    print(f"Asymmetric max error: {error_asym:.6f}")

    # Test 3: Round-trip with clip
    print("\n" + "="*80)
    print("Test 3: Clipping Behavior")
    print("="*80)
    x = np.array([-200.0, -100.0, 0.0, 100.0, 200.0], dtype=np.float32)
    scale = 1.0
    zero_point = 0

    q = quantize_linear(x, scale, zero_point)
    print(f"Input (outside range): {x}")
    print(f"Quantized (clipped):   {q}")
    print(f"Expected clip at:      [{-128}, {127}]")

    assert np.all(q >= -128) and np.all(q <= 127), "Clipping failed!"
    print("[PASS] Clipping works correctly")

    print("\n" + "="*80)
    print("[PASS] All tests passed!")
    print("="*80)


if __name__ == "__main__":
    test_quantization()
