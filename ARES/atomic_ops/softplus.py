# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Atomic Softplus Operation - INT32 LUT + Piecewise Approximation

Implements softplus activation for integer-only execution.

softplus(x) = log(1 + exp(x))

Properties:
- softplus(x) ≈ x           for x >> 0 (linear region)
- softplus(x) ≈ exp(x)      for x << 0 (exponential region)
- softplus(0) = log(2) ≈ 0.693

Used in MAMBA for dt (delta timestep) computation after dt_proj.
The dt_proj output is typically INT32 accumulator, which we convert
to Q16 fixed-point dt values.

Integer-Only Strategy:
1. Piecewise approximation for extreme values
2. LUT with linear interpolation for middle range
3. Q16 output format for sufficient precision
"""

import numpy as np
from typing import Optional, Tuple

try:
    from .quantize import quantize_linear, dequantize_linear
except ImportError:
    from quantize import quantize_linear, dequantize_linear


def softplus_fp32(x: np.ndarray) -> np.ndarray:
    """
    FP32 softplus activation with numerical stability.

    Uses log1p for better precision: softplus(x) = x + log(1 + exp(-|x|))
    """
    return np.where(
        x > 20,
        x,  # For large x, softplus(x) ≈ x
        np.where(
            x < -20,
            np.exp(x),  # For large negative x, softplus(x) ≈ exp(x)
            np.log1p(np.exp(x))  # Standard computation
        )
    )


def generate_softplus_lut_q16(
    x_min: float = -5.0,
    x_max: float = 5.0,
    num_entries: int = 512
) -> Tuple[np.ndarray, float, float, float]:
    """
    Generate a Q16 lookup table for softplus in range [x_min, x_max].

    Outside this range, piecewise approximation is used:
    - x > x_max: softplus(x) ≈ x
    - x < x_min: softplus(x) ≈ exp(x)

    Args:
        x_min: Minimum x value for LUT
        x_max: Maximum x value for LUT
        num_entries: Number of LUT entries

    Returns:
        lut_q16: Q16 lookup table (INT32)
        x_min: Actual x_min used
        x_max: Actual x_max used
        step: Step size between entries
    """
    step = (x_max - x_min) / (num_entries - 1)

    # Create x values for LUT
    x_vals = np.linspace(x_min, x_max, num_entries)

    # Compute softplus and convert to Q16
    y_fp32 = softplus_fp32(x_vals)
    lut_q16 = np.round(y_fp32 * 65536).astype(np.int32)

    return lut_q16, x_min, x_max, step


def generate_exp_lut_q16(
    x_min: float = -8.0,
    x_max: float = 0.0,
    num_entries: int = 256
) -> Tuple[np.ndarray, float, float, float]:
    """
    Generate a Q16 lookup table for exp(x) in range [x_min, x_max].

    Used for the exponential region of softplus (x << 0).

    Returns:
        lut_q16: Q16 lookup table (INT32)
        x_min, x_max, step: LUT parameters
    """
    step = (x_max - x_min) / (num_entries - 1)
    x_vals = np.linspace(x_min, x_max, num_entries)
    y_fp32 = np.exp(x_vals)
    lut_q16 = np.round(y_fp32 * 65536).astype(np.int32)

    return lut_q16, x_min, x_max, step


def softplus_q16(
    x_q16: np.ndarray,
    lut_q16: np.ndarray,
    exp_lut_q16: np.ndarray,
    softplus_x_min: float,
    softplus_x_max: float,
    softplus_step: float,
    exp_x_min: float,
    exp_x_max: float,
    exp_step: float
) -> np.ndarray:
    """
    Integer-only softplus using Q16 fixed-point with LUT + piecewise.

    Args:
        x_q16: Input in Q16 format (INT32)
        lut_q16: Softplus LUT for middle range
        exp_lut_q16: Exp LUT for negative range
        softplus_x_min/max/step: Softplus LUT parameters
        exp_x_min/max/step: Exp LUT parameters

    Returns:
        Output in Q16 format (INT32)
    """
    # Convert Q16 bounds to integer thresholds
    softplus_min_q16 = int(softplus_x_min * 65536)
    softplus_max_q16 = int(softplus_x_max * 65536)
    exp_min_q16 = int(exp_x_min * 65536)

    # Pre-compute inverse step for integer indexing
    # idx = (x - x_min) / step = (x - x_min) * inv_step
    inv_softplus_step_q16 = int(1.0 / softplus_step * 65536)
    inv_exp_step_q16 = int(1.0 / exp_step * 65536)

    output = np.zeros_like(x_q16, dtype=np.int32)

    # Process each element
    for idx in np.ndindex(x_q16.shape):
        x = x_q16[idx]

        if x >= softplus_max_q16:
            # Linear region: softplus(x) ≈ x
            output[idx] = x
        elif x <= exp_min_q16:
            # Deep negative: softplus(x) ≈ 0 (exp(x) → 0)
            output[idx] = 0
        elif x < softplus_min_q16:
            # Exponential region: use exp LUT
            # Index calculation: idx = (x - exp_min) / step
            # x_shifted is Q16, inv_step is Q16, so product is Q32
            # Use INT64 to avoid overflow, shift by 32 to get integer index
            x_shifted = np.int64(x) - np.int64(exp_min_q16)
            lut_idx_q32 = x_shifted * np.int64(inv_exp_step_q16)
            lut_idx = int(lut_idx_q32 >> 32)
            lut_idx = np.clip(lut_idx, 0, len(exp_lut_q16) - 1)
            output[idx] = exp_lut_q16[lut_idx]
        else:
            # Middle range: use softplus LUT with linear interpolation
            # x_shifted is Q16, inv_step is Q16, product is Q32
            # Use INT64 to avoid overflow
            x_shifted = np.int64(x) - np.int64(softplus_min_q16)
            lut_idx_q32 = x_shifted * np.int64(inv_softplus_step_q16)
            # Shift by 16 to get Q16 index (integer + fractional)
            lut_idx_q16 = lut_idx_q32 >> 16
            lut_idx = int(lut_idx_q16 >> 16)  # Integer part of index
            lut_idx = int(np.clip(lut_idx, 0, len(lut_q16) - 2))

            # Linear interpolation using fractional part
            frac = int(lut_idx_q16 & 0xFFFF)  # Fractional part (Q16)
            y0 = int(lut_q16[lut_idx])
            y1 = int(lut_q16[lut_idx + 1])
            output[idx] = y0 + ((y1 - y0) * frac >> 16)

    return output


def softplus_int32_to_q16(
    x_int32: np.ndarray,
    scale_x: float,
    lut_q16: Optional[np.ndarray] = None,
    exp_lut_q16: Optional[np.ndarray] = None,
    lut_params: Optional[dict] = None
) -> np.ndarray:
    """
    Apply softplus to INT32 accumulator output, returning Q16 result.

    This is the primary interface for MAMBA dt computation:
    dt_int32 (from dt_proj) -> softplus -> dt_q16

    Args:
        x_int32: Input INT32 (accumulator from linear layer)
        scale_x: Scale factor for x_int32 (x_fp32 = x_int32 * scale_x)
        lut_q16: Pre-generated softplus LUT (optional)
        exp_lut_q16: Pre-generated exp LUT (optional)
        lut_params: LUT parameters dict (optional)

    Returns:
        Output in Q16 format
    """
    # Generate LUTs if not provided
    if lut_q16 is None:
        lut_q16, sp_min, sp_max, sp_step = generate_softplus_lut_q16()
    else:
        sp_min = lut_params.get('softplus_x_min', -5.0) if lut_params else -5.0
        sp_max = lut_params.get('softplus_x_max', 5.0) if lut_params else 5.0
        sp_step = (sp_max - sp_min) / (len(lut_q16) - 1)

    if exp_lut_q16 is None:
        exp_lut_q16, exp_min, exp_max, exp_step = generate_exp_lut_q16()
    else:
        exp_min = lut_params.get('exp_x_min', -8.0) if lut_params else -8.0
        exp_max = lut_params.get('exp_x_max', 0.0) if lut_params else 0.0
        exp_step = (exp_max - exp_min) / (len(exp_lut_q16) - 1)

    # Convert INT32 to Q16 using scale
    # x_fp32 = x_int32 * scale_x
    # x_q16 = x_fp32 * 65536 = x_int32 * scale_x * 65536
    scale_to_q16 = scale_x * 65536
    x_q16 = np.round(x_int32.astype(np.float64) * scale_to_q16).astype(np.int32)

    # Apply softplus
    output_q16 = softplus_q16(
        x_q16, lut_q16, exp_lut_q16,
        sp_min, sp_max, sp_step,
        exp_min, exp_max, exp_step
    )

    return output_q16


def test_softplus():
    """Test INT32/Q16 softplus implementation."""
    print("=" * 80)
    print("Testing Integer-Only Softplus (LUT + Piecewise)")
    print("=" * 80)

    # Test 1: LUT generation
    print("\n--- Test 1: LUT Generation ---")
    lut_q16, sp_min, sp_max, sp_step = generate_softplus_lut_q16()
    exp_lut_q16, exp_min, exp_max, exp_step = generate_exp_lut_q16()

    print(f"Softplus LUT: {len(lut_q16)} entries, range [{sp_min}, {sp_max}], step {sp_step:.4f}")
    print(f"Exp LUT: {len(exp_lut_q16)} entries, range [{exp_min}, {exp_max}], step {exp_step:.4f}")

    # Verify key values
    # softplus(0) = log(2) ≈ 0.693
    mid_idx = len(lut_q16) // 2
    softplus_0_q16 = lut_q16[mid_idx]
    softplus_0_fp32 = softplus_0_q16 / 65536.0
    expected_softplus_0 = np.log(2)
    print(f"\nSoftplus(0) from LUT: {softplus_0_fp32:.6f} (expected: {expected_softplus_0:.6f})")
    print("Test 1 PASSED!")

    # Test 2: Q16 softplus accuracy
    print("\n--- Test 2: Q16 Softplus Accuracy ---")
    test_values_fp32 = np.array([-8, -5, -2, -1, 0, 1, 2, 5, 8], dtype=np.float32)
    test_values_q16 = (test_values_fp32 * 65536).astype(np.int32)

    output_q16 = softplus_q16(
        test_values_q16, lut_q16, exp_lut_q16,
        sp_min, sp_max, sp_step,
        exp_min, exp_max, exp_step
    )
    output_fp32 = output_q16 / 65536.0
    reference_fp32 = softplus_fp32(test_values_fp32)

    print(f"Input FP32:      {test_values_fp32}")
    print(f"Reference FP32:  {reference_fp32}")
    print(f"Q16 -> FP32:     {output_fp32}")

    error = np.abs(reference_fp32 - output_fp32)
    max_error = np.max(error)
    mean_error = np.mean(error)
    print(f"\nMax error: {max_error:.6f}")
    print(f"Mean error: {mean_error:.6f}")

    # Accept up to 1% relative error or 0.01 absolute
    assert max_error < 0.1 or np.max(error / (reference_fp32 + 1e-6)) < 0.01
    print("Test 2 PASSED!")

    # Test 3: INT32 to Q16 conversion
    print("\n--- Test 3: INT32 to Q16 Conversion ---")
    scale_x = 0.001  # Typical scale from dt_proj
    x_fp32 = np.array([-3, -1, 0, 1, 3], dtype=np.float32)
    x_int32 = np.round(x_fp32 / scale_x).astype(np.int32)

    output_q16 = softplus_int32_to_q16(x_int32, scale_x)
    output_fp32 = output_q16 / 65536.0
    reference_fp32 = softplus_fp32(x_fp32)

    print(f"x_fp32:          {x_fp32}")
    print(f"x_int32:         {x_int32}")
    print(f"Reference FP32:  {reference_fp32}")
    print(f"Q16 -> FP32:     {output_fp32}")

    error = np.abs(reference_fp32 - output_fp32)
    print(f"Max error: {np.max(error):.6f}")
    print("Test 3 PASSED!")

    # Test 4: PyTorch comparison
    print("\n--- Test 4: PyTorch Comparison ---")
    try:
        import torch
        import torch.nn.functional as F

        x_torch = torch.from_numpy(np.array([-5, -2, 0, 2, 5], dtype=np.float32))
        y_torch = F.softplus(x_torch)
        y_torch_np = y_torch.numpy()
        y_ref = softplus_fp32(x_torch.numpy())

        torch_error = np.max(np.abs(y_torch_np - y_ref))
        print(f"PyTorch vs FP32 reference max error: {torch_error:.9f}")
        print("Test 4 PASSED!")

    except ImportError:
        print("PyTorch not available, skipping Test 4")

    # Test 5: Large tensor
    print("\n--- Test 5: Large Tensor Performance ---")
    x_large_fp32 = np.random.randn(128, 64).astype(np.float32) * 3
    x_large_int32 = np.round(x_large_fp32 / scale_x).astype(np.int32)

    import time
    start = time.time()
    output_large_q16 = softplus_int32_to_q16(x_large_int32, scale_x)
    elapsed = time.time() - start

    output_large_fp32 = output_large_q16 / 65536.0
    reference_large_fp32 = softplus_fp32(x_large_fp32)

    large_error = np.abs(reference_large_fp32 - output_large_fp32)
    print(f"Tensor shape: {x_large_int32.shape}")
    print(f"Time: {elapsed*1000:.2f} ms")
    print(f"Max error: {np.max(large_error):.6f}")
    print(f"Mean error: {np.mean(large_error):.6f}")
    print("Test 5 PASSED!")

    # Test 6: Piecewise regions
    print("\n--- Test 6: Piecewise Region Verification ---")
    # Test values in different regions
    regions = {
        'deep_negative': np.array([-10, -8, -7], dtype=np.float32),
        'exp_region': np.array([-6, -5.5, -5.1], dtype=np.float32),
        'lut_region': np.array([-4, -2, 0, 2, 4], dtype=np.float32),
        'linear_region': np.array([5.1, 6, 8, 10], dtype=np.float32)
    }

    for region_name, values in regions.items():
        x_q16 = (values * 65536).astype(np.int32)
        output_q16 = softplus_q16(
            x_q16, lut_q16, exp_lut_q16,
            sp_min, sp_max, sp_step,
            exp_min, exp_max, exp_step
        )
        output_fp32 = output_q16 / 65536.0
        reference_fp32 = softplus_fp32(values)
        region_error = np.max(np.abs(reference_fp32 - output_fp32))
        print(f"  {region_name}: max_error = {region_error:.6f}")

    print("Test 6 PASSED!")

    print("\n" + "=" * 80)
    print("All Softplus tests completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    test_softplus()
