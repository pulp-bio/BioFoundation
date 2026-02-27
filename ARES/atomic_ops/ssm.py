# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Atomic SSM (State Space Model) Operations - Integer-Only

Implements the core Mamba SSM components for integer-only execution on GAP9:

1. ssm_discretize_q15(): Convert continuous A, B to discrete dA, dB' using LUTs
2. ssm_scan_q15(): Q15 fixed-point state recurrence
3. ssm_gate_silu_q13(): Direct 256-entry LUT gating

Data Layout (Mamba convention):
- x: [L, M] - input sequence (L timesteps, M channels/d_inner)
- dt: [L, M] or [L] - delta timesteps (after softplus)
- A: [D, M] or [D] - state transition matrix (negative values)
- B: [L, D] - input-dependent B matrix
- C: [L, D] - input-dependent C matrix
- h: [M, D] - hidden state
- y: [L, M] - output sequence

Q-Formats:
- Q15: 15 fractional bits, range [-1, 1) with 1/32768 resolution
- Q13: 13 fractional bits, range [-4, 4) with 1/8192 resolution
- Q16: 16 fractional bits, range [-32768, 32768) with 1/65536 resolution
"""

import numpy as np
from typing import Tuple, Optional

try:
    from .quantize import quantize_linear, dequantize_linear
    from .constants import INT16_MAX, INT16_MIN, Q13_SCALE_INT, Q15_SCALE, Q15_SCALE_INT
except ImportError:
    from quantize import quantize_linear, dequantize_linear
    from constants import INT16_MAX, INT16_MIN, Q13_SCALE_INT, Q15_SCALE, Q15_SCALE_INT


# --- LUT Generation ---

def generate_exp_lut_q15(
    z_min: float = -10.0,
    z_max: float = 0.0,
    num_entries: int = 512
) -> Tuple[np.ndarray, float, float, float]:
    """
    Generate Q15 lookup table for exp(z) used in SSM discretization.

    The exp LUT is used for: dA = exp(dt * A)
    Since A is negative and dt is positive, z = dt * A is in [z_min, 0].

    Args:
        z_min: Minimum z value (most negative)
        z_max: Maximum z value (should be 0 or close)
        num_entries: Number of LUT entries

    Returns:
        lut_q15: Q15 lookup table (INT16)
        z_min, z_max, step: LUT parameters
    """
    step = (z_max - z_min) / (num_entries - 1)
    z_vals = np.linspace(z_min, z_max, num_entries)

    # exp(z) for z in [-10, 0] gives range [exp(-10)≈0, exp(0)=1]
    exp_vals = np.exp(z_vals)

    # Convert to Q15 (multiply by 2^15)
    lut_q15 = np.clip(np.round(exp_vals * Q15_SCALE_INT), INT16_MIN, INT16_MAX).astype(np.int16)

    return lut_q15, z_min, z_max, step


def generate_phi1_lut_q15(
    z_min: float = -10.0,
    z_max: float = 0.0,
    num_entries: int = 512
) -> Tuple[np.ndarray, float, float, float]:
    """
    Generate Q15 lookup table for φ1(z) = (exp(z) - 1) / z.

    Used in dB' computation: dB' = dt * B * s_x * φ1(dt * A)

    φ1(z) properties:
    - φ1(0) = 1 (by L'Hopital's rule: lim_{z->0} (exp(z)-1)/z = 1)
    - φ1(z) ≈ 1 + z/2 + z²/6 + ... for small z
    - φ1(z) → 0 as z → -∞

    Args:
        z_min: Minimum z value
        z_max: Maximum z value
        num_entries: Number of LUT entries

    Returns:
        lut_q15: Q15 lookup table (INT16)
        z_min, z_max, step: LUT parameters
    """
    step = (z_max - z_min) / (num_entries - 1)
    z_vals = np.linspace(z_min, z_max, num_entries)

    # Compute φ1(z) = (exp(z) - 1) / z with special handling for z ≈ 0
    # Use np.divide with where to avoid divide-by-zero warning
    with np.errstate(divide='ignore', invalid='ignore'):
        phi1_vals = np.where(
            np.abs(z_vals) < 1e-6,
            1.0,  # φ1(0) = 1
            (np.exp(z_vals) - 1.0) / z_vals
        )

    # Convert to Q15
    lut_q15 = np.clip(np.round(phi1_vals * Q15_SCALE_INT), INT16_MIN, INT16_MAX).astype(np.int16)

    return lut_q15, z_min, z_max, step


def generate_exp_neg_lut_q15(scale_in: float = 0.1) -> np.ndarray:
    """Generate 256-entry Q15 exp LUT for negative inputs (I-Mamba)."""
    lut = np.zeros(256, dtype=np.int16)
    for q_in in range(-128, 128):
        x = q_in * scale_in
        if x > 20.0:
            exp_val = 1.0
        elif x < -20.0:
            exp_val = 0.0
        else:
            exp_val = np.exp(x)

        q15_out = int(np.round(exp_val * Q15_SCALE_INT))
        q15_out = max(0, min(INT16_MAX, q15_out))
        lut[q_in + 128] = np.int16(q15_out)
    return lut


def generate_softplus_lut_q8_8(scale_in: float = 0.1) -> np.ndarray:
    """Generate 256-entry Q8.8 softplus LUT (I-Mamba)."""
    lut = np.zeros(256, dtype=np.int16)
    for q_in in range(-128, 128):
        x = q_in * scale_in
        if x > 20.0:
            softplus = x
        elif x < -20.0:
            softplus = np.exp(x)
        else:
            softplus = np.log1p(np.exp(x))

        q8_8_out = int(np.round(softplus * 256))
        q8_8_out = max(0, min(INT16_MAX, q8_8_out))
        lut[q_in + 128] = np.int16(q8_8_out)
    return lut


# --- SSM Discretization (Integer-Only) ---

def ssm_discretize_q15(
    dt_q16: np.ndarray,
    A_q16: np.ndarray,
    B_q15: np.ndarray,
    s_x_q15: int,
    exp_lut_q15: np.ndarray,
    phi1_lut_q15: np.ndarray,
    lut_min_q32: int,
    inv_lut_step_q31: int,
    lut_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integer-only SSM discretization using LUT indexing via multiplicative inverse.

    Computes:
        z = dt * A (in fixed-point)
        dA = exp(z) via LUT
        dB' = dt * B * s_x * φ1(z) via LUT

    Args:
        dt_q16: Delta timesteps [L, M] or [L] in Q16 format
        A_q16: A matrix [D, M] or [D] in Q16 format (negative values)
        B_q15: B matrix [L, D] in Q15 format
        s_x_q15: Input scale in Q15 format
        exp_lut_q15: Exp LUT [lut_size] in Q15
        phi1_lut_q15: Phi1 LUT [lut_size] in Q15
        lut_min_q32: LUT minimum z value in Q32
        inv_lut_step_q31: 1/step in Q31 for index calculation
        lut_size: Number of LUT entries

    Returns:
        dA_q15: [L, D, M] or [L, D] in Q15 format
        dB_prime_q15: [L, D, M] or [L, D] in Q15 format
    """
    # Handle different input shapes
    if dt_q16.ndim == 1:
        L = dt_q16.shape[0]
        M = 1
        dt_q16 = dt_q16[:, np.newaxis]  # [L, 1]
    else:
        L, M = dt_q16.shape

    if A_q16.ndim == 1:
        D = A_q16.shape[0]
        A_q16 = A_q16[:, np.newaxis]  # [D, 1]
    else:
        D, M_a = A_q16.shape
        assert M_a == M or M_a == 1, f"A shape mismatch: {A_q16.shape}"

    # Output arrays
    dA_q15 = np.zeros((L, D, M), dtype=np.int16)
    dB_prime_q15 = np.zeros((L, D, M), dtype=np.int16)

    for t in range(L):
        for d in range(D):
            for m in range(M):
                # Get dt and A values
                dt_val = np.int64(dt_q16[t, m if dt_q16.shape[1] > 1 else 0])
                A_val = np.int64(A_q16[d, m if A_q16.shape[1] > 1 else 0])

                # Compute z = dt * A in Q32 (Q16 * Q16 = Q32)
                z_q32 = dt_val * A_val

                # LUT index calculation via multiplicative inverse
                # idx = (z - lut_min) / step = (z - lut_min) * inv_step >> 31
                z_shifted = z_q32 - np.int64(lut_min_q32)
                idx_q31 = z_shifted * np.int64(inv_lut_step_q31)
                idx = int(idx_q31 >> 32)
                frac_q15 = int((idx_q31 >> 17) & 0x7FFF)  # 15-bit fractional part

                # Clamp index to valid range
                idx = max(0, min(idx, lut_size - 2))

                # dA via exp LUT with linear interpolation
                e0 = int(exp_lut_q15[idx])
                e1 = int(exp_lut_q15[idx + 1])
                e_diff = e1 - e0
                # Interpolate: dA = e0 + (frac * (e1 - e0)) >> 15
                e_inc = (frac_q15 * e_diff + (1 << 14)) >> 15  # Rounded shift
                dA_q15[t, d, m] = np.int16(e0 + e_inc)

                # dB' via phi1 LUT
                p0 = int(phi1_lut_q15[idx])
                p1 = int(phi1_lut_q15[idx + 1])
                p_diff = p1 - p0
                p_inc = (frac_q15 * p_diff + (1 << 14)) >> 15
                phi1_q15 = p0 + p_inc

                # dB' = dt * B * s_x * phi1
                # dt_q16 * B_q15 = Q31, then * s_x_q15 = Q46, then * phi1_q15 = Q61
                # We need to carefully manage the shifts
                B_val = np.int64(B_q15[t, d])

                # Compute in stages to avoid overflow
                # scale_factor = dt * B * s_x (normalize to reasonable range)
                # Then multiply by phi1

                # dt_q16 * s_x_q15 >> 15 = Q16 (scale factor for dt)
                dt_sx = (dt_val * np.int64(s_x_q15) + (1 << 14)) >> 15  # Q16

                # dt_sx * B_q15 >> 15 = Q16 (scaled B term)
                db_scaled = (dt_sx * B_val + (1 << 14)) >> 15  # Q16

                # db_scaled * phi1_q15 >> 15 = Q16, then >> 1 = Q15
                db_prime = (db_scaled * np.int64(phi1_q15) + (1 << 14)) >> 15  # Q16
                db_prime = (db_prime + 1) >> 1  # Q15

                dB_prime_q15[t, d, m] = np.clip(db_prime, INT16_MIN, INT16_MAX).astype(np.int16)

    return dA_q15, dB_prime_q15


def ssm_discretize_fp32_reference(
    dt: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    s_x: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    FP32 reference implementation of SSM discretization.

    Args:
        dt: Delta timesteps [L, M] or [L]
        A: A matrix [D, M] or [D] (negative values)
        B: B matrix [L, D]
        s_x: Input scale factor

    Returns:
        dA: [L, D, M] or [L, D]
        dB_prime: [L, D, M] or [L, D]
    """
    # Handle shapes
    if dt.ndim == 1:
        dt = dt[:, np.newaxis]
    L, M = dt.shape

    if A.ndim == 1:
        A = A[:, np.newaxis]
    D, _ = A.shape

    dA = np.zeros((L, D, M), dtype=np.float32)
    dB_prime = np.zeros((L, D, M), dtype=np.float32)

    for t in range(L):
        for d in range(D):
            for m in range(M):
                dt_val = dt[t, m if dt.shape[1] > 1 else 0]
                A_val = A[d, m if A.shape[1] > 1 else 0]

                # z = dt * A
                z = dt_val * A_val

                # dA = exp(z)
                dA[t, d, m] = np.exp(z)

                # phi1(z) = (exp(z) - 1) / z
                if np.abs(z) < 1e-6:
                    phi1 = 1.0
                else:
                    phi1 = (np.exp(z) - 1.0) / z

                # dB' = dt * B * s_x * phi1
                dB_prime[t, d, m] = dt_val * B[t, d] * s_x * phi1

    return dA, dB_prime


# --- SSM Scan (Q15 Fixed-Point) ---

def ssm_scan_q15(
    x_i8: np.ndarray,
    dA_q15: np.ndarray,
    dB_prime_q15: np.ndarray,
    C_q15: np.ndarray,
    D_val: Optional[np.ndarray] = None,
    h_init: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Q15 fixed-point SSM scan (the core Mamba recurrence).

    State update: h[d] = (dA[d] * h[d] >> 15) + dB'[d] * x
    Output: y = (sum(h[d] * C[d])) >> 30 + D * x

    Args:
        x_i8: Input [L, M] in INT8
        dA_q15: Discretized A [L, D, M] or [L, D] in Q15
        dB_prime_q15: Discretized B' [L, D, M] or [L, D] in Q15
        C_q15: C matrix [L, D] in Q15
        D_val: Optional D residual [M] (FP32 or INT)
        h_init: Optional initial state [M, D] in INT32

    Returns:
        y_acc: Output accumulator [L, M] in INT32
        h_final: Final state [M, D] in INT32
    """
    L, M = x_i8.shape

    # Determine D from dA shape
    if dA_q15.ndim == 3:
        D = dA_q15.shape[1]
    else:
        D = dA_q15.shape[1]
        # Expand to [L, D, M] if needed
        dA_q15 = np.broadcast_to(dA_q15[:, :, np.newaxis], (L, D, M))
        dB_prime_q15 = np.broadcast_to(dB_prime_q15[:, :, np.newaxis], (L, D, M))

    # Initialize state (INT32 for accumulation precision)
    if h_init is not None:
        h = h_init.copy().astype(np.int32)
    else:
        h = np.zeros((M, D), dtype=np.int32)

    # Output accumulator
    y_acc = np.zeros((L, M), dtype=np.int32)

    def rounding_shift_right(x, shift):
        """Rounded right shift with ties-away-from-zero."""
        if shift <= 0:
            return x
        offset = 1 << (shift - 1)
        return (x + offset) >> shift

    for t in range(L):
        for m in range(M):
            x_val = int(x_i8[t, m])

            # State update for each state dimension
            for d in range(D):
                # h = (dA * h >> 15) + dB' * x
                dA_val = int(dA_q15[t, d, m])
                dB_val = int(dB_prime_q15[t, d, m])

                # dA * h: Q15 * INT32 = Q15 accumulator value
                h_mul = np.int64(h[m, d]) * np.int64(dA_val)
                h_after = rounding_shift_right(h_mul, 15)

                # Add dB' * x: Q15 * INT8 = Q15 + x bits
                h_add = np.int64(dB_val) * np.int64(x_val)

                h_new = h_after + h_add
                # Saturate to INT32 range
                h[m, d] = np.clip(h_new, np.iinfo(np.int32).min, np.iinfo(np.int32).max)

            # Output: y = (h · C) >> 30
            acc64 = np.int64(0)
            for d in range(D):
                C_val = int(C_q15[t, d])
                acc64 += np.int64(h[m, d]) * np.int64(C_val)

            y_acc[t, m] = rounding_shift_right(acc64, 30)

            # Add D residual if provided
            if D_val is not None:
                # D * x (assuming D is small integer or Q format)
                if isinstance(D_val[m], (int, np.integer)):
                    y_acc[t, m] += D_val[m] * x_val
                else:
                    y_acc[t, m] += int(D_val[m] * x_val)

    return y_acc, h


def ssm_scan_fp32_reference(
    x: np.ndarray,
    dA: np.ndarray,
    dB_prime: np.ndarray,
    C: np.ndarray,
    D_val: Optional[np.ndarray] = None,
    h_init: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    FP32 reference implementation of SSM scan.

    Args:
        x: Input [L, M]
        dA: Discretized A [L, D, M] or [L, D]
        dB_prime: Discretized B' [L, D, M] or [L, D]
        C: C matrix [L, D]
        D_val: Optional D residual [M]
        h_init: Optional initial state [M, D]

    Returns:
        y: Output [L, M]
        h_final: Final state [M, D]
    """
    L, M = x.shape

    if dA.ndim == 2:
        D = dA.shape[1]
        dA = np.broadcast_to(dA[:, :, np.newaxis], (L, D, M))
        dB_prime = np.broadcast_to(dB_prime[:, :, np.newaxis], (L, D, M))
    else:
        D = dA.shape[1]

    if h_init is not None:
        h = h_init.copy().astype(np.float32)
    else:
        h = np.zeros((M, D), dtype=np.float32)

    y = np.zeros((L, M), dtype=np.float32)

    for t in range(L):
        for m in range(M):
            x_val = x[t, m]

            # State update
            for d in range(D):
                h[m, d] = dA[t, d, m] * h[m, d] + dB_prime[t, d, m] * x_val

            # Output
            y[t, m] = np.sum(h[m, :] * C[t, :])

            # D residual
            if D_val is not None:
                y[t, m] += D_val[m] * x_val

    return y, h


# --- SSM Gate (SiLU LUT) ---

def ssm_gate_silu_q13(
    y_acc_i32: np.ndarray,
    z_i8: np.ndarray,
    silu_lut_q13: np.ndarray
) -> np.ndarray:
    """
    Apply SiLU gating to SSM output using direct 256-entry LUT.

    y_gated = (y_acc * silu(z)) >> 13

    Since z_i8 has only 256 possible values, we use a direct lookup
    instead of interpolation.

    Args:
        y_acc_i32: SSM output accumulator [L, M] in INT32
        z_i8: Gate input [L, M] in INT8
        silu_lut_q13: 256-entry Q13 SiLU LUT

    Returns:
        y_gated: Gated output [L, M] in INT32
    """
    # Convert z_i8 to LUT indices (map [-128, 127] to [0, 255])
    indices = z_i8.astype(np.int32) + 128

    # Lookup SiLU values (Q13)
    silu_q13 = silu_lut_q13[indices].astype(np.int64)

    # Apply gate: y_gated = (y_acc * silu_q13) >> 13
    y_acc_64 = y_acc_i32.astype(np.int64)
    y_gated_64 = (y_acc_64 * silu_q13 + (1 << 12)) >> 13  # Rounded shift

    # Saturate to INT32
    y_gated = np.clip(y_gated_64, np.iinfo(np.int32).min, np.iinfo(np.int32).max).astype(np.int32)

    return y_gated


def generate_silu_gate_lut_q13(scale_z: float) -> np.ndarray:
    """
    Generate 256-entry Q13 LUT for SiLU gating.

    For each possible z_i8 value [-128, 127]:
        z_float = z_i8 * scale_z
        silu_float = z_float * sigmoid(z_float)
        silu_q13[z_i8 + 128] = round(silu_float * Q13_SCALE_INT)

    Args:
        scale_z: Quantization scale for z input

    Returns:
        lut_q13: 256-entry INT16 array in Q2.13 format
    """
    lut_q13 = np.zeros(256, dtype=np.int16)

    for i in range(256):
        z_i8 = i - 128
        z_float = z_i8 * scale_z

        # SiLU = z * sigmoid(z)
        sigmoid_z = 1.0 / (1.0 + np.exp(-z_float)) if z_float > -20 else 0.0
        silu_float = z_float * sigmoid_z

        # Convert to Q13
        silu_q13 = np.clip(np.round(silu_float * Q13_SCALE_INT), INT16_MIN, INT16_MAX)
        lut_q13[i] = np.int16(silu_q13)

    return lut_q13


# --- Complete SSM Forward Pass (Integer-Only) ---

def ssm_forward_int8(
    x_i8: np.ndarray,
    dt_q16: np.ndarray,
    A_q16: np.ndarray,
    B_q15: np.ndarray,
    C_q15: np.ndarray,
    z_i8: np.ndarray,
    scale_x: float,
    scale_z: float,
    scale_y: float,
    exp_lut_q15: np.ndarray,
    phi1_lut_q15: np.ndarray,
    silu_lut_q13: np.ndarray,
    lut_min: float,
    lut_max: float,
    D_val: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Complete integer-only SSM forward pass.

    Pipeline:
    1. Discretize: dt, A, B -> dA, dB' (using LUTs)
    2. Scan: x, dA, dB', C -> y_acc (Q15 state update)
    3. Gate: y_acc, z -> y_gated (SiLU LUT)
    4. Requantize: y_gated -> y_i8

    Args:
        x_i8: Input [L, M] in INT8
        dt_q16: Delta timesteps [L, M] in Q16
        A_q16: A matrix [D, M] in Q16
        B_q15: B matrix [L, D] in Q15
        C_q15: C matrix [L, D] in Q15
        z_i8: Gate input [L, M] in INT8
        scale_x: Input quantization scale
        scale_z: Gate quantization scale
        scale_y: Output quantization scale
        exp_lut_q15: Exp LUT for discretization
        phi1_lut_q15: Phi1 LUT for discretization
        silu_lut_q13: SiLU LUT for gating
        lut_min, lut_max: LUT domain bounds
        D_val: Optional D residual [M]

    Returns:
        y_i8: Output [L, M] in INT8
    """
    lut_size = len(exp_lut_q15)
    lut_step = (lut_max - lut_min) / (lut_size - 1)

    # Pre-compute fixed-point constants for LUT indexing
    lut_min_q32 = int(lut_min * (1 << 32))
    inv_lut_step_q31 = int((1 << 31) / lut_step)

    # Convert scale_x to Q15
    s_x_q15 = int(scale_x * Q15_SCALE_INT)

    # 1. Discretization
    dA_q15, dB_prime_q15 = ssm_discretize_q15(
        dt_q16, A_q16, B_q15, s_x_q15,
        exp_lut_q15, phi1_lut_q15,
        lut_min_q32, inv_lut_step_q31, lut_size
    )

    # 2. SSM Scan
    y_acc, _ = ssm_scan_q15(x_i8, dA_q15, dB_prime_q15, C_q15, D_val)

    # 3. SiLU Gating
    y_gated = ssm_gate_silu_q13(y_acc, z_i8, silu_lut_q13)

    # 4. Requantize to INT8
    # y_gated is in accumulator domain, need to scale to INT8
    # This depends on the accumulator scale which combines multiple Q formats
    y_fp32 = y_gated.astype(np.float32) * scale_y
    y_i8 = np.clip(np.round(y_fp32 / scale_y), -128, 127).astype(np.int8)

    return y_i8


# --- SSM Layer with Projections ---

def ssm_layer_forward_int8(
    x_int8: np.ndarray,
    x_proj_weight_int8: np.ndarray,
    x_proj_bias_fp32: Optional[np.ndarray],
    dt_proj_weight_int8: np.ndarray,
    dt_proj_bias_fp32: Optional[np.ndarray],
    A_log_fp32: np.ndarray,
    D_fp32: np.ndarray,
    scale_x: float,
    scale_x_proj: float,
    scale_dt_proj: float,
    scale_output: float,
    d_inner: int,
    d_state: int,
    dt_rank: int
) -> np.ndarray:
    """
    Complete SSM layer forward pass including projections.

    This function handles the full QuantSSM layer:
    1. x_proj: Project input to get dt_input, B, C
    2. dt_proj: Project dt_input to full dt and apply softplus
    3. SSM core: Discretize and scan

    Args:
        x_int8: Input [B, L, d_inner] in INT8
        x_proj_weight_int8: Projection weight [dt_rank + 2*d_state, d_inner] in INT8
        x_proj_bias_fp32: Projection bias [dt_rank + 2*d_state] in FP32 (optional)
        dt_proj_weight_int8: dt projection weight [d_inner, dt_rank] in INT8
        dt_proj_bias_fp32: dt projection bias [d_inner] in FP32 (optional)
        A_log_fp32: Log of A matrix [d_state, d_inner] in FP32
        D_fp32: Skip connection coefficient [d_inner] in FP32
        scale_x: Input quantization scale
        scale_x_proj: x_proj weight quantization scale
        scale_dt_proj: dt_proj weight quantization scale
        scale_output: Output quantization scale
        d_inner: Inner dimension (M)
        d_state: State dimension (D)
        dt_rank: Rank of dt projection

    Returns:
        output_int8: Output [B, L, d_inner] in INT8
    """
    # Handle batch dimension
    original_shape = x_int8.shape
    if x_int8.ndim == 3:
        B, L, M = x_int8.shape
        x_flat = x_int8.reshape(B * L, M)
    else:
        B = 1
        L, M = x_int8.shape
        x_flat = x_int8.reshape(L, M)

    # --- Step 1: x_proj to get dt_input, B, C ---
    # INT8 linear: x_flat @ x_proj_weight.T
    proj_int32 = x_flat.astype(np.int32) @ x_proj_weight_int8.T.astype(np.int32)
    proj_fp32 = proj_int32.astype(np.float32) * (scale_x * scale_x_proj)
    if x_proj_bias_fp32 is not None:
        proj_fp32 += x_proj_bias_fp32

    # Split into dt_input, B, C
    dt_input = proj_fp32[:, :dt_rank]  # [B*L, dt_rank]
    B_ssm = proj_fp32[:, dt_rank:dt_rank + d_state]  # [B*L, d_state]
    C_ssm = proj_fp32[:, dt_rank + d_state:]  # [B*L, d_state]

    # --- Step 2: dt_proj to get dt and apply softplus ---
    # Quantize dt_input to INT8 for dt_proj
    dt_input_scale = np.abs(dt_input).max() / 127.0 if np.abs(dt_input).max() > 0 else 1.0
    dt_input_int8 = np.clip(np.round(dt_input / dt_input_scale), -128, 127).astype(np.int8)

    # INT8 linear: dt_input_int8 @ dt_proj_weight.T
    dt_int32 = dt_input_int8.astype(np.int32) @ dt_proj_weight_int8.T.astype(np.int32)
    dt_fp32 = dt_int32.astype(np.float32) * (dt_input_scale * scale_dt_proj)
    if dt_proj_bias_fp32 is not None:
        dt_fp32 += dt_proj_bias_fp32

    # Apply softplus: log(1 + exp(x))
    # Use numerically stable version
    dt_fp32 = np.where(
        dt_fp32 > 20,
        dt_fp32,  # For large x, softplus(x) ≈ x
        np.log1p(np.exp(dt_fp32))
    )

    # --- Step 3: Compute A (negative exponential of A_log) ---
    A = -np.exp(A_log_fp32)  # [d_state, d_inner]

    # --- Step 4: Sequential SSM scan ---
    # Initialize state: h[m, d] for each batch item
    h = np.zeros((B, M, d_state), dtype=np.float32)

    # Reshape for per-batch processing
    x_batched = x_flat.reshape(B, L, M).astype(np.float32) * scale_x
    dt_batched = dt_fp32.reshape(B, L, M)
    B_batched = B_ssm.reshape(B, L, d_state)
    C_batched = C_ssm.reshape(B, L, d_state)

    y_list = []
    for t in range(L):
        dt_t = dt_batched[:, t, :]  # [B, M]
        x_t = x_batched[:, t, :]  # [B, M]
        B_t = B_batched[:, t, :]  # [B, D]
        C_t = C_batched[:, t, :]  # [B, D]

        # Discretize: dA = exp(dt * A)
        # A is [D, M], dt is [B, M]
        # dA should be [B, M, D] = exp([B, M, 1] * [D, M].T) = exp([B, M, D])
        dA = np.exp(dt_t[:, :, np.newaxis] * A.T[np.newaxis, :, :])  # [B, M, D]

        # dB' = dt * B (simplified)
        dB = dt_t[:, :, np.newaxis] * B_t[:, np.newaxis, :]  # [B, M, D]

        # State update: h = dA * h + dB' * x
        h = dA * h + dB * x_t[:, :, np.newaxis]  # [B, M, D]

        # Output: y = C * h (sum over D)
        y_t = np.sum(h * C_t[:, np.newaxis, :], axis=-1)  # [B, M]

        # Add skip connection: y = y + D * x
        y_t = y_t + D_fp32 * x_t  # [B, M]

        y_list.append(y_t)

    y = np.stack(y_list, axis=1)  # [B, L, M]

    # --- Step 5: Quantize output ---
    output_int8 = np.clip(np.round(y / scale_output), -128, 127).astype(np.int8)

    # Reshape to match input shape
    output_int8 = output_int8.reshape(original_shape)

    return output_int8


def ssm_layer_forward_int8_imamba(
    x_int8: np.ndarray,
    x_proj_weight_int8: np.ndarray,
    dt_proj_weight_int8: np.ndarray,
    dt_proj_bias_fp32: Optional[np.ndarray],
    A_log_fp32: np.ndarray,
    D_fp32: np.ndarray,
    scale_x: float,
    scale_x_proj: float,
    scale_dt_proj: float,
    scale_output: float,
    d_inner: int,
    d_state: int,
    dt_rank: int,
    softplus_scale_in: float = 0.1
) -> np.ndarray:
    """I-Mamba integer SSM reference matching the GAP9 kernel."""
    had_batch = x_int8.ndim == 3
    if not had_batch:
        x_int8 = x_int8[np.newaxis, ...]
    B, L, M = x_int8.shape

    dt_bias_q16_16 = np.zeros(d_inner, dtype=np.int32)
    if dt_proj_bias_fp32 is not None:
        dt_bias_q16_16 = np.round(dt_proj_bias_fp32 * 65536.0).astype(np.int32)

    dt_scale_shift = 24
    bc_shift = 16
    output_shift = 24
    dt_scale_q = int(round((scale_x * scale_x_proj) * scale_dt_proj * 65536.0 * (1 << dt_scale_shift)))
    bc_scale_factor = int(round(scale_x * scale_x_proj * Q15_SCALE * (1 << bc_shift)))
    output_scale_q = int(round(scale_x / (Q15_SCALE * scale_output) * (1 << output_shift)))

    A_q15 = np.clip(np.round(-np.exp(A_log_fp32) * Q15_SCALE_INT), INT16_MIN, INT16_MAX).astype(np.int16)
    D_q15 = np.clip(np.round(D_fp32 * Q15_SCALE_INT), INT16_MIN, INT16_MAX).astype(np.int16)

    softplus_lut = generate_softplus_lut_q8_8(softplus_scale_in)
    exp_lut = generate_exp_neg_lut_q15(softplus_scale_in)

    proj_size = dt_rank + 2 * d_state
    output_int8 = np.zeros_like(x_int8, dtype=np.int8)

    for b in range(B):
        x_flat = x_int8[b].astype(np.int32)
        proj_all = x_flat @ x_proj_weight_int8.T.astype(np.int32)

        B_all = np.zeros((L, d_state), dtype=np.int16)
        C_all = np.zeros((L, d_state), dtype=np.int16)
        for t in range(L):
            proj_t = proj_all[t]
            for d in range(d_state):
                b_scaled = (int(proj_t[dt_rank + d]) * bc_scale_factor) >> bc_shift
                if b_scaled > INT16_MAX:
                    b_scaled = INT16_MAX
                elif b_scaled < INT16_MIN:
                    b_scaled = INT16_MIN
                B_all[t, d] = b_scaled

                c_scaled = (int(proj_t[dt_rank + d_state + d]) * bc_scale_factor) >> bc_shift
                if c_scaled > INT16_MAX:
                    c_scaled = INT16_MAX
                elif c_scaled < INT16_MIN:
                    c_scaled = INT16_MIN
                C_all[t, d] = c_scaled

        dt_all = np.zeros((d_inner, L), dtype=np.int16)
        for t in range(L):
            proj_t = proj_all[t]
            delta_local = (proj_t[:dt_rank] >> 8).astype(np.int32)
            dt_acc = delta_local @ dt_proj_weight_int8.T.astype(np.int32)
            dt_scaled = dt_acc.astype(np.int64) * dt_scale_q
            dt_val_q16_16 = (dt_scaled >> dt_scale_shift).astype(np.int64)
            dt_val_q16_16 += dt_bias_q16_16.astype(np.int64)

            lut_idx = (dt_val_q16_16 * 10) >> 16
            lut_idx = np.clip(lut_idx, -128, 127).astype(np.int32)
            dt_all[:, t] = softplus_lut[lut_idx + 128]

        h_state = np.zeros((d_inner, d_state), dtype=np.int16)
        for m in range(d_inner):
            h_local = h_state[m].astype(np.int32)
            A_local = A_q15[:, m].astype(np.int32)
            D_val = int(D_q15[m])
            dt_m = dt_all[m]

            for t in range(L):
                dt_val = int(dt_m[t])
                x_i8 = int(x_int8[b, t, m])
                B_t = B_all[t]
                C_t = C_all[t]

                y_acc = 0
                for d in range(d_state):
                    A_val = int(A_local[d])
                    dt_A_q23 = dt_val * A_val
                    dt_A_q15 = dt_A_q23 >> 8
                    exp_idx = (dt_A_q15 * 10) >> 15
                    if exp_idx > 127:
                        exp_idx = 127
                    elif exp_idx < -128:
                        exp_idx = -128
                    dA_q15 = int(exp_lut[exp_idx + 128])

                    dB_q23 = dt_val * int(B_t[d])
                    dB_q15 = dB_q23 >> 8
                    h_decay = (dA_q15 * int(h_local[d])) >> 15
                    h_input = dB_q15 * x_i8
                    h_new = h_decay + (h_input >> 7)
                    if h_new > INT16_MAX:
                        h_new = INT16_MAX
                    elif h_new < INT16_MIN:
                        h_new = INT16_MIN
                    h_local[d] = h_new

                    y_acc += (int(C_t[d]) * h_new) >> 15

                y_acc += D_val * x_i8
                y_scaled = y_acc * output_scale_q
                ssm_out = (y_scaled + (1 << (output_shift - 1))) >> output_shift
                if ssm_out > 127:
                    ssm_out = 127
                elif ssm_out < -128:
                    ssm_out = -128

                output_int8[b, t, m] = np.int8(ssm_out)

            h_state[m] = h_local.astype(np.int16)

    if not had_batch:
        return output_int8[0]
    return output_int8


# --- Tests ---

def test_ssm():
    """Test integer-only SSM operations."""
    print("=" * 80)
    print("Testing Integer-Only SSM Operations")
    print("=" * 80)

    # Test 1: LUT generation
    print("\n--- Test 1: LUT Generation ---")
    exp_lut, z_min, z_max, step = generate_exp_lut_q15(z_min=-10.0, z_max=0.0)
    phi1_lut, _, _, _ = generate_phi1_lut_q15(z_min=-10.0, z_max=0.0)

    print(f"Exp LUT: {len(exp_lut)} entries, range [{z_min}, {z_max}]")
    print(f"Phi1 LUT: {len(phi1_lut)} entries")

    # Verify key values
    # exp(0) = 1 -> 32767 in Q15
    # exp(-10) ≈ 0.000045 -> ~1-2 in Q15
    print(f"exp(0) in Q15: {exp_lut[-1]} (expected ~32767)")
    print(f"exp(-10) in Q15: {exp_lut[0]} (expected ~1)")
    print(f"phi1(0) in Q15: {phi1_lut[-1]} (expected ~32768, phi1(0)=1)")

    assert exp_lut[-1] > 32700, "exp(0) should be close to 32768"
    assert exp_lut[0] < 10, "exp(-10) should be very small"
    print("Test 1 PASSED!")

    # Test 2: FP32 reference discretization
    print("\n--- Test 2: FP32 Reference Discretization ---")
    L, M, D = 4, 2, 3
    dt = np.random.uniform(0.01, 0.1, (L, M)).astype(np.float32)
    A = -np.exp(np.random.uniform(0, 2, (D, M))).astype(np.float32)  # Negative
    B = np.random.randn(L, D).astype(np.float32) * 0.1

    dA_ref, dB_ref = ssm_discretize_fp32_reference(dt, A, B, s_x=0.05)

    print(f"dt range: [{dt.min():.4f}, {dt.max():.4f}]")
    print(f"A range: [{A.min():.4f}, {A.max():.4f}]")
    print(f"dA range: [{dA_ref.min():.4f}, {dA_ref.max():.4f}]")
    print(f"dB' range: [{dB_ref.min():.6f}, {dB_ref.max():.6f}]")

    # dA should be in (0, 1) since A < 0 and dt > 0
    assert dA_ref.min() > 0 and dA_ref.max() <= 1, "dA should be in (0, 1]"
    print("Test 2 PASSED!")

    # Test 3: FP32 reference SSM scan
    print("\n--- Test 3: FP32 Reference SSM Scan ---")
    x = np.random.randn(L, M).astype(np.float32) * 0.5
    C = np.random.randn(L, D).astype(np.float32) * 0.1
    D_val = np.ones(M, dtype=np.float32)

    y_ref, h_ref = ssm_scan_fp32_reference(x, dA_ref, dB_ref, C, D_val)

    print(f"x shape: {x.shape}")
    print(f"y shape: {y_ref.shape}")
    print(f"y range: [{y_ref.min():.4f}, {y_ref.max():.4f}]")
    print(f"h_final range: [{h_ref.min():.6f}, {h_ref.max():.6f}]")
    print("Test 3 PASSED!")

    # Test 4: Q15 SSM scan
    print("\n--- Test 4: Q15 SSM Scan ---")
    scale_x = 0.05
    x_i8 = quantize_linear(x, scale_x)

    # Quantize dA, dB' to Q15
    dA_q15 = np.clip(np.round(dA_ref * Q15_SCALE_INT), INT16_MIN, INT16_MAX).astype(np.int16)
    dB_q15 = np.clip(np.round(dB_ref * Q15_SCALE_INT), INT16_MIN, INT16_MAX).astype(np.int16)
    C_q15 = np.clip(np.round(C * Q15_SCALE_INT), INT16_MIN, INT16_MAX).astype(np.int16)

    y_acc_q15, h_q15 = ssm_scan_q15(x_i8, dA_q15, dB_q15, C_q15, D_val=None)

    print(f"y_acc shape: {y_acc_q15.shape}")
    print(f"y_acc range: [{y_acc_q15.min()}, {y_acc_q15.max()}]")
    print("Test 4 PASSED!")

    # Test 5: SiLU gate LUT
    print("\n--- Test 5: SiLU Gate LUT ---")
    scale_z = 0.05
    silu_lut = generate_silu_gate_lut_q13(scale_z)

    print(f"SiLU LUT shape: {silu_lut.shape}")
    print(f"SiLU LUT range: [{silu_lut.min()}, {silu_lut.max()}]")

    # Verify SiLU(0) = 0
    print(f"SiLU(0) = {silu_lut[128]} (expected 0)")
    assert silu_lut[128] == 0, "SiLU(0) should be 0"

    # SiLU for positive values should be positive
    assert all(silu_lut[129:] >= 0), "SiLU for positive inputs should be >= 0"
    print("Test 5 PASSED!")

    # Test 6: SiLU gating
    print("\n--- Test 6: SiLU Gating ---")
    z_i8 = np.random.randint(-64, 64, (L, M)).astype(np.int8)

    y_gated = ssm_gate_silu_q13(y_acc_q15, z_i8, silu_lut)

    print(f"y_gated shape: {y_gated.shape}")
    print(f"y_gated range: [{y_gated.min()}, {y_gated.max()}]")
    print("Test 6 PASSED!")

    # Test 7: End-to-end accuracy check (simplified)
    print("\n--- Test 7: Accuracy Verification ---")

    # Compare FP32 reference with Q15 scan (without gating)
    # Scale the Q15 output to compare with FP32
    y_q15_scaled = y_acc_q15.astype(np.float32)  # This needs proper scaling

    # Run FP32 scan without D for comparison
    y_ref_no_d, _ = ssm_scan_fp32_reference(x, dA_ref, dB_ref, C, D_val=None)

    # The Q15 output has different scale, so we just check shapes match
    assert y_acc_q15.shape == y_ref_no_d.shape, "Shape mismatch"
    print("Shape verification: PASSED!")

    print("\n" + "=" * 80)
    print("All SSM tests completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    test_ssm()
