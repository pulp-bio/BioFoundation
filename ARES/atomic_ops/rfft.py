# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


# LUNA uses patch_size=40 (fixed, small). We implement a fixed-point DFT for this case.
RFFT_SUPPORTED_PATCH_SIZE = 40
RFFT_NUM_BINS_40 = RFFT_SUPPORTED_PATCH_SIZE // 2 + 1  # 21
Q15_SCALE = 32768

# atan LUT: atan(z)/pi for z in [0, 1], stored in Q15.
ATAN_LUT_SIZE = 1024


def _q15_from_float(x: np.ndarray) -> np.ndarray:
    """Convert float array in [-1,1] to Q15 int16 with 1.0 mapped to 32767."""
    q = np.round(x.astype(np.float64) * Q15_SCALE).astype(np.int64)
    q = np.clip(q, -Q15_SCALE, Q15_SCALE - 1)
    return q.astype(np.int16)


def _mul_shift_round_nearest_even_s32(val: int, mul: int, shift: int) -> int:
    """Match `mul_shift_round_nearest_even` from `network_kernels.h`."""
    prod = int(val) * int(mul)
    if shift <= 0:
        return int(prod)

    abs_prod = prod if prod >= 0 else -prod
    q = abs_prod >> shift
    mask = (1 << shift) - 1
    r = abs_prod & mask
    half = 1 << (shift - 1)

    q_rounded = q
    if r > half or (r == half and (q & 1)):
        q_rounded = q + 1
    return int(q_rounded if prod >= 0 else -q_rounded)


# Base twiddle steps for N=40 (cos/sin of 2*pi*k/N), in Q15.
_RFFT40_COS_STEP_Q15 = np.array(
    [
        32767, 32365, 31164, 29197, 26510, 23170, 19261, 14876, 10126, 5126, 0,
        -5126, -10126, -14876, -19261, -23170, -26510, -29197, -31164, -32365, -32768,
    ],
    dtype=np.int32,
)
_RFFT40_SIN_STEP_Q15 = np.array(
    [
        0, 5126, 10126, 14876, 19261, 23170, 26510, 29197, 31164, 32365, 32767,
        32365, 31164, 29197, 26510, 23170, 19261, 14876, 10126, 5126, 0,
    ],
    dtype=np.int32,
)


def _build_rfft40_trig_tables_q15() -> Tuple[np.ndarray, np.ndarray]:
    """
    Build trig tables using deterministic Q15 recurrence (no libm dependency).

    This matches the on-the-fly recurrence used by the GAP9 kernel.
    """
    cos_q15 = np.zeros((RFFT_NUM_BINS_40, RFFT_SUPPORTED_PATCH_SIZE), dtype=np.int16)
    sin_q15 = np.zeros((RFFT_NUM_BINS_40, RFFT_SUPPORTED_PATCH_SIZE), dtype=np.int16)

    for k in range(RFFT_NUM_BINS_40):
        cos_step = int(_RFFT40_COS_STEP_Q15[k])
        sin_step = int(_RFFT40_SIN_STEP_Q15[k])
        c = 32767  # cos(0) in Q15
        s = 0      # sin(0) in Q15
        for n in range(RFFT_SUPPORTED_PATCH_SIZE):
            cos_q15[k, n] = np.int16(c)
            sin_q15[k, n] = np.int16(s)

            cn = _mul_shift_round_nearest_even_s32(c, cos_step, 15) - _mul_shift_round_nearest_even_s32(s, sin_step, 15)
            sn = _mul_shift_round_nearest_even_s32(s, cos_step, 15) + _mul_shift_round_nearest_even_s32(c, sin_step, 15)
            if cn < -Q15_SCALE:
                cn = -Q15_SCALE
            if cn > Q15_SCALE - 1:
                cn = Q15_SCALE - 1
            if sn < -Q15_SCALE:
                sn = -Q15_SCALE
            if sn > Q15_SCALE - 1:
                sn = Q15_SCALE - 1
            c, s = cn, sn

    return cos_q15, sin_q15


def _build_atan_lut_q15() -> np.ndarray:
    z = np.linspace(0.0, 1.0, ATAN_LUT_SIZE, dtype=np.float64)
    atan_norm = np.arctan(z) / np.pi  # in [0, 0.25]
    lut = np.round(atan_norm * Q15_SCALE).astype(np.int64)
    lut = np.clip(lut, 0, Q15_SCALE - 1)
    return lut.astype(np.int16)


_RFFT40_COS_Q15, _RFFT40_SIN_Q15 = _build_rfft40_trig_tables_q15()
_ATAN_LUT_Q15 = _build_atan_lut_q15()


def _atan_lut_lookup_q15(ratio_q15: int) -> int:
    # ratio_q15 in [0, 32767]
    if ratio_q15 <= 0:
        return 0
    if ratio_q15 >= Q15_SCALE - 1:
        return int(_ATAN_LUT_Q15[-1])
    idx = (ratio_q15 * (ATAN_LUT_SIZE - 1)) // (Q15_SCALE - 1)
    return int(_ATAN_LUT_Q15[int(idx)])


def atan2_pi_q15(y: int, x: int) -> int:
    """
    Approx atan2(y, x) returning angle/pi in Q15 ([-32768, 32767]).

    Uses a LUT for atan(z)/pi with z in [0,1] and quadrant correction.
    """
    if x == 0 and y == 0:
        return 0

    ax = -x if x < 0 else x
    ay = -y if y < 0 else y

    if ax >= ay:
        ratio_q15 = (ay << 15) // ax if ax != 0 else (Q15_SCALE - 1)
        base = _atan_lut_lookup_q15(int(ratio_q15))
        angle = int(base)
    else:
        ratio_q15 = (ax << 15) // ay if ay != 0 else (Q15_SCALE - 1)
        base = _atan_lut_lookup_q15(int(ratio_q15))
        angle = int((Q15_SCALE // 2) - base)  # pi/2 => 0.5 in units of pi

    # Quadrant correction (angle is in [0, 0.5] in Q15)
    if x >= 0:
        angle = angle if y >= 0 else -angle
    else:
        angle = (Q15_SCALE - angle) if y >= 0 else (angle - Q15_SCALE)

    if angle < -Q15_SCALE:
        angle = -Q15_SCALE
    if angle > Q15_SCALE - 1:
        angle = Q15_SCALE - 1
    return int(angle)


def rfft40_features_int8_fixed_point(
    input_int8: np.ndarray,
    scale_input: float,
    scale_output: float,
) -> np.ndarray:
    """
    Fixed-point RFFT (N=40) producing concatenated [magnitude, phase] features.

    Args:
        input_int8: (..., 40) INT8 patches
        scale_input: input quant scale (x_fp32 = x_int8 * scale_input)
        scale_output: output scale for BOTH magnitude and phase features

    Returns:
        output_int8: (..., 42) where 42 = 2*(40//2+1)
    """
    x = np.asarray(input_int8, dtype=np.int8)
    if x.shape[-1] != RFFT_SUPPORTED_PATCH_SIZE:
        raise ValueError(f"Only patch_size={RFFT_SUPPORTED_PATCH_SIZE} supported, got {x.shape[-1]}")
    if scale_output == 0.0:
        raise ValueError("scale_output must be non-zero")

    leading_shape = x.shape[:-1]
    num_patches = int(np.prod(leading_shape)) if leading_shape else 1
    x2 = x.reshape(num_patches, RFFT_SUPPORTED_PATCH_SIZE).astype(np.int32, copy=False)

    # DFT accumulators in INT32/INT64: sum_n x[n] * trig_q15[k,n]
    cos = _RFFT40_COS_Q15.astype(np.int32, copy=False)  # [K, N]
    sin = _RFFT40_SIN_Q15.astype(np.int32, copy=False)

    real_acc = (x2 @ cos.T).astype(np.int64, copy=False)  # [P, K]
    imag_acc = (-(x2 @ sin.T)).astype(np.int64, copy=False)

    mag_int8 = np.empty_like(real_acc, dtype=np.int8)
    phase_int8 = np.empty_like(real_acc, dtype=np.int8)

    scale_in_f = np.float32(scale_input)
    scale_out_f = np.float32(scale_output)
    mag_mul = scale_in_f / np.float32(Q15_SCALE)  # converts Q15 sums to float magnitude
    phase_mul = np.float32(math.pi) / np.float32(Q15_SCALE)

    for p in range(num_patches):
        for k in range(RFFT_NUM_BINS_40):
            re = int(real_acc[p, k])
            im = int(imag_acc[p, k])

            mag_sq = (re * re) + (im * im)
            mag_acc = math.isqrt(mag_sq)  # floor sqrt, deterministic

            mag_fp32 = np.float32(mag_acc) * mag_mul  # = mag_acc * scale_in / 32768
            q_mag = int(np.round(mag_fp32 / scale_out_f))
            if q_mag > 127:
                q_mag = 127
            if q_mag < -128:
                q_mag = -128
            mag_int8[p, k] = np.int8(q_mag)

            angle_q15 = atan2_pi_q15(im, re)
            phase_fp32 = np.float32(angle_q15) * phase_mul  # radians
            q_phase = int(np.round(phase_fp32 / scale_out_f))
            if q_phase > 127:
                q_phase = 127
            if q_phase < -128:
                q_phase = -128
            phase_int8[p, k] = np.int8(q_phase)

    out = np.concatenate([mag_int8, phase_int8], axis=-1)
    out = out.reshape(tuple(leading_shape) + (2 * RFFT_NUM_BINS_40,))
    return out


def test_rfft40_zero():
    x = np.zeros((1, 1, 1, RFFT_SUPPORTED_PATCH_SIZE), dtype=np.int8)
    y = rfft40_features_int8_fixed_point(x, scale_input=0.1, scale_output=0.05)
    assert y.shape[-1] == 2 * RFFT_NUM_BINS_40
    assert np.all(y == 0)


def test_rfft40_impulse():
    # Impulse at n=0: all bins have the same real value, imag=0 -> phase=0.
    x = np.zeros((RFFT_SUPPORTED_PATCH_SIZE,), dtype=np.int8)
    x[0] = 10
    y = rfft40_features_int8_fixed_point(x, scale_input=0.1, scale_output=0.01)
    assert y.shape == (2 * RFFT_NUM_BINS_40,)
    mag = y[:RFFT_NUM_BINS_40]
    phase = y[RFFT_NUM_BINS_40:]
    assert np.all(phase == 0)
    # Magnitude should be constant across bins for an impulse at n=0.
    assert np.all(mag == mag[0])
