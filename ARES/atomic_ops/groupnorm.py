# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""
Atomic GroupNorm Operation - Integer-only Implementation

Implements Group Normalization using integer arithmetic for bit-exact
matching with C/hardware implementations.

Uses:
- INT64 accumulation for mean/variance computation
- C-style truncated division (rounds toward zero)
- Integer square root (I-BERT Algorithm 4 or binary search)

Reference:
    "I-BERT: Integer-only BERT Quantization" (Kim et al., 2021)
"""

import numpy as np

try:
    from .layernorm import sqrt_q64, i_sqrt_newton
except ImportError:
    from layernorm import sqrt_q64, i_sqrt_newton


def _trunc_div_s64(num: np.int64, den: np.int64) -> np.int64:
    """
    C-style truncated integer division (rounds toward zero).

    Python's // performs floor division (rounds toward -inf).
    C's / performs truncated division (rounds toward 0).

    Example:
        -10 // 3 = -4 (Python floor)
        -10 / 3 = -3 (C truncate)

    Args:
        num: Dividend (INT64)
        den: Divisor (INT64)

    Returns:
        Truncated division result
    """
    if den == 0:
        raise ZeroDivisionError("division by zero")
    if num >= 0:
        return num // den
    return -((-num) // den)


def groupnorm_int8_fixed_point(
    input_int8: np.ndarray,
    weight_fp32: np.ndarray | None,
    bias_fp32: np.ndarray | None,
    scale_input: float,
    scale_output: float,
    num_groups: int,
) -> np.ndarray:
    """
    INT8 GroupNorm with fixed-point mean/variance (bit-exact friendly with C).

    Normalizes per (batch, group) across:
      - channels_per_group channels
      - all spatial dimensions

    Uses:
      - INT64 accumulation for mean/variance (in INT8 domain)
      - integer division (truncate toward zero for mean)
      - integer epsilon = 1 and integer sqrt via sqrt_q64
      - FP32 affine (gamma/beta) and requantization to INT8
    """
    x = np.asarray(input_int8, dtype=np.int8)
    if x.ndim < 2:
        raise ValueError(f"GroupNorm expects at least 2D [B,C,...], got shape {x.shape}")

    batch = int(x.shape[0])
    channels = int(x.shape[1])
    if num_groups <= 0:
        raise ValueError(f"GroupNorm num_groups must be > 0, got {num_groups}")
    if channels % num_groups != 0:
        raise ValueError(f"GroupNorm channels ({channels}) must be divisible by num_groups ({num_groups})")

    spatial_size = int(np.prod(x.shape[2:])) if x.ndim > 2 else 1
    channels_per_group = channels // num_groups
    group_elems = np.int64(channels_per_group * spatial_size)

    w = None if weight_fp32 is None else np.asarray(weight_fp32, dtype=np.float32)
    b = None if bias_fp32 is None else np.asarray(bias_fp32, dtype=np.float32)
    if w is None:
        w = np.ones((channels,), dtype=np.float32)
    if b is None:
        b = np.zeros((channels,), dtype=np.float32)
    if w.shape[0] != channels or b.shape[0] != channels:
        raise ValueError(f"GroupNorm weight/bias must be length C={channels}, got w={w.shape}, b={b.shape}")

    x_reshaped = x.reshape(batch, channels, spatial_size)
    out_fp32 = np.zeros((batch, channels, spatial_size), dtype=np.float32)

    scale_in_f = np.float32(scale_input)
    scale_out_f = np.float32(scale_output)

    for bs in range(batch):
        for g in range(num_groups):
            c0 = g * channels_per_group
            c1 = c0 + channels_per_group

            # Compute mean in INT8 domain (INT64 sum, trunc toward 0)
            sum_val = np.int64(0)
            sumsq_val = np.int64(0)
            for c in range(c0, c1):
                for s in range(spatial_size):
                    v = np.int64(x_reshaped[bs, c, s])
                    sum_val += v
                    sumsq_val += v * v

            mean = _trunc_div_s64(sum_val, group_elems)

            # Variance: sum((x-mean)^2) / N (integer), epsilon=1
            var_sum = (
                sumsq_val
                - np.int64(2) * mean * sum_val
                + group_elems * mean * mean
            )
            variance = var_sum // group_elems
            variance += 1
            std = np.int64(sqrt_q64(variance, frac_bits=0))
            if std <= 0:
                std = np.int64(1)

            inv_std_f = np.float32(1.0) / np.float32(std)

            # Normalize + affine per channel
            for c in range(c0, c1):
                gamma = np.float32(w[c])
                beta = np.float32(b[c])
                for s in range(spatial_size):
                    x_centered = np.int64(x_reshaped[bs, c, s]) - mean
                    x_norm = np.float32(x_centered) * inv_std_f
                    x_norm = x_norm * scale_in_f
                    out_fp32[bs, c, s] = gamma * x_norm + beta

    out_fp32 = out_fp32.reshape(x.shape)
    out_int8 = np.clip(np.round(out_fp32 / scale_out_f), -128, 127).astype(np.int8)
    return out_int8


def test_groupnorm_constant_zero():
    x = np.full((1, 8, 2, 2), 5, dtype=np.int8)
    w = np.ones((8,), dtype=np.float32)
    b = np.zeros((8,), dtype=np.float32)
    y = groupnorm_int8_fixed_point(x, w, b, scale_input=0.1, scale_output=0.1, num_groups=4)
    assert y.shape == x.shape
    assert np.all(y == 0), "Constant input should normalize to zeros (gamma=1,beta=0)"


def test_groupnorm_shape_and_dtype():
    rng = np.random.default_rng(0)
    x = rng.integers(-128, 128, size=(1, 12, 5), dtype=np.int16).astype(np.int8)
    w = rng.standard_normal(12, dtype=np.float32)
    b = rng.standard_normal(12, dtype=np.float32)
    y = groupnorm_int8_fixed_point(x, w, b, scale_input=0.02, scale_output=0.03, num_groups=3)
    assert y.shape == x.shape
    assert y.dtype == np.int8


def test_groupnorm_invalid_groups_raises():
    x = np.zeros((1, 10, 4), dtype=np.int8)
    try:
        _ = groupnorm_int8_fixed_point(x, None, None, 0.1, 0.1, num_groups=4)
    except ValueError:
        return
    raise AssertionError("Expected ValueError when channels not divisible by num_groups")

