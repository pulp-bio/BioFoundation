# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Tuple

import numpy as np

Q15_SCALE = 32768


def _q15_from_float(x: np.ndarray) -> np.ndarray:
    """Convert float array in [-1, 1] to Q15 int16 with 1.0 mapped to 32767."""
    q = np.round(x.astype(np.float64) * Q15_SCALE).astype(np.int64)
    q = np.clip(q, -Q15_SCALE, Q15_SCALE - 1)
    return q.astype(np.int16)


def rope_precompute_sin_cos_q15(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute RoPE sin/cos tables in Q15.

    Tables match the standard RoPE formulation:
      inv_freq[i] = 1 / base^( (2*i) / head_dim )
      angles[pos, i] = pos * inv_freq[i]

    Returns:
        (cos_q15, sin_q15) with shape [seq_len, head_dim//2] and dtype int16.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")
    if head_dim <= 0 or (head_dim % 2) != 0:
        raise ValueError(f"head_dim must be positive and even, got {head_dim}")

    inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float64) / np.float64(head_dim)))
    positions = np.arange(seq_len, dtype=np.float64)
    angles = np.outer(positions, inv_freq)  # [seq_len, head_dim/2]

    cos = np.cos(angles)
    sin = np.sin(angles)
    return _q15_from_float(cos), _q15_from_float(sin)


def _shift_round_nearest_even_int32(x: np.ndarray, shift: int) -> np.ndarray:
    """
    Divide by 2^shift with ties-to-even rounding (vectorized).

    Matches the "nearest-even" behavior used in GAP9 kernels.
    """
    if shift <= 0:
        return x.astype(np.int32, copy=False)

    x64 = x.astype(np.int64, copy=False)
    sign = np.where(x64 < 0, -1, 1).astype(np.int64)
    ax = np.abs(x64)

    q = ax >> shift
    r = ax & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    round_up = (r > half) | ((r == half) & ((q & 1) == 1))
    q_rounded = q + round_up.astype(np.int64)
    return (q_rounded * sign).astype(np.int32)


def rope_apply_int8_q15(
    x_int8: np.ndarray,
    cos_q15: np.ndarray,
    sin_q15: np.ndarray,
    pos_offset: int = 0,
) -> np.ndarray:
    """
    Apply RoPE rotation in INT8 domain using Q15 sin/cos tables.

    Input layout matches attention internal layout: [B, H, N, D].
    The output uses the same quantization scale as the input (rotation is orthonormal).
    """
    x = np.asarray(x_int8, dtype=np.int8)
    if x.ndim != 4:
        raise ValueError(f"Expected x_int8 shape [B, H, N, D], got {x.shape}")
    head_dim = int(x.shape[-1])
    if (head_dim % 2) != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")

    seq_len = int(x.shape[-2])
    half = head_dim // 2
    if pos_offset < 0:
        raise ValueError(f"pos_offset must be >= 0, got {pos_offset}")

    cos = np.asarray(cos_q15, dtype=np.int16)
    sin = np.asarray(sin_q15, dtype=np.int16)
    if cos.shape != sin.shape:
        raise ValueError(f"cos_q15 and sin_q15 shapes must match, got {cos.shape} vs {sin.shape}")
    if cos.ndim != 2 or cos.shape[1] != half:
        raise ValueError(f"Expected cos/sin shape [N_total, D/2]=[*, {half}], got {cos.shape}")
    if cos.shape[0] < pos_offset + seq_len:
        raise ValueError(f"cos/sin tables too short for pos_offset={pos_offset}, seq_len={seq_len} (have {cos.shape[0]})")

    # [B, H, N, D/2]
    x_even = x[..., 0::2].astype(np.int32, copy=False)
    x_odd = x[..., 1::2].astype(np.int32, copy=False)

    cos_i32 = cos[pos_offset:pos_offset + seq_len].astype(np.int32, copy=False)[None, None, :, :]
    sin_i32 = sin[pos_offset:pos_offset + seq_len].astype(np.int32, copy=False)[None, None, :, :]

    # Numerators in Q15: (int8 * q15) ± (int8 * q15)
    even_num = x_even * cos_i32 - x_odd * sin_i32
    odd_num = x_even * sin_i32 + x_odd * cos_i32

    even = _shift_round_nearest_even_int32(even_num, 15)
    odd = _shift_round_nearest_even_int32(odd_num, 15)

    even = np.clip(even, -128, 127).astype(np.int8)
    odd = np.clip(odd, -128, 127).astype(np.int8)

    out = np.empty_like(x, dtype=np.int8)
    out[..., 0::2] = even
    out[..., 1::2] = odd
    return out


def test_rope_zero():
    cos, sin = rope_precompute_sin_cos_q15(seq_len=8, head_dim=32)
    x = np.zeros((1, 2, 8, 32), dtype=np.int8)
    y = rope_apply_int8_q15(x, cos, sin)
    assert y.shape == x.shape
    assert np.all(y == 0)


def test_rope_pos0_identity():
    cos, sin = rope_precompute_sin_cos_q15(seq_len=4, head_dim=8)
    x = np.random.randint(-128, 128, size=(1, 3, 4, 8), dtype=np.int8)
    y = rope_apply_int8_q15(x, cos, sin)
    assert np.all(y[:, :, 0, :] == x[:, :, 0, :])

