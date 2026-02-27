# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def embedding_int8(indices: np.ndarray, weight_int8: np.ndarray) -> np.ndarray:
    """
    INT8 embedding lookup (gather rows).

    Args:
        indices: Integer tensor of indices (any shape), dtype int*/uint*
        weight_int8: [vocab_size, embed_dim] INT8 embedding table

    Returns:
        output_int8: indices.shape + (embed_dim,) INT8 tensor
    """
    indices_arr = np.asarray(indices)
    if indices_arr.dtype.kind not in {"i", "u"}:
        raise TypeError(f"Embedding indices must be integer type, got {indices_arr.dtype}")

    weight = np.asarray(weight_int8, dtype=np.int8)
    if weight.ndim != 2:
        raise ValueError(f"Embedding weight must be 2D [vocab, dim], got shape {weight.shape}")

    vocab_size, embed_dim = weight.shape
    flat = indices_arr.reshape(-1)
    if flat.size:
        min_idx = int(flat.min())
        max_idx = int(flat.max())
        if min_idx < 0 or max_idx >= vocab_size:
            raise ValueError(f"Embedding index out of range [{min_idx}, {max_idx}] for vocab_size={vocab_size}")

    gathered = weight[flat.astype(np.int64, copy=False)]
    out_shape = tuple(indices_arr.shape) + (embed_dim,)
    return gathered.reshape(out_shape).astype(np.int8, copy=False)


def test_embedding_int8_basic():
    weight = np.arange(5 * 3, dtype=np.int8).reshape(5, 3)
    indices = np.array([0, 3, 1, 4], dtype=np.int32)
    out = embedding_int8(indices, weight)
    assert out.shape == (4, 3)
    assert np.array_equal(out, weight[indices])


def test_embedding_int8_nd_indices():
    weight = (np.arange(7 * 4, dtype=np.int8) - 13).reshape(7, 4)
    indices = np.array([[1, 2], [6, 0]], dtype=np.int64)
    out = embedding_int8(indices, weight)
    assert out.shape == (2, 2, 4)
    assert np.array_equal(out, weight[indices.reshape(-1)].reshape(2, 2, 4))


def test_embedding_int8_out_of_range_raises():
    weight = np.zeros((4, 8), dtype=np.int8)
    indices = np.array([0, 4], dtype=np.int32)
    try:
        _ = embedding_int8(indices, weight)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for out-of-range embedding index")

