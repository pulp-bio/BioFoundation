# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Transpose (Permute) operation for INT8 tensors.

This module implements transpose/permute operations that physically rearrange
data in memory according to dimension permutation.
"""

import numpy as np
from typing import List, Tuple


def transpose_int8(
    x_int8: np.ndarray,
    dims: List[int]
) -> np.ndarray:
    """
    Transpose (permute) INT8 tensor according to dimension specification.

    This is a physical data movement operation, not a view. In PyTorch/NumPy,
    permute() returns a view, but in C with flat memory layout, we must
    actually copy and rearrange the data.

    Args:
        x_int8: Input INT8 tensor of shape (d0, d1, ..., dn)
        dims: Permutation of dimensions. E.g., [0, 2, 1] swaps last two dims

    Returns:
        Transposed INT8 tensor with permuted dimensions

    Example:
        Input shape: (1, 64, 196), dims=[0, 2, 1]
        Output shape: (1, 196, 64)

        Output[b, i, j] = Input[b, j, i]
    """
    # NumPy's transpose handles the permutation for us
    return np.transpose(x_int8, dims).astype(np.int8)


def transpose_2d_batch_int8(
    x_int8: np.ndarray,
    batch_size: int,
    dim1: int,
    dim2: int
) -> np.ndarray:
    """
    Specialized transpose for 3D tensors with fixed batch dimension.

    Transposes last two dimensions while keeping batch dimension fixed.
    Equivalent to dims=[0, 2, 1] permutation.

    Args:
        x_int8: Input INT8 tensor, shape (batch_size, dim1, dim2)
        batch_size: Size of batch dimension (dimension 0)
        dim1: Size of dimension 1 (will become dimension 2 after transpose)
        dim2: Size of dimension 2 (will become dimension 1 after transpose)

    Returns:
        Transposed INT8 tensor, shape (batch_size, dim2, dim1)

    Note:
        This is optimized for the common case of swapping sequence/channel dims
        in Transformer models.
    """
    assert x_int8.shape == (batch_size, dim1, dim2), \
        f"Shape mismatch: expected ({batch_size}, {dim1}, {dim2}), got {x_int8.shape}"

    # Reshape to (batch_size, dim1, dim2)
    x_reshaped = x_int8.reshape(batch_size, dim1, dim2)

    # Transpose last two dimensions
    x_transposed = np.transpose(x_reshaped, (0, 2, 1))

    # Result shape: (batch_size, dim2, dim1)
    assert x_transposed.shape == (batch_size, dim2, dim1)

    return x_transposed.astype(np.int8)


def test_transpose_int8():
    """Unit test for transpose_int8."""
    print("Testing transpose_int8...")

    # Test 1: Simple 3D transpose [0, 2, 1]
    x = np.arange(24, dtype=np.int8).reshape(2, 3, 4)
    y = transpose_int8(x, [0, 2, 1])

    assert y.shape == (2, 4, 3), f"Shape mismatch: {y.shape}"
    assert y.dtype == np.int8, f"Dtype mismatch: {y.dtype}"

    # Verify correctness
    for b in range(2):
        for i in range(4):
            for j in range(3):
                expected = x[b, j, i]
                actual = y[b, i, j]
                assert expected == actual, \
                    f"Mismatch at [{b},{i},{j}]: expected {expected}, got {actual}"

    print("  [OK] Test 1 passed: 3D transpose [0, 2, 1]")

    # Test 2: 2D transpose
    x2 = np.arange(12, dtype=np.int8).reshape(3, 4)
    y2 = transpose_int8(x2, [1, 0])

    assert y2.shape == (4, 3)
    assert np.array_equal(y2, x2.T)

    print("  [OK] Test 2 passed: 2D transpose")

    # Test 3: Identity permutation (no-op)
    x3 = np.arange(24, dtype=np.int8).reshape(2, 3, 4)
    y3 = transpose_int8(x3, [0, 1, 2])

    assert y3.shape == (2, 3, 4)
    assert np.array_equal(y3, x3)

    print("  [OK] Test 3 passed: Identity permutation")

    print("[PASS] All transpose_int8 tests passed!")


def test_transpose_2d_batch_int8():
    """Unit test for transpose_2d_batch_int8."""
    print("\nTesting transpose_2d_batch_int8...")

    # Test with Transformer-like dimensions: [1, 64, 196] -> [1, 196, 64]
    batch_size = 1
    dim1 = 64
    dim2 = 196

    x = np.arange(batch_size * dim1 * dim2, dtype=np.int8).reshape(batch_size, dim1, dim2)
    y = transpose_2d_batch_int8(x, batch_size, dim1, dim2)

    assert y.shape == (batch_size, dim2, dim1), f"Shape mismatch: {y.shape}"
    assert y.dtype == np.int8, f"Dtype mismatch: {y.dtype}"

    # Verify correctness
    for b in range(batch_size):
        for i in range(dim2):
            for j in range(dim1):
                expected = x[b, j, i]
                actual = y[b, i, j]
                assert expected == actual, \
                    f"Mismatch at [{b},{i},{j}]: expected {expected}, got {actual}"

    print("  [OK] Test passed: [1, 64, 196] -> [1, 196, 64] transpose")

    # Test with larger batch
    batch_size2 = 4
    dim1_2 = 8
    dim2_2 = 12

    x2 = np.random.randint(-128, 127, size=(batch_size2, dim1_2, dim2_2), dtype=np.int8)
    y2 = transpose_2d_batch_int8(x2, batch_size2, dim1_2, dim2_2)

    assert y2.shape == (batch_size2, dim2_2, dim1_2)

    # Verify with NumPy transpose
    expected = np.transpose(x2, (0, 2, 1))
    assert np.array_equal(y2, expected)

    print("  [OK] Test passed: Batch transpose with random data")

    print("[PASS] All transpose_2d_batch_int8 tests passed!")


if __name__ == "__main__":
    test_transpose_int8()
    test_transpose_2d_batch_int8()
    print("\n[PASS] All tests passed!")
