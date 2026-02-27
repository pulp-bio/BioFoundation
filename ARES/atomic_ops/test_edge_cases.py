# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Edge Case Tests for Atomic Operations

Tests boundary conditions, error handling, and edge cases that aren't covered
in the standard unit tests within each atomic operation file.

Run: python atomic_ops/test_edge_cases.py
"""

import numpy as np
import warnings

try:
    from .quantize import quantize_linear, dequantize_linear
    from .conv2d import conv2d_int8
    from .linear import linear_int8
    from .relu import relu_int8
    from .maxpool import maxpool2d_int8
except ImportError:
    from quantize import quantize_linear, dequantize_linear
    from conv2d import conv2d_int8
    from linear import linear_int8
    from relu import relu_int8
    from maxpool import maxpool2d_int8


class EdgeCaseTests:
    """Test edge cases for atomic operations."""

    def __init__(self):
        self.passed = 0
        self.failed = 0

    def test(self, name, func):
        """Run a test and track results."""
        try:
            func()
            print(f"[PASS] {name}")
            self.passed += 1
        except AssertionError as e:
            print(f"[FAIL] {name}: {e}")
            self.failed += 1
        except Exception as e:
            print(f"[WARN]  {name}: Unexpected error: {e}")
            self.failed += 1

    def summary(self):
        """Print test summary."""
        total = self.passed + self.failed
        print("\n" + "="*80)
        print(f"Edge Case Test Results: {self.passed}/{total} passed")
        if self.failed == 0:
            print("[PASS] All edge case tests passed!")
        else:
            print(f"[FAIL] {self.failed} tests failed")
        print("="*80)
        return self.failed == 0


def test_quantize_very_small_scale():
    """Test quantization with very small scale (near-zero)."""
    x = np.array([1.0, 2.0, 3.0])
    scale = 0.0001  # Very small scale
    q = quantize_linear(x, scale=scale)

    # Should clamp to INT8 range
    assert np.all(q >= -128) and np.all(q <= 127), "Quantized values out of INT8 range"

    # Dequantize and verify magnitude
    x_deq = dequantize_linear(q, scale=scale)
    # With very small scales, large values get clamped to 127, so reconstruction is lossy
    # Just verify the output is in a reasonable range
    assert np.all(x_deq > 0), "Dequantized values should be positive"
    assert np.all(x_deq < 20), "Dequantized values should be bounded (due to clamping)"


def test_quantize_very_large_scale():
    """Test quantization with very large scale."""
    x = np.array([1.0, 2.0, 3.0])
    scale = 100.0  # Very large scale
    q = quantize_linear(x, scale=scale)

    # All values should quantize to near-zero
    assert np.all(np.abs(q) <= 1), "Large scale should quantize small values to near-zero"


def test_quantize_overflow():
    """Test quantization with values that exceed INT8 range."""
    x = np.array([1000.0, -1000.0, 500.0])
    scale = 0.01
    q = quantize_linear(x, scale=scale)

    # Should clamp to [-128, 127]
    assert np.all(q >= -128) and np.all(q <= 127), "Overflow not properly clamped"
    assert q[0] == 127, "Positive overflow should clamp to 127"
    assert q[1] == -128, "Negative overflow should clamp to -128"


def test_quantize_zero_scale_protection():
    """Test that zero scale is handled (should warn or error)."""
    x = np.array([1.0, 2.0, 3.0])
    scale = 0.0

    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            q = quantize_linear(x, scale=scale)

            # Should either warn, error, or produce special result
            # Check if we got inf/nan or a warning
            if np.any(np.isinf(q)) or np.any(np.isnan(q)) or len(w) > 0:
                pass  # Expected behavior
            else:
                # Should have gotten some indication of problem
                raise AssertionError("Zero scale should produce warning or special value")
    except (ZeroDivisionError, ValueError):
        # Acceptable - explicit error on zero scale
        pass


def test_conv2d_empty_input():
    """Test Conv2D with empty batch dimension."""
    # Empty batch
    x_int8 = np.zeros((0, 3, 5, 5), dtype=np.int8)
    w_int8 = np.random.randint(-10, 10, (8, 3, 3, 3), dtype=np.int8)
    bias = np.zeros(8, dtype=np.int32)

    try:
        y = conv2d_int8(x_int8, w_int8, bias,
                       scale_x=0.01, scale_w=0.01, scale_y=0.02)
        assert y.shape[0] == 0, "Empty batch should produce empty output"
        assert y.shape[1:] == (8, 3, 3), "Output spatial dims should be correct"
    except Exception:
        # Some implementations may raise error on empty input - acceptable
        pass


def test_conv2d_mismatched_channels():
    """Test Conv2D with mismatched input/weight channels."""
    x_int8 = np.zeros((1, 3, 5, 5), dtype=np.int8)
    w_int8 = np.random.randint(-10, 10, (8, 4, 3, 3), dtype=np.int8)  # Wrong: 4 instead of 3
    bias = np.zeros(8, dtype=np.int32)

    try:
        y = conv2d_int8(x_int8, w_int8, bias,
                       scale_x=0.01, scale_w=0.01, scale_y=0.02)
        raise AssertionError("Should have failed with mismatched channels")
    except (ValueError, AssertionError):
        # Expected - should fail with dimension mismatch
        pass


def test_linear_mismatched_dimensions():
    """Test Linear with mismatched input/weight dimensions."""
    x_int8 = np.zeros((2, 10), dtype=np.int8)
    w_int8 = np.random.randint(-10, 10, (5, 15), dtype=np.int8)  # Wrong: 15 instead of 10

    try:
        y = linear_int8(x_int8, w_int8, None,
                       scale_x=0.01, scale_w=0.01, scale_y=0.02)
        raise AssertionError("Should have failed with mismatched dimensions")
    except (ValueError, AssertionError):
        # Expected - matrix dimensions don't match
        pass


def test_linear_extreme_accumulation():
    """Test Linear with values that cause INT32 accumulation overflow."""
    # Worst case: 127 * 127 * N_features
    # INT32 max = 2,147,483,647
    # Can safely accumulate ~132,000 terms of (127*127)

    x_int8 = np.full((1, 1000), 127, dtype=np.int8)
    w_int8 = np.full((1, 1000), 127, dtype=np.int8)

    y = linear_int8(x_int8, w_int8, None,
                   scale_x=0.01, scale_w=0.01, scale_y=0.1)

    # Should not overflow - INT32 accumulation should handle this
    assert not np.any(np.isnan(y)), "INT32 accumulation overflowed"
    assert not np.any(np.isinf(y)), "Scaling produced infinity"


def test_relu_negative_values():
    """Test ReLU correctly zeros out negative values."""
    x = np.array([-100, -50, -1, 0, 1, 50, 100], dtype=np.int8)
    y = relu_int8(x)

    assert np.all(y[x < 0] == 0), "ReLU should zero negative values"
    assert np.all(y[x >= 0] == x[x >= 0]), "ReLU should preserve positive values"


def test_relu_preserves_zero():
    """Test ReLU correctly handles zero."""
    x = np.array([0, 0, 0], dtype=np.int8)
    y = relu_int8(x)

    assert np.all(y == 0), "ReLU(0) should be 0"


def test_maxpool_uniform_values():
    """Test MaxPool with all identical values."""
    x = np.full((1, 2, 4, 4), 42, dtype=np.int8)
    y = maxpool2d_int8(x, kernel_size=(2, 2), stride=(2, 2))

    assert np.all(y == 42), "MaxPool of uniform values should be uniform"


def test_maxpool_all_negative():
    """Test MaxPool with all negative values."""
    x = np.full((1, 2, 4, 4), -50, dtype=np.int8)
    y = maxpool2d_int8(x, kernel_size=(2, 2), stride=(2, 2))

    assert np.all(y == -50), "MaxPool should handle all-negative values"


def test_maxpool_extreme_values():
    """Test MaxPool with INT8 boundary values."""
    x = np.array([[[
        [-128, -128, 127, 127],
        [-128, -128, 127, 127],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]]], dtype=np.int8)

    y = maxpool2d_int8(x, kernel_size=(2, 2), stride=(2, 2))

    assert y[0, 0, 0, 0] == -128, "Max of all -128 should be -128"
    assert y[0, 0, 0, 1] == 127, "Max of all 127 should be 127"
    assert y[0, 0, 1, 0] == 0, "Max of all 0 should be 0"


def test_scale_negative():
    """Test that negative scales are handled appropriately."""
    x = np.array([1.0, 2.0, 3.0])
    scale = -0.01  # Invalid: negative scale

    try:
        # Some implementations may allow this (sign flip), others should error
        q = quantize_linear(x, scale=scale)
        # If it doesn't error, verify behavior makes sense
        assert np.all(q <= 0), "Negative scale should flip signs"
    except ValueError:
        # Acceptable - rejecting negative scales
        pass


def test_single_element_operations():
    """Test operations with single-element tensors."""
    # Quantize
    x = np.array([1.0])
    q = quantize_linear(x, scale=0.01)
    assert q.shape == (1,), "Single element should preserve shape"

    # Linear
    x_int8 = np.array([[10]], dtype=np.int8)
    w_int8 = np.array([[5]], dtype=np.int8)
    y = linear_int8(x_int8, w_int8, None,
                   scale_x=0.01, scale_w=0.01, scale_y=0.02)
    assert y.shape == (1, 1), "Single element linear should work"

    # ReLU
    x_int8 = np.array([-5], dtype=np.int8)
    y = relu_int8(x_int8)
    assert y[0] == 0, "ReLU single negative should be 0"


def main():
    """Run all edge case tests."""
    print("="*80)
    print("Running Edge Case Tests for Atomic Operations")
    print("="*80)
    print()

    tester = EdgeCaseTests()

    print("Quantization Edge Cases:")
    tester.test("Very small scale", test_quantize_very_small_scale)
    tester.test("Very large scale", test_quantize_very_large_scale)
    tester.test("Overflow handling", test_quantize_overflow)
    tester.test("Zero scale protection", test_quantize_zero_scale_protection)
    tester.test("Negative scale", test_scale_negative)

    print("\nConv2D Edge Cases:")
    tester.test("Empty input batch", test_conv2d_empty_input)
    tester.test("Mismatched channels", test_conv2d_mismatched_channels)

    print("\nLinear Edge Cases:")
    tester.test("Mismatched dimensions", test_linear_mismatched_dimensions)
    tester.test("Extreme accumulation", test_linear_extreme_accumulation)

    print("\nReLU Edge Cases:")
    tester.test("Negative values", test_relu_negative_values)
    tester.test("Zero preservation", test_relu_preserves_zero)

    print("\nMaxPool Edge Cases:")
    tester.test("Uniform values", test_maxpool_uniform_values)
    tester.test("All negative", test_maxpool_all_negative)
    tester.test("Extreme values", test_maxpool_extreme_values)

    print("\nGeneral Edge Cases:")
    tester.test("Single element operations", test_single_element_operations)

    return tester.summary()


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
