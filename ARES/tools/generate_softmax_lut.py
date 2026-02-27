#!/usr/bin/env python3
# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Softmax LUT Generation Utility

Generates a lookup table for integer-only softmax implementation (i-Softmax).
This enables bit-exact matching between Python and C code by eliminating
FP32 precision differences in exp() computation.

Usage:
    # Generate LUT for a specific output directory
    python generate_softmax_lut.py --output bin/softmax_lut.bin

    # Use as module
    from tools.generate_softmax_lut import generate_softmax_lut
    lut = generate_softmax_lut()
"""

import argparse
import numpy as np
import struct
from pathlib import Path


def generate_softmax_lut(
    input_range=(-12.8, 0.0),
    num_entries=256,
    output_scale=1000.0,
    dtype=np.int16
):
    """
    Generate softmax lookup table for exp() approximation.

    The LUT maps integer-quantized attention scores to exp() values.
    This is used in the softmax computation: exp(x_i - x_max) / sum(exp(x_j - x_max))

    Parameters
    ----------
    input_range : tuple of float
        (min, max) range of input values (after max subtraction for stability).
        Default: (-12.8, 0.0) covers typical softmax inputs.
        Note: exp(-12.8) ≈ 2.7e-6 (negligible), exp(0) = 1.0
    num_entries : int
        Number of LUT entries (must be power of 2 for efficient indexing).
        Default: 256 (8-bit LUT)
        Alternatives: 512 (9-bit), 1024 (10-bit), 2048 (11-bit)
    output_scale : float
        Scale factor for quantizing exp() output to integers.
        Default: 1000.0
        This determines precision: larger scale = more precision but risk of overflow.
    dtype : numpy dtype
        Output data type for LUT entries.
        Default: np.int16 (range -32768 to 32767)

    Returns
    -------
    lut : numpy.ndarray
        Lookup table of shape (num_entries,) with dtype specified.
        lut[i] contains the quantized exp() value for input at index i.

    lut_metadata : dict
        Metadata for using the LUT:
        - 'input_min': Minimum input value covered
        - 'input_max': Maximum input value covered
        - 'input_step': Step size between LUT entries
        - 'output_scale': Scale factor used for quantization
        - 'num_entries': Number of LUT entries
        - 'dtype': Data type of LUT entries

    Algorithm
    ---------
    1. Divide input range into num_entries equal steps
    2. For each step, compute exp(input_value)
    3. Quantize to integer: lut[i] = round(exp(input[i]) * output_scale)
    4. Clip to dtype range to prevent overflow

    Example
    -------
    >>> lut, meta = generate_softmax_lut()
    >>> # To use: given attention score x in range [-12.8, 0]
    >>> # 1. Compute index: idx = int((x - meta['input_min']) / meta['input_step'])
    >>> # 2. Lookup: exp_x_quantized = lut[idx]
    >>> # 3. Accumulate and normalize for softmax
    """
    input_min, input_max = input_range
    input_step = (input_max - input_min) / num_entries

    # Generate input values (uniformly spaced)
    input_values = np.linspace(input_min, input_max, num_entries, dtype=np.float64)

    # Compute exp() in high precision (float64)
    exp_values = np.exp(input_values)

    # Quantize to integer
    lut_int = np.round(exp_values * output_scale).astype(np.int64)

    # Clip to dtype range
    dtype_info = np.iinfo(dtype)
    lut_int = np.clip(lut_int, dtype_info.min, dtype_info.max)

    # Cast to target dtype
    lut = lut_int.astype(dtype)

    # Metadata for using the LUT
    lut_metadata = {
        'input_min': input_min,
        'input_max': input_max,
        'input_step': input_step,
        'output_scale': output_scale,
        'num_entries': num_entries,
        'dtype': str(dtype),
    }

    return lut, lut_metadata


def save_lut_binary(lut, output_path):
    """
    Save LUT as binary file (little-endian int16).

    Parameters
    ----------
    lut : numpy.ndarray
        Lookup table to save
    output_path : str or Path
        Output file path (e.g., 'bin/softmax_lut.bin')

    File Format
    -----------
    - Binary file with little-endian int16 values
    - No header (raw data)
    - Size: num_entries * 2 bytes (for int16)
    - Example: 256 entries → 512 bytes
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as binary (little-endian int16)
    with open(output_path, 'wb') as f:
        for value in lut:
            f.write(struct.pack('<h', value))  # '<h' = little-endian int16

    print(f"Saved LUT to: {output_path}")
    print(f"  Size: {output_path.stat().st_size} bytes ({len(lut)} entries x 2 bytes)")


def save_lut_c_header(lut, lut_metadata, output_path):
    """
    Save LUT as C header file for debugging/validation.

    Parameters
    ----------
    lut : numpy.ndarray
        Lookup table to save
    lut_metadata : dict
        Metadata for LUT
    output_path : str or Path
        Output file path (e.g., 'bin/softmax_lut.h')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("// Softmax LUT (generated by generate_softmax_lut.py)\n")
        f.write("// DO NOT EDIT MANUALLY\n\n")
        f.write(f"#define SOFTMAX_LUT_SIZE {lut_metadata['num_entries']}\n")
        f.write(f"#define SOFTMAX_LUT_INPUT_MIN {lut_metadata['input_min']:.6f}f\n")
        f.write(f"#define SOFTMAX_LUT_INPUT_MAX {lut_metadata['input_max']:.6f}f\n")
        f.write(f"#define SOFTMAX_LUT_INPUT_STEP {lut_metadata['input_step']:.6f}f\n")
        f.write(f"#define SOFTMAX_LUT_OUTPUT_SCALE {lut_metadata['output_scale']:.1f}f\n\n")

        f.write("static const int16_t softmax_lut[SOFTMAX_LUT_SIZE] = {\n")
        for i in range(0, len(lut), 8):
            chunk = lut[i:i+8]
            values_str = ', '.join(f'{v:6d}' for v in chunk)
            f.write(f"    {values_str},\n")
        f.write("};\n")

    print(f"Saved C header to: {output_path}")


def validate_lut(lut, lut_metadata, num_test_points=1000):
    """
    Validate LUT accuracy against numpy.exp().

    Parameters
    ----------
    lut : numpy.ndarray
        Lookup table to validate
    lut_metadata : dict
        Metadata for LUT
    num_test_points : int
        Number of random test points

    Returns
    -------
    validation_results : dict
        Statistics on LUT accuracy:
        - 'max_abs_error': Maximum absolute error
        - 'mean_abs_error': Mean absolute error
        - 'max_rel_error': Maximum relative error (%)
        - 'mean_rel_error': Mean relative error (%)
    """
    input_min = lut_metadata['input_min']
    input_max = lut_metadata['input_max']
    input_step = lut_metadata['input_step']
    output_scale = lut_metadata['output_scale']

    # Generate random test points in input range
    test_inputs = np.random.uniform(input_min, input_max, num_test_points)

    # Ground truth: numpy exp()
    exp_true = np.exp(test_inputs)

    # LUT approximation
    indices = ((test_inputs - input_min) / input_step).astype(int)
    indices = np.clip(indices, 0, len(lut) - 1)  # Handle boundary cases
    exp_lut_quantized = lut[indices]
    exp_lut = exp_lut_quantized.astype(np.float64) / output_scale

    # Compute errors
    abs_errors = np.abs(exp_true - exp_lut)
    rel_errors = np.abs((exp_true - exp_lut) / (exp_true + 1e-10)) * 100  # %

    validation_results = {
        'max_abs_error': np.max(abs_errors),
        'mean_abs_error': np.mean(abs_errors),
        'max_rel_error': np.max(rel_errors),
        'mean_rel_error': np.mean(rel_errors),
    }

    return validation_results


def print_validation_report(validation_results):
    """Print validation report in human-readable format."""
    print("\n" + "="*60)
    print("LUT VALIDATION REPORT")
    print("="*60)
    print(f"Max Absolute Error:  {validation_results['max_abs_error']:.6e}")
    print(f"Mean Absolute Error: {validation_results['mean_abs_error']:.6e}")
    print(f"Max Relative Error:  {validation_results['max_rel_error']:.3f}%")
    print(f"Mean Relative Error: {validation_results['mean_rel_error']:.3f}%")
    print("="*60)

    # Interpret results
    if validation_results['mean_rel_error'] < 0.1:
        print("[PASS] EXCELLENT: <0.1% mean error (production ready)")
    elif validation_results['mean_rel_error'] < 1.0:
        print("[PASS] GOOD: <1% mean error (acceptable for most use cases)")
    elif validation_results['mean_rel_error'] < 5.0:
        print("[WARN]  ACCEPTABLE: <5% mean error (consider larger LUT)")
    else:
        print("[FAIL] POOR: >5% mean error (increase num_entries or output_scale)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate softmax lookup table for i-Softmax implementation"
    )
    parser.add_argument(
        '--output',
        type=str,
        default='bin/softmax_lut.bin',
        help='Output path for binary LUT (default: bin/softmax_lut.bin)'
    )
    parser.add_argument(
        '--header',
        type=str,
        default=None,
        help='Optional output path for C header (e.g., bin/softmax_lut.h)'
    )
    parser.add_argument(
        '--input-min',
        type=float,
        default=-12.8,
        help='Minimum input value (default: -12.8)'
    )
    parser.add_argument(
        '--input-max',
        type=float,
        default=0.0,
        help='Maximum input value (default: 0.0)'
    )
    parser.add_argument(
        '--num-entries',
        type=int,
        default=256,
        help='Number of LUT entries (default: 256 for 8-bit LUT)'
    )
    parser.add_argument(
        '--output-scale',
        type=float,
        default=1000.0,
        help='Scale factor for quantization (default: 1000.0)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run validation tests on generated LUT'
    )

    args = parser.parse_args()

    # Generate LUT
    print("Generating softmax LUT...")
    print(f"  Input range: [{args.input_min}, {args.input_max}]")
    print(f"  Num entries: {args.num_entries}")
    print(f"  Output scale: {args.output_scale}")
    print()

    lut, lut_metadata = generate_softmax_lut(
        input_range=(args.input_min, args.input_max),
        num_entries=args.num_entries,
        output_scale=args.output_scale,
        dtype=np.int16
    )

    print(f"Generated LUT:")
    print(f"  Shape: {lut.shape}")
    print(f"  Dtype: {lut.dtype}")
    print(f"  Value range: [{np.min(lut)}, {np.max(lut)}]")
    print(f"  Input step size: {lut_metadata['input_step']:.6f}")
    print()

    # Save binary file
    save_lut_binary(lut, args.output)

    # Save C header (optional)
    if args.header:
        save_lut_c_header(lut, lut_metadata, args.header)

    # Validate (optional)
    if args.validate:
        validation_results = validate_lut(lut, lut_metadata)
        print_validation_report(validation_results)

    print("[PASS] LUT generation complete!")


if __name__ == '__main__':
    main()
