# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Constants used across the codegen module.

These constants define tolerances and thresholds used in code generation.
"""

# Tolerance for comparing quantization scales (e.g., determining if requantization needed)
SCALE_EPSILON = 1e-8

# Tolerance for shape dimension matching (e.g., pattern matching in optimization)
SHAPE_EPSILON = 1e-6
