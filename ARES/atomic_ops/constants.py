# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Shared numeric constants for atomic INT8 reference ops.

These Python reference implementations intentionally mirror the generated C
runtime behavior. Centralizing Q-format scales and common LUT sizing avoids
“magic numbers” drifting between modules.
"""

# Q15 fixed-point representation:
# - signed int16 storing values in [-1.0, 1.0) with 15 fractional bits
Q15_FRACTION_BITS = 15
Q15_SCALE_INT = 1 << Q15_FRACTION_BITS  # 32768
Q15_SCALE = float(Q15_SCALE_INT)        # 32768.0 (for FP math)
INT16_MAX_Q15 = Q15_SCALE_INT - 1       # 32767 (~0.99997 in Q15)

# int16 bounds (useful for LUT outputs and clipping)
INT16_MIN = -(1 << 15)                  # -32768
INT16_MAX = (1 << 15) - 1               # 32767

# LUT sizing: 12-bit index balances accuracy vs memory for int16 LUTs.
LUT_INDEX_BITS = 12
LUT_SIZE = 1 << LUT_INDEX_BITS          # 4096

# Other common Q-formats used in this repo
Q13_FRACTION_BITS = 13
Q13_SCALE_INT = 1 << Q13_FRACTION_BITS  # 8192

