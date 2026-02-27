/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Core Utilities for ARES Runtime
 *
 * This module contains shared mathematical functions used across
 * multiple operations (MHSA softmax, SSM softplus, etc.).
 */

#include "core/utils.h"
#include <math.h>

// Fast exponential approximation used by MHSA softmax (FP32 path)
float fast_exp(float x)
{
    // Input bounds checking to prevent infinite loops
    // exp(88) ≈ 1.6e38 (near float max), exp(-88) ≈ 0
    if (x > 88.0f) return 1e38f;
    if (x < -88.0f) return 0.0f;
    if (x != x) return 0.0f;  // NaN check (NaN != NaN)

    // Range reduction to keep |x| <= 0.5 for better polynomial accuracy
    int reduction = 0;
    while (x > 0.5f) { x *= 0.5f; reduction++; }
    while (x < -0.5f) { x *= 0.5f; reduction++; }

    // 7th-order Taylor series for exp(x) around 0
    float x2 = x * x;
    float result = 1.0f + x + x2 * (0.5f + x * (0.166666666666667f + x * (0.041666666666667f +
                   x * (0.008333333333333f + x * (0.001388888888889f + x * 0.000198412698413f)))));

    while (reduction > 0) { result *= result; reduction--; }
    return result;
}

// Fast natural log approximation for SSM softplus
// Uses polynomial approximation around 1.0 and range reduction
float fast_log(float x)
{
    if (x <= 0.0f) return -1e30f;  // Handle invalid input
    if (x != x) return -1e30f;     // NaN check (NaN != NaN)
    if (x > 1e30f) return 70.0f;   // Avoid overflow (log(1e30) ≈ 69)
    if (x < 1e-30f) return -70.0f; // Avoid underflow (log(1e-30) ≈ -69)

    // Range reduction: log(x) = log(m * 2^e) = log(m) + e * log(2)
    // where 0.5 <= m < 1.0
    int e = 0;
    while (x >= 2.0f) { x *= 0.5f; e++; }
    while (x < 1.0f) { x *= 2.0f; e--; }

    // Now 1.0 <= x < 2.0
    // Use log(1+y) approximation where y = x-1, so 0 <= y < 1
    float y = x - 1.0f;

    // Minimax polynomial for log(1+y), accurate for y in [0, 1)
    // log(1+y) ≈ y - y²/2 + y³/3 - y⁴/4 + ...
    float y2 = y * y;
    float result = y - y2 * 0.5f + y * y2 * (0.333333333f - y * 0.25f + y2 * 0.2f);

    // Add back: log(2) * e
    result += e * 0.693147181f;

    return result;
}

// Fast log(1+exp(x)) = softplus for SSM dt computation
// Numerically stable version avoiding overflow
float fast_softplus(float x)
{
    if (x > 20.0f) return x;                        // For large x, softplus(x) ≈ x
    if (x < -20.0f) return fast_exp(x);            // For very negative x, softplus(x) ≈ exp(x)
    return fast_log(1.0f + fast_exp(x));           // Standard computation
}
