/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Core Utilities Header for ARES Runtime
 *
 * Shared mathematical functions used across multiple operations.
 */

#ifndef ARES_CORE_UTILS_H
#define ARES_CORE_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Fast exponential approximation.
 * Uses 7th-order Taylor series with range reduction.
 * Used by MHSA softmax (FP32 path) and SSM.
 */
float fast_exp(float x);

/**
 * Fast natural logarithm approximation.
 * Uses polynomial approximation with range reduction.
 * Used by SSM softplus computation.
 */
float fast_log(float x);

/**
 * Fast softplus: log(1 + exp(x))
 * Numerically stable version avoiding overflow.
 * Used by SSM dt (delta time) computation.
 */
float fast_softplus(float x);

#ifdef __cplusplus
}
#endif

#endif // ARES_CORE_UTILS_H
