/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Normalization Operations Header for ARES Runtime
 *
 * LayerNorm, GroupNorm, RMSNorm operations with integer-only variants.
 */

#ifndef ARES_OPS_NORM_H
#define ARES_OPS_NORM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef ARES_LLAMA_SUPPORT
/* ---
 * RMSNorm (Root Mean Square Layer Normalization)
 * Used in Llama, Llama2, and other modern LLMs.
 * Simpler than LayerNorm - no mean subtraction.
 * Formula: y = (x / rms(x)) * weight, where rms(x) = sqrt(mean(x^2) + eps)
 * --- */

/* RMSNorm with FP32-based statistics (sequential) */
void network_rmsnorm_int8_fp32(
    const int8_t *input, int8_t *output,
    const float *weight,
    uint32_t total_elements, uint32_t normalized_dim,
    float scale_in, float scale_out, float eps
);

/* i-RMSNorm: Integer-only for bit-exact matching (sequential) */
void network_rmsnorm_int8_integer(
    const int8_t *input, int8_t *output,
    const float *weight,
    uint32_t total_elements, uint32_t normalized_dim,
    float scale_in, float scale_out
);

/* i-RMSNorm with parallel execution across vectors (8 cores) */
void network_rmsnorm_int8_integer_parallel(
    const int8_t *input, int8_t *output,
    const float *weight,
    uint32_t total_elements, uint32_t normalized_dim,
    float scale_in, float scale_out
);
#endif // ARES_LLAMA_SUPPORT

/* --- LayerNorm --- */

/* LayerNorm with FP32-based statistics (sequential) */
void network_layernorm_int8_fixed_point(
    const int8_t *input, int8_t *output,
    const float *weight, const float *bias,
    uint32_t total_elements, uint32_t normalized_dim,
    float scale_in, float scale_out, float eps
);

/* i-LayerNorm: Integer-only for bit-exact matching (sequential) */
void network_layernorm_int8_integer(
    const int8_t *input, int8_t *output,
    const float *weight, const float *bias,
    uint32_t total_elements, uint32_t normalized_dim,
    float scale_in, float scale_out
);

/* i-LayerNorm with parallel execution across vectors (8 cores) */
void network_layernorm_int8_integer_parallel(
    const int8_t *input, int8_t *output,
    const float *weight, const float *bias,
    uint32_t total_elements, uint32_t normalized_dim,
    float scale_in, float scale_out
);

/* GroupNorm with integer statistics (parallel, 8 cores) */
void network_groupnorm_int8_integer_parallel(
    const int8_t *input,
    int8_t *output,
    const float *weight,
    const float *bias,
    uint32_t batch,
    uint32_t channels,
    uint32_t spatial_size,
    uint32_t num_groups,
    float scale_input,
    float scale_output
);

#ifdef __cplusplus
}
#endif

#endif /* ARES_OPS_NORM_H */
