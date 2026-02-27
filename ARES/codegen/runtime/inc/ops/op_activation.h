/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Activation Operations Header for ARES Runtime
 *
 * ReLU, GELU, SiLU, and Requantize operations.
 */

#ifndef ARES_OPS_ACTIVATION_H
#define ARES_OPS_ACTIVATION_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* i-GELU LUT parameters (must match Python atomic_ops/gelu.py) */
#define I_GELU_INPUT_MIN (-4.0f)
#define I_GELU_INPUT_MAX (4.0f)
#define I_GELU_INPUT_STEP (0.03125f)
#define I_GELU_OUTPUT_SCALE (32767)
#define I_GELU_NUM_ENTRIES (256)

/* Pre-computed i-GELU lookup table */
extern const int16_t i_gelu_lut[I_GELU_NUM_ENTRIES];

/* GELU using lookup table (bit-exact with Python) */
void network_gelu_int8_lut_inplace(
    int8_t *buffer, uint32_t num_elements,
    float scale_input, float scale_output
);

/* GELU using LUT with parallel execution (8 cores) */
void network_gelu_int8_lut_inplace_parallel(
    int8_t *buffer, uint32_t num_elements,
    float scale_input, float scale_output
);

/* Non-inplace GELU using LUT */
void network_gelu_int8(
    const int8_t *input, int8_t *output,
    uint32_t num_elements,
    float scale_input, float scale_output
);

/* GELU using tanh approximation (sequential) */
void network_gelu_int8_inplace(
    int8_t *buffer, uint32_t num_elements,
    float scale_input, float scale_output
);

/* SiLU via pre-computed LUT (parallel) */
void network_silu_int8_lut(
    const int8_t *input,
    int8_t *output,
    const int8_t *lut,
    int num_elements
);

/* SiLU in-place via LUT (parallel) */
void network_silu_int8_lut_inplace(
    int8_t *buffer,
    const int8_t *lut,
    int num_elements
);

/* Generate SiLU INT8 LUT at runtime */
void generate_silu_lut_int8(
    int8_t *lut,
    float scale_in,
    float scale_out
);

/* ReLU inplace: max(0, x) for INT8 data (parallel) */
void relu_int8_inplace(int8_t *data, size_t size);

/* Requantize inplace: converts between quantization scales (parallel) */
void requantize_int8_inplace(int8_t *data, size_t size, float scale_in, float scale_out);

#ifdef __cplusplus
}
#endif

#endif /* ARES_OPS_ACTIVATION_H */
