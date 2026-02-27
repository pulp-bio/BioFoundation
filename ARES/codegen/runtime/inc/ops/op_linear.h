/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Linear Tile Workers Header for ARES Runtime
 *
 * L1-tiled and pipelined linear operations.
 * Type definitions are in network_kernels.h.
 */

#ifndef ARES_OPS_LINEAR_H
#define ARES_OPS_LINEAR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Linear tile worker for L1-tiled execution */
void linear_tile_worker(void *arg);

/* Linear to FP32 tile worker for classifier layers */
void linear_to_fp32_tile_worker(void *arg);

/* Sequential strided linear (single output feature at a time with output stride) */
void linear_int8_sequential_strided(
    const int8_t *input, const int8_t *weights, const float *bias, int8_t *output,
    uint16_t in_features, uint16_t out_features, uint16_t out_stride,
    float scale_input, float scale_weight, float scale_output
);

/* L1-tiled linear worker: OPTIMIZED VERSION with 4x output + 4x timestep unrolling */
void linear_int8_l1_worker(void *arg);

#ifdef __cplusplus
}
#endif

#endif /* ARES_OPS_LINEAR_H */
