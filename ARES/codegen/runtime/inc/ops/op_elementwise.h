/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Elementwise Operations Header for ARES Runtime
 *
 * Add, Concat, Transpose operations.
 */

#ifndef ARES_OPS_ELEMENTWISE_H
#define ARES_OPS_ELEMENTWISE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Elementwise Add INT8 with requantization (parallel) */
void network_add_int8(
    const int8_t *input_a, const int8_t *input_b, int8_t *output,
    uint32_t size,
    float scale_a, float scale_b, float scale_out
);

/* Concat along channel axis with requantization (parallel) */
void network_concat_int8(
    const int8_t **inputs, const float *input_scales, int8_t *output,
    uint16_t num_inputs, uint16_t batch,
    const uint16_t *channels_per_input,
    uint16_t height, uint16_t width,
    float scale_output
);

/* 2D Transpose [B, D1, D2] -> [B, D2, D1] (parallel) */
void network_transpose_2d_int8(
    const int8_t *input, int8_t *output,
    int batch_size, int dim1, int dim2
);

/* 2D Zero Padding [B, C, H, W] -> [B, C, H+pad_top+pad_bottom, W+pad_left+pad_right] (parallel) */
void network_zeropad2d_int8(
    const int8_t *input, int8_t *output,
    int channels, int in_h, int in_w,
    int pad_left, int pad_right, int pad_top, int pad_bottom
);

/* ---
 * HWC Layout Operations
 * ---
 * For Height-Width-Channel layout where channels are contiguous at each
 * spatial position. Enables efficient SIMD for small channel counts.
 */

/* 2D Zero Padding for HWC layout [H, W, C] -> [H+pad, W+pad, C] (parallel) */
void network_zeropad2d_int8_hwc(
    const int8_t *input, int8_t *output,
    int channels, int in_h, int in_w,
    int pad_left, int pad_right, int pad_top, int pad_bottom
);

/* Layout conversion: CHW [C, H, W] -> HWC [H, W, C] (parallel) */
void network_chw_to_hwc_int8(
    const int8_t *input, int8_t *output,
    int channels, int height, int width
);

/* Layout conversion: HWC [H, W, C] -> CHW [C, H, W] (parallel) */
void network_hwc_to_chw_int8(
    const int8_t *input, int8_t *output,
    int channels, int height, int width
);

#ifdef __cplusplus
}
#endif

#endif /* ARES_OPS_ELEMENTWISE_H */
