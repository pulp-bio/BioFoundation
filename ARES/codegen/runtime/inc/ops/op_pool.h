/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Pooling Operations Header for ARES Runtime
 *
 * MaxPool, AvgPool, GlobalAvgPool, AdaptiveAvgPool1D operations.
 */

#ifndef ARES_OPS_POOL_H
#define ARES_OPS_POOL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* MaxPool INT8 (parallel across channels) - supports non-square kernels */
void network_maxpool_int8(
    const int8_t *input, int8_t *output,
    uint16_t in_h, uint16_t in_w, uint16_t in_ch,
    uint16_t out_h, uint16_t out_w,
    uint16_t kernel_h, uint16_t kernel_w,
    uint16_t stride_h, uint16_t stride_w,
    uint16_t pad_h, uint16_t pad_w
);

/* AvgPool INT8 with requantization support (parallel) - supports non-square kernels */
void network_avgpool_int8(
    const int8_t *input, int8_t *output,
    uint16_t in_h, uint16_t in_w, uint16_t in_ch,
    uint16_t out_h, uint16_t out_w,
    uint16_t kernel_h, uint16_t kernel_w,
    uint16_t stride_h, uint16_t stride_w,
    float scale_in, float scale_out
);

/* Global Average Pool INT8 (parallel across channels) */
void network_global_avgpool_int8(
    const int8_t *input, int8_t *output,
    uint16_t batch, uint16_t channels,
    uint16_t h, uint16_t w,
    float scale_in, float scale_out
);

/* Adaptive Average Pool 1D INT8 (parallel) */
void network_adaptive_avgpool1d_int8(
    const int8_t *input, int8_t *output,
    uint16_t batch, uint16_t channels,
    uint16_t input_len, uint16_t output_size,
    uint16_t input_stride_ch, uint16_t input_stride_len,
    uint32_t input_batch_stride
);

/* Mean Pool INT8 over sequence dimension (parallel across features)
 * Input: [B, seq_len, features], Output: [B, features]
 * Used for transformer classification heads */
void network_mean_pool_int8(
    const int8_t *input, int8_t *output,
    uint32_t batch, uint32_t seq_len, uint32_t features,
    float scale_in, float scale_out
);

/* ---
 * HWC Layout Pool Operations
 * ---
 * For Height-Width-Channel layout where channels are contiguous at each
 * spatial position. Better memory access for small channel counts.
 */

/* MaxPool INT8 HWC (parallel across spatial positions) - supports non-square kernels */
void network_maxpool_int8_hwc(
    const int8_t *input, int8_t *output,
    uint16_t in_h, uint16_t in_w, uint16_t channels,
    uint16_t out_h, uint16_t out_w,
    uint16_t kernel_h, uint16_t kernel_w,
    uint16_t stride_h, uint16_t stride_w,
    uint16_t pad_h, uint16_t pad_w
);

/* AvgPool INT8 HWC with requantization support (parallel) - supports non-square kernels */
void network_avgpool_int8_hwc(
    const int8_t *input, int8_t *output,
    uint16_t in_h, uint16_t in_w, uint16_t channels,
    uint16_t out_h, uint16_t out_w,
    uint16_t kernel_h, uint16_t kernel_w,
    uint16_t stride_h, uint16_t stride_w,
    float scale_in, float scale_out
);

#ifdef __cplusplus
}
#endif

#endif /* ARES_OPS_POOL_H */
