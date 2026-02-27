/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * NE16 Neural Accelerator Driver for GAP9
 *
 * This module provides a low-level interface to the NE16 hardware accelerator.
 * NE16 performs accelerated 1x1 and 3x3 convolutions with INT8 weights and
 * activations, achieving up to ~11 MACs/cycle.
 *
 * Key Features:
 * - 1x1 convolution (also used for linear/matmul)
 * - 3x3 convolution with optional padding
 * - Depthwise 3x3 convolution
 * - Optional output quantization (8-bit or 32-bit output)
 *
 * Data Format Requirements:
 * - Input: Unsigned INT8 (0-255)
 * - Weights: Pre-packed in NE16 bit-sliced format
 * - Output: Signed INT8 (-128 to 127) or INT32
 *
 * Usage Pattern:
 * 1. Call ne16_init() once at startup
 * 2. Convert signed inputs to unsigned (s8 + 128 -> u8)
 * 3. Pre-pack weights offline using ne16_packing.py
 * 4. Call ne16_conv1x1_u8_u8_to_s32() or ne16_conv1x1_u8_u8_to_s8()
 */

#ifndef NE16_DRIVER_H
#define NE16_DRIVER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef ARES_USE_NE16

/* NE16 configuration constants */
#define NE16_INPUT_ZP 128        /* Input zero-point (signed -> unsigned) */
#define NE16_WEIGHT_OFFSET (-128) /* Weight offset for unsigned storage */

/**
 * Initialize NE16 accelerator.
 *
 * Enables clock gating, sets HCI priority, and performs soft reset.
 * Must be called before any NE16 operations.
 * Safe to call multiple times (idempotent).
 */
void ne16_init(void);

/**
 * Perform soft reset of NE16 accelerator.
 *
 * Clears all internal state. Call between jobs if needed to reset
 * accumulated state or recover from errors.
 */
void ne16_soft_clear_all(void);

/**
 * Run NE16 1x1 convolution producing 32-bit accumulators.
 *
 * This is the "safe" path that produces full-precision INT32 output,
 * allowing bit-exact requantization in software.
 *
 * Tensor layout (HWC / token-major):
 * - Input:  [H, W, C_in] unsigned INT8
 * - Output: [H, W, C_out] signed INT32
 * - Weights: Pre-packed for NE16 1x1 mode
 *
 * For linear/matmul, treat tokens as spatial dimension:
 * - H = 1, W = seq_len, C_in = in_features, C_out = out_features
 *
 * @param infeat        Input tensor (unsigned INT8)
 * @param weights_packed Pre-packed weights from ne16_packing.py
 * @param outfeat       Output tensor (signed INT32)
 * @param in_w          Input width (tokens for linear)
 * @param in_h          Input height (1 for linear)
 * @param in_feat       Input features/channels
 * @param out_feat      Output features/channels
 * @param weight_offset Weight offset (typically -128)
 */
void ne16_conv1x1_u8_u8_to_s32(
    const uint8_t *infeat,
    const uint8_t *weights_packed,
    int32_t *outfeat,
    int in_w,
    int in_h,
    int in_feat,
    int out_feat,
    int8_t weight_offset
);

/**
 * Run NE16 1x1 convolution with output quantization.
 *
 * Uses NE16's internal normalization to produce INT8 output directly.
 * Faster than s32 path + software requant, but may have different
 * rounding behavior than ARES's standard quantization.
 *
 * @param infeat        Input tensor (unsigned INT8)
 * @param weights_packed Pre-packed weights
 * @param bias          Corrected bias [out_feat] (includes input_zp adjustment)
 * @param scale         Per-channel scale [out_feat] (uint8)
 * @param scale_shift   Per-channel shift [out_feat] (uint8)
 * @param outfeat       Output tensor (signed INT8)
 * @param in_w          Input width
 * @param in_h          Input height
 * @param in_feat       Input features
 * @param out_feat      Output features
 * @param weight_offset Weight offset (typically -128)
 */
void ne16_conv1x1_u8_u8_to_s8(
    const uint8_t *infeat,
    const uint8_t *weights_packed,
    const int32_t *bias,
    const uint8_t *scale,
    const uint8_t *scale_shift,
    int8_t *outfeat,
    int in_w,
    int in_h,
    int in_feat,
    int out_feat,
    int8_t weight_offset
);

/**
 * Submit NE16 1x1 convolution asynchronously (INT32 output).
 *
 * Programs and triggers NE16 but returns immediately without waiting.
 * Call ne16_wait_job() with the returned job_id to wait for completion.
 *
 * @return job_id to pass to ne16_wait_job()
 */
int ne16_conv1x1_s32_submit_async(
    const uint8_t *infeat,
    const uint8_t *weights_packed,
    int32_t *outfeat,
    int in_w,
    int in_h,
    int in_feat,
    int out_feat,
    int8_t weight_offset
);

/**
 * Submit NE16 1x1 convolution asynchronously (INT8 output with HW requant).
 *
 * Programs and triggers NE16 but returns immediately without waiting.
 * Call ne16_wait_job() with the returned job_id to wait for completion.
 * Uses NE16 hardware requantization to produce INT8 output directly.
 *
 * @return job_id to pass to ne16_wait_job()
 */
int ne16_conv1x1_submit_async(
    const uint8_t *infeat,
    const uint8_t *weights_packed,
    const int32_t *bias,
    const uint8_t *scale,
    const uint8_t *scale_shift,
    int8_t *outfeat,
    int in_w,
    int in_h,
    int in_feat,
    int out_feat,
    int8_t weight_offset
);

/**
 * Wait for an NE16 job to complete.
 *
 * @param job_id Job ID returned by ne16_conv1x1_*_submit_async()
 */
void ne16_wait_job(int job_id);

/**
 * Run NE16 3x3 convolution producing 32-bit accumulators.
 *
 * Standard 3x3 convolution with stride 1 and optional padding.
 *
 * @param infeat        Input tensor [H_in, W_in, C_in] (unsigned INT8)
 * @param weights_packed Pre-packed 3x3 weights
 * @param outfeat       Output tensor [H_out, W_out, C_out] (signed INT32)
 * @param in_w          Input width
 * @param in_h          Input height
 * @param in_feat       Input channels
 * @param out_feat      Output channels
 * @param pad_h         Vertical padding
 * @param pad_w         Horizontal padding
 * @param weight_offset Weight offset (typically -128)
 */
void ne16_conv3x3_u8_u8_to_s32(
    const uint8_t *infeat,
    const uint8_t *weights_packed,
    int32_t *outfeat,
    int in_w,
    int in_h,
    int in_feat,
    int out_feat,
    int pad_h,
    int pad_w,
    int8_t weight_offset
);

/**
 * Run NE16 depthwise 3x3 convolution producing 32-bit accumulators.
 *
 * Depthwise convolution: each output channel processes only its corresponding
 * input channel. Uses NE16 FILTER_MODE=1 (depthwise 3x3 mode).
 *
 * @param infeat        Input tensor [H_in, W_in, C] (unsigned INT8)
 * @param weights_packed Pre-packed depthwise weights (from ne16_pack_conv3x3_depthwise_weights)
 * @param outfeat       Output tensor [H_out, W_out, C] (signed INT32)
 * @param in_w          Input width
 * @param in_h          Input height
 * @param channels      Number of channels (same for input and output in depthwise)
 * @param pad_h         Vertical padding
 * @param pad_w         Horizontal padding
 * @param weight_offset Weight offset (typically -128)
 */
void ne16_conv3x3_dw_u8_u8_to_s32(
    const uint8_t *infeat,
    const uint8_t *weights_packed,
    int32_t *outfeat,
    int in_w,
    int in_h,
    int channels,
    int pad_h,
    int pad_w,
    int8_t weight_offset
);

/**
 * Run NE16 depthwise 3x3 convolution with hardware requantization to INT8.
 *
 * Same as ne16_conv3x3_dw_u8_u8_to_s32 but uses NE16's internal normalization
 * to produce INT8 output directly.
 *
 * @param infeat        Input tensor [H_in, W_in, C] (unsigned INT8)
 * @param weights_packed Pre-packed depthwise weights
 * @param bias          Corrected bias [channels]
 * @param scale         Per-channel scale [channels] (uint8)
 * @param scale_shift   Per-channel shift [channels] (uint8)
 * @param outfeat       Output tensor [H_out, W_out, C] (signed INT8)
 * @param in_w          Input width
 * @param in_h          Input height
 * @param channels      Number of channels
 * @param pad_h         Vertical padding
 * @param pad_w         Horizontal padding
 * @param weight_offset Weight offset (typically -128)
 */
void ne16_conv3x3_dw_u8_u8_to_s8(
    const uint8_t *infeat,
    const uint8_t *weights_packed,
    const int32_t *bias,
    const uint8_t *scale,
    const uint8_t *scale_shift,
    int8_t *outfeat,
    int in_w,
    int in_h,
    int channels,
    int pad_h,
    int pad_w,
    int8_t weight_offset
);

#endif /* ARES_USE_NE16 */

#ifdef __cplusplus
}
#endif

#endif /* NE16_DRIVER_H */
