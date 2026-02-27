/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * NE16 Linear Layer Kernels
 *
 * High-level linear layer execution using the NE16 accelerator.
 * These functions handle input conversion, NE16 execution, and postprocessing.
 *
 * Usage:
 * 1. Pre-pack weights offline using ne16_packing.py
 * 2. Allocate scratch buffers using ne16_linear_*_scratch_size()
 * 3. Call ne16_linear_int8_packed() for each forward pass
 */

#ifndef NE16_LINEAR_H
#define NE16_LINEAR_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef ARES_USE_NE16

/**
 * Execute linear layer using NE16 with pre-packed weights.
 *
 * This is the main entry point for NE16-accelerated linear layers.
 * Weights must be pre-packed offline using ne16_packing.py.
 *
 * The function tiles over tokens to limit scratch buffer usage, performing:
 * 1. Parallel S8->U8 input conversion (all cores)
 * 2. NE16 1x1 convolution (core 0 only)
 * 3. Parallel INT32->INT8 requantization (all cores)
 *
 * @param input             Signed INT8 input [num_tokens, in_features]
 * @param weights_packed    Pre-packed weights from ne16_packing.py
 * @param bias_corr         Corrected bias [out_features] (includes input_zp adjustment)
 * @param output            Signed INT8 output [num_tokens, out_stride]
 * @param num_tokens        Number of tokens (sequence length)
 * @param in_features       Input feature dimension
 * @param out_features      Output feature dimension
 * @param out_stride        Stride between rows in output (for strided output)
 * @param scale_input       Input quantization scale
 * @param scale_weight      Weight quantization scale
 * @param scale_output      Output quantization scale
 * @param tile_tokens       Number of tokens per tile (controls scratch size)
 * @param input_u8_scratch  Scratch buffer for U8 input [tile_tokens * in_features]
 * @param output_s32_scratch Scratch buffer for S32 accumulators [tile_tokens * out_features]
 */
void ne16_linear_int8_packed(
    const int8_t *input,
    const int8_t *weights_packed,
    const int32_t *bias_corr,
    int8_t *output,
    int num_tokens,
    int in_features,
    int out_features,
    int out_stride,
    float scale_input,
    float scale_weight,
    float scale_output,
    int tile_tokens,
    uint8_t *input_u8_scratch,
    int32_t *output_s32_scratch
);

/**
 * Execute linear layer using NE16 with hardware outquant (OUTQUANT=1).
 *
 * This uses NE16's internal requantization to produce INT8 outputs directly.
 * It requires `scale`/`scale_shift` arrays (Scale / ScaleN) and a bias array.
 *
 *
 * @param input                  Signed INT8 input [num_tokens, in_features]
 * @param weights_packed         Pre-packed NE16 weights
 * @param bias_corr              Bias array [out_features] in accumulator domain
 * @param scale                  Per-output-channel multiplier [out_features]
 * @param scale_shift            Per-output-channel shift [out_features]
 * @param output                 Signed INT8 output base [num_tokens, out_stride]
 * @param num_tokens             Total number of tokens
 * @param in_features            Input feature dimension
 * @param out_features           Output feature dimension
 * @param out_stride             Stride between tokens in output
 * @param tile_tokens            Tokens per tile
 * @param input_u8_ping          U8 input scratch ping buffer [tile_tokens * in_features]
 * @param input_u8_pong          Optional U8 input scratch pong buffer [tile_tokens * in_features]
 *                               When provided, input conversion is overlapped with NE16 execution.
 * @param output_s8_tile_scratch Optional contiguous output scratch
 *                               - If provided, NE16 streams output here and the result is copied/scattered into
 *                                 the final output layout.
 *                               - If NULL, NE16 streams directly to `output` (only valid when the target is L1).
 */
void ne16_linear_int8_packed_hw_requant(
    const int8_t *input,
    const int8_t *weights_packed,
    const int32_t *bias_corr,
    const uint8_t *scale,
    const uint8_t *scale_shift,
    int8_t *output,
    int num_tokens,
    int in_features,
    int out_features,
    int out_stride,
    int tile_tokens,
    uint8_t *input_u8_ping,
    uint8_t *input_u8_pong,
    int8_t *output_s8_tile_scratch
);

/**
 * Execute batched linear layer using NE16 with pre-packed weights.
 *
 * Same as ne16_linear_int8_packed but handles batch dimension.
 * Each batch is processed sequentially.
 *
 * @param input             [batch, num_tokens, in_features]
 * @param weights_packed    Pre-packed weights
 * @param bias_corr         Corrected bias [out_features]
 * @param output            [batch, num_tokens, out_features]
 * @param batch             Batch size
 * @param num_tokens        Tokens per batch
 * @param in_features       Input features
 * @param out_features      Output features
 * @param scale_input       Input scale
 * @param scale_weight      Weight scale
 * @param scale_output      Output scale
 * @param tile_tokens       Tokens per tile
 * @param input_u8_scratch  U8 input scratch
 * @param output_s32_scratch S32 output scratch
 */
void ne16_linear_int8_batch_packed(
    const int8_t *input,
    const int8_t *weights_packed,
    const int32_t *bias_corr,
    int8_t *output,
    int batch,
    int num_tokens,
    int in_features,
    int out_features,
    float scale_input,
    float scale_weight,
    float scale_output,
    int tile_tokens,
    uint8_t *input_u8_scratch,
    int32_t *output_s32_scratch
);

/**
 * Calculate required input scratch buffer size.
 *
 * @param tile_tokens   Tokens per tile
 * @param in_features   Input feature dimension
 * @return Size in bytes
 */
size_t ne16_linear_input_scratch_size(int tile_tokens, int in_features);

/**
 * Calculate required output scratch buffer size.
 *
 * @param tile_tokens   Tokens per tile
 * @param out_features  Output feature dimension
 * @return Size in bytes
 */
size_t ne16_linear_output_scratch_size(int tile_tokens, int out_features);

/**
 * Calculate total scratch buffer size (input + output).
 *
 * @param tile_tokens   Tokens per tile
 * @param in_features   Input feature dimension
 * @param out_features  Output feature dimension
 * @return Total size in bytes
 */
size_t ne16_linear_total_scratch_size(int tile_tokens, int in_features, int out_features);

/**
 * Execute pipelined linear layer with double-buffered input conversion.
 *
 * This function overlaps CPU input conversion (S8->U8) with NE16 compute,
 * using ping-pong buffers. While NE16 processes tile N, the CPU converts
 * tile N+1 in parallel across all cores.
 *
 * Two modes are supported:
 * - HW requantization (use_hw_requant=1): NE16 outputs INT8 directly
 * - SW requantization (use_hw_requant=0): NE16 outputs INT32, CPU postprocesses
 *
 * @param input             Signed INT8 input [num_tokens, in_features]
 * @param weights_packed    Pre-packed NE16 weights
 * @param bias_corr         Bias correction [out_features]
 * @param scale             Per-channel scale [out_features] (HW mode, else NULL)
 * @param scale_shift       Per-channel shift [out_features] (HW mode, else NULL)
 * @param output            Signed INT8 output [num_tokens, out_stride]
 * @param num_tokens        Total number of tokens
 * @param in_features       Input feature dimension
 * @param out_features      Output feature dimension
 * @param out_stride        Stride between rows in output
 * @param scale_input       Input quantization scale (SW mode)
 * @param scale_weight      Weight quantization scale (SW mode)
 * @param scale_output      Output quantization scale (SW mode)
 * @param tile_tokens       Number of tokens per tile
 * @param input_u8_ping     Ping buffer for U8 input [tile_tokens * in_features]
 * @param input_u8_pong     Pong buffer for U8 input [tile_tokens * in_features]
 * @param output_s32_scratch INT32 accumulator scratch (SW mode, else NULL)
 * @param use_hw_requant    1 for HW requantization, 0 for SW
 */
void ne16_linear_int8_pipelined(
    const int8_t *input,
    const int8_t *weights_packed,
    const int32_t *bias_corr,
    const uint8_t *scale,
    const uint8_t *scale_shift,
    int8_t *output,
    int num_tokens,
    int in_features,
    int out_features,
    int out_stride,
    float scale_input,
    float scale_weight,
    float scale_output,
    int tile_tokens,
    uint8_t *input_u8_ping,
    uint8_t *input_u8_pong,
    int32_t *output_s32_scratch,
    int use_hw_requant
);

/**
 * Compute NE16 outquant multiplier+shift (Scale / ScaleN) from a float scale.
 *
 * This ports the reference `compute_scales()` logic (GAP SDK) used for NE16 OUTQUANT.
 *
 * @param scale         Floating-point scale factor (typically s_in * s_w / s_out)
 * @param qbias_out     Output multiplier (Scale), stored as a raw 8-bit value
 * @param qnorm_out     Output right-shift (ScaleN), stored as a raw 8-bit value
 */
void ne16_compute_outquant_scale(float scale, uint8_t *qbias_out, uint8_t *qnorm_out);

/**
 * Calculate scratch size for pipelined execution.
 *
 * Includes double-buffered input scratch (ping + pong).
 * For SW requantization, also includes output INT32 scratch.
 *
 * @param tile_tokens   Tokens per tile
 * @param in_features   Input features
 * @param out_features  Output features
 * @param use_hw_requant 1 if using HW requantization
 * @return Total scratch size in bytes
 */
size_t ne16_linear_pipelined_scratch_size(int tile_tokens, int in_features, int out_features, int use_hw_requant);

/**
 * Pack weights at runtime (diagnostic helper for DMA investigation).
 *
 * @param w_s8         Original signed INT8 weights [out_features, in_features]
 * @param out_packed   Output buffer for packed weights
 * @param in_features  Input feature dimension
 * @param out_features Output feature dimension
 */
void ne16_pack_weights_runtime(
    const int8_t *w_s8,
    uint8_t *out_packed,
    int in_features,
    int out_features
);

/**
 * Run NE16 selftest to verify hardware/simulator is working correctly.
 *
 * Creates known inputs and weights, runs both SW reference and NE16,
 * and compares results.
 *
 * @param l1_scratch    L1 buffer for scratch (input_u8 + acc_s32). If NULL, will allocate.
 * @param l1_size       Size of l1_scratch buffer in bytes
 * @return 0 on success, -1 on failure
 */
int ne16_linear_selftest(void *l1_scratch, size_t l1_size);

#endif /* ARES_USE_NE16 */

#ifdef __cplusplus
}
#endif

#endif /* NE16_LINEAR_H */
