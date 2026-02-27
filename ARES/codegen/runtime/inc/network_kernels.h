/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

// INT8 kernel declarations for PULP cluster execution.
#pragma once
#include <stdint.h>
#include <math.h>
#include "pmsis.h"
#include "ares_config.h"

#ifdef __cplusplus
extern "C" {
#endif

// INT8 SIMD types for PULP cores
typedef int8_t v4s __attribute__((vector_size(4)));
typedef uint8_t v4u __attribute__((vector_size(4)));

// SIMD dot product with accumulation: acc += a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
// __builtin_pulp_sdotsp4: signed-signed dot product (both operands signed)
#define SumDotpSS(a, b, c) __builtin_pulp_sdotsp4(a, b, c)

// Match NumPy/ONNX rounding (ties-to-even) for quantization paths.
// On GAP9 (RISC-V + FPU), `lrintf()` can be expensive; use a direct FCVT with
// round-to-nearest-even to keep behavior while reducing overhead.
static inline int32_t qround(float x) {
#if defined(__riscv)
    int32_t out;
    __asm__ __volatile__("fcvt.w.s %0, %1, rne" : "=r"(out) : "f"(x));
    return out;
#else
    return (int32_t)lrintf(x);
#endif
}

// Divide with round-to-nearest-even (ties to even). Denominator must be > 0.
static inline int32_t div_round_nearest_even_s64(int64_t num, int64_t den) {
    if (den == 0) return 0;
    int64_t q = num / den;
    int64_t r = num % den;
    if (r == 0) return (int32_t)q;
    int64_t abs_r = r >= 0 ? r : -r;
    int64_t twice_r = abs_r << 1;
    if (twice_r > den || (twice_r == den && (q & 1))) {
        q += (num >= 0) ? 1 : -1;
    }
    return (int32_t)q;
}

// Fixed-point multiply + shift with round-to-nearest-even.
static inline int32_t mul_shift_round_nearest_even(int32_t val, int32_t mul, int shift) {
    int64_t prod = (int64_t)val * (int64_t)mul;
    if (shift <= 0) return (int32_t)prod;

    // Fast path for power-of-two denominator (2^shift) with ties-to-even rounding.
    // Avoids 64-bit division/mod which is expensive on RV32.
    const uint64_t abs_prod = (uint64_t)(prod >= 0 ? prod : -prod);
    const uint64_t q = abs_prod >> shift;
    const uint64_t mask = (1ULL << shift) - 1ULL;
    const uint64_t r = abs_prod & mask;
    const uint64_t half = 1ULL << (shift - 1);

    uint64_t q_rounded = q;
    if (r > half || (r == half && (q & 1ULL))) {
        q_rounded = q + 1ULL;
    }
    return (prod >= 0) ? (int32_t)q_rounded : (int32_t)(-(int64_t)q_rounded);
}

// GAP9 Cluster Configuration: 9 total cores
// Cores 0-7: Worker cores for computation (CL_NUM_CORES=8, defined in ares_config.h)
// Core 8: Cluster controller - receives cluster entry point, orchestrates DMA and forks work to cores 0-7
#define CL_ORCHESTRATOR_CORE_ID (8)  // Cluster controller core (network_cl_entry runs here)

// Reference Conv2D kernel (Single-Core, Debug)
void network_conv2d_reference(
    const int8_t *input,          // INT8 input [C_in, H, W]
    const int8_t *weights,        // INT8 weights [C_out, C_in, K_h, K_w]
    const void *bias,             // INT32 bias [C_out] (void* for flexibility)
    int8_t *output,               // INT8 output [C_out, H_out, W_out]
    uint16_t in_h,
    uint16_t in_w,
    uint16_t in_ch,
    uint16_t out_h,
    uint16_t out_w,
    uint16_t out_ch,
    uint16_t kernel_h,
    uint16_t kernel_w,
    uint16_t weight_row_stride,   // Bytes per output channel in weights (>= in_ch*kernel_h*kernel_w)
    uint16_t stride_h,
    uint16_t stride_w,
    uint16_t pad_h,
    uint16_t pad_w,
    float scale_input,
    float scale_weight,
    float scale_output,
    struct pi_device *cluster_dev);

// True INT8 Conv2D kernel: signed activations/weights, INT32 accumulators with FP rescale
void network_conv2d_int8(
    const int8_t *input,          // INT8 input [C_in, H, W]
    const int8_t *weights,        // INT8 weights [C_out, C_in, K_h, K_w]
    const void *bias,             // INT32 bias [C_out] (void* for flexibility)
    int8_t *output,               // INT8 output [C_out, H_out, W_out]
    uint16_t in_h,
    uint16_t in_w,
    uint16_t in_ch,
    uint16_t out_h,
    uint16_t out_w,
    uint16_t out_ch,
    uint16_t kernel_h,
    uint16_t kernel_w,
    uint16_t weight_row_stride,   // Bytes per output channel in weights (>= in_ch*kernel_h*kernel_w)
    uint16_t stride_h,
    uint16_t stride_w,
    uint16_t pad_h,
    uint16_t pad_w,
    float scale_input,
    float scale_weight,
    float scale_output,
    struct pi_device *cluster_dev);

// HWC Conv2D kernel: optimized for Height-Width-Channel layout
// For networks with small channel counts, HWC enables efficient SIMD.
// Input/Output: [H, W, C] layout where channels are contiguous at each spatial position.
// Specialized fast paths for 1xK and Kx1 kernels.
void network_conv2d_int8_hwc(
    const int8_t *input,          // INT8 input [H, W, C_in] in HWC layout
    const int8_t *weights,        // INT8 weights [C_out, kernel_h * kernel_w * C_in]
    const void *bias,             // INT32 bias [C_out]
    int8_t *output,               // INT8 output [H_out, W_out, C_out] in HWC layout
    uint16_t in_h,
    uint16_t in_w,
    uint16_t in_ch,
    uint16_t out_h,
    uint16_t out_w,
    uint16_t out_ch,
    uint16_t kernel_h,
    uint16_t kernel_w,
    uint16_t weight_row_stride,   // Weight stride per output channel (0 = use in_ch*kh*kw, for SIMD padding)
    uint16_t stride_h,
    uint16_t stride_w,
    uint16_t pad_h,
    uint16_t pad_w,
    float scale_input,
    float scale_weight,
    float scale_output,
    uint16_t out_ch_stride,       // Output channel stride (0 = use out_ch, for Ko-tiling use total_out_ch)
    uint16_t out_ch_offset);      // Output channel offset (for Ko-tiling)

// Depthwise Conv2D kernel: each output channel convolves with one input channel (groups=channels)
// Input: [H, W, C] in HWC layout, Weights: [C, 1, kernel_h, kernel_w], Output: [H_out, W_out, C]
// Supports Ko-tiling (weight tiling over output channels) via total_channels and ch_offset
void network_conv2d_depthwise_int8(
    const int8_t *input,          // INT8 input [H, W, C] in HWC layout
    const int8_t *weights,        // INT8 weights [tile_ch, kernel_h * kernel_w]
    const void *bias,             // INT32 bias [tile_ch] (already offset to current tile)
    int8_t *output,               // INT8 output [H_out, W_out, C] in HWC layout
    uint16_t in_h,
    uint16_t in_w,
    uint16_t channels,            // Number of channels to process (tile size)
    uint16_t out_h,
    uint16_t out_w,
    uint16_t kernel_h,
    uint16_t kernel_w,
    uint16_t stride_h,
    uint16_t stride_w,
    uint16_t pad_h,
    uint16_t pad_w,
    float scale_input,
    float scale_weight,
    float scale_output,
    uint16_t total_channels,      // Total channels in input/output (0 = use channels)
    uint16_t ch_offset);          // Starting channel offset for Ko-tiling (0 = no offset)

// Simple inplace operations (parallel across cores)
void relu_int8_inplace(int8_t *data, size_t size);
void requantize_int8_inplace(int8_t *data, size_t size, float scale_in, float scale_out);

// MaxPool wrapper: INT8 → INT8 (supports non-square kernels)
void network_maxpool_int8(
    const int8_t *input,          // INT8 input [C, H, W]
    int8_t *output,               // INT8 output [C, H', W']
    uint16_t in_h,
    uint16_t in_w,
    uint16_t channels,
    uint16_t out_h,
    uint16_t out_w,
    uint16_t kernel_h,
    uint16_t kernel_w,
    uint16_t stride_h,
    uint16_t stride_w,
    uint16_t pad_h,
    uint16_t pad_w
);

// Linear layer producing INT8 output (for intermediate FC layers)
void network_linear_int8(
    const int8_t *input,          // INT8 input vector [dim_in]
    const int8_t *weights,        // INT8 weights [dim_out, dim_in]
    const void *bias,             // INT32 bias [dim_out] (void* for flexibility)
    int8_t *output,               // INT8 output [dim_out]
    uint16_t in_features,
    uint16_t out_features,
    float scale_input,
    float scale_weight,
    float scale_output
);

// Sequential (non-parallelized) Linear layer for use when outer loop is already parallel
// Used by transformer MLP layers where we parallelize over tokens
void network_linear_int8_sequential(
    const int8_t *input,          // INT8 input vector [dim_in]
    const int8_t *weights,        // INT8 weights [dim_out, dim_in]
    const void *bias,             // INT32 bias [dim_out] (void* for flexibility)
    int8_t *output,               // INT8 output [dim_out]
    uint16_t in_features,
    uint16_t out_features,
    float scale_input,
    float scale_weight,
    float scale_output
);

// 2-bit packed weight Linear (parallel) - unpacks on-the-fly
// Weights are packed 4 per byte: bits[1:0]=w0, bits[3:2]=w1, bits[5:4]=w2, bits[7:6]=w3
// Mapping: 0->-1, 1->0, 2->+1 (ternary weights)
void network_linear_2bit_int8(
    const int8_t *input,          // INT8 input vector [dim_in]
    const uint8_t *weights_packed, // Packed 2-bit weights [dim_out, ceil(dim_in/4)]
    const void *bias,             // INT32 bias [dim_out]
    int8_t *output,               // INT8 output [dim_out]
    uint16_t in_features,
    uint16_t out_features,
    float scale_input,
    float scale_weight,
    float scale_output
);

// 2-bit packed weight Linear (sequential) - for use in loops already parallelized
void network_linear_2bit_int8_sequential(
    const int8_t *input,
    const uint8_t *weights_packed,
    const void *bias,
    int8_t *output,
    uint16_t in_features,
    uint16_t out_features,
    float scale_input,
    float scale_weight,
    float scale_output
);

// Multi-token parallel Linear: processes all tokens in parallel across cores
// Weights should be in L1 for best performance (used by MHSA projections)
void network_linear_int8_parallel_tokens(
    const int8_t *input,       // [seq_len, in_features] in L2
    const int8_t *weights,     // [out_features, in_features] (preferably in L1)
    const int32_t *bias,       // [out_features] in L2
    int8_t *output,            // [seq_len, out_features] in L2
    int seq_len,
    int in_features,
    int out_features,
    float scale_input,
    float scale_weight,
    float scale_output
);

// Multi-token parallel Linear with strided output:
// - Writes a contiguous output tile per token into a larger [seq_len, out_stride] output tensor.
// - `output` should point at output[0, out_offset] and `out_stride` must be the full row stride.
void network_linear_int8_parallel_tokens_strided_out(
    const int8_t *input,       // [seq_len, in_features] in L2
    const int8_t *weights,     // [out_features, in_features] (preferably in L1)
    const int32_t *bias,       // [out_features] in L2 (tile start)
    int8_t *output,            // base pointer to output[0, out_offset]
    int seq_len,
    int in_features,
    int out_features,
    int out_stride,
    float scale_input,
    float scale_weight,
    float scale_output
);

// Fused Q/K/V projection: computes all 3 projections in a single kernel launch
// This reduces fork/barrier overhead from 3 to 1 for MHSA projections.
void network_linear_int8_fused_qkv(
    const int8_t *input,
    const int8_t *q_weights, const int8_t *k_weights, const int8_t *v_weights,
    const int32_t *q_bias, const int32_t *k_bias, const int32_t *v_bias,
    int8_t *q_output, int8_t *k_output, int8_t *v_output,
    int seq_len, int embed_dim,
    float scale_input,
    float scale_q_weight, float scale_k_weight, float scale_v_weight,
    float scale_q, float scale_k, float scale_v
);

// Fused Q/K/V projection (rectangular): computes 3 projections in a single launch.
// Used to fuse head-slice projections (e.g. out_features=head_dim, in_features=embed_dim).
void network_linear_int8_fused_qkv_parallel_tokens_rect(
    const int8_t *input,
    const int8_t *q_weights, const int8_t *k_weights, const int8_t *v_weights,
    const int32_t *q_bias, const int32_t *k_bias, const int32_t *v_bias,
    int8_t *q_output, int8_t *k_output, int8_t *v_output,
    int seq_len, int in_features, int out_features,
    float scale_input,
    float scale_q_weight, float scale_k_weight, float scale_v_weight,
    float scale_q, float scale_k, float scale_v
);

// Linear classifier: INT8 → FP32 (adds FP32 bias internally)
void network_linear_int8_to_fp32(
    const int8_t *input,          // INT8 input vector [dim_in]
    const int8_t *weights,        // INT8 weights [dim_out, dim_in]
    const void *bias,             // FP32 bias [dim_out] (void* for flexibility)
    float *output,                // FP32 logits [dim_out]
    uint16_t in_features,
    uint16_t out_features,
    float scale_input,
    float scale_weight
);

// AvgPool2d: INT8 average pooling with rescaling (supports non-square kernels)
void network_avgpool_int8(
    const int8_t *input,          // INT8 input [C, H, W]
    int8_t *output,               // INT8 output [C, H', W']
    uint16_t in_h,
    uint16_t in_w,
    uint16_t channels,
    uint16_t out_h,
    uint16_t out_w,
    uint16_t kernel_h,
    uint16_t kernel_w,
    uint16_t stride_h,
    uint16_t stride_w,
    float scale_input,
    float scale_output
);

// Element-wise Add: INT8 + INT8 → INT8 with scale matching
void network_add_int8(
    const int8_t *input1,         // INT8 input 1
    const int8_t *input2,         // INT8 input 2
    int8_t *output,               // INT8 output
    uint32_t size,                // Total number of elements
    float scale_x1,
    float scale_x2,
    float scale_output
);

// Channel Concatenation: Multiple INT8 → INT8 with rescaling
void network_concat_int8(
    const int8_t **inputs,        // Array of INT8 input pointers
    const float *scales_input,    // Array of input scales
    int8_t *output,               // INT8 output
    uint16_t num_inputs,          // Number of inputs to concatenate
    uint16_t batch,
    const uint16_t *channels_per_input,  // Array of channel counts
    uint16_t height,
    uint16_t width,
    float scale_output
);

// Global Average Pooling: INT8 [B,C,H,W] → INT8 [B,C,1,1]
void network_global_avgpool_int8(
    const int8_t *input,          // INT8 input [B, C, H, W]
    int8_t *output,               // INT8 output [B, C, 1, 1]
    uint16_t batch,
    uint16_t channels,
    uint16_t height,
    uint16_t width,
    float scale_input,
    float scale_output
);

// Global Average Pooling HWC: INT8 [H,W,C] → INT8 [1,1,C]
void network_global_avgpool_int8_hwc(
    const int8_t *input,          // INT8 input [H, W, C]
    int8_t *output,               // INT8 output [C]
    uint16_t channels,
    uint16_t height,
    uint16_t width,
    float scale_input,
    float scale_output
);

// AdaptiveAvgPool1d: INT8 [B, C, L] -> INT8 [B, C, output_size]
// Uses integer sum/avg (scale preserved).
void network_adaptive_avgpool1d_int8(
    const int8_t *input,
    int8_t *output,
    uint16_t batch,
    uint16_t channels,
    uint16_t input_len,
    uint16_t output_size,
    uint16_t input_stride_ch,
    uint16_t input_stride_len,
    uint32_t input_batch_stride
);

// Mean pooling over sequence dimension: INT8 [B, seq_len, features] -> INT8 [B, features]
// Used for transformer classification heads (mean over sequence tokens).
void network_mean_pool_int8(
    const int8_t *input,
    int8_t *output,
    uint32_t batch,
    uint32_t seq_len,
    uint32_t features,
    float scale_input,
    float scale_output
);

// Embedding lookup: indices (INT32) -> embeddings (INT8 rows)
void network_embedding_int8_parallel(
    const int32_t *indices,
    const int8_t *weight,
    int8_t *output,
    uint32_t num_indices,
    uint32_t embed_dim,
    uint32_t vocab_size
);

// i-GroupNorm: Integer mean/variance + integer sqrt for bit-exact matching.
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

// Fixed-point RFFT feature extractor (currently optimized for patch_size=40).
// Output layout per patch: [mag[0..N/2], phase[0..N/2]] as INT8 with a single scale.
void network_rfft_features_int8_parallel(
    const int8_t *input,
    int8_t *output,
    uint32_t num_patches,
    uint32_t patch_size,
    float scale_input,
    float scale_output
);

// RoPE (rotary position embeddings) applied in-place to a head-contiguous tensor:
//   data layout: [num_heads, seq_len, head_dim] (INT8), with Q15 sin/cos tables.
void network_rope_int8_inplace_parallel(
    int8_t *data,
    const int16_t *rope_cos_q15,
    const int16_t *rope_sin_q15,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim,
    uint32_t pos_offset
);

// Cross-Attention (learned queries attend to KV input), integer-softmax path.
// Layout:
//   kv_input:     [batch, kv_len, embed_dim] INT8
//   query_embed:  [num_queries, embed_dim] INT8
//   output:       [batch, num_queries, embed_dim] INT8
//
// Scratch buffers must be in L2 and correctly sized:
//   q_proj_out:   [num_queries, embed_dim]
//   k_proj_out:   [batch*kv_len, embed_dim]
//   v_proj_out:   [batch*kv_len, embed_dim]
//   context_out:  [batch*num_queries, embed_dim]
void network_cross_attention_int8_parallel(
    const int8_t *kv_input,
    int8_t *output,
    int8_t *q_proj_out,
    int8_t *k_proj_out,
    int8_t *v_proj_out,
    int8_t *context_out,
    const int8_t *query_embed,
    const int8_t *q_weight, const int32_t *q_bias,
    const int8_t *k_weight, const int32_t *k_bias,
    const int8_t *v_weight, const int32_t *v_bias,
    const int8_t *out_weight, const int32_t *out_bias,
    uint32_t batch,
    uint32_t kv_len,
    uint32_t num_queries,
    uint32_t embed_dim,
    uint32_t num_heads,
    float scale_kv_input,
    float scale_query_input,
    float scale_q_weight,
    float scale_k_weight,
    float scale_v_weight,
    float scale_out_weight,
    float scale_q,
    float scale_k,
    float scale_v,
    float scale_output,
    float softmax_scale,
    int32_t requant_mul,
    int32_t requant_shift,
    int8_t *l1_scratch,
    size_t l1_scratch_size
);

// Reusable attention core: takes pre-projected Q/K/V and computes
// integer softmax → context → output projection.
//   q_projected:    [q_tokens, embed_dim] INT8 (q_tokens = batch*q_len for cross-attn,
//                    or just num_queries for cross-attn with shared Q across batch)
//   k_projected:    [batch*kv_len, embed_dim] INT8
//   v_projected:    [batch*kv_len, embed_dim] INT8
//   output:         [batch*q_len, embed_dim] INT8
//   context_scratch:[batch*q_len, embed_dim] INT8 (intermediate buffer)
//
// When q_shared_across_batch=1, q_projected has shape [q_len, embed_dim] and is
// reused for every batch element (optimization for learned queries).
void network_attention_core_int8(
    const int8_t *q_projected,
    const int8_t *k_projected,
    const int8_t *v_projected,
    const int8_t *out_weight, const int32_t *out_bias,
    int8_t *output,
    int8_t *context_scratch,
    uint32_t batch, uint32_t q_len, uint32_t kv_len,
    uint32_t embed_dim, uint32_t num_heads,
    float scale_q, float scale_k, float scale_v,
    float scale_out_weight, float scale_output,
    int32_t requant_mul, int32_t requant_shift,
    int q_shared_across_batch,
    int8_t *l1_scratch, size_t l1_scratch_size
);

/* ---
 * RMSNorm - Root Mean Square Layer Normalization (used in Llama/LLMs)
 * Simpler than LayerNorm - no mean subtraction, only RMS normalization.
 * Formula: y = (x / rms(x)) * weight, where rms(x) = sqrt(mean(x^2) + eps)
 * --- */

/* RMSNorm with FP32-based statistics (sequential) */
void network_rmsnorm_int8_fp32(
    const int8_t *input,
    int8_t *output,
    const float *weight,
    uint32_t total_elements,
    uint32_t normalized_dim,
    float scale_input,
    float scale_output,
    float eps
);

/* i-RMSNorm: Integer-only for bit-exact matching (sequential) */
void network_rmsnorm_int8_integer(
    const int8_t *input,
    int8_t *output,
    const float *weight,
    uint32_t total_elements,
    uint32_t normalized_dim,
    float scale_input,
    float scale_output
);

/* i-RMSNorm with parallel execution across vectors (8 cores) */
void network_rmsnorm_int8_integer_parallel(
    const int8_t *input,
    int8_t *output,
    const float *weight,
    uint32_t total_elements,
    uint32_t normalized_dim,
    float scale_input,
    float scale_output
);

/* --- LayerNorm --- */

void network_layernorm_int8(
    const int8_t *input,
    int8_t *output,
    const float *weight,
    const float *bias,
    uint32_t total_elements,
    uint32_t normalized_dim,
    float scale_input,
    float scale_output,
    float eps
);

void network_layernorm_int8_fixed_point(
    const int8_t *input,
    int8_t *output,
    const float *weight,
    const float *bias,
    uint32_t total_elements,
    uint32_t normalized_dim,
    float scale_input,
    float scale_output,
    float eps
);

// i-LayerNorm: Integer-only LayerNorm for bit-exact matching with Python
// Uses INT64 accumulation, integer division, and binary search sqrt
void network_layernorm_int8_integer(
    const int8_t *input,
    int8_t *output,
    const float *weight,
    const float *bias,
    uint32_t total_elements,
    uint32_t normalized_dim,
    float scale_input,
    float scale_output
);

void network_layernorm_int8_integer_parallel(
    const int8_t *input,
    int8_t *output,
    const float *weight,
    const float *bias,
    uint32_t total_elements,
    uint32_t normalized_dim,
    float scale_input,
    float scale_output
);

void network_gelu_int8(
    const int8_t *input,
    int8_t *output,
    uint32_t num_elements,
    float scale_input,
    float scale_output
);

void network_gelu_int8_inplace(
    int8_t *buffer,
    uint32_t num_elements,
    float scale_input,
    float scale_output
);

// i-GELU: LUT-based GELU for bit-exact matching with Python
// Uses precomputed lookup table instead of tanhf() to avoid FP variations
void network_gelu_int8_lut_inplace(
    int8_t *buffer,
    uint32_t num_elements,
    float scale_input,
    float scale_output
);

// i-GELU: parallel LUT-based GELU (cluster cores)
void network_gelu_int8_lut_inplace_parallel(
    int8_t *buffer,
    uint32_t num_elements,
    float scale_input,
    float scale_output
);

// i-GELU LUT (defined in network_kernels.c)
extern const int16_t i_gelu_lut[256];

void network_mhsa_int8(
    const int8_t *input,
    int8_t *output,
    const int8_t *q_weight,
    const float *q_bias,
    const int8_t *k_weight,
    const float *k_bias,
    const int8_t *v_weight,
    const float *v_bias,
    const int8_t *out_weight,
    const float *out_bias,
    uint16_t seq_len,
    uint16_t embed_dim,
    uint16_t num_heads,
    float scale_input,
    float scale_q_weight,
    float scale_k_weight,
    float scale_v_weight,
    float scale_out_weight,
    float softmax_scale,
    float scale_output,
    uint8_t pool_mode,
    struct pi_device *cluster_dev
);

// Helper: Project input to Q/K/V and quantize (Path B hybrid)
// Computes ACTUAL scales based on FP32 ranges (not theoretical scale_input x scale_weight)
void project_and_quantize_qkv(
    const int8_t *input_int8,
    int8_t *q_int8,
    int8_t *k_int8,
    int8_t *v_int8,
    const int8_t *q_weight,
    const float *q_bias,
    const int8_t *k_weight,
    const float *k_bias,
    const int8_t *v_weight,
    const float *v_bias,
    uint16_t seq_len,
    uint16_t embed_dim,
    float scale_input,
    float scale_q_weight,
    float scale_k_weight,
    float scale_v_weight,
    float scale_q_output_theoretical,
    float scale_k_output_theoretical,
    float scale_v_output_theoretical,
    float *actual_scale_q_out,
    float *actual_scale_k_out,
    float *actual_scale_v_out
);

// Helper: Project input to Q/K/V using INT8 projections (Path A pure INT8)
// Uses dynamic scale calibration + INT8 network_linear_int8() for projections
void project_qkv_int8(
    const int8_t *input_int8,
    int8_t *q_int8,
    int8_t *k_int8,
    int8_t *v_int8,
    const int8_t *q_weight,
    const float *q_bias_fp32,
    const int8_t *k_weight,
    const float *k_bias_fp32,
    const int8_t *v_weight,
    const float *v_bias_fp32,
    uint16_t seq_len,
    uint16_t embed_dim,
    float scale_input,
    float scale_q_weight,
    float scale_k_weight,
    float scale_v_weight,
    float *actual_scale_q_out,
    float *actual_scale_k_out,
    float *actual_scale_v_out
);

// Helper: Apply output projection and pooling (Path B hybrid)
void output_project_and_pool(
    const int8_t *context_int8,
    int8_t *output_int8,
    const int8_t *out_weight,
    const float *out_bias,
    uint16_t seq_len,
    uint16_t embed_dim,
    float scale_context,
    float scale_out_weight,
    float scale_output,
    uint8_t pool_mode
);

// Helper: Fast exp approximation (used by MHSA softmax)
float fast_exp(float x);

// i-Softmax LUT for bit-exact matching (defined in network_kernels.c)
// LUT contains exp(x) * 32767 for x in [-8.0, 0.0] with 1024 entries
extern const int16_t i_softmax_lut[1024];

// i-Softmax row function - applies LUT-based softmax to a single row
void i_softmax_row(
    const float *scores_row,
    float *attn_row,
    int seq_len,
    const int16_t *softmax_lut
);

// ---
// Performance Counters (Milestone 2.1)
// ---

#ifdef ENABLE_PERF_COUNTERS

// Per-layer performance metrics using GAP9 hardware counters
typedef struct {
    uint32_t compute_cycles;      // Cores 0-7 execution time (cycles)
    uint32_t dma_load_cycles;     // L2→L1 transfer time (cycles)
    uint32_t dma_store_cycles;    // L1→L2 transfer time (cycles)
    uint32_t total_cycles;        // Total layer execution time
    uint32_t idle_cycles;         // Waiting time (compute or DMA stalls)
    float overlap_percent;        // Compute/transfer overlap: 100 * (1 - idle/total)
    size_t l1_peak_bytes;         // Peak L1 memory usage for this layer
} layer_perf_t;

// Performance counter utilities
void perf_counter_init(void);
void perf_counter_reset(void);
uint32_t perf_counter_get_cycles(void);

// Per-layer performance tracking
void perf_layer_start(const char *layer_name);
void perf_layer_end(const char *layer_name, layer_perf_t *perf);
void perf_layer_record(const char *layer_name, const layer_perf_t *perf);
void perf_summary_print(void);

// DMA-specific timing (for measuring load/store separately)
void perf_dma_load_start(void);
uint32_t perf_dma_load_end(void);
void perf_dma_store_start(void);
uint32_t perf_dma_store_end(void);

// Compute-specific timing
void perf_compute_start(void);
uint32_t perf_compute_end(void);

#endif // ENABLE_PERF_COUNTERS

// ---
// Kernel Worker Types and Functions
// ---

// Conv2D tile worker function (executed by Cores 0-7)
// For L1 tiling: Core 8 has already loaded tile to L1 via DMA, padding=0 (tile has halo)
// For L2-only: Operates directly on L2 buffers, padding=actual padding value
typedef struct {
    int8_t *tile_input_l1;
    int8_t *tile_output_l1;
    const int8_t *weights_l2;
    uint16_t weight_row_stride;   // Bytes per output channel in weights_l2 (>= in_ch*kernel_h*kernel_w); 0 = default
    const int32_t *bias_l2;
    uint16_t tile_in_h;
    uint16_t tile_in_w;
    uint16_t tile_out_h;
    uint16_t tile_out_w;
    uint16_t in_ch;
    uint16_t out_ch;
    uint16_t groups;             // 1=standard conv, in_ch=depthwise conv
    uint16_t kernel_h;
    uint16_t kernel_w;
    uint16_t stride_h;
    uint16_t stride_w;
    uint16_t pad_h;
    uint16_t pad_w;  // 0 for L1 tiles (halo already included), actual value for L2-only
    float scale_input;
    float scale_weight;
    float scale_output;
    struct pi_device *cluster_dev;
    // Fusion parameters (for look-ahead/look-behind pattern)
    uint8_t fusion_relu;       // 1 if ReLU fusion enabled
    uint8_t fusion_quant;      // 1 if requantization fusion enabled
    float quant_scale_in;      // Input scale for requantization
    float quant_scale_out;     // Output scale for requantization
    // Layout parameter for HWC support
    uint8_t layout;            // 0 = LAYOUT_CHW (default), 1 = LAYOUT_HWC
    // For HWC Ko-tiling: kernel writes to interleaved positions
    uint16_t total_out_ch;     // Total output channels (0 = use out_ch, for Ko-tiling)
    uint16_t out_ch_offset;    // Starting channel offset for this tile (for Ko-tiling)
} conv2d_tile_args_t;

void conv2d_tile_worker(void *arg);
void conv2d_tile_worker_with_fusion(void *arg);

// Linear tile args (for L1 tiling with INT8 output)
typedef struct {
    const int8_t *input_l1;        // Input in L1 (loaded once, reused across tiles)
    const int8_t *weights_l2;      // Weight tile in L2 (or L1 for tiled execution)
    const float *bias_l2;          // Bias in L2 (INT32 for intermediate, float for final)
    int8_t *output_l1;             // Output tile in L1
    uint16_t dim_in;               // Input features
    uint16_t dim_out;              // Output features (tile size)
    float scale_input;
    float scale_weight;
    float scale_output;
} linear_tile_args_t;

void linear_tile_worker(void *arg);

// Linear tile args for batched token processing (reduces fork overhead)
// Processes multiple tokens in a single fork call
typedef struct {
    const int8_t *input_base;      // Base input pointer in L2
    const int8_t *weights_l2;      // Weights in L2
    const float *bias_l2;          // Bias in L2
    int8_t *output_base;           // Base output pointer in L2
    uint16_t dim_in;               // Input features per token
    uint16_t dim_out;              // Output features per token
    uint16_t batch_tokens;         // Number of tokens to process
    float scale_input;
    float scale_weight;
    float scale_output;
} linear_tile_batched_args_t;

void linear_tile_batched_worker(void *arg);

// Linear to FP32 tile args (for final classifier with L1 tiling)
typedef struct {
    const int8_t *input_l1;        // Input in L1 (loaded once, reused across tiles)
    const int8_t *weights_l2;      // Weight tile in L2 (or L1 for tiled execution)
    const float *bias_l2;          // Bias in L2
    float *output_l1;              // Output tile in L1 (FP32)
    uint16_t dim_in;               // Input features
    uint16_t dim_out;              // Output features (tile size)
    float scale_input;
    float scale_weight;
} linear_to_fp32_tile_args_t;

void linear_to_fp32_tile_worker(void *arg);

// Requantize worker for L1 tiles (used by fusion in network.c and network_dma_pipeline.c)
typedef struct { int8_t *data; size_t size; float scale_in; float scale_out; } requantize_l1_args_t;
void requantize_l1_worker(void *arg);

// Transpose 2D INT8 tensor (swap last two dimensions)
void network_transpose_2d_int8(
    const int8_t *input,
    int8_t *output,
    int batch_size,
    int dim1,
    int dim2
);

// Zero-padding 2D INT8 tensor: [C, H, W] -> [C, H+pad_top+pad_bottom, W+pad_left+pad_right]
void network_zeropad2d_int8(
    const int8_t *input,
    int8_t *output,
    int channels,
    int in_h,
    int in_w,
    int pad_left,
    int pad_right,
    int pad_top,
    int pad_bottom
);

// ---
// MHSA Permute Functions (for head-contiguous data layout optimization)
// ---
// Enables bulk DMA per head instead of strided row-by-row transfers

// Permute: [seq_len, embed_dim] -> [num_heads, seq_len, head_dim]
// After permute, each head's data is contiguous for bulk DMA
void mhsa_permute_to_heads(const int8_t *input, int8_t *output,
                           int seq_len, int embed_dim, int num_heads);

// Inverse permute: [num_heads, seq_len, head_dim] -> [seq_len, embed_dim]
void mhsa_permute_from_heads(const int8_t *input, int8_t *output,
                             int seq_len, int embed_dim, int num_heads);

// ---
// MAMBA-Specific Kernels (Integer-Only SSM)
// ---

// Depthwise 1D Convolution: INT8 [B, C, L] -> INT8 [B, C, L]
// Each channel has its own filter (groups=C). Supports causal (left-only) padding.
void network_conv1d_depthwise_int8(
    const int8_t *input,          // INT8 input [C, L] or [B, C, L]
    const int8_t *weights,        // INT8 weights [C, K] (one K-tap filter per channel)
    const int32_t *bias,          // INT32 bias [C] (optional, can be NULL)
    int8_t *output,               // INT8 output [C, L] or [B, C, L]
    int channels,                 // Number of channels
    int length,                   // Input sequence length
    int kernel_size,              // Convolution kernel size (typically 4)
    int causal,                   // 1 for causal (left-only padding), 0 otherwise
    float scale_input,
    float scale_weight,
    float scale_output
);

// FUSED: Conv1D Depthwise + SiLU + Transpose
// Combines three operations into single pass, eliminating intermediate buffer
// Input:  [B, d_inner, L] (channels-first for conv1d)
// Output: [B, L, d_inner] (channels-last for SSM)
void network_conv1d_silu_transpose_fused(
    const int8_t *input,          // INT8 input [B, d_inner, L]
    const int8_t *weights,        // INT8 weights [d_inner, K]
    const int32_t *bias,          // INT32 bias [d_inner] (optional)
    const int8_t *silu_lut,       // INT8 SiLU LUT [256]
    int8_t *output,               // INT8 output [B, L, d_inner] (transposed!)
    int batch,
    int channels,                 // d_inner
    int length,                   // L
    int kernel_size,
    int causal,
    float scale_input,
    float scale_weight,
    float scale_output
);

// FUSED: Conv1D Depthwise + SiLU + Transpose (strided input)
// Input layout: [B, L, input_stride], where x-values are at offsets [t * input_stride + c]
// Output: [B, L, d_inner] (transposed)
void network_conv1d_silu_transpose_fused_strided(
    const int8_t *input,          // INT8 input [B, L, input_stride]
    int input_stride,             // Stride between timesteps (elements)
    const int8_t *weights,        // INT8 weights [d_inner, K]
    const int32_t *bias,          // INT32 bias [d_inner] (optional)
    const int8_t *silu_lut,       // INT8 SiLU LUT [256]
    int8_t *output,               // INT8 output [B, L, d_inner] (transposed!)
    int batch,
    int channels,                 // d_inner
    int length,                   // L
    int kernel_size,
    int causal,
    float scale_input,
    float scale_weight,
    float scale_output
);

// SiLU (Swish) Activation via 256-entry LUT: INT8 -> INT8
// SiLU(x) = x * sigmoid(x), pre-computed for all INT8 values
void network_silu_int8_lut(
    const int8_t *input,          // INT8 input
    int8_t *output,               // INT8 output
    const int8_t *lut,            // 256-entry INT8 LUT (indexed by x+128)
    int num_elements              // Total number of elements
);

// SiLU in-place via 256-entry LUT
void network_silu_int8_lut_inplace(
    int8_t *buffer,               // INT8 buffer (modified in-place)
    const int8_t *lut,            // 256-entry INT8 LUT
    int num_elements
);

// Generate SiLU LUT at runtime (for code generation - typically pre-computed)
void generate_silu_lut_int8(
    int8_t *lut,                  // Output: 256-entry INT8 LUT
    float scale_in,
    float scale_out
);

// Generate SiLU Q2.13 LUT for gating operation
void generate_silu_lut_q13(
    int16_t *lut,                 // Output: 256-entry INT16 Q2.13 LUT
    float scale_in
);

// SSM Forward Pass: Integer-only State Space Model core
// Implements: h_t = dA * h_{t-1} + dB' * x_t; y_t = C * h_t + D * x_t
// I-Mamba step 2c: D parameter now in Q15 format
void network_ssm_forward_int8(
    const int8_t *x_i8,           // Input [seq_len, d_inner] in INT8
    const int32_t *dt_q16,        // Delta timesteps [seq_len, d_inner] in Q16
    const int16_t *A_q15,         // A matrix [d_state, d_inner] in Q15
    const int16_t *B_q15,         // B matrix [seq_len, d_state] in Q15
    const int16_t *C_q15,         // C matrix [seq_len, d_state] in Q15
    const int16_t *D_q15,         // D skip connection [d_inner] in Q15 (can be NULL)
    const int8_t *z_i8,           // Gate input [seq_len, d_inner] in INT8
    int8_t *y_i8,                 // Output [seq_len, d_inner] in INT8
    int16_t *dA_buf,              // Temp buffer [seq_len, d_inner, d_state]
    int16_t *dB_buf,              // Temp buffer [seq_len, d_inner, d_state]
    int32_t *y_acc_buf,           // Temp buffer [seq_len, d_inner]
    int16_t *h_buf,               // Hidden state buffer [d_inner, d_state]
    const int16_t *silu_lut_q13,  // SiLU LUT for gating (256 entries)
    int seq_len,
    int d_inner,
    int d_state,
    float scale_x,
    float scale_y
);

// SSM Discretization: Convert continuous A, B to discrete dA, dB' using Q15 fixed-point
// dA = exp(dt * A), dB' = phi1(-dt * A) * B * dt where phi1(x) = (exp(x) - 1) / x
void ssm_discretize_q15(
    const int32_t *dt_q16,        // Delta timesteps [seq_len, d_inner] in Q16
    const int16_t *A_q15,         // A matrix [d_state, d_inner] in Q15 (negative values)
    const int16_t *B_q15,         // B matrix [seq_len, d_state] in Q15
    int16_t *dA_q15,              // Output: discretized A [seq_len, d_inner, d_state] in Q15
    int16_t *dB_prime_q15,        // Output: discretized B' [seq_len, d_inner, d_state] in Q15
    int seq_len,
    int d_inner,
    int d_state,
    int16_t s_x_q15               // Input scale in Q15
);

// SSM Scan: State update loop h = dA * h + dB' * x; y = C * h + D * x
// I-Mamba step 2c: D parameter in Q15 format for dyadic arithmetic
void ssm_scan_q15(
    const int8_t *x_i8,           // Input [seq_len, d_inner] in INT8
    const int16_t *dA_q15,        // Discretized A [seq_len, d_inner, d_state] in Q15
    const int16_t *dB_prime_q15,  // Discretized B' [seq_len, d_inner, d_state] in Q15
    const int16_t *C_q15,         // C matrix [seq_len, d_state] in Q15
    const int16_t *D_q15,         // D skip connection [d_inner] in Q15 (can be NULL)
    int32_t *y_acc,               // Output accumulator [seq_len, d_inner] in INT32
    int16_t *h_q15,               // Hidden state buffer [d_inner, d_state] in Q15
    int seq_len,
    int d_inner,
    int d_state
);

// Complete SSM Layer: Handles x_proj, dt_proj, softplus, scan, and output
// This matches the Python ssm_layer_forward_int8 function
// I-Mamba: Fully Integer SSM Layer (step 10 - Zero FP32)
// All computations use fixed-point arithmetic with precomputed scale factors
// - dt_proj: INT8 weights + Q16.16 bias + precomputed dt_scale_q
// - B/C: Computed to Q15 using precomputed bc_scale_factor
// - Hidden state: Q15 fixed-point
// - Output: Direct INT8 using precomputed output_scale_q
void network_ssm_layer_int8(
    const int8_t *x_int8,           // Input [batch, seq_len, d_inner]
    int8_t *output_int8,            // Output [batch, seq_len, d_inner]
    const int8_t *x_proj_weight,    // x_proj weights [dt_rank + 2*d_state, d_inner]
    const int8_t *dt_proj_weight,   // dt_proj weights [d_inner, dt_rank] (INT8)
    const int32_t *dt_proj_bias_q16_16,  // dt_proj bias [d_inner] (Q16.16 fixed-point)
    const int16_t *A_q15,           // A matrix [d_state, d_inner] in Q15 (pre-computed -exp(A_log))
    const int16_t *D_q15,           // D skip connection [d_inner] in Q15
    const int16_t *softplus_lut,    // Q8.8 softplus LUT [256]
    const int16_t *exp_lut,         // Q15 exp LUT for discretization [256]
    void *scratch,                  // Scratch buffer for integer intermediates
    int batch,
    int seq_len,
    int d_inner,
    int d_state,
    int dt_rank,
    int32_t dt_scale_q,             // Precomputed: (scale_x * scale_x_proj * scale_dt_proj) * 65536 * 2^24
    int dt_scale_shift,             // Shift amount for dt_scale_q (typically 24)
    int32_t bc_scale_factor,        // Precomputed: scale_x * scale_x_proj * 32768 * 2^16
    int32_t output_scale_q          // Precomputed: scale_x / (32768 * scale_output) * 2^24
);

// SSM Gate: y_output = y_ssm * silu(z)
void ssm_gate_silu_q13(
    const int8_t *y_ssm_int8,         // SSM output [seq_len, d_inner]
    const int8_t *z_int8,             // Gate input [seq_len, d_inner]
    int8_t *output_int8,              // Gated output [seq_len, d_inner]
    const int16_t *silu_lut_q13,      // SiLU LUT in Q2.13
    int num_elements
);

// Pre-computed LUTs for SSM discretization (defined in network_kernels.c)
#define SSM_EXP_LUT_SIZE 512
#define SSM_PHI1_LUT_SIZE 512
#define SSM_SOFTPLUS_LUT_SIZE 512
extern const int16_t ssm_exp_lut_q15[SSM_EXP_LUT_SIZE];
extern const int16_t ssm_phi1_lut_q15[SSM_PHI1_LUT_SIZE];

// Softplus activation for MAMBA dt computation
// softplus(x) = log(1 + exp(x))
void network_softplus_q16(
    const int32_t *input,         // INT32 accumulator from dt_proj
    int32_t *output_q16,          // Q16 fixed-point output
    int num_elements,
    float scale_in                // Scale to convert input to FP32
);

// Element-wise multiply with requantization
// out = a * b (used for MAMBA gating: x * silu(z))
void network_elementwise_mul_int8(
    const int8_t *a,
    const int8_t *b,
    int8_t *output,
    int num_elements,
    float scale_a,
    float scale_b,
    float scale_out
);


// ---
// L1-Tiled Linear Workers (moved from template for code reuse)
// ---

// Args struct for L1-tiled linear INT8 worker
// Weights are pre-loaded to L1 for faster access
typedef struct {
    const int8_t *input;           // Input data in L2
    const int8_t *weights_l1;      // Weights pre-loaded to L1
    const int32_t *bias;           // Bias in L2 (small, okay to not tile)
    int8_t *output;                // Output data in L2
    uint16_t in_features;
    uint16_t out_features;
    uint16_t batch_tokens;
    float scale_in, scale_w, scale_out;
} linear_int8_l1_args_t;

// L1-tiled linear worker: OPTIMIZED VERSION with 4x output + 4x timestep unrolling
// Achieves ~8-12 MACs/cycle vs ~0.05 in naive implementation
// Must be called via pi_cl_team_fork() from cluster controller
void linear_int8_l1_worker(void *arg);

// Sequential strided linear (single output feature at a time with output stride)
void linear_int8_sequential_strided(
    const int8_t *input, const int8_t *weights, const float *bias, int8_t *output,
    uint16_t in_features, uint16_t out_features, uint16_t out_stride,
    float scale_input, float scale_weight, float scale_output
);

#ifdef __cplusplus
}
#endif
