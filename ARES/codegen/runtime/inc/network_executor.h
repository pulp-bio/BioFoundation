/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Network Executor - Data-Driven Layer Dispatch
 *
 * This module provides a runtime dispatch loop that executes layers based on
 * their LayerSpec configurations. It replaces template-time unrolling with
 * a compact runtime dispatcher.
 *
 * The executor iterates over the network_layers[] array and dispatches each
 * layer to the appropriate execute function based on layer->type.
 *
 * IMPORTANT: This file does NOT modify the optimized pipeline functions in
 * network_dma_pipeline.c. All pipeline logic (row-batched DMA, look-ahead,
 * pipelining) remains unchanged.
 */

#ifndef NETWORK_EXECUTOR_H
#define NETWORK_EXECUTOR_H

#include <stdint.h>
#include <stddef.h>
#include "layer_descriptors.h"
#include "network_dma_pipeline.h"
#include "network_kernels.h"

// The executor uses layer_runtime_ctx_t instead of network_cl_args_t directly

// Maximum number of inputs for concat operations (DenseNet-style networks)
#define MAX_CONCAT_INPUTS 8

// Runtime context passed to executor functions
// This struct holds the runtime pointers that can't be known at compile time
typedef struct {
    // L1 buffer for tiled execution
    int8_t *l1_buffer;
    size_t l1_buffer_size;

    // Device handles
    struct pi_device *cluster_dev;
    struct pi_device *ram_dev;

    // Runtime buffer pointers (patched by network_cl_entry)
    int8_t *input_buffer_l2;
    int8_t *output_buffer_l2;
    int8_t *weight_l2;
    void *bias_l2;  // Can be int32_t* or float* depending on layer

    // L3 tiling (for streamed layers)
    void *l3_weight_addr;
    void *l3_bias_addr;
    void *l3_output_addr;
    int l3_tiling_enabled;

    // For layers needing two inputs (Add, Concat)
    int8_t *input_b_buffer_l2;

    // For fused Conv+MaxPool: pooled output goes to a different buffer than conv output
    int8_t *fused_output_buffer_l2;

    // For concat operations (variable number of inputs)
    const int8_t *concat_inputs[MAX_CONCAT_INPUTS];
    float concat_scales[MAX_CONCAT_INPUTS];
    uint16_t concat_channels[MAX_CONCAT_INPUTS];
    int concat_num_inputs;

    // For LayerNorm (weight and bias are FP32)
    const float *layernorm_weight;
    const float *layernorm_bias;

    // For MHSA (4 projection weights + biases)
    int8_t *mhsa_q_weight_l2;
    int32_t *mhsa_q_bias_l2;
    int8_t *mhsa_k_weight_l2;
    int32_t *mhsa_k_bias_l2;
    int8_t *mhsa_v_weight_l2;
    int32_t *mhsa_v_bias_l2;
    int8_t *mhsa_out_weight_l2;
    int32_t *mhsa_out_bias_l2;

    // MHSA intermediate projection buffers
    int8_t *mhsa_q_buffer_l2;
    int8_t *mhsa_k_buffer_l2;
    int8_t *mhsa_v_buffer_l2;

    // Cross-attention specific (query embedding + scratch buffers)
    int8_t *cross_attn_query_embed;
    int8_t *cross_attn_q_proj_out;
    int8_t *cross_attn_k_proj_out;
    int8_t *cross_attn_v_proj_out;
    int8_t *cross_attn_context_out;

    // Alternating attention (Cerebro transformer) weights
    int8_t *alt_attn_qkv_weight_l2;   // Combined QKV projection weight [3*D, D]
    int32_t *alt_attn_qkv_bias_l2;    // Combined QKV projection bias [3*D]
    int8_t *alt_attn_out_weight_l2;   // Output projection weight [D, D]
    int32_t *alt_attn_out_bias_l2;    // Output projection bias [D]

    // NE16-packed alternating attention weights (when use_ne16_qkv/use_ne16_out is set)
    const int8_t *alt_attn_qkv_ne16_packed_l2;   // NE16-packed QKV weights
    const int32_t *alt_attn_qkv_ne16_bias_l2;    // NE16 bias with correction
    const uint8_t *alt_attn_qkv_ne16_scale_l2;   // NE16 HW requant scale
    const uint8_t *alt_attn_qkv_ne16_shift_l2;   // NE16 HW requant shift
    const int8_t *alt_attn_out_ne16_packed_l2;   // NE16-packed output proj weights
    const int32_t *alt_attn_out_ne16_bias_l2;    // NE16 bias with correction
    const uint8_t *alt_attn_out_ne16_scale_l2;   // NE16 HW requant scale
    const uint8_t *alt_attn_out_ne16_shift_l2;   // NE16 HW requant shift

    // Mamba/FEMBA specific context
    void *mamba_scratch;          // Large scratch buffer for Mamba operations
    size_t mamba_scratch_size;    // Size of scratch buffer
    void *pos_embed_weight_l3;    // L3 address of positional embedding weights


    // NE16 accelerator context (packed weights passed at runtime)
    const int8_t *ne16_weights_packed;   // Pre-packed NE16 weights (L2)
    const int32_t *ne16_bias_corrected;  // Bias with -128*sum(weights) correction (L2)
    const uint8_t *ne16_hw_scale;        // Per-channel HW outquant scale (qbias) (L2)
    const uint8_t *ne16_hw_scale_shift;  // Per-channel HW outquant shift (qnorm) (L2)
    int ne16_use_hw_requant;             // >0 force HW outquant, <0 force SW requant, 0 = default

    // NE16 persistent L1 buffers (allocated once, reused across layers)
    // This avoids per-layer L1 alloc/free overhead
    uint8_t *ne16_weight_l1;             // Persistent L1 buffer for weights
    size_t ne16_weight_l1_size;          // Size of L1 weight buffer
    int32_t *ne16_bias_l1;               // Persistent L1 buffer for bias
    size_t ne16_bias_l1_size;            // Size of L1 bias buffer

    // Network-specific args pointer (for complex ops like Mamba that need full context)
    // This allows execute functions to access network_cl_args_t for Mamba-specific buffers
    void *network_args;

#ifdef ARES_LLAMA_SUPPORT
    // KV cache for autoregressive generation
    int8_t *kv_cache_k;            // [n_layers * max_seq_len * kv_dim] in L2 arena
    int8_t *kv_cache_v;            // [n_layers * max_seq_len * kv_dim] in L2 arena
    int kv_cache_pos;              // Current sequence position (shared across layers)
    int kv_cache_max_seq_len;      // Maximum sequence length
    int kv_cache_n_layers;         // Number of transformer layers
    int kv_cache_kv_dim;           // KV dimension (n_kv_heads * head_dim)
    int current_layer_idx;         // Current transformer block index (for KV cache offset)

    // RoPE precomputed tables (Q15 fixed point, shared across layers)
    const int16_t *rope_cos_q15;   // [max_seq_len, head_dim/2]
    const int16_t *rope_sin_q15;   // [max_seq_len, head_dim/2]

    // Autoregressive MHSA scratch buffers
    int8_t *mhsa_ar_q_buffer;     // Scratch for Q projection [dim]
    int8_t *mhsa_ar_k_buffer;     // Scratch for K projection [kv_dim]
    int8_t *mhsa_ar_v_buffer;     // Scratch for V projection [kv_dim]
    float *mhsa_ar_scores;        // Scratch for attention scores [max_seq_len]
    float *mhsa_ar_context;       // Scratch for context vector [dim]

    // Residual buffer for autoregressive loop
    int8_t *residual_buffer_l2;
    float residual_scale;

    // For RMSNorm (weight is FP32, no bias)
    const float *rmsnorm_weight;

    // SwiGLU FFN (Llama-style) weights and scratch buffers
    const int8_t *swiglu_w1_weight;   // Gate projection [hidden_dim, dim]
    const int32_t *swiglu_w1_bias;    // Gate bias (optional, usually NULL for Llama)
    const int8_t *swiglu_w3_weight;   // Up projection [hidden_dim, dim]
    const int32_t *swiglu_w3_bias;    // Up bias (optional, usually NULL for Llama)
    const int8_t *swiglu_w2_weight;   // Down projection [dim, hidden_dim]
    const int32_t *swiglu_w2_bias;    // Down bias (optional, usually NULL for Llama)
    int8_t *swiglu_scratch;           // Scratch buffer for intermediate results
    size_t swiglu_scratch_size;       // Size of scratch buffer
#endif // ARES_LLAMA_SUPPORT

    // Golden comparison (optional)
    int8_t *golden_buffer;
    size_t golden_size;

    // Performance counter (when ENABLE_PERF_COUNTERS is defined)
#ifdef ENABLE_PERF_COUNTERS
    layer_perf_t *perf_counter;
#endif
} layer_runtime_ctx_t;

// Execute a single layer using the appropriate kernel
// Returns 0 on success, non-zero on error
int execute_layer(const LayerSpec *layer, layer_runtime_ctx_t *ctx);

// Execute all layers in sequence
// layers: array of LayerSpec structs
// num_layers: number of layers
// Returns 0 on success, non-zero on error
int execute_all_layers(const LayerSpec layers[], int num_layers,
                       void *network_args);

// Per-op-type execute functions (called by execute_layer)
void execute_conv2d(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
void execute_linear_int8(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
void execute_linear_fp32(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
void execute_maxpool(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
void execute_avgpool(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
void execute_global_avgpool(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
void execute_mhsa(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
void execute_cross_attention(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
void execute_relu(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
void execute_requantize(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
void execute_add(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
void execute_concat(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
void execute_layernorm(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
void execute_gelu(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
void execute_transpose_2d(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
void execute_embedding(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
void execute_groupnorm(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
void execute_rfft(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
void execute_conv1d_depthwise(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
void execute_silu(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
void execute_ssm(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
void execute_mamba_block(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
void execute_mamba_wrapper(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
void execute_patch_embed(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
void execute_pos_embed(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
void execute_zeropad2d(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
void execute_chw_to_hwc(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
void execute_hwc_to_chw(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
void execute_mean(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
void execute_alternating_attention(const LayerSpec *layer, layer_runtime_ctx_t *ctx);

// LUNA composite operations (dispatched via template-generated code)
void execute_cross_attn_self_refine(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
void execute_classification_head_mlp(const LayerSpec *layer, layer_runtime_ctx_t *ctx);

// Llama/LLM operations (conditional to avoid code bloat)
#ifdef ARES_LLAMA_SUPPORT
void execute_rmsnorm(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
void execute_swiglu_ffn(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
void execute_llama_block(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
void execute_mhsa_autoregressive(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
void execute_residual_add(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
int argmax_fp32(const float *logits, int vocab_size);
#endif // ARES_LLAMA_SUPPORT

// NE16 accelerator operations
#ifdef ARES_USE_NE16
void execute_linear_ne16(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
void execute_conv2d_1x1_ne16(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
void execute_conv2d_3x3_ne16(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
#ifdef ARES_NE16_DEPTHWISE
void execute_conv2d_3x3_dw_ne16(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
void execute_conv2d_3x3_dw_ne16_tiled(const LayerSpec *layer, layer_runtime_ctx_t *ctx);
#endif // ARES_NE16_DEPTHWISE
#endif // ARES_USE_NE16

// Golden validation helper
void validate_layer_output(const char *layer_name, const int8_t *output,
                           const int8_t *golden, size_t size);

#endif // NETWORK_EXECUTOR_H
