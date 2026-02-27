/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

// Layer type enums and per-layer parameter structs for the data-driven executor.
#ifndef LAYER_DESCRIPTORS_H
#define LAYER_DESCRIPTORS_H

#include "pmsis.h"
#include "network_dma_pipeline.h" // For pipeline configs

// ---
// Simple Layer Types (for data-driven executor)
// ---
// These are ops that have uniform interfaces and can be dispatched via tables
// rather than explicit generated code.

typedef enum {
    SIMPLE_RELU = 0,
    SIMPLE_GELU,
    SIMPLE_SILU,
    SIMPLE_SOFTMAX,
    SIMPLE_ADD,
    SIMPLE_FLATTEN,
    SIMPLE_SQUEEZE,
    SIMPLE_TRANSPOSE_2D,
    SIMPLE_MAXPOOL2D,
    SIMPLE_AVGPOOL2D,
    SIMPLE_ADAPTIVE_AVGPOOL1D,
    SIMPLE_REQUANTIZE,
    // Sentinel for bounds checking
    SIMPLE_OP_COUNT
} SimpleOpType;

// Buffer slot: maps slot index to L2 arena offset
typedef struct {
    uint32_t offset;    // Offset in L2 arena
    uint32_t size;      // Buffer size in bytes
} BufferSlot;

// Compact descriptor for data-driven ops
// Uses buffer slot indices instead of direct pointers for compactness
typedef struct {
    SimpleOpType type;
    uint16_t input_slot;      // Index into buffer table
    uint16_t output_slot;     // Index into buffer table (same for in-place)

    union {
        // Activation ops (ReLU, GELU, SiLU, Softmax)
        struct {
            uint32_t numel;
            float scale_in;
            float scale_out;
        } activation;

        // Element-wise binary (Add)
        struct {
            uint16_t input2_slot;
            uint16_t _pad;
            uint32_t numel;
            float scale_a;
            float scale_b;
            float scale_out;
        } binary;

        // Pooling (MaxPool, AvgPool)
        struct {
            uint16_t h, w, c;         // Input dimensions
            uint16_t out_h, out_w;    // Output dimensions
            uint16_t kh, kw;          // Kernel size
            uint16_t stride_h, stride_w;
            uint16_t pad_h, pad_w;
            float scale_in;           // For AvgPool (ignored by MaxPool)
            float scale_out;          // For AvgPool (ignored by MaxPool)
            uint8_t layout;           // 0=CHW, 1=HWC
            uint8_t _pad[3];
        } pool;

        // Transpose 2D
        struct {
            uint16_t dim0, dim1, dim2;
            uint8_t perm[3];
            uint8_t _pad;
            float scale;
        } transpose;

        // Shape ops (Flatten, Squeeze) - no extra params needed
        struct {
            uint32_t numel;  // For verification only
        } shape;

        // Requantize
        struct {
            uint32_t numel;
            float scale_in;
            float scale_out;
        } requantize;

        // Adaptive average pooling 1D
        struct {
            uint16_t batch;
            uint16_t channels;
            uint16_t length;
            uint16_t output_size;
            uint16_t stride_ch;       // Channel stride in input
            uint16_t stride_len;      // Length stride in input
            uint32_t batch_stride;    // Batch stride in input
        } adaptive_pool;
    } params;
} SimpleLayerSpec;

// ---
// Layer Types (for explicit/complex ops - existing enum)
// ---
typedef enum {
    OP_CONV2D = 0,
    OP_LINEAR_INT8,
    OP_LINEAR_FP32,
    OP_MAXPOOL,
    OP_AVGPOOL,
    OP_GLOBAL_AVGPOOL,
    OP_MHSA,
    OP_CROSS_ATTENTION,
    OP_ADD,
    OP_CONCAT,
    OP_RELU,
    OP_REQUANTIZE,
    OP_LAYERNORM,
    OP_GELU,
    OP_SOFTMAX,
    OP_FLATTEN,
    OP_SQUEEZE,
    OP_RESHAPE,
    OP_TRANSPOSE_2D,
    OP_ZEROPAD2D,
    OP_EMBEDDING,
    OP_GROUPNORM,
    OP_RFFT,
    // MAMBA-specific operations
    OP_CONV1D_DEPTHWISE,
    OP_SILU,
    OP_SSM,
    OP_MAMBA_BLOCK,
    OP_MAMBA_WRAPPER,
    // FEMBA-specific operations
    OP_PATCH_EMBED,
    OP_POS_EMBED,
    // Layout conversion operations
    OP_CHW_TO_HWC,
    OP_HWC_TO_CHW,
    // NE16 accelerator operations
    OP_LINEAR_NE16,
    OP_CONV2D_1X1_NE16,
    OP_CONV2D_3X3_NE16,
    OP_CONV2D_3X3_DW_NE16,  // Depthwise 3x3 using NE16 hardware
    // Pooling operations
    OP_MEAN,
    // Cerebro transformer operations
    OP_ALTERNATING_ATTENTION,
    // LUNA composite operations (template-generated, bypass executor)
    OP_CROSS_ATTN_SELF_REFINE,
    OP_CLASSIFICATION_HEAD_MLP,
#ifdef ARES_LLAMA_SUPPORT
    // Llama/LLM operations (conditional to avoid code bloat)
    OP_RMSNORM,                 // RMSNorm for Llama/LLMs (simpler than LayerNorm)
    OP_SWIGLU_FFN,              // SwiGLU Feed-Forward Network (Llama-style)
    OP_LLAMA_BLOCK,             // Complete Llama transformer decoder block
    OP_MHSA_AUTOREGRESSIVE,     // Single-token MHSA with KV cache for autoregressive generation
    OP_RESIDUAL_ADD,            // Residual add (FP32 accumulation for autoregressive loop)
#endif
    OP_UNKNOWN
} OpType;

// Generic Layer Descriptor
typedef struct {
    OpType type;
    const char *name;
    
    union {
        conv2d_pipeline_config_t      conv2d;
        linear_int8_pipeline_config_t linear_int8;
        linear_fp32_pipeline_config_t linear_fp32;
        mhsa_pipeline_config_t        mhsa;

        // NE16-accelerated linear layer parameters
        struct {
            int batch;
            int num_tokens;           // Sequence length (tokens)
            int in_features;          // Input feature dimension
            int out_features;         // Output feature dimension
            int out_stride;           // Stride between output rows
            float scale_input;        // Input quantization scale
            float scale_weight;       // Weight quantization scale
            float scale_output;       // Output quantization scale
            int tile_tokens;          // Number of tokens per NE16 tile
            // Pointers to pre-packed weights and corrected bias
            const int8_t *weights_packed;   // Pre-packed weights from ne16_packing.py
            const int32_t *bias_corrected;  // Bias with input_zp correction
            // HW outquant parameters (reference-compatible)
            const uint8_t *hw_scale;        // Per-output-channel multiplier (qbias)
            const uint8_t *hw_scale_shift;  // Per-output-channel shift (qnorm)
            int use_hw_requant;             // 1 = HW outquant, 0 = SW requant
            // Scratch buffer sizes (for arena allocation)
            size_t scratch_input_size;      // tile_tokens * in_features
            size_t scratch_output_size;     // tile_tokens * out_features * 4 (or * 1 for HW outquant)
        } linear_ne16;

        struct {
            int batch;
            int kv_len;
            int num_queries;
            int embed_dim;
            int num_heads;
            int head_dim;
            float scale_kv_in;
            float scale_query_in;
            float scale_q_weight, scale_k_weight, scale_v_weight, scale_out_weight;
            float scale_q, scale_k, scale_v, scale_output;
            float softmax_scale;
        } cross_attention;
        
        //  Added structs for simple layers so union size is correct
        struct {
            int in_h, in_w, channels;
            int out_h, out_w;
            int kernel_h, kernel_w;     // Non-square pooling support
            int stride_h, stride_w;     // Non-square stride support
            int pad_h, pad_w;           // Non-square padding support
            int fusion_quant;
            float quant_scale_in, quant_scale_out;
            int num_tiles, num_tiles_h, num_tiles_w;
            int tile_h, tile_w;
            size_t l1_input_size, l1_output_size;
            int out_tile_h, out_tile_w;
            int l3_tiling_enabled;
            int l3_tile_h, l3_tile_h_halo, num_l3_tiles;
            TensorLayout layout;  // LAYOUT_CHW or LAYOUT_HWC
        } maxpool;

        struct {
            int in_h, in_w, channels;
            int out_h, out_w;
            int kernel_h, kernel_w;     // Non-square pooling support
            int stride_h, stride_w;     // Non-square stride support
            float scale_in, scale_out;
            int fusion_quant;
            float quant_scale_in, quant_scale_out;
            int num_tiles, num_tiles_h, num_tiles_w;
            int tile_h, tile_w;
            size_t l1_input_size, l1_output_size;
            int out_tile_h, out_tile_w;
            int l3_tiling_enabled;
            int l3_tile_h, l3_tile_h_halo, num_l3_tiles;
            TensorLayout layout;  // LAYOUT_CHW or LAYOUT_HWC
        } avgpool;

        struct {
            int batch, channels, h, w;
            float scale_in, scale_out;
            TensorLayout layout;  // LAYOUT_CHW or LAYOUT_HWC
        } global_avgpool;
        
        struct {
            void *buffer;
            int size;
            float scale_in, scale_out;  // For requantization in ReLU
        } relu;

        struct {
            void *buffer;
            int size;
            float scale_in, scale_out;
        } requantize;
        
        struct {
            int size;
            float scale_a, scale_b, scale_out;
        } add;

        struct {
            int num_inputs;
            int height, width;
            float scale_output;
        } concat;

        struct {
            int num_tokens, embed_dim;
            float scale_in, scale_out;
        } layernorm;

        struct {
            int num_elements;
            float scale_in, scale_out;
        } gelu;
        
        struct {
            int batch, dim0, dim1;
            float scale;
        } transpose_2d;

        struct {
            int vocab_size;
            int embed_dim;
            int num_indices;
            float scale_out;
        } embedding;

        struct {
            int batch;
            int channels;
            int spatial_size;
            int num_groups;
            float scale_in, scale_out;
        } groupnorm;

        // Mean pooling (e.g., mean over sequence dimension)
        struct {
            int batch;
            int seq_len;          // Dimension to reduce over
            int features;         // Feature dimension (preserved)
            int dim;              // Axis to reduce (typically 1 for sequence)
            int keepdim;          // Whether to keep the reduced dimension
            float scale_in, scale_out;
        } mean;

        // Alternating Attention (Cerebro transformer)
        struct {
            int batch;
            int seq_len;           // = num_channels * temporal_len
            int embed_dim;
            int num_heads;
            int head_dim;
            int num_channels;      // Number of EEG channels
            int temporal_len;      // Number of time steps
            int block_idx;         // Block index (even=channel attn, odd=temporal attn)
            float scaling_factor;  // 1/sqrt(head_dim)
            float scale_in;
            float scale_qkv_weight;
            float scale_qkv_out;
            float scale_q, scale_k, scale_v;
            float scale_out_weight;
            float scale_out;
            // NE16 support for QKV projection
            int use_ne16_qkv;
            const int8_t *qkv_ne16_packed;
            const int32_t *qkv_ne16_bias;
            const uint8_t *qkv_ne16_scale;
            const uint8_t *qkv_ne16_scale_shift;
            // NE16 support for output projection
            int use_ne16_out;
            const int8_t *out_ne16_packed;
            const int32_t *out_ne16_bias;
            const uint8_t *out_ne16_scale;
            const uint8_t *out_ne16_scale_shift;
        } alternating_attention;

        struct {
            int num_patches;
            int patch_size;
            int num_bins;
            float scale_in, scale_out;
        } rfft;

        // CrossAttention with Self-Refinement (LUNA per-patch block)
        // All per-sub-operation scales are emitted as template-time constants.
        struct {
            int batch;              // Per-patch batch (e.g. 32 for LUNA)
            int kv_len;             // Key/value sequence length (e.g. 22)
            int num_queries;        // Number of learned queries (e.g. 4)
            int embed_dim;          // Embedding dimension (e.g. 64)
            int num_heads;          // Attention heads (e.g. 2)
            int head_dim;           // embed_dim / num_heads
            int ff_dim;             // FFN hidden dim (e.g. 256)
            int num_sa_blocks;      // Self-attention refinement blocks (e.g. 3)
            float softmax_scale;    // 1/sqrt(head_dim)
            float scale_input;      // Input quantization scale
            float scale_output;     // Output quantization scale
        } cross_attn_self_refine;

        // Classification Head with MLP (LUNA output block)
        struct {
            int batch;              // Batch size
            int seq_len;            // Input sequence length (e.g. 32)
            int hidden_dim;         // Hidden dim (e.g. 256)
            int num_heads;          // Attention heads (e.g. 8)
            int head_dim;           // hidden_dim / num_heads
            int mlp_hidden_dim;     // MLP intermediate dim (e.g. 1024)
            int num_classes;        // Output classes (e.g. 2)
            float softmax_scale;    // 1/sqrt(head_dim)
            float scale_input;      // Input quantization scale
            float scale_output;     // Output quantization scale
        } classification_head_mlp;
        
        struct {} flatten;
        struct {} squeeze;
        struct {} reshape;

        struct {
            int in_h, in_w, channels;
            int out_h, out_w;
            int pad_left, pad_right, pad_top, pad_bottom;
            TensorLayout layout;  // LAYOUT_CHW or LAYOUT_HWC
        } zeropad2d;

        // Layout conversion parameters
        struct {
            int channels;
            int height;
            int width;
        } layout_convert;

        // NE16-accelerated Conv2D layer parameters
        struct {
            int batch;
            int in_h, in_w;           // Input spatial dimensions
            int in_channels;          // Input channels
            int out_channels;         // Output channels
            int kernel_h, kernel_w;   // Kernel size (1x1 or 3x3)
            int stride_h, stride_w;   // Stride
            int pad_h, pad_w;         // Padding
            float scale_input;        // Input quantization scale
            float scale_weight;       // Weight quantization scale
            float scale_output;       // Output quantization scale
            // Pointers to pre-packed weights and corrected bias
            const int8_t *weights_packed;   // Pre-packed weights
            const int32_t *bias_corrected;  // Bias with input_zp correction
            // Hardware output quantization (per-channel)
            const uint8_t *hw_scale;        // Per-channel multiplier (qbias)
            const uint8_t *hw_scale_shift;  // Per-channel shift (qnorm)
            int use_hw_requant;             // 1 = HW outquant, 0 = SW requant
            // Scratch buffer sizes
            size_t scratch_input_size;
            size_t scratch_output_size;
            // NE16 depthwise spatial tiling parameters (for large activations)
            int ne16_dw_spatial_tiling;   // 0=disabled, 1=enabled
            int ne16_dw_num_tiles;        // Number of spatial tiles
            int ne16_dw_tile_h_out;       // Output height per tile
            int ne16_dw_tile_h_in;        // Input height with halo
        } conv2d_ne16;

        // MAMBA-specific layer parameters
        struct {
            int channels;           // Number of channels (= groups for depthwise)
            int length;             // Sequence length
            int kernel_size;        // Convolution kernel size (typically 4)
            int causal;             // 1 for causal (left-only padding), 0 otherwise
            float scale_in, scale_w, scale_out;
        } conv1d_depthwise;

        struct {
            int num_elements;       // Total number of elements
            float scale_in, scale_out;
            const int8_t *lut;      // Pointer to 256-entry INT8 LUT
        } silu;

        struct {
            int batch;              // Batch size
            int seq_len;            // Sequence length
            int d_inner;            // Inner dimension
            int d_state;            // State dimension (N)
            int dt_rank;            // dt projection rank
            float scale_x;          // Input scale
            float scale_output;     // Output scale
            // Pointers to SSM parameters (loaded from L3/L2)
            const int8_t *x_proj_weight;
            const int8_t *dt_proj_weight;
            const float *A;  // Pre-computed -exp(A_log)
            const float *D;
        } ssm;

        struct {
            int batch;
            int seq_len;
            int d_model;
            int d_inner;
            int d_state;
            int dt_rank;
            int kernel_size;
            float scale_in, scale_out;
            // Pointers to all MambaBlock weights
            const int8_t *in_proj_weight;
            const int8_t *conv1d_weight;
            const int8_t *x_proj_weight;
            const int8_t *dt_proj_weight;
            const float *A;  // Pre-computed -exp(A_log)
            const float *D;
            const int8_t *out_proj_weight;
        } mamba_block;

        struct {
            int batch;
            int seq_len;
            int d_model;
            int d_inner;
            int d_state;
            int dt_rank;
            int kernel_size;
            float scale_in, scale_out;
        } mamba_wrapper;


        // FEMBA Patch Embedding layer parameters
        struct {
            int batch;              // Batch size
            int in_chans;           // Input channels
            int inp_h, inp_w;       // Input spatial dimensions
            int patch_h, patch_w;   // Patch size (kernel size) - supports non-square
            int stride_h, stride_w; // Stride for patch extraction - supports non-square
            int embed_dim;          // Embedding dimension per patch row
            int grid_h, grid_w;     // Grid dimensions after patching
            int seq_len;            // Output sequence length (= grid_w)
            int d_model;            // Output model dimension (= grid_h * embed_dim)
            float scale_in;         // Input scale
            float scale_weight;     // Weight scale
            float scale_out;        // Output scale
        } patch_embed;

        // FEMBA Positional Embedding layer parameters
        struct {
            int batch;              // Batch size
            int seq_len;            // Sequence length
            int d_model;            // Model dimension
            float scale_pos;        // Positional embedding scale
            float scale_input;      // Input scale (from patch_embed output)
            float scale_out;        // Output scale after addition
        } pos_embed;

#ifdef ARES_LLAMA_SUPPORT
        // RMSNorm for Llama/LLMs (simpler than LayerNorm - no mean subtraction)
        struct {
            int num_vectors;        // Number of vectors to normalize (batch * seq_len)
            int normalized_dim;     // Dimension of each vector (embed_dim)
            float scale_in, scale_out;
            float eps;              // Epsilon for numerical stability (default 1e-5)
        } rmsnorm;

        // SwiGLU FFN (Llama-style feed-forward: gate=W1@x, up=W3@x, out=W2@(silu(gate)*up))
        struct {
            int batch;              // Batch size
            int seq_len;            // Sequence length
            int dim;                // Input/output dimension (d_model)
            int hidden_dim;         // Hidden dimension (typically 8/3 * dim for Llama)
            // Quantization scales
            float scale_input;      // Input scale
            float scale_w1;         // W1 (gate) weight scale
            float scale_w3;         // W3 (up) weight scale
            float scale_hidden;     // Hidden (after SiLU*up) scale
            float scale_w2;         // W2 (down) weight scale
            float scale_output;     // Output scale
        } swiglu_ffn;

        // Autoregressive MHSA with KV cache (single-token attention)
        struct {
            int dim;                // Model dimension
            int num_heads;          // Number of query heads
            int n_kv_heads;         // Number of KV heads (for GQA)
            int head_dim;           // Dimension per head
            int max_seq_len;        // Max sequence length (KV cache size)
            float scale_input;      // Input quantization scale
            float scale_q_weight, scale_k_weight, scale_v_weight, scale_out_weight;
            float scale_q, scale_k, scale_v;  // QKV output scales
            float scale_output;     // Output scale
            float softmax_scale;    // 1/sqrt(head_dim)
        } mhsa_autoregressive;

        // Residual add (with FP32 intermediate for precision)
        struct {
            int size;               // Number of elements
            float scale_a, scale_b; // Input scales
            float scale_out;        // Output scale
        } residual_add;

        // Llama transformer decoder block (RMSNorm + MHSA + RMSNorm + SwiGLU FFN)
        struct {
            int batch;              // Batch size
            int seq_len;            // Sequence length
            int dim;                // Model dimension
            int hidden_dim;         // FFN hidden dimension
            int num_heads;          // Number of attention heads
            int n_kv_heads;         // Number of KV heads (for GQA, <= num_heads)
            int head_dim;           // Dimension per head
            int max_seq_len;        // Maximum sequence length (for RoPE/KV cache)
            float eps;              // RMSNorm epsilon
            // Quantization scales (simplified - per-subblock scales in practice)
            float scale_input;
            float scale_output;
            // KV cache mode
            int use_kv_cache;       // 1 = autoregressive mode with KV cache
            int kv_cache_pos;       // Current position in sequence
        } llama_block;
#endif // ARES_LLAMA_SUPPORT

    } params;

    // L3 Tiling / Streaming State (Generic)
    void *l3_input_addr;
    void *l3_output_addr;
    void *l3_weight_addr;
    int   l3_tiling_enabled;
} LayerSpec;

#endif // LAYER_DESCRIPTORS_H
