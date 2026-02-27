/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * MHSA (Multi-Head Self-Attention) Operations Header
 *
 * Contains permute and RoPE functions for attention computations.
 */

#ifndef ARES_OPS_MHSA_H
#define ARES_OPS_MHSA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Permute from [seq_len, embed_dim] to [num_heads, seq_len, head_dim] */
void mhsa_permute_to_heads(const int8_t *input, int8_t *output,
                           int seq_len, int embed_dim, int num_heads);

/* Permute from [num_heads, seq_len, head_dim] to [seq_len, embed_dim] */
void mhsa_permute_from_heads(const int8_t *input, int8_t *output,
                             int seq_len, int embed_dim, int num_heads);

/* RoPE (Rotary Position Embeddings) - in-place rotation */
void network_rope_int8_inplace_parallel(
    int8_t *data,
    const int16_t *rope_cos_q15,
    const int16_t *rope_sin_q15,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim,
    uint32_t pos_offset
);

/* GQA (Grouped Query Attention) Support */

/* Repeat K/V heads for GQA: [n_kv_heads, seq_len, head_dim] -> [num_heads, seq_len, head_dim] */
void mhsa_repeat_kv_parallel(
    const int8_t *kv_in,    // [n_kv_heads, seq_len, head_dim]
    int8_t *kv_out,         // [num_heads, seq_len, head_dim]
    int n_kv_heads,
    int num_heads,
    int seq_len,
    int head_dim
);

/* GQA K/V projection with fewer output heads than Q */
void mhsa_gqa_project_kv_parallel(
    const int8_t *input,           // [seq_len, embed_dim]
    const int8_t *k_weight,        // [n_kv_heads * head_dim, embed_dim]
    const int8_t *v_weight,        // [n_kv_heads * head_dim, embed_dim]
    const int32_t *k_bias,         // [n_kv_heads * head_dim] (or NULL)
    const int32_t *v_bias,         // [n_kv_heads * head_dim] (or NULL)
    int8_t *k_out,                 // [seq_len, n_kv_heads * head_dim]
    int8_t *v_out,                 // [seq_len, n_kv_heads * head_dim]
    int seq_len,
    int embed_dim,
    int n_kv_heads,
    int head_dim,
    float scale_input,
    float scale_k_weight, float scale_v_weight,
    float scale_k, float scale_v
);

#ifdef ARES_LLAMA_SUPPORT
/* Autoregressive single-query attention with KV cache.
 *
 * Processes a single query token against all cached K/V positions.
 * KV cache is updated in-place before attention computation.
 *
 * Args:
 *   q:       Query vector for current token [num_heads * head_dim] (INT8)
 *   kv_cache_k: KV cache keys [n_layers][max_seq_len][kv_dim]
 *   kv_cache_v: KV cache values [n_layers][max_seq_len][kv_dim]
 *   k_new:   New K vector [kv_dim] (INT8, stored into cache)
 *   v_new:   New V vector [kv_dim] (INT8, stored into cache)
 *   output:  Output vector [dim] (INT8)
 *   scores_scratch: FP32 scratch for attention scores [max_seq_len]
 *   context_scratch: FP32 scratch for context [dim]
 *   params:  Attention parameters
 *   layer_idx: Current layer index
 *   pos:     Current sequence position
 *   max_seq_len: Maximum sequence length
 *   scale_q, scale_k, scale_v: QKV quantization scales
 *   scale_output: Output quantization scale
 *   softmax_scale: 1/sqrt(head_dim)
 */
void mhsa_single_query_attention(
    const int8_t *q,
    int8_t *kv_cache_k,
    int8_t *kv_cache_v,
    const int8_t *k_new,
    const int8_t *v_new,
    int8_t *output,
    float *scores_scratch,
    float *context_scratch,
    int num_heads,
    int n_kv_heads,
    int head_dim,
    int layer_idx,
    int pos,
    int max_seq_len,
    int kv_dim,
    float scale_q,
    float scale_k,
    float scale_v,
    float scale_output,
    float softmax_scale
);
#endif // ARES_LLAMA_SUPPORT

#ifdef __cplusplus
}
#endif

#endif /* ARES_OPS_MHSA_H */
