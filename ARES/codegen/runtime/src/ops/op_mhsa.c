/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * MHSA (Multi-Head Self-Attention) Operations for ARES Runtime
 *
 * Contains permute and RoPE functions for attention computations.
 */

#include "ops/op_mhsa.h"
#include "network_kernels.h"
#include "pmsis.h"


/* ---
 * MHSA Permute: Reorganize Q/K/V from [seq_len, embed_dim] to [num_heads, seq_len, head_dim]
 * This enables bulk DMA per head instead of strided row-by-row DMA
 * --- */

typedef struct {
    const int8_t *input;
    int8_t *output;
    int seq_len;
    int embed_dim;
    int num_heads;
    int head_dim;
} mhsa_permute_args_t;

static void mhsa_permute_to_heads_worker(void *args) {
    mhsa_permute_args_t *p = (mhsa_permute_args_t *)args;
    const int core_id = pi_core_id();

    const int tokens_per_core = (p->seq_len + CL_NUM_CORES - 1) / CL_NUM_CORES;
    const int token_start = core_id * tokens_per_core;
    int token_end = token_start + tokens_per_core;
    if (token_end > p->seq_len) token_end = p->seq_len;

    const int head_dim = p->head_dim;
    for (int t = token_start; t < token_end; t++) {
        const int8_t *src_row = p->input + t * p->embed_dim;
        for (int h = 0; h < p->num_heads; h++) {
            int8_t *dst = p->output + h * p->seq_len * head_dim + t * head_dim;
            const int8_t *src = src_row + h * head_dim;
            int d = 0;
            for (; d + 4 <= head_dim; d += 4) {
                *((v4s *)(dst + d)) = *((const v4s *)(src + d));
            }
            for (; d < head_dim; d++) {
                dst[d] = src[d];
            }
        }
    }
    pi_cl_team_barrier();
}

void mhsa_permute_to_heads(const int8_t *input, int8_t *output,
                           int seq_len, int embed_dim, int num_heads) {
    mhsa_permute_args_t args = {
        .input = input,
        .output = output,
        .seq_len = seq_len,
        .embed_dim = embed_dim,
        .num_heads = num_heads,
        .head_dim = embed_dim / num_heads
    };
    pi_cl_team_fork(CL_NUM_CORES, mhsa_permute_to_heads_worker, &args);
}

static void mhsa_permute_from_heads_worker(void *args) {
    mhsa_permute_args_t *p = (mhsa_permute_args_t *)args;
    const int core_id = pi_core_id();

    const int tokens_per_core = (p->seq_len + CL_NUM_CORES - 1) / CL_NUM_CORES;
    const int token_start = core_id * tokens_per_core;
    int token_end = token_start + tokens_per_core;
    if (token_end > p->seq_len) token_end = p->seq_len;

    for (int t = token_start; t < token_end; t++) {
        int8_t *dst_row = p->output + t * p->embed_dim;
        for (int h = 0; h < p->num_heads; h++) {
            const int8_t *src = p->input + h * p->seq_len * p->head_dim + t * p->head_dim;
            int8_t *dst = dst_row + h * p->head_dim;
            int d = 0;
            for (; d + 4 <= p->head_dim; d += 4) {
                *((v4s *)(dst + d)) = *((const v4s *)(src + d));
            }
            for (; d < p->head_dim; d++) {
                dst[d] = src[d];
            }
        }
    }
    pi_cl_team_barrier();
}

void mhsa_permute_from_heads(const int8_t *input, int8_t *output,
                             int seq_len, int embed_dim, int num_heads) {
    mhsa_permute_args_t args = {
        .input = input,
        .output = output,
        .seq_len = seq_len,
        .embed_dim = embed_dim,
        .num_heads = num_heads,
        .head_dim = embed_dim / num_heads
    };
    pi_cl_team_fork(CL_NUM_CORES, mhsa_permute_from_heads_worker, &args);
}

/* ---
 * RoPE (Rotary Position Embeddings): in-place rotation on head-contiguous tensors
 * Layout: [num_heads, seq_len, head_dim]
 * --- */

static inline int32_t shift_round_nearest_even_s32(int32_t val, int shift) {
    if (shift <= 0) return val;

    const int32_t sign = (val < 0) ? -1 : 1;
    uint32_t abs_val = (uint32_t)(val < 0 ? -val : val);

    uint32_t q = abs_val >> shift;
    uint32_t r = abs_val & ((1U << shift) - 1U);
    uint32_t half = 1U << (shift - 1);

    if (r > half || (r == half && (q & 1U))) {
        q += 1U;
    }
    return (int32_t)q * sign;
}

typedef struct {
    int8_t *data;
    const int16_t *cos_q15;
    const int16_t *sin_q15;
    uint32_t num_heads;
    uint32_t seq_len;
    uint32_t head_dim;
    uint32_t pos_offset;
} rope_int8_args_t;

static void rope_int8_worker(void *arg) {
    rope_int8_args_t *a = (rope_int8_args_t *)arg;
    const uint32_t core_id = (uint32_t)pi_core_id();

    const uint32_t total_rows = a->num_heads * a->seq_len;
    const uint32_t chunk = (total_rows + (uint32_t)CL_NUM_CORES - 1U) / (uint32_t)CL_NUM_CORES;
    const uint32_t start = core_id * chunk;
    uint32_t end = start + chunk;
    if (end > total_rows) end = total_rows;

    const uint32_t half = a->head_dim >> 1;

    for (uint32_t row = start; row < end; row++) {
        const uint32_t head = row / a->seq_len;
        const uint32_t pos = row - head * a->seq_len;

        int8_t *vec = a->data + (head * a->seq_len + pos) * a->head_dim;
        const int16_t *cos_row = a->cos_q15 + (a->pos_offset + pos) * half;
        const int16_t *sin_row = a->sin_q15 + (a->pos_offset + pos) * half;

        for (uint32_t i = 0; i < half; i++) {
            const int32_t x_even = (int32_t)vec[2U * i];
            const int32_t x_odd  = (int32_t)vec[2U * i + 1U];
            const int32_t c = (int32_t)cos_row[i];
            const int32_t s = (int32_t)sin_row[i];

            const int32_t even_num = x_even * c - x_odd * s;
            const int32_t odd_num  = x_even * s + x_odd * c;

            int32_t out_even = shift_round_nearest_even_s32(even_num, 15);
            int32_t out_odd  = shift_round_nearest_even_s32(odd_num, 15);

            if (out_even > 127) out_even = 127;
            if (out_even < -128) out_even = -128;
            if (out_odd > 127) out_odd = 127;
            if (out_odd < -128) out_odd = -128;

            vec[2U * i] = (int8_t)out_even;
            vec[2U * i + 1U] = (int8_t)out_odd;
        }
    }
}

void network_rope_int8_inplace_parallel(
    int8_t *data,
    const int16_t *rope_cos_q15,
    const int16_t *rope_sin_q15,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim,
    uint32_t pos_offset
) {
    if (!data || !rope_cos_q15 || !rope_sin_q15) return;
    if ((head_dim & 1U) != 0U) return;

    rope_int8_args_t args = {
        .data = data,
        .cos_q15 = rope_cos_q15,
        .sin_q15 = rope_sin_q15,
        .num_heads = num_heads,
        .seq_len = seq_len,
        .head_dim = head_dim,
        .pos_offset = pos_offset,
    };
    pi_cl_team_fork(NUM_CORES, rope_int8_worker, &args);
}

/* ---
 * Repeat KV: Expand K/V from [n_kv_heads, seq_len, head_dim] to [num_heads, seq_len, head_dim]
 * Used for Grouped Query Attention (GQA) where n_kv_heads < num_heads.
 * Each KV head is repeated (num_heads / n_kv_heads) times.
 * --- */

typedef struct {
    const int8_t *input;   // [n_kv_heads, seq_len, head_dim]
    int8_t *output;        // [num_heads, seq_len, head_dim]
    int n_kv_heads;
    int num_heads;
    int seq_len;
    int head_dim;
    int kv_rep;            // num_heads / n_kv_heads
} repeat_kv_args_t;

static void repeat_kv_worker(void *args) {
    repeat_kv_args_t *p = (repeat_kv_args_t *)args;
    const int core_id = pi_core_id();

    // Distribute output heads across cores
    const int heads_per_core = (p->num_heads + CL_NUM_CORES - 1) / CL_NUM_CORES;
    const int head_start = core_id * heads_per_core;
    int head_end = head_start + heads_per_core;
    if (head_end > p->num_heads) head_end = p->num_heads;

    const int head_size = p->seq_len * p->head_dim;

    for (int h_out = head_start; h_out < head_end; h_out++) {
        // Determine which KV head to copy from
        const int h_kv = h_out / p->kv_rep;

        const int8_t *src = p->input + h_kv * head_size;
        int8_t *dst = p->output + h_out * head_size;

        // Copy head_size bytes using SIMD where possible
        int i = 0;
        for (; i + 4 <= head_size; i += 4) {
            *((v4s *)(dst + i)) = *((const v4s *)(src + i));
        }
        for (; i < head_size; i++) {
            dst[i] = src[i];
        }
    }
    pi_cl_team_barrier();
}

void mhsa_repeat_kv_parallel(
    const int8_t *kv_in,    // [n_kv_heads, seq_len, head_dim]
    int8_t *kv_out,         // [num_heads, seq_len, head_dim]
    int n_kv_heads,
    int num_heads,
    int seq_len,
    int head_dim
) {
    if (n_kv_heads == num_heads) {
        // No repeat needed, just copy
        const int total_size = num_heads * seq_len * head_dim;
        for (int i = 0; i < total_size; i += 4) {
            if (i + 4 <= total_size) {
                *((v4s *)(kv_out + i)) = *((const v4s *)(kv_in + i));
            } else {
                for (int j = i; j < total_size; j++) {
                    kv_out[j] = kv_in[j];
                }
            }
        }
        return;
    }

    if (n_kv_heads <= 0 || num_heads % n_kv_heads != 0) {
        // Invalid configuration
        return;
    }

    repeat_kv_args_t args = {
        .input = kv_in,
        .output = kv_out,
        .n_kv_heads = n_kv_heads,
        .num_heads = num_heads,
        .seq_len = seq_len,
        .head_dim = head_dim,
        .kv_rep = num_heads / n_kv_heads,
    };
    pi_cl_team_fork(CL_NUM_CORES, repeat_kv_worker, &args);
}

/* ---
 * GQA Projection: Project K/V with fewer output heads than Q
 * K/V: [seq_len, dim] -> [n_kv_heads, seq_len, head_dim]
 * Q:   [seq_len, dim] -> [num_heads, seq_len, head_dim]
 * --- */

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
) {
    const int kv_dim = n_kv_heads * head_dim;

    // Project K
    network_linear_int8_parallel_tokens(
        input,
        k_weight,
        k_bias,
        k_out,
        seq_len,
        embed_dim,
        kv_dim,
        scale_input,
        scale_k_weight,
        scale_k
    );

    // Project V
    network_linear_int8_parallel_tokens(
        input,
        v_weight,
        v_bias,
        v_out,
        seq_len,
        embed_dim,
        kv_dim,
        scale_input,
        scale_v_weight,
        scale_v
    );
}

#ifdef ARES_LLAMA_SUPPORT
/* ---
 * Autoregressive Single-Query Attention with KV Cache
 *
 * For each head h:
 *   1. Compute dot(q_h, k_t) for all cached positions t=0..pos
 *   2. Softmax over scores
 *   3. Weighted sum of v_t to get context_h
 *   4. Quantize context to INT8
 *
 * K/V cache layout: [layer, pos, kv_dim] where kv_dim = n_kv_heads * head_dim
 * The cache is flat: offset = layer_idx * max_seq_len * kv_dim + pos * kv_dim
 * --- */

typedef struct {
    const int8_t *q;            // [num_heads * head_dim]
    const int8_t *kv_cache_k;   // [max_seq_len, kv_dim] (for this layer)
    const int8_t *kv_cache_v;   // [max_seq_len, kv_dim] (for this layer)
    float *scores_scratch;      // [max_seq_len] per-head scratch (shared, sequential heads)
    float *context_scratch;     // [dim] output context
    int num_heads;
    int n_kv_heads;
    int head_dim;
    int pos;                    // Number of valid K/V positions (current_pos + 1)
    int kv_rep;                 // num_heads / n_kv_heads
    float scale_q;
    float scale_k;
    float scale_v;
    float softmax_scale;
} mhsa_ar_attn_args_t;

static void mhsa_ar_attention_worker(void *args) {
    mhsa_ar_attn_args_t *a = (mhsa_ar_attn_args_t *)args;
    const int core_id = pi_core_id();
    const int heads_per_core = (a->num_heads + CL_NUM_CORES - 1) / CL_NUM_CORES;
    const int head_start = core_id * heads_per_core;
    int head_end = head_start + heads_per_core;
    if (head_end > a->num_heads) head_end = a->num_heads;

    const int head_dim = a->head_dim;
    const int seq_len = a->pos;  // Number of valid positions
    const float dequant_qk = a->scale_q * a->scale_k;
    const float dequant_v = a->scale_v;
    const float sm_scale = a->softmax_scale;

    for (int h = head_start; h < head_end; h++) {
        // Map query head to KV head (for GQA)
        int kv_h = h / a->kv_rep;

        const int8_t *q_h = a->q + h * head_dim;

        // Compute attention scores: dot(q_h, k_t) / sqrt(head_dim)
        float max_score = -1e30f;
        for (int t = 0; t < seq_len; t++) {
            const int8_t *k_t = a->kv_cache_k + t * (a->n_kv_heads * head_dim) + kv_h * head_dim;
            int32_t dot = 0;
            for (int d = 0; d < head_dim; d++) {
                dot += (int32_t)q_h[d] * (int32_t)k_t[d];
            }
            float score = (float)dot * dequant_qk * sm_scale;
            a->scores_scratch[h * seq_len + t] = score;  // Use per-head offset
            if (score > max_score) max_score = score;
        }

        // Softmax (numerically stable)
        float sum_exp = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            float e = expf(a->scores_scratch[h * seq_len + t] - max_score);
            a->scores_scratch[h * seq_len + t] = e;
            sum_exp += e;
        }
        float inv_sum = 1.0f / (sum_exp + 1e-10f);
        for (int t = 0; t < seq_len; t++) {
            a->scores_scratch[h * seq_len + t] *= inv_sum;
        }

        // Weighted sum of V: context_h = sum(score[t] * v_t)
        float *ctx_h = a->context_scratch + h * head_dim;
        for (int d = 0; d < head_dim; d++) {
            ctx_h[d] = 0.0f;
        }
        for (int t = 0; t < seq_len; t++) {
            float w = a->scores_scratch[h * seq_len + t];
            const int8_t *v_t = a->kv_cache_v + t * (a->n_kv_heads * head_dim) + kv_h * head_dim;
            for (int d = 0; d < head_dim; d++) {
                ctx_h[d] += w * ((float)v_t[d] * dequant_v);
            }
        }
    }
    pi_cl_team_barrier();
}

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
) {
    // Compute layer offset into KV cache
    int layer_offset = layer_idx * max_seq_len * kv_dim;

    // Store new K/V at current position
    int8_t *k_dest = kv_cache_k + layer_offset + pos * kv_dim;
    int8_t *v_dest = kv_cache_v + layer_offset + pos * kv_dim;
    for (int i = 0; i < kv_dim; i++) {
        k_dest[i] = k_new[i];
        v_dest[i] = v_new[i];
    }

    // Compute attention over positions 0..pos (inclusive)
    int seq_len = pos + 1;

    mhsa_ar_attn_args_t args = {
        .q = q,
        .kv_cache_k = kv_cache_k + layer_offset,
        .kv_cache_v = kv_cache_v + layer_offset,
        .scores_scratch = scores_scratch,
        .context_scratch = context_scratch,
        .num_heads = num_heads,
        .n_kv_heads = n_kv_heads,
        .head_dim = head_dim,
        .pos = seq_len,
        .kv_rep = num_heads / n_kv_heads,
        .scale_q = scale_q,
        .scale_k = scale_k,
        .scale_v = scale_v,
        .softmax_scale = softmax_scale,
    };

    pi_cl_team_fork(CL_NUM_CORES, mhsa_ar_attention_worker, &args);

    // Quantize context (FP32) to output (INT8)
    // context is in head-interleaved order [h0_d0..h0_dH, h1_d0..] = [dim]
    int dim = num_heads * head_dim;
    for (int i = 0; i < dim; i++) {
        float val = context_scratch[i];
        int32_t q_val = (int32_t)(val / scale_output + 0.5f);
        if (val < 0) q_val = (int32_t)(val / scale_output - 0.5f);
        if (q_val > 127) q_val = 127;
        if (q_val < -128) q_val = -128;
        output[i] = (int8_t)q_val;
    }
}
#endif // ARES_LLAMA_SUPPORT
