/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * kernel_mhsa.c - Multi-Head Self-Attention Kernels for GAP9
 *
 * Contains MHSA-related kernels:
 *   - mhsa_fused_projection: Fused Q/K/V projections (3→1 fork/barrier)
 *   - mhsa_fused_projection_rectangular: Head-slice projections
 *   - network_cross_attention_isoftmax: Cross-attention with integer softmax
 *   - mhsa_projection_parallel: Generic projection kernel
 *
 * Features:
 *   - Integer softmax via LUT for bit-exact matching
 *   - SIMD via SumDotpSS intrinsic
 *   - Reduced fork/barrier overhead through kernel fusion
 *   - Support for both self-attention and cross-attention
 *
 * Part of the ARES modular kernel system.
 */

// FUSED Q/K/V PROJECTION KERNEL - Computes all 3 projections in single launch
// ---
// This reduces fork/barrier overhead from 3 to 1 for MHSA projections.
typedef struct {
    const int8_t *input;       // [seq_len, embed_dim]
    const int8_t *q_weights;   // [embed_dim, embed_dim]
    const int8_t *k_weights;   // [embed_dim, embed_dim]
    const int8_t *v_weights;   // [embed_dim, embed_dim]
    const int32_t *q_bias;     // [embed_dim]
    const int32_t *k_bias;     // [embed_dim]
    const int32_t *v_bias;     // [embed_dim]
    int8_t *q_output;          // [seq_len, embed_dim]
    int8_t *k_output;          // [seq_len, embed_dim]
    int8_t *v_output;          // [seq_len, embed_dim]
    int seq_len;
    int embed_dim;
    float scale_input;
    float scale_q_weight, scale_k_weight, scale_v_weight;
    float scale_q, scale_k, scale_v;
} mhsa_fused_proj_args_t;

static void mhsa_fused_projection_worker(void *args) {
    mhsa_fused_proj_args_t *p = (mhsa_fused_proj_args_t *)args;
    const int core_id = pi_core_id();

    // Distribute tokens across cores
    const int tokens_per_core = (p->seq_len + CL_NUM_CORES - 1) / CL_NUM_CORES;
    const int token_start = core_id * tokens_per_core;
    int token_end = token_start + tokens_per_core;
    if (token_end > p->seq_len) token_end = p->seq_len;

    // OPTIMIZED: Precompute combined requantization scale (FP32, but only once per core)
    const float requant_scale_q = p->scale_input * p->scale_q_weight / p->scale_q;
    const float requant_scale_k = p->scale_input * p->scale_k_weight / p->scale_k;
    const float requant_scale_v = p->scale_input * p->scale_v_weight / p->scale_v;

    // SIMD loop parameters
    const int simd_count = p->embed_dim >> 2;
    const int embed_dim = p->embed_dim;

#if LINEAR_INT8_INPUT_L1_CACHE
    const int cache_input = (embed_dim <= LINEAR_INT8_INPUT_L1_CACHE_MAX_BYTES);
    v4s input_cache[cache_input ? simd_count : 1] __attribute__((aligned(4)));
#endif

    // Process assigned tokens
    for (int t = token_start; t < token_end; t++) {
        const int8_t *input_row = p->input + t * embed_dim;
        int8_t *q_row = p->q_output + t * embed_dim;
        int8_t *k_row = p->k_output + t * embed_dim;
        int8_t *v_row = p->v_output + t * embed_dim;

        // Pre-cast input row for SIMD access (done once per token)
        const v4s *pA = (const v4s *)input_row;
#if LINEAR_INT8_INPUT_L1_CACHE
        if (cache_input) {
            const v4s *pA_src = (const v4s *)input_row;
            for (int k = 0; k < simd_count; k++) {
                input_cache[k] = pA_src[k];
            }
            pA = (const v4s *)input_cache;
        }
#endif

        // Process output features in groups of 2 for better input reuse
        // 2x unrolling works best for fused QKV (6 accumulators vs 12 for 4x which causes register spills)
        for (int o = 0; o < embed_dim; o += 2) {
            // Weight row pointers for 2 output features
            const v4s *pQ0 = (const v4s *)(p->q_weights + o * embed_dim);
            const v4s *pQ1 = (const v4s *)(p->q_weights + (o+1) * embed_dim);
            const v4s *pK0 = (const v4s *)(p->k_weights + o * embed_dim);
            const v4s *pK1 = (const v4s *)(p->k_weights + (o+1) * embed_dim);
            const v4s *pV0 = (const v4s *)(p->v_weights + o * embed_dim);
            const v4s *pV1 = (const v4s *)(p->v_weights + (o+1) * embed_dim);

            // 6 accumulators for 2 outputs x 3 projections
            int32_t acc_q0 = 0, acc_q1 = 0;
            int32_t acc_k0 = 0, acc_k1 = 0;
            int32_t acc_v0 = 0, acc_v1 = 0;

            // SIMD inner loop with 2x unrolling
            for (int k = 0; k < simd_count; k++) {
                v4s a = pA[k];  // Load input once, use 6 times
                acc_q0 = SumDotpSS(a, pQ0[k], acc_q0);
                acc_q1 = SumDotpSS(a, pQ1[k], acc_q1);
                acc_k0 = SumDotpSS(a, pK0[k], acc_k0);
                acc_k1 = SumDotpSS(a, pK1[k], acc_k1);
                acc_v0 = SumDotpSS(a, pV0[k], acc_v0);
                acc_v1 = SumDotpSS(a, pV1[k], acc_v1);
            }

            // Add biases
            if (p->q_bias) { acc_q0 += p->q_bias[o]; acc_q1 += p->q_bias[o+1]; }
            if (p->k_bias) { acc_k0 += p->k_bias[o]; acc_k1 += p->k_bias[o+1]; }
            if (p->v_bias) { acc_v0 += p->v_bias[o]; acc_v1 += p->v_bias[o+1]; }

            // Requantize and clamp
            int32_t q0 = qround((float)acc_q0 * requant_scale_q);
            int32_t q1 = qround((float)acc_q1 * requant_scale_q);
            int32_t k0 = qround((float)acc_k0 * requant_scale_k);
            int32_t k1 = qround((float)acc_k1 * requant_scale_k);
            int32_t v0 = qround((float)acc_v0 * requant_scale_v);
            int32_t v1 = qround((float)acc_v1 * requant_scale_v);

            q_row[o]   = (int8_t)(q0 > 127 ? 127 : (q0 < -128 ? -128 : q0));
            q_row[o+1] = (int8_t)(q1 > 127 ? 127 : (q1 < -128 ? -128 : q1));
            k_row[o]   = (int8_t)(k0 > 127 ? 127 : (k0 < -128 ? -128 : k0));
            k_row[o+1] = (int8_t)(k1 > 127 ? 127 : (k1 < -128 ? -128 : k1));
            v_row[o]   = (int8_t)(v0 > 127 ? 127 : (v0 < -128 ? -128 : v0));
            v_row[o+1] = (int8_t)(v1 > 127 ? 127 : (v1 < -128 ? -128 : v1));
        }
    }
}

// Wrapper to launch fused Q/K/V projection
void network_linear_int8_fused_qkv(
    const int8_t *input,
    const int8_t *q_weights, const int8_t *k_weights, const int8_t *v_weights,
    const int32_t *q_bias, const int32_t *k_bias, const int32_t *v_bias,
    int8_t *q_output, int8_t *k_output, int8_t *v_output,
    int seq_len, int embed_dim,
    float scale_input,
    float scale_q_weight, float scale_k_weight, float scale_v_weight,
    float scale_q, float scale_k, float scale_v
) {
    mhsa_fused_proj_args_t args = {
        .input = input,
        .q_weights = q_weights, .k_weights = k_weights, .v_weights = v_weights,
        .q_bias = q_bias, .k_bias = k_bias, .v_bias = v_bias,
        .q_output = q_output, .k_output = k_output, .v_output = v_output,
        .seq_len = seq_len, .embed_dim = embed_dim,
        .scale_input = scale_input,
        .scale_q_weight = scale_q_weight, .scale_k_weight = scale_k_weight, .scale_v_weight = scale_v_weight,
        .scale_q = scale_q, .scale_k = scale_k, .scale_v = scale_v
    };
    pi_cl_team_fork(CL_NUM_CORES, mhsa_fused_projection_worker, &args);
}

// ---
// FUSED Q/K/V PROJECTION KERNEL (RECTANGULAR) - Computes 3 projections in single launch
// ---
// Used for head-slice projections (out_features=head_dim, in_features=embed_dim) to
// reduce fork/barrier overhead (3 launches -> 1) and reuse input loads across Q/K/V.
typedef struct {
    const int8_t *input;       // [seq_len, in_features]
    const int8_t *q_weights;   // [out_features, in_features]
    const int8_t *k_weights;   // [out_features, in_features]
    const int8_t *v_weights;   // [out_features, in_features]
    const int32_t *q_bias;     // [out_features] (optional)
    const int32_t *k_bias;     // [out_features] (optional)
    const int32_t *v_bias;     // [out_features] (optional)
    int8_t *q_output;          // [seq_len, out_features]
    int8_t *k_output;          // [seq_len, out_features]
    int8_t *v_output;          // [seq_len, out_features]
    int seq_len;
    int in_features;
    int out_features;
    float scale_input;
    float scale_q_weight, scale_k_weight, scale_v_weight;
    float scale_q, scale_k, scale_v;
} mhsa_fused_proj_rect_args_t;

static void mhsa_fused_projection_rect_worker(void *args) {
    mhsa_fused_proj_rect_args_t *p = (mhsa_fused_proj_rect_args_t *)args;
    const int core_id = pi_core_id();

    // Distribute tokens across cores
    const int tokens_per_core = (p->seq_len + CL_NUM_CORES - 1) / CL_NUM_CORES;
    const int token_start = core_id * tokens_per_core;
    int token_end = token_start + tokens_per_core;
    if (token_end > p->seq_len) token_end = p->seq_len;

    // Precompute combined requantization scales (FP32, once per core)
    const float requant_scale_q = p->scale_input * p->scale_q_weight / p->scale_q;
    const float requant_scale_k = p->scale_input * p->scale_k_weight / p->scale_k;
    const float requant_scale_v = p->scale_input * p->scale_v_weight / p->scale_v;

    const int in_features = p->in_features;
    const int out_features = p->out_features;
    const int simd_count = in_features >> 2;
    const int in_features_aligned4 = simd_count << 2;

#if LINEAR_INT8_INPUT_L1_CACHE
    const int cache_input = (in_features <= LINEAR_INT8_INPUT_L1_CACHE_MAX_BYTES);
    v4s input_cache[cache_input ? simd_count : 1] __attribute__((aligned(4)));
#endif

    for (int t = token_start; t < token_end; t++) {
        const int8_t *input_row = p->input + t * in_features;
        int8_t *q_row = p->q_output + t * out_features;
        int8_t *k_row = p->k_output + t * out_features;
        int8_t *v_row = p->v_output + t * out_features;

        const v4s *pA = (const v4s *)input_row;
#if LINEAR_INT8_INPUT_L1_CACHE
        if (cache_input) {
            const v4s *pA_src = (const v4s *)input_row;
            for (int k = 0; k < simd_count; k++) {
                input_cache[k] = pA_src[k];
            }
            pA = (const v4s *)input_cache;
        }
#endif

        int o = 0;
        for (; o + 1 < out_features; o += 2) {
            const int8_t *q_w_row0 = p->q_weights + o * in_features;
            const int8_t *q_w_row1 = p->q_weights + (o + 1) * in_features;
            const int8_t *k_w_row0 = p->k_weights + o * in_features;
            const int8_t *k_w_row1 = p->k_weights + (o + 1) * in_features;
            const int8_t *v_w_row0 = p->v_weights + o * in_features;
            const int8_t *v_w_row1 = p->v_weights + (o + 1) * in_features;

            const v4s *pQ0 = (const v4s *)q_w_row0;
            const v4s *pQ1 = (const v4s *)q_w_row1;
            const v4s *pK0 = (const v4s *)k_w_row0;
            const v4s *pK1 = (const v4s *)k_w_row1;
            const v4s *pV0 = (const v4s *)v_w_row0;
            const v4s *pV1 = (const v4s *)v_w_row1;

            int32_t acc_q0 = 0, acc_q1 = 0;
            int32_t acc_k0 = 0, acc_k1 = 0;
            int32_t acc_v0 = 0, acc_v1 = 0;

            for (int k = 0; k < simd_count; k++) {
                v4s a = pA[k];
                acc_q0 = SumDotpSS(a, pQ0[k], acc_q0);
                acc_q1 = SumDotpSS(a, pQ1[k], acc_q1);
                acc_k0 = SumDotpSS(a, pK0[k], acc_k0);
                acc_k1 = SumDotpSS(a, pK1[k], acc_k1);
                acc_v0 = SumDotpSS(a, pV0[k], acc_v0);
                acc_v1 = SumDotpSS(a, pV1[k], acc_v1);
            }

            for (int j = in_features_aligned4; j < in_features; j++) {
                const int32_t a = (int32_t)input_row[j];
                acc_q0 += a * (int32_t)q_w_row0[j];
                acc_q1 += a * (int32_t)q_w_row1[j];
                acc_k0 += a * (int32_t)k_w_row0[j];
                acc_k1 += a * (int32_t)k_w_row1[j];
                acc_v0 += a * (int32_t)v_w_row0[j];
                acc_v1 += a * (int32_t)v_w_row1[j];
            }

            if (p->q_bias) { acc_q0 += p->q_bias[o]; acc_q1 += p->q_bias[o + 1]; }
            if (p->k_bias) { acc_k0 += p->k_bias[o]; acc_k1 += p->k_bias[o + 1]; }
            if (p->v_bias) { acc_v0 += p->v_bias[o]; acc_v1 += p->v_bias[o + 1]; }

            int32_t q0 = qround((float)acc_q0 * requant_scale_q);
            int32_t q1 = qround((float)acc_q1 * requant_scale_q);
            int32_t k0 = qround((float)acc_k0 * requant_scale_k);
            int32_t k1 = qround((float)acc_k1 * requant_scale_k);
            int32_t v0 = qround((float)acc_v0 * requant_scale_v);
            int32_t v1 = qround((float)acc_v1 * requant_scale_v);

            q_row[o]     = (int8_t)(q0 > 127 ? 127 : (q0 < -128 ? -128 : q0));
            q_row[o + 1] = (int8_t)(q1 > 127 ? 127 : (q1 < -128 ? -128 : q1));
            k_row[o]     = (int8_t)(k0 > 127 ? 127 : (k0 < -128 ? -128 : k0));
            k_row[o + 1] = (int8_t)(k1 > 127 ? 127 : (k1 < -128 ? -128 : k1));
            v_row[o]     = (int8_t)(v0 > 127 ? 127 : (v0 < -128 ? -128 : v0));
            v_row[o + 1] = (int8_t)(v1 > 127 ? 127 : (v1 < -128 ? -128 : v1));
        }

        if (o < out_features) {
            const int8_t *q_w_row0 = p->q_weights + o * in_features;
            const int8_t *k_w_row0 = p->k_weights + o * in_features;
            const int8_t *v_w_row0 = p->v_weights + o * in_features;

            const v4s *pQ0 = (const v4s *)q_w_row0;
            const v4s *pK0 = (const v4s *)k_w_row0;
            const v4s *pV0 = (const v4s *)v_w_row0;

            int32_t acc_q0 = 0;
            int32_t acc_k0 = 0;
            int32_t acc_v0 = 0;

            for (int k = 0; k < simd_count; k++) {
                v4s a = pA[k];
                acc_q0 = SumDotpSS(a, pQ0[k], acc_q0);
                acc_k0 = SumDotpSS(a, pK0[k], acc_k0);
                acc_v0 = SumDotpSS(a, pV0[k], acc_v0);
            }

            for (int j = in_features_aligned4; j < in_features; j++) {
                const int32_t a = (int32_t)input_row[j];
                acc_q0 += a * (int32_t)q_w_row0[j];
                acc_k0 += a * (int32_t)k_w_row0[j];
                acc_v0 += a * (int32_t)v_w_row0[j];
            }

            if (p->q_bias) acc_q0 += p->q_bias[o];
            if (p->k_bias) acc_k0 += p->k_bias[o];
            if (p->v_bias) acc_v0 += p->v_bias[o];

            int32_t q0 = qround((float)acc_q0 * requant_scale_q);
            int32_t k0 = qround((float)acc_k0 * requant_scale_k);
            int32_t v0 = qround((float)acc_v0 * requant_scale_v);

            q_row[o] = (int8_t)(q0 > 127 ? 127 : (q0 < -128 ? -128 : q0));
            k_row[o] = (int8_t)(k0 > 127 ? 127 : (k0 < -128 ? -128 : k0));
            v_row[o] = (int8_t)(v0 > 127 ? 127 : (v0 < -128 ? -128 : v0));
        }
    }
}

void network_linear_int8_fused_qkv_parallel_tokens_rect(
    const int8_t *input,
    const int8_t *q_weights, const int8_t *k_weights, const int8_t *v_weights,
    const int32_t *q_bias, const int32_t *k_bias, const int32_t *v_bias,
    int8_t *q_output, int8_t *k_output, int8_t *v_output,
    int seq_len, int in_features, int out_features,
    float scale_input,
    float scale_q_weight, float scale_k_weight, float scale_v_weight,
    float scale_q, float scale_k, float scale_v
) {
    mhsa_fused_proj_rect_args_t args = {
        .input = input,
        .q_weights = q_weights, .k_weights = k_weights, .v_weights = v_weights,
        .q_bias = q_bias, .k_bias = k_bias, .v_bias = v_bias,
        .q_output = q_output, .k_output = k_output, .v_output = v_output,
        .seq_len = seq_len,
        .in_features = in_features,
        .out_features = out_features,
        .scale_input = scale_input,
        .scale_q_weight = scale_q_weight, .scale_k_weight = scale_k_weight, .scale_v_weight = scale_v_weight,
        .scale_q = scale_q, .scale_k = scale_k, .scale_v = scale_v
    };
    pi_cl_team_fork(CL_NUM_CORES, mhsa_fused_projection_rect_worker, &args);
}

// MHSA permute and RoPE functions moved to ops/op_mhsa.c

// ---
// Cross-Attention (learned queries attend to KV input) - integer softmax path
// ---

// Pre-computed LUT for integer softmax: exp(x) * 2^24 for x in [-128, 0]
// Index: x + 128 (so index 0 = exp(-128), index 128 = exp(0))
static const uint32_t i_softmax_lut_int8[129] = {
             0,          0,          0,          0,          0,          0,          0,          0,
             0,          0,          0,          0,          0,          0,          0,          0,
             0,          0,          0,          0,          0,          0,          0,          0,
             0,          0,          0,          0,          0,          0,          0,          0,
             0,          0,          0,          0,          0,          0,          0,          0,
             0,          0,          0,          0,          0,          0,          0,          0,
             0,          0,          0,          0,          0,          0,          0,          0,
             0,          0,          0,          0,          0,          0,          0,          0,
             0,          0,          0,          0,          0,          0,          0,          0,
             0,          0,          0,          0,          0,          0,          0,          0,
             0,          0,          0,          0,          0,          0,          0,          0,
             0,          0,          0,          0,          0,          0,          0,          0,
             0,          0,          0,          0,          0,          0,          0,          0,
             0,          0,          0,          0,          0,          0,          0,          0,
             1,          5,         13,         37,        103,        280,        761,       2070,
          5628,      15298,      41586,     113043,     307285,     835288,    2270549,    6171992,
      16777216
};

typedef struct {
    const int8_t *q_proj;   // [Q, D]
    const int8_t *k_proj;   // [B*KV, D]
    const int8_t *v_proj;   // [B*KV, D]
    int8_t *context;        // [B*Q, D]
    int32_t *scores;        // Optional L1 scratch: [NUM_CORES, KV] INT32 scores cache
    const int8_t *k_head;   // Optional staged K head: [B*KV, head_dim] in L1
    const int8_t *v_head;   // Optional staged V head: [B*KV, head_dim] in L1
    uint32_t fixed_head;    // Used when k_head/v_head are provided
    uint32_t batch;
    uint32_t kv_len;
    uint32_t num_queries;
    uint32_t embed_dim;
    uint32_t num_heads;
    uint32_t head_dim;
    int32_t requant_mul;
    int32_t requant_shift;
} cross_attn_args_t;

static void cross_attention_linear_int8_tiled_weights(
    const char *layer_name,
    const int8_t *input,
    const int8_t *weight,
    const int32_t *bias,
    int8_t *output,
    int batch_tokens,
    int features,
    float scale_input,
    float scale_weight,
    float scale_output,
    int8_t *l1_buffer,
    size_t l1_buffer_size
) {
    linear_int8_pipeline_config_t cfg = {0};
    cfg.layer_name = layer_name;
    cfg.input_buffer_l2 = (int8_t *)input;
    cfg.output_buffer_l2 = output;
    cfg.weight_l2 = (int8_t *)weight;
    cfg.bias_l2 = (int32_t *)bias;
    cfg.l1_buffer = l1_buffer;
    cfg.l1_buffer_size = l1_buffer_size;
    cfg.in_features = features;
    cfg.out_features = features;
    cfg.batch_tokens = batch_tokens;
    cfg.tile_out_features = 0;  // Auto-tune from available L1
    cfg.num_tiles = 0;
    cfg.l1_input_size = 0;
    cfg.l1_output_size = 0;
    cfg.l1_weight_size = 0;
    cfg.scale_input = scale_input;
    cfg.scale_weight = scale_weight;
    cfg.scale_output = scale_output;
    cfg.fusion_relu = 0;
    cfg.fusion_quant = 0;
    cfg.relu_output_scale = 0.0f;
    cfg.quant_scale_in = 0.0f;
    cfg.quant_scale_out = 0.0f;
#ifdef ENABLE_PERF_COUNTERS
    cfg.perf_counter = NULL;
#endif
    cfg.golden_buffer = NULL;
    cfg.golden_size = 0;
    cfg.compare_buffer = NULL;
    cfg.ram_dev = NULL;
    cfg.l3_tiling_enabled = 0;
    cfg.l3_weight_addr = NULL;
    cfg.l3_output_addr = NULL;
    cfg.l3_bias_addr = NULL;
    cfg.l3_tile_out_features = 0;
    cfg.num_l3_tiles = 0;
    linear_int8_tiled_l1_pipeline(&cfg);
}

static inline int32_t dot_int8_simd(const int8_t *a, const int8_t *b, uint32_t len) {
    int32_t acc = 0;
    const uint32_t simd_count = len >> 2;
    const v4s *va = (const v4s *)a;
    const v4s *vb = (const v4s *)b;
    for (uint32_t i = 0; i < simd_count; i++) {
        acc = SumDotpSS(va[i], vb[i], acc);
    }
    for (uint32_t i = (simd_count << 2); i < len; i++) {
        acc += (int32_t)a[i] * (int32_t)b[i];
    }
    return acc;
}

static void cross_attention_softmax_av_worker(void *arg) {
    cross_attn_args_t *a = (cross_attn_args_t *)arg;
    const uint32_t core_id = (uint32_t)pi_core_id();

    if (a->kv_len == 0 || a->head_dim == 0) return;

    const int use_staged_kv = (a->k_head != NULL) && (a->v_head != NULL);
    const uint32_t total_rows = use_staged_kv
        ? (a->batch * a->num_queries)
        : (a->batch * a->num_heads * a->num_queries);
    const uint32_t kv_len = a->kv_len;
    const uint32_t head_dim = a->head_dim;
    const uint32_t embed_dim = a->embed_dim;
    const uint32_t kv_stride = use_staged_kv ? head_dim : embed_dim;
    const int32_t requant_mul = a->requant_mul;
    const int32_t requant_shift = a->requant_shift;
    const int64_t round_val = ((requant_shift > 0) && (requant_shift < 63)) ? (1LL << (requant_shift - 1)) : 0;
    int32_t *scores = a->scores ? (a->scores + core_id * kv_len) : NULL;

    for (uint32_t row = core_id; row < total_rows; row += (uint32_t)NUM_CORES) {
        uint32_t q_idx;
        uint32_t head;
        uint32_t b;
        const int8_t *k_base;
        const int8_t *v_base;

        if (use_staged_kv) {
            q_idx = row % a->num_queries;
            b = row / a->num_queries;
            head = a->fixed_head;
            k_base = a->k_head + (b * kv_len) * head_dim;
            v_base = a->v_head + (b * kv_len) * head_dim;
        } else {
            q_idx = row % a->num_queries;
            const uint32_t tmp = row / a->num_queries;
            head = tmp % a->num_heads;
            b = tmp / a->num_heads;
            k_base = a->k_proj + (b * kv_len) * embed_dim + head * head_dim;
            v_base = a->v_proj + (b * kv_len) * embed_dim + head * head_dim;
        }

        const int8_t *q_vec = a->q_proj + q_idx * embed_dim + head * head_dim;

        // Cache Q head vector in L1 (stack) once per row to avoid repeated L2 reads in dot products.
        const uint32_t q_words = (head_dim + 3U) >> 2;
        v4s q_cache[q_words] __attribute__((aligned(4)));
        const v4s *q_src = (const v4s *)q_vec;
        const uint32_t q_full_words = head_dim >> 2;
        for (uint32_t i = 0; i < q_full_words; i++) {
            q_cache[i] = q_src[i];
        }
        for (uint32_t i = (q_full_words << 2); i < head_dim; i++) {
            ((int8_t *)q_cache)[i] = q_vec[i];
        }
        const int8_t *q_ptr = (const int8_t *)q_cache;

        // Step 1: Find max INT32 score
        int32_t max_score;
        if (scores) {
            max_score = dot_int8_simd(q_ptr, k_base, head_dim);
            scores[0] = max_score;
            for (uint32_t n = 1; n < kv_len; n++) {
                const int8_t *k_vec = k_base + n * kv_stride;
                int32_t s = dot_int8_simd(q_ptr, k_vec, head_dim);
                scores[n] = s;
                if (s > max_score) max_score = s;
            }
        } else {
            max_score = dot_int8_simd(q_ptr, k_base, head_dim);
            for (uint32_t n = 1; n < kv_len; n++) {
                const int8_t *k_vec = k_base + n * kv_stride;
                int32_t s = dot_int8_simd(q_ptr, k_vec, head_dim);
                if (s > max_score) max_score = s;
            }
        }

        // Step 2: Requantize diffs to LUT indices and accumulate y_sum
        uint64_t y_sum = 0;
        uint8_t attn_row[kv_len];
        for (uint32_t n = 0; n < kv_len; n++) {
            const int32_t score = scores ? scores[n] : dot_int8_simd(q_ptr, k_base + n * kv_stride, head_dim);
            int32_t diff = score - max_score;  // <= 0

            int32_t x_int = ((int64_t)diff * (int64_t)requant_mul + round_val) >> requant_shift;
            if (x_int < -128) x_int = -128;
            if (x_int > 0) x_int = 0;
            const uint8_t idx = (uint8_t)(x_int + 128);
            attn_row[n] = idx;  // cache idx
            y_sum += i_softmax_lut_int8[idx];
        }

        // Step 3: Normalize indices -> UINT8 attention weights in [0,255]
        if (y_sum > 0) {
            const uint64_t inv_sum = (255ULL << 24) / y_sum;
            const uint64_t round_norm = (1ULL << 23);
            for (uint32_t n = 0; n < kv_len; n++) {
                const uint32_t y = i_softmax_lut_int8[attn_row[n]];
                attn_row[n] = (uint8_t)(((uint64_t)y * inv_sum + round_norm) >> 24);
            }
        } else {
            const uint8_t uniform = (uint8_t)(255U / kv_len);
            for (uint32_t n = 0; n < kv_len; n++) {
                attn_row[n] = uniform;
            }
        }

        // Step 4: Context = attn x V (UINT8xINT8 -> INT8 with (acc+128)>>8)
        //
        // Keep accumulators in registers by processing 4 output channels at once.
        // This avoids updating a stack-backed accumulator array inside the hot loop.
        int8_t *ctx_row = a->context + (b * a->num_queries + q_idx) * a->embed_dim + head * a->head_dim;
        uint32_t d = 0;
        for (; d + 3 < head_dim; d += 4) {
            int32_t acc0 = 0;
            int32_t acc1 = 0;
            int32_t acc2 = 0;
            int32_t acc3 = 0;

            for (uint32_t n = 0; n < kv_len; n++) {
                const int32_t w = (int32_t)attn_row[n];
                const int8_t *v_vec = v_base + n * kv_stride + d;
                acc0 += w * (int32_t)v_vec[0];
                acc1 += w * (int32_t)v_vec[1];
                acc2 += w * (int32_t)v_vec[2];
                acc3 += w * (int32_t)v_vec[3];
            }

            int32_t q0 = (acc0 + 128) >> 8;
            int32_t q1 = (acc1 + 128) >> 8;
            int32_t q2 = (acc2 + 128) >> 8;
            int32_t q3 = (acc3 + 128) >> 8;

            if (q0 > 127) q0 = 127;
            if (q0 < -128) q0 = -128;
            if (q1 > 127) q1 = 127;
            if (q1 < -128) q1 = -128;
            if (q2 > 127) q2 = 127;
            if (q2 < -128) q2 = -128;
            if (q3 > 127) q3 = 127;
            if (q3 < -128) q3 = -128;

            ctx_row[d] = (int8_t)q0;
            ctx_row[d + 1] = (int8_t)q1;
            ctx_row[d + 2] = (int8_t)q2;
            ctx_row[d + 3] = (int8_t)q3;
        }

        for (; d < head_dim; d++) {
            int32_t acc = 0;
            for (uint32_t n = 0; n < kv_len; n++) {
                const int32_t w = (int32_t)attn_row[n];
                acc += w * (int32_t)v_base[n * kv_stride + d];
            }
            int32_t q_val = (acc + 128) >> 8;
            if (q_val > 127) q_val = 127;
            if (q_val < -128) q_val = -128;
            ctx_row[d] = (int8_t)q_val;
        }
    }
}

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
) {
    if (!q_projected || !k_projected || !v_projected || !output || !context_scratch) return;
    if (embed_dim == 0 || num_heads == 0 || (embed_dim % num_heads) != 0) return;

    const uint32_t head_dim = embed_dim / num_heads;

    // L1 scratch allocation for scores cache and optional K/V head staging
    int32_t *scores_l1 = NULL;
    int8_t *k_head_l1 = NULL;
    int8_t *v_head_l1 = NULL;

    if (l1_scratch && l1_scratch_size) {
        const uintptr_t base = ((uintptr_t)l1_scratch + 3U) & ~((uintptr_t)3U);
        const size_t pad = (size_t)(base - (uintptr_t)l1_scratch);
        if (l1_scratch_size > pad) {
            const size_t avail = l1_scratch_size - pad;
            const size_t scores_bytes = (size_t)NUM_CORES * (size_t)kv_len * sizeof(int32_t);
            const size_t head_kv_bytes = (size_t)batch * (size_t)kv_len * (size_t)head_dim;
            if (avail >= scores_bytes) {
                scores_l1 = (int32_t *)base;
                if (avail >= scores_bytes + 2U * head_kv_bytes) {
                    k_head_l1 = (int8_t *)((uint8_t *)scores_l1 + scores_bytes);
                    v_head_l1 = k_head_l1 + head_kv_bytes;
                }
            }
        }
    }

    // Use q_projected directly — if q_shared_across_batch, the softmax worker
    // indexes Q as [q_idx * embed_dim + head * head_dim] which works for both
    // shared Q [q_len, embed_dim] and per-batch Q [batch*q_len, embed_dim].
    const uint32_t rows_per_head = batch * q_len;
    const int use_head_staging =
        (k_head_l1 != NULL) && (v_head_l1 != NULL) && (rows_per_head >= (uint32_t)NUM_CORES);

    if (use_head_staging) {
        pi_cl_dma_copy_2d_t kv_dma;
        kv_dma.dir = PI_CL_DMA_DIR_EXT2LOC;
        kv_dma.merge = 0;
        kv_dma.size = (uint32_t)(kv_len * head_dim);
        kv_dma.length = (uint32_t)head_dim;
        kv_dma.stride = (uint32_t)embed_dim;

        for (uint32_t head = 0; head < num_heads; head++) {
            for (uint32_t b = 0; b < batch; b++) {
                kv_dma.ext = (uint32_t)(k_projected + (b * kv_len) * embed_dim + head * head_dim);
                kv_dma.loc = (uint32_t)(k_head_l1 + (b * kv_len) * head_dim);
                pi_cl_dma_memcpy_2d(&kv_dma);
                pi_cl_dma_wait(&kv_dma);

                kv_dma.ext = (uint32_t)(v_projected + (b * kv_len) * embed_dim + head * head_dim);
                kv_dma.loc = (uint32_t)(v_head_l1 + (b * kv_len) * head_dim);
                pi_cl_dma_memcpy_2d(&kv_dma);
                pi_cl_dma_wait(&kv_dma);
            }

            cross_attn_args_t attn_args = {
                .q_proj = q_projected,
                .k_proj = k_projected,
                .v_proj = v_projected,
                .context = context_scratch,
                .scores = scores_l1,
                .k_head = k_head_l1,
                .v_head = v_head_l1,
                .fixed_head = head,
                .batch = batch,
                .kv_len = kv_len,
                .num_queries = q_len,
                .embed_dim = embed_dim,
                .num_heads = num_heads,
                .head_dim = head_dim,
                .requant_mul = requant_mul,
                .requant_shift = requant_shift,
            };
            pi_cl_team_fork(NUM_CORES, cross_attention_softmax_av_worker, &attn_args);
        }
    } else {
        cross_attn_args_t attn_args = {
            .q_proj = q_projected,
            .k_proj = k_projected,
            .v_proj = v_projected,
            .context = context_scratch,
            .scores = scores_l1,
            .k_head = NULL,
            .v_head = NULL,
            .fixed_head = 0,
            .batch = batch,
            .kv_len = kv_len,
            .num_queries = q_len,
            .embed_dim = embed_dim,
            .num_heads = num_heads,
            .head_dim = head_dim,
            .requant_mul = requant_mul,
            .requant_shift = requant_shift,
        };
        pi_cl_team_fork(NUM_CORES, cross_attention_softmax_av_worker, &attn_args);
    }

    // Output projection: context → output
    const int out_tokens = (int)(batch * q_len);
    network_linear_int8_parallel_tokens(
        context_scratch,
        out_weight,
        out_bias,
        output,
        out_tokens,
        (int)embed_dim,
        (int)embed_dim,
        scale_v,              // context scale matches V scale (post-AV)
        scale_out_weight,
        scale_output
    );
}

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
) {
    (void)softmax_scale;  // softmax_scale is folded into requant_mul in network.c.mako
    if (!kv_input || !output || !q_proj_out || !k_proj_out || !v_proj_out || !context_out) return;
    if (!query_embed || !q_weight || !k_weight || !v_weight || !out_weight) return;
    if (embed_dim == 0 || num_heads == 0 || (embed_dim % num_heads) != 0) return;

    // 1) Q projection (constant across batch)
    if (num_queries > 1) {
        cross_attention_linear_int8_tiled_weights(
            "cross_attn.q_proj",
            query_embed,
            q_weight,
            q_bias,
            q_proj_out,
            (int)num_queries,
            (int)embed_dim,
            scale_query_input,
            scale_q_weight,
            scale_q,
            l1_scratch,
            l1_scratch_size
        );
    } else {
        network_linear_int8_parallel_tokens(
            query_embed,
            q_weight,
            q_bias,
            q_proj_out,
            (int)num_queries,
            (int)embed_dim,
            (int)embed_dim,
            scale_query_input,
            scale_q_weight,
            scale_q
        );
    }

    // 2) K/V projections over all KV tokens (flattened batch)
    const int kv_tokens = (int)(batch * kv_len);
    if (kv_tokens > 1) {
        cross_attention_linear_int8_tiled_weights(
            "cross_attn.k_proj",
            kv_input,
            k_weight,
            k_bias,
            k_proj_out,
            kv_tokens,
            (int)embed_dim,
            scale_kv_input,
            scale_k_weight,
            scale_k,
            l1_scratch,
            l1_scratch_size
        );
        cross_attention_linear_int8_tiled_weights(
            "cross_attn.v_proj",
            kv_input,
            v_weight,
            v_bias,
            v_proj_out,
            kv_tokens,
            (int)embed_dim,
            scale_kv_input,
            scale_v_weight,
            scale_v,
            l1_scratch,
            l1_scratch_size
        );
    } else {
        network_linear_int8_parallel_tokens(
            kv_input,
            k_weight,
            k_bias,
            k_proj_out,
            kv_tokens,
            (int)embed_dim,
            (int)embed_dim,
            scale_kv_input,
            scale_k_weight,
            scale_k
        );
        network_linear_int8_parallel_tokens(
            kv_input,
            v_weight,
            v_bias,
            v_proj_out,
            kv_tokens,
            (int)embed_dim,
            (int)embed_dim,
            scale_kv_input,
            scale_v_weight,
            scale_v
        );
    }

    // 3+4) Attention core (softmax + AV) + output projection via reusable function
    network_attention_core_int8(
        q_proj_out, k_proj_out, v_proj_out,
        out_weight, out_bias,
        output, context_out,
        batch, num_queries, kv_len,
        embed_dim, num_heads,
        scale_q, scale_k, scale_v,
        scale_out_weight, scale_output,
        requant_mul, requant_shift,
        1,  // q_shared_across_batch: Q is [num_queries, D] shared across batch
        l1_scratch, l1_scratch_size
    );
}

void mhsa_projection_worker(void *args) {
    mhsa_proj_args_t *p = (mhsa_proj_args_t *)args;
    const int core_id = pi_core_id();

    // Distribute tokens across cores
    const int tokens_per_core = (p->seq_len + CL_NUM_CORES - 1) / CL_NUM_CORES;
    const int token_start = core_id * tokens_per_core;
    int token_end = token_start + tokens_per_core;
    if (token_end > p->seq_len) token_end = p->seq_len;

    // OPTIMIZED: Single requantization factor (computed once per core)
#ifdef LINEAR_INT8_FIXEDPOINT_REQUANT
    const int32_t requant_mul = p->requant_mul;
    const int requant_shift = p->requant_shift;
    const int64_t requant_round = (requant_shift > 0) ? (1LL << (requant_shift - 1)) : 0;
#else
    const float requant_scale = p->scale_input * p->scale_weight / p->scale_output;
#endif

    // SIMD loop parameters
    const int simd_count = p->in_features >> 2;
    const int in_features = p->in_features;
    const int out_features = p->out_features;

    // ---
    // 4x Output Unrolling (PULP-NN style)
    // ---
    // Process 4 output features per iteration to amortize input loads.
    for (int t = token_start; t < token_end; t++) {
        const int8_t *input_row = p->input + t * in_features;
        int8_t *output_row = p->output + t * out_features;
        const v4s *pA = (const v4s *)input_row;

        for (int o = 0; o < out_features; o += 4) {
            const v4s *pW0 = (const v4s *)(p->weights + o * in_features);
            const v4s *pW1 = (const v4s *)(p->weights + (o + 1) * in_features);
            const v4s *pW2 = (const v4s *)(p->weights + (o + 2) * in_features);
            const v4s *pW3 = (const v4s *)(p->weights + (o + 3) * in_features);

            int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
            for (int k = 0; k < simd_count; k++) {
                v4s a = pA[k];
                acc0 = SumDotpSS(a, pW0[k], acc0);
                acc1 = SumDotpSS(a, pW1[k], acc1);
                acc2 = SumDotpSS(a, pW2[k], acc2);
                acc3 = SumDotpSS(a, pW3[k], acc3);
            }

            if (p->bias) {
                acc0 += p->bias[o];
                acc1 += p->bias[o + 1];
                acc2 += p->bias[o + 2];
                acc3 += p->bias[o + 3];
            }

#ifdef LINEAR_INT8_FIXEDPOINT_REQUANT
            int64_t prod0 = (int64_t)acc0 * (int64_t)requant_mul;
            int64_t prod1 = (int64_t)acc1 * (int64_t)requant_mul;
            int64_t prod2 = (int64_t)acc2 * (int64_t)requant_mul;
            int64_t prod3 = (int64_t)acc3 * (int64_t)requant_mul;
            int32_t q0 = (int32_t)((prod0 + (prod0 >= 0 ? requant_round : -requant_round)) >> requant_shift);
            int32_t q1 = (int32_t)((prod1 + (prod1 >= 0 ? requant_round : -requant_round)) >> requant_shift);
            int32_t q2 = (int32_t)((prod2 + (prod2 >= 0 ? requant_round : -requant_round)) >> requant_shift);
            int32_t q3 = (int32_t)((prod3 + (prod3 >= 0 ? requant_round : -requant_round)) >> requant_shift);
#else
            int32_t q0 = qround((float)acc0 * requant_scale);
            int32_t q1 = qround((float)acc1 * requant_scale);
            int32_t q2 = qround((float)acc2 * requant_scale);
            int32_t q3 = qround((float)acc3 * requant_scale);
#endif
            output_row[o]     = (int8_t)(q0 > 127 ? 127 : (q0 < -128 ? -128 : q0));
            output_row[o + 1] = (int8_t)(q1 > 127 ? 127 : (q1 < -128 ? -128 : q1));
            output_row[o + 2] = (int8_t)(q2 > 127 ? 127 : (q2 < -128 ? -128 : q2));
            output_row[o + 3] = (int8_t)(q3 > 127 ? 127 : (q3 < -128 ? -128 : q3));
        }
    }
}

// Wrapper to launch parallel projection
void network_linear_int8_parallel_tokens(
    const int8_t *input,       // [seq_len, in_features]
    const int8_t *weights,     // [out_features, in_features] (preferably in L1)
    const int32_t *bias,       // [out_features]
    int8_t *output,            // [seq_len, out_features]
    int seq_len,
    int in_features,
    int out_features,
    float scale_input,
    float scale_weight,
    float scale_output
) {
    mhsa_proj_args_t args = {
        .input = input,
        .weights = weights,
        .bias = bias,
        .output = output,
        .seq_len = seq_len,
        .in_features = in_features,
        .out_features = out_features,
        .scale_input = scale_input,
        .scale_weight = scale_weight,
        .scale_output = scale_output
    };
#ifdef LINEAR_INT8_FIXEDPOINT_REQUANT
    // Use Q8.24 fixed-point for requantization (mul/shift computed once per call)
    args.requant_shift = 24;
    float requant_scale = scale_input * scale_weight / scale_output;
    args.requant_mul = qround(requant_scale * (float)(1 << args.requant_shift));
#endif
    pi_cl_team_fork(CL_NUM_CORES, mhsa_projection_worker, &args);
}

// ---
