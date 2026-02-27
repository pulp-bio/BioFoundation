/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Network Kernels - Shared Runtime Implementation
 *
 * This file contains all neural network computational kernels for the GAP9 platform.
 * Large kernel groups are extracted into separate files in kernels/ for maintainability.
 *
 * MODULAR STRUCTURE:
 *   - kernels/kernel_softmax.c: Integer softmax LUT and i_softmax_row()
 *   - kernels/kernel_perf.c: Performance counters
 *   - kernels/kernel_ssm.c: Conv1D, SiLU, SSM (Mamba)
 *   - kernels/kernel_conv2d.c: Conv2D variants (CHW and HWC layouts)
 *   - kernels/kernel_linear.c: Linear layer variants (INT8, 2-bit, FP32)
 *   - kernels/kernel_mhsa.c: MHSA projections and cross-attention
 */
#include "network_kernels.h"
#include "network_dma_pipeline.h"
#include "core/utils.h"
#include "ops/op_activation.h"
#include "ops/op_pool.h"
#include "ops/op_norm.h"
#include "ops/op_elementwise.h"
#include "ops/op_linear.h"
#include "ops/op_conv2d.h"
#include "ops/op_mhsa.h"
#include <stddef.h>
#include <math.h>
#include <string.h>
#include "pmsis.h"


// Linear INT8 tuning macros are centralized in `codegen/runtime/inc/ares_config.h`.

// for shared use between network.c and network_kernels.c

// Core math utilities (fast_exp, fast_log, fast_softplus) are in core/utils.c

// ---
// MODULAR KERNEL INCLUDES
// ---
// These files contain extracted kernel implementations for better maintainability.
// They are #included (not separately compiled) to share static symbols.

#include "kernels/kernel_softmax.c"
#include "kernels/kernel_perf.c"
#include "kernels/kernel_ssm.c"
#include "kernels/kernel_conv2d.c"
#include "kernels/kernel_linear.c"
#include "kernels/kernel_mhsa.c"

// GELU functions moved to ops/op_activation.c


// Pool operations moved to ops/op_pool.c

typedef struct {
    const int32_t *indices;
    const int8_t *weight;
    int8_t *output;
    uint32_t num_indices;
    uint32_t embed_dim;
    uint32_t vocab_size;
} embedding_int8_args_t;

static void embedding_int8_worker(void *arg) {
    embedding_int8_args_t *a = (embedding_int8_args_t *)arg;
    const uint32_t core_id = (uint32_t)pi_core_id();
    const uint32_t chunk = (a->num_indices + (uint32_t)CL_NUM_CORES - 1U) / (uint32_t)CL_NUM_CORES;
    const uint32_t start = core_id * chunk;
    uint32_t end = start + chunk;
    if (end > a->num_indices) end = a->num_indices;

    for (uint32_t i = start; i < end; i++) {
        int32_t idx = a->indices[i];
        if ((uint32_t)idx >= a->vocab_size) {
            // Invalid index: clamp to 0 (should not happen for captured indices).
            idx = 0;
        }
        const int8_t *src = a->weight + (uint32_t)idx * a->embed_dim;
        int8_t *dst = a->output + i * a->embed_dim;
        memcpy(dst, src, a->embed_dim);
    }
}

void network_embedding_int8_parallel(
    const int32_t *indices,
    const int8_t *weight,
    int8_t *output,
    uint32_t num_indices,
    uint32_t embed_dim,
    uint32_t vocab_size
) {
    embedding_int8_args_t args = {
        .indices = indices,
        .weight = weight,
        .output = output,
        .num_indices = num_indices,
        .embed_dim = embed_dim,
        .vocab_size = vocab_size,
    };
    pi_cl_team_fork(NUM_CORES, embedding_int8_worker, &args);
}

// ---
// RFFT (patch_size=40): Fixed-point feature extractor (magnitude + phase)
// ---

#define RFFT40_PATCH_SIZE 40
#define RFFT40_NUM_BINS   21  // N/2 + 1 for N=40
#define RFFT40_OUT_FEATURES (2 * RFFT40_NUM_BINS)
#define RFFT_Q15_SCALE    32768
#define RFFT_ATAN_LUT_SIZE 1024

// Base twiddle steps for N=40 (cos/sin of 2*pi*k/N), in Q15.
static const int32_t rfft40_cos_step_q15[RFFT40_NUM_BINS] = {
    32767, 32365, 31164, 29197, 26510, 23170, 19261, 14876, 10126, 5126, 0,
    -5126, -10126, -14876, -19261, -23170, -26510, -29197, -31164, -32365, -32768,
};

static const int32_t rfft40_sin_step_q15[RFFT40_NUM_BINS] = {
    0, 5126, 10126, 14876, 19261, 23170, 26510, 29197, 31164, 32365, 32767,
    32365, 31164, 29197, 26510, 23170, 19261, 14876, 10126, 5126, 0,
};

// atan(z)/pi for z in [0,1], stored in Q15. Generated from Python via atomic_ops.rfft.
static const int16_t rfft_atan_lut_q15[RFFT_ATAN_LUT_SIZE] = {
    0, 10, 20, 31, 41, 51, 61, 71, 82, 92, 102, 112, 122, 133, 143, 153,
    163, 173, 184, 194, 204, 214, 224, 234, 245, 255, 265, 275, 285, 296, 306, 316,
    326, 336, 347, 357, 367, 377, 387, 397, 408, 418, 428, 438, 448, 459, 469, 479,
    489, 499, 509, 520, 530, 540, 550, 560, 570, 581, 591, 601, 611, 621, 631, 642,
    652, 662, 672, 682, 692, 702, 713, 723, 733, 743, 753, 763, 774, 784, 794, 804,
    814, 824, 834, 845, 855, 865, 875, 885, 895, 906, 916, 926, 936, 946, 956, 966,
    977, 987, 997, 1007, 1017, 1027, 1037, 1048, 1058, 1068, 1078, 1088, 1098, 1108, 1119, 1129,
    1139, 1149, 1159, 1169, 1179, 1189, 1200, 1210, 1220, 1230, 1240, 1250, 1260, 1270, 1281, 1291,
    1301, 1311, 1321, 1331, 1341, 1351, 1362, 1372, 1382, 1392, 1402, 1412, 1422, 1432, 1443, 1453,
    1463, 1473, 1483, 1493, 1503, 1513, 1524, 1534, 1544, 1554, 1564, 1574, 1584, 1594, 1605, 1615,
    1625, 1635, 1645, 1655, 1665, 1675, 1686, 1696, 1706, 1716, 1726, 1736, 1746, 1756, 1767, 1777,
    1787, 1797, 1807, 1817, 1827, 1837, 1848, 1858, 1868, 1878, 1888, 1898, 1908, 1918, 1929, 1939,
    1949, 1959, 1969, 1979, 1989, 1999, 2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2091, 2101,
    2111, 2121, 2131, 2141, 2151, 2161, 2172, 2182, 2192, 2202, 2212, 2222, 2232, 2242, 2253, 2263,
    2273, 2283, 2293, 2303, 2313, 2323, 2334, 2344, 2354, 2364, 2374, 2384, 2394, 2404, 2415, 2425,
    2435, 2445, 2455, 2465, 2475, 2485, 2496, 2506, 2516, 2526, 2536, 2546, 2556, 2566, 2577, 2587,
    2597, 2607, 2617, 2627, 2637, 2647, 2658, 2668, 2678, 2688, 2698, 2708, 2718, 2728, 2739, 2749,
    2759, 2769, 2779, 2789, 2799, 2809, 2820, 2830, 2840, 2850, 2860, 2870, 2880, 2890, 2901, 2911,
    2921, 2931, 2941, 2951, 2961, 2971, 2982, 2992, 3002, 3012, 3022, 3032, 3042, 3052, 3063, 3073,
    3083, 3093, 3103, 3113, 3123, 3133, 3144, 3154, 3164, 3174, 3184, 3194, 3204, 3214, 3225, 3235,
    3245, 3255, 3265, 3275, 3285, 3295, 3306, 3316, 3326, 3336, 3346, 3356, 3366, 3376, 3387, 3397,
    3407, 3417, 3427, 3437, 3447, 3457, 3468, 3478, 3488, 3498, 3508, 3518, 3528, 3538, 3549, 3559,
    3569, 3579, 3589, 3599, 3609, 3619, 3630, 3640, 3650, 3660, 3670, 3680, 3690, 3700, 3711, 3721,
    3731, 3741, 3751, 3761, 3771, 3781, 3792, 3802, 3812, 3822, 3832, 3842, 3852, 3862, 3873, 3883,
    3893, 3903, 3913, 3923, 3933, 3943, 3954, 3964, 3974, 3984, 3994, 4004, 4014, 4024, 4035, 4045,
    4055, 4065, 4075, 4085, 4095, 4105, 4116, 4126, 4136, 4146, 4156, 4166, 4176, 4186, 4197, 4207,
    4217, 4227, 4237, 4247, 4257, 4267, 4278, 4288, 4298, 4308, 4318, 4328, 4338, 4348, 4359, 4369,
    4379, 4389, 4399, 4409, 4419, 4429, 4440, 4450, 4460, 4470, 4480, 4490, 4500, 4510, 4521, 4531,
    4541, 4551, 4561, 4571, 4581, 4591, 4602, 4612, 4622, 4632, 4642, 4652, 4662, 4672, 4683, 4693,
    4703, 4713, 4723, 4733, 4743, 4753, 4764, 4774, 4784, 4794, 4804, 4814, 4824, 4834, 4845, 4855,
    4865, 4875, 4885, 4895, 4905, 4915, 4926, 4936, 4946, 4956, 4966, 4976, 4986, 4996, 5007, 5017,
    5027, 5037, 5047, 5057, 5067, 5077, 5088, 5098, 5108, 5118, 5128, 5138, 5148, 5158, 5169, 5179,
    5189, 5199, 5209, 5219, 5229, 5239, 5250, 5260, 5270, 5280, 5290, 5300, 5310, 5320, 5331, 5341,
    5351, 5361, 5371, 5381, 5391, 5401, 5412, 5422, 5432, 5442, 5452, 5462, 5472, 5482, 5493, 5503,
    5513, 5523, 5533, 5543, 5553, 5563, 5574, 5584, 5594, 5604, 5614, 5624, 5634, 5644, 5655, 5665,
    5675, 5685, 5695, 5705, 5715, 5725, 5736, 5746, 5756, 5766, 5776, 5786, 5796, 5806, 5817, 5827,
    5837, 5847, 5857, 5867, 5877, 5887, 5898, 5908, 5918, 5928, 5938, 5948, 5958, 5968, 5979, 5989,
    5999, 6009, 6019, 6029, 6039, 6049, 6060, 6070, 6080, 6090, 6100, 6110, 6120, 6130, 6141, 6151,
    6161, 6171, 6181, 6191, 6201, 6211, 6222, 6232, 6242, 6252, 6262, 6272, 6282, 6292, 6303, 6313,
    6323, 6333, 6343, 6353, 6363, 6373, 6384, 6394, 6404, 6414, 6424, 6434, 6444, 6454, 6465, 6475,
    6485, 6495, 6505, 6515, 6525, 6535, 6546, 6556, 6566, 6576, 6586, 6596, 6606, 6616, 6627, 6637,
    6647, 6657, 6667, 6677, 6687, 6697, 6708, 6718, 6728, 6738, 6748, 6758, 6768, 6778, 6789, 6799,
    6809, 6819, 6829, 6839, 6849, 6859, 6870, 6880, 6890, 6900, 6910, 6920, 6930, 6940, 6951, 6961,
    6971, 6981, 6991, 7001, 7011, 7021, 7032, 7042, 7052, 7062, 7072, 7082, 7092, 7102, 7113, 7123,
    7133, 7143, 7153, 7163, 7173, 7183, 7194, 7204, 7214, 7224, 7234, 7244, 7254, 7264, 7275, 7285,
    7295, 7305, 7315, 7325, 7335, 7345, 7356, 7366, 7376, 7386, 7396, 7406, 7416, 7426, 7437, 7447,
    7457, 7467, 7477, 7487, 7497, 7507, 7518, 7528, 7538, 7548, 7558, 7568, 7578, 7588, 7599, 7609,
    7619, 7629, 7639, 7649, 7659, 7669, 7680, 7690, 7700, 7710, 7720, 7730, 7740, 7750, 7761, 7771,
    7781, 7791, 7801, 7811, 7821, 7831, 7842, 7852, 7862, 7872, 7882, 7892, 7902, 7912, 7923, 7933,
    7943, 7953, 7963, 7973, 7983, 7993, 8004, 8014, 8024, 8034, 8044, 8054, 8064, 8074, 8085, 8095,
    8105, 8115, 8125, 8135, 8145, 8155, 8166, 8176, 8186, 8192,
};

static inline int16_t rfft_atan_lut_lookup_q15(uint16_t ratio_q15) {
    if (ratio_q15 == 0) {
        return 0;
    }
    if (ratio_q15 >= (uint16_t)(RFFT_Q15_SCALE - 1)) {
        return rfft_atan_lut_q15[RFFT_ATAN_LUT_SIZE - 1];
    }
    const uint32_t idx = ((uint32_t)ratio_q15 * (uint32_t)(RFFT_ATAN_LUT_SIZE - 1U)) / (uint32_t)(RFFT_Q15_SCALE - 1U);
    return rfft_atan_lut_q15[idx];
}

// Approx atan2(y,x) returning angle/pi in Q15 ([-32768, 32767]) with quadrant correction.
static inline int32_t rfft_atan2_pi_q15(int32_t y, int32_t x) {
    if (x == 0 && y == 0) {
        return 0;
    }

    const int32_t ax = (x < 0) ? -x : x;
    const int32_t ay = (y < 0) ? -y : y;

    int32_t angle;
    if (ax >= ay) {
        const uint16_t ratio_q15 = (ax != 0) ? (uint16_t)(((uint64_t)ay << 15) / (uint64_t)ax) : (uint16_t)(RFFT_Q15_SCALE - 1);
        const int16_t base = rfft_atan_lut_lookup_q15(ratio_q15);
        angle = (int32_t)base;
    } else {
        const uint16_t ratio_q15 = (ay != 0) ? (uint16_t)(((uint64_t)ax << 15) / (uint64_t)ay) : (uint16_t)(RFFT_Q15_SCALE - 1);
        const int16_t base = rfft_atan_lut_lookup_q15(ratio_q15);
        angle = (int32_t)((RFFT_Q15_SCALE / 2) - (int32_t)base);  // pi/2 => 0.5 in units of pi
    }

    if (x >= 0) {
        angle = (y >= 0) ? angle : -angle;
    } else {
        angle = (y >= 0) ? (RFFT_Q15_SCALE - angle) : (angle - RFFT_Q15_SCALE);
    }

    if (angle < -RFFT_Q15_SCALE) angle = -RFFT_Q15_SCALE;
    if (angle > (RFFT_Q15_SCALE - 1)) angle = (RFFT_Q15_SCALE - 1);
    return angle;
}

// Deterministic floor(sqrt(x)) for 64-bit integers.
static inline uint32_t rfft_isqrt_u64(uint64_t x) {
    uint64_t op = x;
    uint64_t res = 0;
    uint64_t one = 1ULL << 62;

    while (one > op) {
        one >>= 2;
    }

    while (one != 0) {
        if (op >= res + one) {
            op -= res + one;
            res += 2 * one;
        }
        res >>= 1;
        one >>= 2;
    }

    return (uint32_t)res;
}

typedef struct {
    const int8_t *input;
    int8_t *output;
    uint32_t num_patches;
    float mag_factor;
    float phase_factor;
} rfft_features_args_t;

static void rfft40_features_worker(void *arg) {
    rfft_features_args_t *a = (rfft_features_args_t *)arg;
    const uint32_t core_id = (uint32_t)pi_core_id();
    const uint32_t chunk = (a->num_patches + (uint32_t)CL_NUM_CORES - 1U) / (uint32_t)CL_NUM_CORES;
    const uint32_t start = core_id * chunk;
    uint32_t end = start + chunk;
    if (end > a->num_patches) end = a->num_patches;

    // Precompute per-bin twiddle sequences once per core (on stack, in L1),
    // then reuse across patches handled by this core.
    int16_t cos_seq[RFFT40_PATCH_SIZE];
    int16_t sin_seq[RFFT40_PATCH_SIZE];

    for (uint32_t k = 0; k < RFFT40_NUM_BINS; k++) {
        const int32_t cos_step = rfft40_cos_step_q15[k];
        const int32_t sin_step = rfft40_sin_step_q15[k];

        int32_t c = (RFFT_Q15_SCALE - 1);  // cos(0) in Q15
        int32_t s = 0;                    // sin(0) in Q15

        for (uint32_t n = 0; n < RFFT40_PATCH_SIZE; n++) {
            cos_seq[n] = (int16_t)c;
            sin_seq[n] = (int16_t)s;

            int32_t cn = mul_shift_round_nearest_even(c, cos_step, 15) - mul_shift_round_nearest_even(s, sin_step, 15);
            int32_t sn = mul_shift_round_nearest_even(s, cos_step, 15) + mul_shift_round_nearest_even(c, sin_step, 15);

            if (cn < -RFFT_Q15_SCALE) cn = -RFFT_Q15_SCALE;
            if (cn > (RFFT_Q15_SCALE - 1)) cn = (RFFT_Q15_SCALE - 1);
            if (sn < -RFFT_Q15_SCALE) sn = -RFFT_Q15_SCALE;
            if (sn > (RFFT_Q15_SCALE - 1)) sn = (RFFT_Q15_SCALE - 1);
            c = cn;
            s = sn;
        }

        for (uint32_t p = start; p < end; p++) {
            const int8_t *in_patch = a->input + p * RFFT40_PATCH_SIZE;
            int8_t *out_patch = a->output + p * RFFT40_OUT_FEATURES;

            int32_t re = 0;
            int32_t im = 0;

            for (uint32_t n = 0; n < RFFT40_PATCH_SIZE; n++) {
                const int32_t x = (int32_t)in_patch[n];
                re += x * (int32_t)cos_seq[n];
                im -= x * (int32_t)sin_seq[n];  // match Python: imag_acc = -(x @ sin)
            }

            const uint64_t re_abs = (uint64_t)((re < 0) ? -re : re);
            const uint64_t im_abs = (uint64_t)((im < 0) ? -im : im);
            const uint64_t mag_sq = re_abs * re_abs + im_abs * im_abs;
            const uint32_t mag_acc = rfft_isqrt_u64(mag_sq);

            int32_t q_mag = qround((float)mag_acc * a->mag_factor);
            if (q_mag > 127) q_mag = 127;
            if (q_mag < -128) q_mag = -128;
            out_patch[k] = (int8_t)q_mag;

            const int32_t angle_q15 = rfft_atan2_pi_q15(im, re);
            int32_t q_phase = qround((float)angle_q15 * a->phase_factor);
            if (q_phase > 127) q_phase = 127;
            if (q_phase < -128) q_phase = -128;
            out_patch[RFFT40_NUM_BINS + k] = (int8_t)q_phase;
        }
    }
}

void network_rfft_features_int8_parallel(
    const int8_t *input,
    int8_t *output,
    uint32_t num_patches,
    uint32_t patch_size,
    float scale_input,
    float scale_output
) {
    if (patch_size != RFFT40_PATCH_SIZE || scale_output == 0.0f) {
        // Codegen currently guarantees patch_size=40. If violated, produce a safe (zero) output.
        memset(output, 0, (size_t)num_patches * (size_t)RFFT40_OUT_FEATURES);
        return;
    }

    // q_mag = round_to_even(mag_acc * scale_input / (32768 * scale_output))
    const float mag_factor = scale_input / ((float)RFFT_Q15_SCALE * scale_output);

    // q_phase = round_to_even(angle_q15 * pi / (32768 * scale_output)), angle_q15 = angle/pi in Q15.
    const float pi = 3.14159265358979323846f;
    const float phase_factor = pi / ((float)RFFT_Q15_SCALE * scale_output);

    rfft_features_args_t args = {
        .input = input,
        .output = output,
        .num_patches = num_patches,
        .mag_factor = mag_factor,
        .phase_factor = phase_factor,
    };
    pi_cl_team_fork(NUM_CORES, rfft40_features_worker, &args);
}

// Elementwise ops (Add, Concat, Transpose) moved to ops/op_elementwise.c
// Normalization ops (LayerNorm, GroupNorm) moved to ops/op_norm.c

