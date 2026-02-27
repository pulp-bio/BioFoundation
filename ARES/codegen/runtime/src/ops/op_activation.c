/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Activation Operations for ARES Runtime
 *
 * ReLU, GELU (LUT and tanh), SiLU (LUT), and Requantize.
 * All parallel ops use the GAP9 cluster (8 cores).
 */

#include "ops/op_activation.h"
#include "network_kernels.h"
#include "core/utils.h"
#include <math.h>
#include "pmsis.h"

/* ---
 * i-GELU: Integer-only GELU using lookup table for bit-exact matching
 * ---
 * LUT stores gate = 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 * Input range: [-4.0, 4.0], GELU gate ranges from ~0 to ~1
 */
const int16_t i_gelu_lut[I_GELU_NUM_ENTRIES] = {
    /* gate values scaled by 32767: gate=0 -> 0, gate=0.5 -> 16384, gate=1 -> 32767 */
        1,     1,     1,     1,     1,     1,     1,     2,     2,     2,     3,     3,     3,     4,     4,     5,
        6,     7,     8,     9,    10,    11,    13,    14,    16,    18,    20,    23,    26,    29,    32,    36,
       40,    45,    50,    56,    62,    69,    76,    85,    93,   103,   114,   126,   138,   152,   167,   183,
      201,   220,   241,   263,   287,   313,   341,   371,   403,   438,   475,   515,   557,   603,   651,   703,
      758,   817,   879,   945,  1015,  1090,  1168,  1252,  1340,  1433,  1531,  1634,  1743,  1857,  1977,  2104,
     2236,  2374,  2519,  2671,  2829,  2994,  3167,  3346,  3533,  3727,  3928,  4137,  4354,  4578,  4810,  5050,
     5297,  5553,  5816,  6087,  6365,  6652,  6945,  7247,  7555,  7871,  8194,  8524,  8860,  9203,  9553,  9908,
    10270, 10637, 11009, 11386, 11768, 12154, 12545, 12939, 13336, 13736, 14139, 14544, 14951, 15359, 15769, 16178,
    16589, 16998, 17408, 17816, 18223, 18628, 19031, 19431, 19828, 20222, 20613, 20999, 21381, 21758, 22130, 22497,
    22859, 23214, 23564, 23907, 24243, 24573, 24896, 25212, 25520, 25822, 26115, 26402, 26680, 26951, 27214, 27470,
    27717, 27957, 28189, 28413, 28630, 28839, 29040, 29234, 29421, 29600, 29773, 29938, 30096, 30248, 30393, 30531,
    30663, 30790, 30910, 31024, 31133, 31236, 31334, 31427, 31515, 31599, 31677, 31752, 31822, 31888, 31950, 32009,
    32064, 32116, 32164, 32210, 32252, 32292, 32329, 32364, 32396, 32426, 32454, 32480, 32504, 32526, 32547, 32566,
    32584, 32600, 32615, 32629, 32641, 32653, 32664, 32674, 32682, 32691, 32698, 32705, 32711, 32717, 32722, 32727,
    32731, 32735, 32738, 32741, 32744, 32747, 32749, 32751, 32753, 32754, 32756, 32757, 32758, 32759, 32760, 32761,
    32762, 32763, 32763, 32764, 32764, 32764, 32765, 32765, 32765, 32766, 32766, 32766, 32766, 32766, 32766, 32766,
};

void network_gelu_int8_lut_inplace(
    int8_t *buffer, uint32_t num_elements,
    float scale_input, float scale_output
) {
    for (uint32_t i = 0; i < num_elements; i++) {
        float x = (float)buffer[i] * scale_input;
        int idx = (int)lrintf((x - I_GELU_INPUT_MIN) / I_GELU_INPUT_STEP);
        if (idx < 0) idx = 0;
        if (idx >= I_GELU_NUM_ENTRIES) idx = I_GELU_NUM_ENTRIES - 1;

        int16_t gate_int16 = i_gelu_lut[idx];
        float gate_fp32 = (float)gate_int16 / (float)I_GELU_OUTPUT_SCALE;
        float result_fp32 = x * gate_fp32;

        int32_t q = qround(result_fp32 / scale_output);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        buffer[i] = (int8_t)q;
    }
}

typedef struct {
    int8_t *buffer;
    uint32_t num_elements;
    const int8_t *map_int8;
} gelu_lut_map_args_t;

static void gelu_lut_map_worker(void *arg) {
    gelu_lut_map_args_t *a = (gelu_lut_map_args_t *)arg;
    const int core_id = pi_core_id();

    for (uint32_t i = core_id; i < a->num_elements; i += NUM_CORES) {
        const int v = (int)a->buffer[i];
        a->buffer[i] = a->map_int8[(uint32_t)(v + 128)];
    }
}

void network_gelu_int8_lut_inplace_parallel(
    int8_t *buffer, uint32_t num_elements,
    float scale_input, float scale_output
) {
    int8_t map_int8[256];
    for (int v = -128; v <= 127; v++) {
        float x = (float)v * scale_input;
        int idx = (int)lrintf((x - I_GELU_INPUT_MIN) / I_GELU_INPUT_STEP);
        if (idx < 0) idx = 0;
        if (idx >= I_GELU_NUM_ENTRIES) idx = I_GELU_NUM_ENTRIES - 1;

        int16_t gate_int16 = i_gelu_lut[idx];
        float gate_fp32 = (float)gate_int16 / (float)I_GELU_OUTPUT_SCALE;
        float result_fp32 = x * gate_fp32;

        int32_t q = qround(result_fp32 / scale_output);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        map_int8[(uint32_t)(v + 128)] = (int8_t)q;
    }

    gelu_lut_map_args_t args = {
        .buffer = buffer,
        .num_elements = num_elements,
        .map_int8 = map_int8,
    };
    pi_cl_team_fork(NUM_CORES, gelu_lut_map_worker, &args);
}

void network_gelu_int8(
    const int8_t *input, int8_t *output,
    uint32_t num_elements,
    float scale_input, float scale_output
) {
    /* Build precomputed map for all 256 possible INT8 values */
    int8_t map_int8[256];
    for (int v = -128; v <= 127; v++) {
        float x = (float)v * scale_input;
        int idx = (int)lrintf((x - I_GELU_INPUT_MIN) / I_GELU_INPUT_STEP);
        if (idx < 0) idx = 0;
        if (idx >= I_GELU_NUM_ENTRIES) idx = I_GELU_NUM_ENTRIES - 1;

        int16_t gate_int16 = i_gelu_lut[idx];
        float gate_fp32 = (float)gate_int16 / (float)I_GELU_OUTPUT_SCALE;
        float result_fp32 = x * gate_fp32;

        int32_t q = qround(result_fp32 / scale_output);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        map_int8[(uint32_t)(v + 128)] = (int8_t)q;
    }

    /* Apply map (single-threaded for orchestrator core context) */
    for (uint32_t i = 0; i < num_elements; i++) {
        output[i] = map_int8[(uint32_t)(input[i] + 128)];
    }
}

void network_gelu_int8_inplace(
    int8_t *buffer, uint32_t num_elements,
    float scale_input, float scale_output
) {
    /* GELU activation - sequential processing
     * Called from orchestrator core (Core 8), not forked to workers
     * For small buffers (typical: 64-1024 elements), sequential is acceptable */
    for (uint32_t i = 0; i < num_elements; i++) {
        float x = (float)buffer[i] * scale_input;
        /* GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        float res = x * cdf;

        int32_t q = qround(res / scale_output);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        buffer[i] = (int8_t)q;
    }
}

/* ---
 * SiLU (Swish) Activation via 256-entry LUT
 * ---
 * SiLU(x) = x * sigmoid(x)
 * For INT8 input, we can pre-compute all 256 possible outputs.
 * LUT is indexed by (x_int8 + 128) to map [-128, 127] -> [0, 255]
 */
void network_silu_int8_lut(
    const int8_t *input,
    int8_t *output,
    const int8_t *lut,
    int num_elements
) {
    int core_id = pi_core_id();
    int chunk = (num_elements + CL_NUM_CORES - 1) / CL_NUM_CORES;
    int start = core_id * chunk;
    int end = (start + chunk < num_elements) ? (start + chunk) : num_elements;

    for (int i = start; i < end; i++) {
        int idx = (int)input[i] + 128;
        output[i] = lut[idx];
    }

    pi_cl_team_barrier();
}

void network_silu_int8_lut_inplace(
    int8_t *buffer,
    const int8_t *lut,
    int num_elements
) {
    int core_id = pi_core_id();
    int chunk = (num_elements + CL_NUM_CORES - 1) / CL_NUM_CORES;
    int start = core_id * chunk;
    int end = (start + chunk < num_elements) ? (start + chunk) : num_elements;

    for (int i = start; i < end; i++) {
        int idx = (int)buffer[i] + 128;
        buffer[i] = lut[idx];
    }

    pi_cl_team_barrier();
}

void generate_silu_lut_int8(
    int8_t *lut,
    float scale_in,
    float scale_out
) {
    for (int i = 0; i < 256; i++) {
        int8_t x_int8 = (int8_t)(i - 128);
        float x_fp32 = (float)x_int8 * scale_in;

        /* SiLU(x) = x * sigmoid(x) */
        float sigmoid_x = 1.0f / (1.0f + fast_exp(-x_fp32));
        float y_fp32 = x_fp32 * sigmoid_x;

        /* Quantize to INT8 */
        int32_t y_int = qround(y_fp32 / scale_out);
        if (y_int < -128) y_int = -128;
        if (y_int > 127) y_int = 127;
        lut[i] = (int8_t)y_int;
    }
}

/* --- Simple inplace operations (parallel across 8 cores) --- */

void relu_int8_inplace(int8_t *data, size_t size) {
    int core_id = pi_core_id();
    int chunk = (size + CL_NUM_CORES - 1) / CL_NUM_CORES;
    int start = core_id * chunk;
    int end = (start + chunk < (int)size) ? (start + chunk) : (int)size;

    for (int i = start; i < end; i++) {
        if (data[i] < 0) data[i] = 0;
    }
    pi_cl_team_barrier();
}

void requantize_int8_inplace(int8_t *data, size_t size, float scale_in, float scale_out) {
    float ratio = scale_in / scale_out;
    int core_id = pi_core_id();
    int chunk = (size + CL_NUM_CORES - 1) / CL_NUM_CORES;
    int start = core_id * chunk;
    int end = (start + chunk < (int)size) ? (start + chunk) : (int)size;

    for (int i = start; i < end; i++) {
        int32_t val = qround((float)data[i] * ratio);
        if (val > 127) val = 127;
        if (val < -128) val = -128;
        data[i] = (int8_t)val;
    }
    pi_cl_team_barrier();
}
