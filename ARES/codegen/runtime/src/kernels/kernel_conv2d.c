/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * kernel_conv2d.c - Conv2D Kernels for GAP9
 *
 * Contains all Conv2D variants:
 *   - im2col_patch: Extract input patches for SIMD MatMul
 *   - matmul_simd_one_pixel: SIMD matrix multiply for one output pixel
 *   - network_conv2d_reference: Single-core debug kernel
 *   - network_conv2d_int8: Multi-core CHW layout with im2col + SIMD
 *   - network_conv2d_int8_hwc: Multi-core HWC layout for small channels
 *
 * Optimizations:
 *   - Fast paths for 1x1, 3x3, 1xK kernels
 *   - SIMD via SumDotpSS intrinsic (4 MACs per cycle)
 *   - Output channel unrolling (4x)
 *   - Pixel unrolling for 1x1 convolutions (2x)
 *
 * Part of the ARES modular kernel system.
 */

// ---
// IM2COL + MATMUL UTILITIES (PULP-NN style)
// ---

// im2col: Extract a single output pixel's patch into a column buffer
// Input: CHW format [in_ch, in_h, in_w]
// Output: Column buffer [in_ch * kernel_h * kernel_w] (contiguous for SIMD)
static inline void im2col_patch(
    const int8_t *input,
    int8_t *col_buffer,
    int in_h, int in_w, int in_ch,
    int kernel_h, int kernel_w,
    int out_y, int out_x,
    int stride_h, int stride_w,
    int pad_h, int pad_w
) {
    const int in_plane_size = in_h * in_w;

    // Fast path: 1x1 (common for bottlenecks / ResNet)
    if ((kernel_h == 1) && (kernel_w == 1) && (pad_h == 0) && (pad_w == 0)) {
        const int in_y = out_y * stride_h;
        const int in_x = out_x * stride_w;
        if (((unsigned)in_y < (unsigned)in_h) && ((unsigned)in_x < (unsigned)in_w)) {
            const int offset = in_y * in_w + in_x;
            const int8_t *p = input + offset;
            int8_t *dst = col_buffer;
            for (int c = 0; c < in_ch; c++) {
                *dst++ = *p;
                p += in_plane_size;
            }
        } else {
            memset(col_buffer, 0, (size_t)in_ch);
        }
        return;
    }

    // Fast path: 3x3 (common)
    if ((kernel_h == 3) && (kernel_w == 3) && (in_h >= 3) && (in_w >= 3)) {
        const int in_y0 = out_y * stride_h - pad_h;
        const int in_x0 = out_x * stride_w - pad_w;

        // Interior pixels: no bounds checks, fully unrolled copy
        if (((unsigned)in_y0 < (unsigned)(in_h - 2)) && ((unsigned)in_x0 < (unsigned)(in_w - 2))) {
            const int base_offset = in_y0 * in_w + in_x0;
            const int8_t *ch_ptr = input + base_offset;
            int8_t *dst = col_buffer;
            for (int c = 0; c < in_ch; c++) {
                const int8_t *p0 = ch_ptr;
                const int8_t *p1 = p0 + in_w;
                const int8_t *p2 = p1 + in_w;

                dst[0] = p0[0]; dst[1] = p0[1]; dst[2] = p0[2];
                dst[3] = p1[0]; dst[4] = p1[1]; dst[5] = p1[2];
                dst[6] = p2[0]; dst[7] = p2[1]; dst[8] = p2[2];

                dst += 9;
                ch_ptr += in_plane_size;
            }
        } else {
            // Border pixels: zero-fill then conditional loads (still unrolled)
            const int x0 = in_x0;
            const int x1 = in_x0 + 1;
            const int x2 = in_x0 + 2;

            const int y0 = in_y0;
            const int y1 = in_y0 + 1;
            const int y2 = in_y0 + 2;

            const int8_t *in_ch_base = input;
            int8_t *dst = col_buffer;
            for (int c = 0; c < in_ch; c++) {
                dst[0] = 0; dst[1] = 0; dst[2] = 0;
                dst[3] = 0; dst[4] = 0; dst[5] = 0;
                dst[6] = 0; dst[7] = 0; dst[8] = 0;

                if ((unsigned)y0 < (unsigned)in_h) {
                    const int8_t *row = in_ch_base + y0 * in_w;
                    if ((unsigned)x0 < (unsigned)in_w) dst[0] = row[x0];
                    if ((unsigned)x1 < (unsigned)in_w) dst[1] = row[x1];
                    if ((unsigned)x2 < (unsigned)in_w) dst[2] = row[x2];
                }
                if ((unsigned)y1 < (unsigned)in_h) {
                    const int8_t *row = in_ch_base + y1 * in_w;
                    if ((unsigned)x0 < (unsigned)in_w) dst[3] = row[x0];
                    if ((unsigned)x1 < (unsigned)in_w) dst[4] = row[x1];
                    if ((unsigned)x2 < (unsigned)in_w) dst[5] = row[x2];
                }
                if ((unsigned)y2 < (unsigned)in_h) {
                    const int8_t *row = in_ch_base + y2 * in_w;
                    if ((unsigned)x0 < (unsigned)in_w) dst[6] = row[x0];
                    if ((unsigned)x1 < (unsigned)in_w) dst[7] = row[x1];
                    if ((unsigned)x2 < (unsigned)in_w) dst[8] = row[x2];
                }

                in_ch_base += in_plane_size;
                dst += 9;
            }
        }
        return;
    }

    // Generic fallback
    int col_idx = 0;

    for (int c = 0; c < in_ch; c++) {
        const int8_t *in_ch_base = input + c * in_plane_size;
        for (int ky = 0; ky < kernel_h; ky++) {
            int in_y = out_y * stride_h + ky - pad_h;
            for (int kx = 0; kx < kernel_w; kx++) {
                int in_x = out_x * stride_w + kx - pad_w;
                if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                    col_buffer[col_idx] = in_ch_base[in_y * in_w + in_x];
                } else {
                    col_buffer[col_idx] = 0;  // Zero padding
                }
                col_idx++;
            }
        }
    }
}

// SIMD MatMul for Conv2D: Compute one output pixel for all output channels
// col_buffer: [col_size] = input patch for this pixel
// weights: [out_ch, col_size] where col_size = in_ch * kernel_h * kernel_w
// output: [out_ch] output values for this pixel (INT32 accumulators)
static inline void matmul_simd_one_pixel(
    const int8_t *col_buffer,
    const int8_t *weights,
    int32_t *output_acc,
    int out_ch, int col_size,
    int start_ch, int end_ch
) {
    const int simd_count = col_size >> 2;
    const int remainder = col_size & 0x3;

    for (int k = start_ch; k < end_ch; k++) {
        int32_t acc = 0;
        const int8_t *w_row = weights + k * col_size;

        // SIMD inner loop: 4 MACs per iteration
        const v4s *pA = (const v4s *)col_buffer;
        const v4s *pB = (const v4s *)w_row;
        for (int j = 0; j < simd_count; j++) {
            acc = SumDotpSS(pA[j], pB[j], acc);
        }

        // Handle remainder
        const int8_t *pA_rem = col_buffer + (simd_count << 2);
        const int8_t *pB_rem = w_row + (simd_count << 2);
        for (int j = 0; j < remainder; j++) {
            acc += (int32_t)pA_rem[j] * (int32_t)pB_rem[j];
        }

        output_acc[k] = acc;
    }
}

// ---
// CONV2D REFERENCE KERNEL (Single-Core, Debug)
// ---
void network_conv2d_reference(
    const int8_t *input, const int8_t *weights, const void *bias, int8_t *output,
    uint16_t in_h, uint16_t in_w, uint16_t in_ch,
    uint16_t out_h, uint16_t out_w, uint16_t out_ch,
    uint16_t kernel_h, uint16_t kernel_w, uint16_t weight_row_stride,
    uint16_t stride_h, uint16_t stride_w,
    uint16_t pad_h, uint16_t pad_w,
    float scale_input, float scale_weight, float scale_output,
    struct pi_device *cluster_dev
) {
    if (pi_core_id() != 0) return; // Single core only

    const int col_size = in_ch * kernel_h * kernel_w;
    const int w_stride = (weight_row_stride != 0) ? (int)weight_row_stride : col_size;

    for (int k = 0; k < out_ch; k++) {
        const int8_t *w_base = weights + k * w_stride;
        for (int y = 0; y < out_h; y++) {
            for (int x = 0; x < out_w; x++) {
                int32_t acc = 0;
                for (int c = 0; c < in_ch; c++) {
                    for (int ky = 0; ky < kernel_h; ky++) {
                        for (int kx = 0; kx < kernel_w; kx++) {
                            int in_y = y * stride_h + ky - pad_h;
                            int in_x = x * stride_w + kx - pad_w;
                            if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                                int8_t i_val = input[(c * in_h + in_y) * in_w + in_x];
                                int8_t w_val = w_base[(c * kernel_h + ky) * kernel_w + kx];
                                acc += i_val * w_val;
                            }
                        }
                    }
                }
                if (bias) acc += ((int32_t*)bias)[k];

                float val_fp32 = (float)acc * scale_input * scale_weight;
                int32_t q = qround(val_fp32 / scale_output);
                if (q > 127) q = 127;
                if (q < -128) q = -128;
                output[(k * out_h + y) * out_w + x] = (int8_t)q;
            }
        }
    }
}

// ---
// CONV2D KERNEL (INT8) - Multi-Core with im2col + SIMD MatMul
// ---
// Strategy: Distribute output rows across cores. Each core:
// 1. Builds im2col buffer for its assigned output pixels
// 2. Uses SIMD MatMul to compute all output channels for each pixel
//
// This enables SIMD because the weight matrix inner product is contiguous.

// Per-core im2col buffer in L1 (aligned for SIMD)
// Default: 8 cores x 1280 bytes = 10KB total L1 usage for im2col
// Default supports up to: 1280 = 142 channels * 3x3 kernel, or 26 channels * 7x7 kernel
#ifndef IM2COL_BUF_SIZE
#define IM2COL_BUF_SIZE 1280
#endif
static int8_t __attribute__((section(".l1_data"))) im2col_buffers[CL_NUM_CORES][IM2COL_BUF_SIZE] __attribute__((aligned(4)));

// Minimum col_size threshold for im2col benefit (im2col overhead amortized by SIMD gains)
// Below this, direct convolution is faster
#ifndef IM2COL_MIN_COL_SIZE
#define IM2COL_MIN_COL_SIZE 32
#endif

// Unroll factor for the im2col+SIMD output-channel loop.
// Higher reduces repeated col_buffer reads but increases register pressure.
#ifndef CONV2D_IM2COL_OUTCH_UNROLL
#define CONV2D_IM2COL_OUTCH_UNROLL 4
#endif

// Pixel unroll factor for 1x1 convolutions.
// When >= 2, processes 2 output pixels together to reuse weight loads.
// Set to 1 to disable (e.g., for debugging or if L1 col_buffer is too small).
#ifndef CONV1X1_PX_UNROLL
#define CONV1X1_PX_UNROLL 2
#endif

// Direct GEMM bypass for 1x1 convolutions (no im2col overhead).
// When enabled, 1x1 convs are mapped directly to GEMM: [HW, Cin] x [Cin, Cout] → [HW, Cout]
// Requires stride=1, pad=0 to be applicable.
#ifndef CONV2D_1X1_USE_GEMM
#define CONV2D_1X1_USE_GEMM 0  // 0 = use im2col, 1 = use direct GEMM path
#endif

// Use one final SumDotpSS for the tail (col_size % 4) instead of a scalar loop.
// This helps when the im2col column size isn't a multiple of 4 (e.g., RGB 7x7).
#ifndef CONV2D_IM2COL_SIMD_TAIL_DOTP
#define CONV2D_IM2COL_SIMD_TAIL_DOTP 0
#endif

void network_conv2d_int8(
    const int8_t *input, const int8_t *weights, const void *bias, int8_t *output,
    uint16_t in_h, uint16_t in_w, uint16_t in_ch,
    uint16_t out_h, uint16_t out_w, uint16_t out_ch,
    uint16_t kernel_h, uint16_t kernel_w, uint16_t weight_row_stride,
    uint16_t stride_h, uint16_t stride_w,
    uint16_t pad_h, uint16_t pad_w,
    float scale_input, float scale_weight, float scale_output,
    struct pi_device *cluster_dev
) {
    int core_id = pi_core_id();
    const int col_size = in_ch * kernel_h * kernel_w;
    const int w_stride = (weight_row_stride != 0) ? (int)weight_row_stride : col_size;

#ifndef CONV2D_INT8_FIXEDPOINT_REQUANT
#define CONV2D_INT8_FIXEDPOINT_REQUANT 0
#endif
#ifndef CONV2D_INT8_REQUANT_SHIFT
#define CONV2D_INT8_REQUANT_SHIFT 24
#endif
#if CONV2D_INT8_FIXEDPOINT_REQUANT
    const int requant_shift = CONV2D_INT8_REQUANT_SHIFT;
    const float combined_scale = (scale_input * scale_weight) / scale_output;
    const int32_t requant_mul = qround(combined_scale * (float)(1 << requant_shift));
#else
    const float combined_scale = (scale_input * scale_weight) / scale_output;
#endif

#if CONV2D_1X1_USE_GEMM
    // Direct GEMM bypass for 1x1 convolutions (stride=1, no padding).
    // Maps to [HW, Cin] x [Cin, Cout] → [HW, Cout], avoiding im2col overhead.
    if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && pad_h == 0 && pad_w == 0) {
        // Reuse linear kernel: input is [H*W, Cin], weights are [Cout, Cin]
        const int batch_tokens = out_h * out_w;
        network_linear_int8_parallel(
            input, weights, bias, output,
            in_ch,           // in_features = input channels
            out_ch,          // out_features = output channels
            batch_tokens,    // batch_tokens = spatial positions
            scale_input, scale_weight, scale_output,
            /* fusion_relu */ 0, /* fusion_quant */ 0,
            /* relu_output_scale */ 0.0f, /* quant_scale_in */ 0.0f, /* quant_scale_out */ 0.0f
        );
        return;
    }
#endif

    // Fast path for patch-embedding style Conv2D:
    // - in_ch=1, kernel_h=1, kernel_w=20
    // - stride_w=20, no padding
    // This avoids the scalar inner loop by using 5x `SumDotpSS` (20 MACs) and
    // reuses the loaded input patch across all output channels owned by a core.
    if (
        (in_ch == 1) &&
        (kernel_h == 1) &&
        (kernel_w == 20) &&
        (stride_h == 1) &&
        (stride_w == 20) &&
        (pad_h == 0) &&
        (pad_w == 0)
    ) {
        // Distribute output channels across cores (same partitioning as the generic path)
        int chunk = (out_ch + CL_NUM_CORES - 1) / CL_NUM_CORES;
        int start_ch = core_id * chunk;
        int end_ch = (start_ch + chunk > out_ch) ? out_ch : (start_ch + chunk);

        const int out_plane_size = out_h * out_w;

        for (int y = 0; y < out_h; y++) {
            const int8_t *in_row = input + y * in_w;  // in_ch==1, kernel_h==1
            for (int x = 0; x < out_w; x++) {
                const int8_t *in_patch = in_row + x * stride_w;  // stride_w == kernel_w == 20

                const v4s in0 = *((const v4s *)(in_patch + 0));
                const v4s in1 = *((const v4s *)(in_patch + 4));
                const v4s in2 = *((const v4s *)(in_patch + 8));
                const v4s in3 = *((const v4s *)(in_patch + 12));
                const v4s in4 = *((const v4s *)(in_patch + 16));

                for (int k = start_ch; k < end_ch; k++) {
                    int32_t acc = 0;
                    const v4s *w_row = (const v4s *)(weights + k * w_stride);

                    acc = SumDotpSS(in0, w_row[0], acc);
                    acc = SumDotpSS(in1, w_row[1], acc);
                    acc = SumDotpSS(in2, w_row[2], acc);
                    acc = SumDotpSS(in3, w_row[3], acc);
                    acc = SumDotpSS(in4, w_row[4], acc);

                    if (bias) acc += ((int32_t *)bias)[k];

#if CONV2D_INT8_FIXEDPOINT_REQUANT
                    int32_t q = mul_shift_round_nearest_even(acc, requant_mul, requant_shift);
#else
                    float val_fp32 = (float)acc * combined_scale;
                    int32_t q = qround(val_fp32);
#endif
                    if (q > 127) q = 127;
                    if (q < -128) q = -128;

                    output[k * out_plane_size + y * out_w + x] = (int8_t)q;
                }
            }
        }
        return;
    }

    // Use im2col+SIMD for common, safe cases where it improves throughput.
    // Fallback to the original direct path for other shapes.
#ifdef CONV2D_IM2COL_LEGACY_GATING
    const int stride_ok = (stride_h == 1) && (stride_w == 1);
    const int kernel_ok =
        // Common 3x3 conv (pad can be 0 for tiled halo or 1 for full input)
        (((kernel_h == 3) && (kernel_w == 3) && (pad_h <= 1) && (pad_w <= 1))) ||
        // Common 1x1 conv (no padding)
        (((kernel_h == 1) && (kernel_w == 1) && (pad_h == 0) && (pad_w == 0)));
#else
    const int stride_ok =
        ((stride_h == 1) || (stride_h == 2)) &&
        ((stride_w == 1) || (stride_w == 2));

    const int kernel_ok =
        // Common 3x3 conv (pad can be 0 for tiled halo or 1 for full input)
        (((kernel_h == 3) && (kernel_w == 3) && (pad_h <= 1) && (pad_w <= 1))) ||
        // Common 1x1 conv (typically pad=0)
        (((kernel_h == 1) && (kernel_w == 1) && (pad_h == 0) && (pad_w == 0))) ||
        // Common 5x5 conv (pad=2)
        (((kernel_h == 5) && (kernel_w == 5) && (pad_h <= 2) && (pad_w <= 2))) ||
        // Occasional 7x7 conv (pad=3)
        (((kernel_h == 7) && (kernel_w == 7) && (pad_h <= 3) && (pad_w <= 3)));
#endif

    const int use_im2col =
        (col_size >= IM2COL_MIN_COL_SIZE) &&
        (w_stride <= IM2COL_BUF_SIZE) &&
        stride_ok &&
        kernel_ok;

    if (!use_im2col) {
        // Distribute output channels across cores (original approach)
        int chunk = (out_ch + CL_NUM_CORES - 1) / CL_NUM_CORES;
        int start_ch = core_id * chunk;
        int end_ch = (start_ch + chunk > out_ch) ? out_ch : (start_ch + chunk);

        for (int k = start_ch; k < end_ch; k++) {
            const int8_t *w_base = weights + k * w_stride;
            for (int y = 0; y < out_h; y++) {
                for (int x = 0; x < out_w; x++) {
                    int32_t acc = 0;

                    // Convolution Loop (original proven implementation)
                    for (int c = 0; c < in_ch; c++) {
                        for (int ky = 0; ky < kernel_h; ky++) {
                            for (int kx = 0; kx < kernel_w; kx++) {
                                int in_y = y * stride_h + ky - pad_h;
                                int in_x = x * stride_w + kx - pad_w;

                                if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                                    int8_t in_val = input[(c * in_h + in_y) * in_w + in_x];
                                    int8_t w_val = w_base[(c * kernel_h + ky) * kernel_w + kx];
                                    acc += (int32_t)in_val * (int32_t)w_val;
                                }
                            }
                        }
                    }

                    // Add bias to accumulator (original approach)
                    if (bias) {
                        acc += ((int32_t*)bias)[k];
                    }

                    // Scale and requantize
#if CONV2D_INT8_FIXEDPOINT_REQUANT
                    int32_t q = mul_shift_round_nearest_even(acc, requant_mul, requant_shift);
#else
                    float val_fp32 = (float)acc * scale_input * scale_weight;
                    int32_t q = qround(val_fp32 / scale_output);
#endif
                    if (q > 127) q = 127;
                    if (q < -128) q = -128;
                    output[(k * out_h + y) * out_w + x] = (int8_t)q;
                }
            }
        }
        return;
    }

    // For large col_size, use im2col + SIMD (distribute by output rows)
    int chunk = (out_h + CL_NUM_CORES - 1) / CL_NUM_CORES;
    int start_row = core_id * chunk;
    int end_row = (start_row + chunk > out_h) ? out_h : (start_row + chunk);

    const int out_plane_size = out_h * out_w;
    const int simd_count = w_stride >> 2;
    const int remainder = w_stride & 0x3;

    int8_t *col_buffer = im2col_buffers[core_id];
    const int col_stride = (w_stride + 3) & ~3;  // keep secondary buffer 4B-aligned for SIMD loads
    int can_unroll_1x1_2px = 0;
#if CONV1X1_PX_UNROLL >= 2
    can_unroll_1x1_2px =
        (kernel_h == 1) && (kernel_w == 1) && (pad_h == 0) && (pad_w == 0) &&
        (col_stride + w_stride <= IM2COL_BUF_SIZE);
#endif

    for (int out_y = start_row; out_y < end_row; out_y++) {
        if (can_unroll_1x1_2px) {
            for (int out_x = 0; out_x < out_w; ) {
                // For 1x1 convs, unroll two output pixels to reuse weight loads across pixels.
                // This is a small GEMM-like microkernel that trades a bit of extra register pressure
                // for ~2x fewer weight reads per pixel.

                int8_t *col0 = col_buffer;
                int8_t *col1 = col_buffer + col_stride;

                im2col_patch(input, col0, in_h, in_w, in_ch,
                             kernel_h, kernel_w, out_y, out_x,
                             stride_h, stride_w, pad_h, pad_w);
                if (w_stride > col_size) {
                    memset(col0 + col_size, 0, (size_t)(w_stride - col_size));
                }

                const int out_idx0 = out_y * out_w + out_x;
                const v4s *pA0 = (const v4s *)col0;

                if (out_x + 1 < out_w) {
                    // Two-pixel path
                    im2col_patch(input, col1, in_h, in_w, in_ch,
                                 kernel_h, kernel_w, out_y, out_x + 1,
                                 stride_h, stride_w, pad_h, pad_w);
                    if (w_stride > col_size) {
                        memset(col1 + col_size, 0, (size_t)(w_stride - col_size));
                    }

                    const int out_idx1 = out_idx0 + 1;
                    const v4s *pA1 = (const v4s *)col1;

                    int k = 0;

#if CONV2D_IM2COL_OUTCH_UNROLL >= 4
                    for (; k + 3 < out_ch; k += 4) {
                        int32_t acc0_0 = 0, acc1_0 = 0, acc2_0 = 0, acc3_0 = 0;
                        int32_t acc0_1 = 0, acc1_1 = 0, acc2_1 = 0, acc3_1 = 0;

                        const int8_t *w0 = weights + (k + 0) * w_stride;
                        const int8_t *w1 = weights + (k + 1) * w_stride;
                        const int8_t *w2 = weights + (k + 2) * w_stride;
                        const int8_t *w3 = weights + (k + 3) * w_stride;

                        const v4s *pB0 = (const v4s *)w0;
                        const v4s *pB1 = (const v4s *)w1;
                        const v4s *pB2 = (const v4s *)w2;
                        const v4s *pB3 = (const v4s *)w3;

                        for (int j = 0; j < simd_count; j++) {
                            const v4s b0 = pB0[j];
                            const v4s b1 = pB1[j];
                            const v4s b2 = pB2[j];
                            const v4s b3 = pB3[j];

                            const v4s a0 = pA0[j];
                            acc0_0 = SumDotpSS(a0, b0, acc0_0);
                            acc1_0 = SumDotpSS(a0, b1, acc1_0);
                            acc2_0 = SumDotpSS(a0, b2, acc2_0);
                            acc3_0 = SumDotpSS(a0, b3, acc3_0);

                            const v4s a1 = pA1[j];
                            acc0_1 = SumDotpSS(a1, b0, acc0_1);
                            acc1_1 = SumDotpSS(a1, b1, acc1_1);
                            acc2_1 = SumDotpSS(a1, b2, acc2_1);
                            acc3_1 = SumDotpSS(a1, b3, acc3_1);
                        }

                        if (remainder > 0) {
                            const int8_t *pA0_rem = col0 + (simd_count << 2);
                            const int8_t *pA1_rem = col1 + (simd_count << 2);
#if CONV2D_IM2COL_SIMD_TAIL_DOTP
                            const int8_t *pB0_rem = w0 + (simd_count << 2);
                            const int8_t *pB1_rem = w1 + (simd_count << 2);
                            const int8_t *pB2_rem = w2 + (simd_count << 2);
                            const int8_t *pB3_rem = w3 + (simd_count << 2);

                            if (remainder == 1) {
                                const int32_t a00 = (int32_t)pA0_rem[0];
                                const int32_t a01 = (int32_t)pA1_rem[0];
                                acc0_0 += a00 * (int32_t)pB0_rem[0];
                                acc1_0 += a00 * (int32_t)pB1_rem[0];
                                acc2_0 += a00 * (int32_t)pB2_rem[0];
                                acc3_0 += a00 * (int32_t)pB3_rem[0];
                                acc0_1 += a01 * (int32_t)pB0_rem[0];
                                acc1_1 += a01 * (int32_t)pB1_rem[0];
                                acc2_1 += a01 * (int32_t)pB2_rem[0];
                                acc3_1 += a01 * (int32_t)pB3_rem[0];
                            } else {
                                const v4s a0_tail = {
                                    pA0_rem[0],
                                    (remainder > 1) ? pA0_rem[1] : 0,
                                    (remainder > 2) ? pA0_rem[2] : 0,
                                    0
                                };
                                const v4s a1_tail = {
                                    pA1_rem[0],
                                    (remainder > 1) ? pA1_rem[1] : 0,
                                    (remainder > 2) ? pA1_rem[2] : 0,
                                    0
                                };

                                const v4s b0_tail = {
                                    pB0_rem[0],
                                    (remainder > 1) ? pB0_rem[1] : 0,
                                    (remainder > 2) ? pB0_rem[2] : 0,
                                    0
                                };
                                const v4s b1_tail = {
                                    pB1_rem[0],
                                    (remainder > 1) ? pB1_rem[1] : 0,
                                    (remainder > 2) ? pB1_rem[2] : 0,
                                    0
                                };
                                const v4s b2_tail = {
                                    pB2_rem[0],
                                    (remainder > 1) ? pB2_rem[1] : 0,
                                    (remainder > 2) ? pB2_rem[2] : 0,
                                    0
                                };
                                const v4s b3_tail = {
                                    pB3_rem[0],
                                    (remainder > 1) ? pB3_rem[1] : 0,
                                    (remainder > 2) ? pB3_rem[2] : 0,
                                    0
                                };

                                acc0_0 = SumDotpSS(a0_tail, b0_tail, acc0_0);
                                acc1_0 = SumDotpSS(a0_tail, b1_tail, acc1_0);
                                acc2_0 = SumDotpSS(a0_tail, b2_tail, acc2_0);
                                acc3_0 = SumDotpSS(a0_tail, b3_tail, acc3_0);

                                acc0_1 = SumDotpSS(a1_tail, b0_tail, acc0_1);
                                acc1_1 = SumDotpSS(a1_tail, b1_tail, acc1_1);
                                acc2_1 = SumDotpSS(a1_tail, b2_tail, acc2_1);
                                acc3_1 = SumDotpSS(a1_tail, b3_tail, acc3_1);
                            }
#else
                            const int8_t *pB0_rem = w0 + (simd_count << 2);
                            const int8_t *pB1_rem = w1 + (simd_count << 2);
                            const int8_t *pB2_rem = w2 + (simd_count << 2);
                            const int8_t *pB3_rem = w3 + (simd_count << 2);
                            for (int j = 0; j < remainder; j++) {
                                const int32_t a00 = (int32_t)pA0_rem[j];
                                const int32_t a01 = (int32_t)pA1_rem[j];
                                const int32_t b0s = (int32_t)pB0_rem[j];
                                const int32_t b1s = (int32_t)pB1_rem[j];
                                const int32_t b2s = (int32_t)pB2_rem[j];
                                const int32_t b3s = (int32_t)pB3_rem[j];

                                acc0_0 += a00 * b0s;
                                acc1_0 += a00 * b1s;
                                acc2_0 += a00 * b2s;
                                acc3_0 += a00 * b3s;

                                acc0_1 += a01 * b0s;
                                acc1_1 += a01 * b1s;
                                acc2_1 += a01 * b2s;
                                acc3_1 += a01 * b3s;
                            }
#endif
                        }

                        if (bias) {
                            const int32_t *b = (const int32_t *)bias;
                            acc0_0 += b[k + 0];
                            acc1_0 += b[k + 1];
                            acc2_0 += b[k + 2];
                            acc3_0 += b[k + 3];
                            acc0_1 += b[k + 0];
                            acc1_1 += b[k + 1];
                            acc2_1 += b[k + 2];
                            acc3_1 += b[k + 3];
                        }

#if CONV2D_INT8_FIXEDPOINT_REQUANT
                        int32_t q0_0 = mul_shift_round_nearest_even(acc0_0, requant_mul, requant_shift);
                        int32_t q1_0 = mul_shift_round_nearest_even(acc1_0, requant_mul, requant_shift);
                        int32_t q2_0 = mul_shift_round_nearest_even(acc2_0, requant_mul, requant_shift);
                        int32_t q3_0 = mul_shift_round_nearest_even(acc3_0, requant_mul, requant_shift);
                        int32_t q0_1 = mul_shift_round_nearest_even(acc0_1, requant_mul, requant_shift);
                        int32_t q1_1 = mul_shift_round_nearest_even(acc1_1, requant_mul, requant_shift);
                        int32_t q2_1 = mul_shift_round_nearest_even(acc2_1, requant_mul, requant_shift);
                        int32_t q3_1 = mul_shift_round_nearest_even(acc3_1, requant_mul, requant_shift);
#else
                        int32_t q0_0 = qround((float)acc0_0 * combined_scale);
                        int32_t q1_0 = qround((float)acc1_0 * combined_scale);
                        int32_t q2_0 = qround((float)acc2_0 * combined_scale);
                        int32_t q3_0 = qround((float)acc3_0 * combined_scale);
                        int32_t q0_1 = qround((float)acc0_1 * combined_scale);
                        int32_t q1_1 = qround((float)acc1_1 * combined_scale);
                        int32_t q2_1 = qround((float)acc2_1 * combined_scale);
                        int32_t q3_1 = qround((float)acc3_1 * combined_scale);
#endif
                        if (q0_0 > 127) q0_0 = 127; if (q0_0 < -128) q0_0 = -128;
                        if (q1_0 > 127) q1_0 = 127; if (q1_0 < -128) q1_0 = -128;
                        if (q2_0 > 127) q2_0 = 127; if (q2_0 < -128) q2_0 = -128;
                        if (q3_0 > 127) q3_0 = 127; if (q3_0 < -128) q3_0 = -128;
                        if (q0_1 > 127) q0_1 = 127; if (q0_1 < -128) q0_1 = -128;
                        if (q1_1 > 127) q1_1 = 127; if (q1_1 < -128) q1_1 = -128;
                        if (q2_1 > 127) q2_1 = 127; if (q2_1 < -128) q2_1 = -128;
                        if (q3_1 > 127) q3_1 = 127; if (q3_1 < -128) q3_1 = -128;

                        output[(k + 0) * out_plane_size + out_idx0] = (int8_t)q0_0;
                        output[(k + 1) * out_plane_size + out_idx0] = (int8_t)q1_0;
                        output[(k + 2) * out_plane_size + out_idx0] = (int8_t)q2_0;
                        output[(k + 3) * out_plane_size + out_idx0] = (int8_t)q3_0;

                        output[(k + 0) * out_plane_size + out_idx1] = (int8_t)q0_1;
                        output[(k + 1) * out_plane_size + out_idx1] = (int8_t)q1_1;
                        output[(k + 2) * out_plane_size + out_idx1] = (int8_t)q2_1;
                        output[(k + 3) * out_plane_size + out_idx1] = (int8_t)q3_1;
                    }
#endif

                    for (; k < out_ch; k++) {
                        int32_t acc0 = 0;
                        int32_t acc1 = 0;
                        const int8_t *w_row = weights + k * w_stride;
                        const v4s *pB = (const v4s *)w_row;

                        for (int j = 0; j < simd_count; j++) {
                            const v4s b = pB[j];
                            acc0 = SumDotpSS(pA0[j], b, acc0);
                            acc1 = SumDotpSS(pA1[j], b, acc1);
                        }

                        if (remainder > 0) {
                            const int8_t *pA0_rem = col0 + (simd_count << 2);
                            const int8_t *pA1_rem = col1 + (simd_count << 2);
                            const int8_t *pB_rem = w_row + (simd_count << 2);
#if CONV2D_IM2COL_SIMD_TAIL_DOTP
                            if (remainder == 1) {
                                acc0 += (int32_t)pA0_rem[0] * (int32_t)pB_rem[0];
                                acc1 += (int32_t)pA1_rem[0] * (int32_t)pB_rem[0];
                            } else {
                                const v4s a0_tail = {
                                    pA0_rem[0],
                                    (remainder > 1) ? pA0_rem[1] : 0,
                                    (remainder > 2) ? pA0_rem[2] : 0,
                                    0
                                };
                                const v4s a1_tail = {
                                    pA1_rem[0],
                                    (remainder > 1) ? pA1_rem[1] : 0,
                                    (remainder > 2) ? pA1_rem[2] : 0,
                                    0
                                };
                                const v4s b_tail = {
                                    pB_rem[0],
                                    (remainder > 1) ? pB_rem[1] : 0,
                                    (remainder > 2) ? pB_rem[2] : 0,
                                    0
                                };
                                acc0 = SumDotpSS(a0_tail, b_tail, acc0);
                                acc1 = SumDotpSS(a1_tail, b_tail, acc1);
                            }
#else
                            for (int j = 0; j < remainder; j++) {
                                acc0 += (int32_t)pA0_rem[j] * (int32_t)pB_rem[j];
                                acc1 += (int32_t)pA1_rem[j] * (int32_t)pB_rem[j];
                            }
#endif
                        }

                        if (bias) {
                            acc0 += ((int32_t*)bias)[k];
                            acc1 += ((int32_t*)bias)[k];
                        }

#if CONV2D_INT8_FIXEDPOINT_REQUANT
                        int32_t q0 = mul_shift_round_nearest_even(acc0, requant_mul, requant_shift);
                        int32_t q1 = mul_shift_round_nearest_even(acc1, requant_mul, requant_shift);
#else
                        int32_t q0 = qround((float)acc0 * combined_scale);
                        int32_t q1 = qround((float)acc1 * combined_scale);
#endif
                        if (q0 > 127) q0 = 127;
                        if (q0 < -128) q0 = -128;
                        if (q1 > 127) q1 = 127;
                        if (q1 < -128) q1 = -128;

                        output[k * out_plane_size + out_idx0] = (int8_t)q0;
                        output[k * out_plane_size + out_idx1] = (int8_t)q1;
                    }

                    out_x += 2;
                    continue;
                }

                // One-pixel tail (rare): fall back to the standard im2col compute path below.
                im2col_patch(input, col_buffer, in_h, in_w, in_ch,
                             kernel_h, kernel_w, out_y, out_x,
                             stride_h, stride_w, pad_h, pad_w);
                if (w_stride > col_size) {
                    memset(col_buffer + col_size, 0, (size_t)(w_stride - col_size));
                }

                const int out_idx = out_y * out_w + out_x;
                const v4s *pA = (const v4s *)col_buffer;

                int k = 0;

#if CONV2D_IM2COL_OUTCH_UNROLL >= 4
                for (; k + 3 < out_ch; k += 4) {
                    int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

                    const int8_t *w0 = weights + (k + 0) * w_stride;
                    const int8_t *w1 = weights + (k + 1) * w_stride;
                    const int8_t *w2 = weights + (k + 2) * w_stride;
                    const int8_t *w3 = weights + (k + 3) * w_stride;

                    const v4s *pB0 = (const v4s *)w0;
                    const v4s *pB1 = (const v4s *)w1;
                    const v4s *pB2 = (const v4s *)w2;
                    const v4s *pB3 = (const v4s *)w3;

                    for (int j = 0; j < simd_count; j++) {
                        const v4s a = pA[j];
                        acc0 = SumDotpSS(a, pB0[j], acc0);
                        acc1 = SumDotpSS(a, pB1[j], acc1);
                        acc2 = SumDotpSS(a, pB2[j], acc2);
                        acc3 = SumDotpSS(a, pB3[j], acc3);
                    }

                    if (remainder > 0) {
                        const int8_t *pA_rem = col_buffer + (simd_count << 2);
#if CONV2D_IM2COL_SIMD_TAIL_DOTP
                        const int8_t *pB0_rem = w0 + (simd_count << 2);
                        const int8_t *pB1_rem = w1 + (simd_count << 2);
                        const int8_t *pB2_rem = w2 + (simd_count << 2);
                        const int8_t *pB3_rem = w3 + (simd_count << 2);

                        if (remainder == 1) {
                            const int32_t a0 = (int32_t)pA_rem[0];
                            acc0 += a0 * (int32_t)pB0_rem[0];
                            acc1 += a0 * (int32_t)pB1_rem[0];
                            acc2 += a0 * (int32_t)pB2_rem[0];
                            acc3 += a0 * (int32_t)pB3_rem[0];
                        } else {
                            const v4s a_tail = {
                                pA_rem[0],
                                (remainder > 1) ? pA_rem[1] : 0,
                                (remainder > 2) ? pA_rem[2] : 0,
                                0
                            };

                            const v4s b0_tail = {
                                pB0_rem[0],
                                (remainder > 1) ? pB0_rem[1] : 0,
                                (remainder > 2) ? pB0_rem[2] : 0,
                                0
                            };
                            const v4s b1_tail = {
                                pB1_rem[0],
                                (remainder > 1) ? pB1_rem[1] : 0,
                                (remainder > 2) ? pB1_rem[2] : 0,
                                0
                            };
                            const v4s b2_tail = {
                                pB2_rem[0],
                                (remainder > 1) ? pB2_rem[1] : 0,
                                (remainder > 2) ? pB2_rem[2] : 0,
                                0
                            };
                            const v4s b3_tail = {
                                pB3_rem[0],
                                (remainder > 1) ? pB3_rem[1] : 0,
                                (remainder > 2) ? pB3_rem[2] : 0,
                                0
                            };

                            acc0 = SumDotpSS(a_tail, b0_tail, acc0);
                            acc1 = SumDotpSS(a_tail, b1_tail, acc1);
                            acc2 = SumDotpSS(a_tail, b2_tail, acc2);
                            acc3 = SumDotpSS(a_tail, b3_tail, acc3);
                        }
#else
                        const int8_t *pB0_rem = w0 + (simd_count << 2);
                        const int8_t *pB1_rem = w1 + (simd_count << 2);
                        const int8_t *pB2_rem = w2 + (simd_count << 2);
                        const int8_t *pB3_rem = w3 + (simd_count << 2);
                        for (int j = 0; j < remainder; j++) {
                            const int32_t a = (int32_t)pA_rem[j];
                            acc0 += a * (int32_t)pB0_rem[j];
                            acc1 += a * (int32_t)pB1_rem[j];
                            acc2 += a * (int32_t)pB2_rem[j];
                            acc3 += a * (int32_t)pB3_rem[j];
                        }
#endif
                    }

                    if (bias) {
                        const int32_t *b = (const int32_t *)bias;
                        acc0 += b[k + 0];
                        acc1 += b[k + 1];
                        acc2 += b[k + 2];
                        acc3 += b[k + 3];
                    }

#if CONV2D_INT8_FIXEDPOINT_REQUANT
                    int32_t q0 = mul_shift_round_nearest_even(acc0, requant_mul, requant_shift);
                    int32_t q1 = mul_shift_round_nearest_even(acc1, requant_mul, requant_shift);
                    int32_t q2 = mul_shift_round_nearest_even(acc2, requant_mul, requant_shift);
                    int32_t q3 = mul_shift_round_nearest_even(acc3, requant_mul, requant_shift);
#else
                    int32_t q0 = qround((float)acc0 * combined_scale);
                    int32_t q1 = qround((float)acc1 * combined_scale);
                    int32_t q2 = qround((float)acc2 * combined_scale);
                    int32_t q3 = qround((float)acc3 * combined_scale);
#endif
                    if (q0 > 127) q0 = 127; if (q0 < -128) q0 = -128;
                    if (q1 > 127) q1 = 127; if (q1 < -128) q1 = -128;
                    if (q2 > 127) q2 = 127; if (q2 < -128) q2 = -128;
                    if (q3 > 127) q3 = 127; if (q3 < -128) q3 = -128;

                    output[(k + 0) * out_plane_size + out_idx] = (int8_t)q0;
                    output[(k + 1) * out_plane_size + out_idx] = (int8_t)q1;
                    output[(k + 2) * out_plane_size + out_idx] = (int8_t)q2;
                    output[(k + 3) * out_plane_size + out_idx] = (int8_t)q3;
                }
#endif

                for (; k < out_ch; k++) {
                    int32_t acc = 0;
                    const int8_t *w_row = weights + k * w_stride;
                    const v4s *pB = (const v4s *)w_row;

                    for (int j = 0; j < simd_count; j++) {
                        acc = SumDotpSS(pA[j], pB[j], acc);
                    }

                    if (remainder > 0) {
                        const int8_t *pA_rem = col_buffer + (simd_count << 2);
                        const int8_t *pB_rem = w_row + (simd_count << 2);
#if CONV2D_IM2COL_SIMD_TAIL_DOTP
                        if (remainder == 1) {
                            acc += (int32_t)pA_rem[0] * (int32_t)pB_rem[0];
                        } else {
                            const v4s a_tail = {
                                pA_rem[0],
                                (remainder > 1) ? pA_rem[1] : 0,
                                (remainder > 2) ? pA_rem[2] : 0,
                                0
                            };
                            const v4s b_tail = {
                                pB_rem[0],
                                (remainder > 1) ? pB_rem[1] : 0,
                                (remainder > 2) ? pB_rem[2] : 0,
                                0
                            };
                            acc = SumDotpSS(a_tail, b_tail, acc);
                        }
#else
                        for (int j = 0; j < remainder; j++) {
                            acc += (int32_t)pA_rem[j] * (int32_t)pB_rem[j];
                        }
#endif
                    }

                    if (bias) acc += ((int32_t*)bias)[k];

#if CONV2D_INT8_FIXEDPOINT_REQUANT
                    int32_t q = mul_shift_round_nearest_even(acc, requant_mul, requant_shift);
#else
                    int32_t q = qround((float)acc * combined_scale);
#endif
                    if (q > 127) q = 127;
                    if (q < -128) q = -128;

                    output[k * out_plane_size + out_idx] = (int8_t)q;
                }

                out_x += 1;
            }
        } else {
            for (int out_x = 0; out_x < out_w; out_x++) {
            im2col_patch(input, col_buffer, in_h, in_w, in_ch,
                         kernel_h, kernel_w, out_y, out_x,
                         stride_h, stride_w, pad_h, pad_w);
            if (w_stride > col_size) {
                memset(col_buffer + col_size, 0, (size_t)(w_stride - col_size));
            }

            const int out_idx = out_y * out_w + out_x;
            const v4s *pA = (const v4s *)col_buffer;

            int k = 0;

#if CONV2D_IM2COL_OUTCH_UNROLL >= 4
            for (; k + 3 < out_ch; k += 4) {
                int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

                const int8_t *w0 = weights + (k + 0) * w_stride;
                const int8_t *w1 = weights + (k + 1) * w_stride;
                const int8_t *w2 = weights + (k + 2) * w_stride;
                const int8_t *w3 = weights + (k + 3) * w_stride;

                const v4s *pB0 = (const v4s *)w0;
                const v4s *pB1 = (const v4s *)w1;
                const v4s *pB2 = (const v4s *)w2;
                const v4s *pB3 = (const v4s *)w3;

                for (int j = 0; j < simd_count; j++) {
                    const v4s a = pA[j];
                    acc0 = SumDotpSS(a, pB0[j], acc0);
                    acc1 = SumDotpSS(a, pB1[j], acc1);
                    acc2 = SumDotpSS(a, pB2[j], acc2);
                    acc3 = SumDotpSS(a, pB3[j], acc3);
                }

                if (remainder > 0) {
                    const int8_t *pA_rem = col_buffer + (simd_count << 2);
#if CONV2D_IM2COL_SIMD_TAIL_DOTP
                    const int8_t *pB0_rem = w0 + (simd_count << 2);
                    const int8_t *pB1_rem = w1 + (simd_count << 2);
                    const int8_t *pB2_rem = w2 + (simd_count << 2);
                    const int8_t *pB3_rem = w3 + (simd_count << 2);

                    if (remainder == 1) {
                        const int32_t a0 = (int32_t)pA_rem[0];
                        acc0 += a0 * (int32_t)pB0_rem[0];
                        acc1 += a0 * (int32_t)pB1_rem[0];
                        acc2 += a0 * (int32_t)pB2_rem[0];
                        acc3 += a0 * (int32_t)pB3_rem[0];
                    } else {
                        const v4s a_tail = {
                            pA_rem[0],
                            (remainder > 1) ? pA_rem[1] : 0,
                            (remainder > 2) ? pA_rem[2] : 0,
                            0
                        };

                        const v4s b0_tail = {
                            pB0_rem[0],
                            (remainder > 1) ? pB0_rem[1] : 0,
                            (remainder > 2) ? pB0_rem[2] : 0,
                            0
                        };
                        const v4s b1_tail = {
                            pB1_rem[0],
                            (remainder > 1) ? pB1_rem[1] : 0,
                            (remainder > 2) ? pB1_rem[2] : 0,
                            0
                        };
                        const v4s b2_tail = {
                            pB2_rem[0],
                            (remainder > 1) ? pB2_rem[1] : 0,
                            (remainder > 2) ? pB2_rem[2] : 0,
                            0
                        };
                        const v4s b3_tail = {
                            pB3_rem[0],
                            (remainder > 1) ? pB3_rem[1] : 0,
                            (remainder > 2) ? pB3_rem[2] : 0,
                            0
                        };

                        acc0 = SumDotpSS(a_tail, b0_tail, acc0);
                        acc1 = SumDotpSS(a_tail, b1_tail, acc1);
                        acc2 = SumDotpSS(a_tail, b2_tail, acc2);
                        acc3 = SumDotpSS(a_tail, b3_tail, acc3);
                    }
#else
                    const int8_t *pB0_rem = w0 + (simd_count << 2);
                    const int8_t *pB1_rem = w1 + (simd_count << 2);
                    const int8_t *pB2_rem = w2 + (simd_count << 2);
                    const int8_t *pB3_rem = w3 + (simd_count << 2);
                    for (int j = 0; j < remainder; j++) {
                        const int32_t a = (int32_t)pA_rem[j];
                        acc0 += a * (int32_t)pB0_rem[j];
                        acc1 += a * (int32_t)pB1_rem[j];
                        acc2 += a * (int32_t)pB2_rem[j];
                        acc3 += a * (int32_t)pB3_rem[j];
                    }
#endif
                }

                if (bias) {
                    const int32_t *b = (const int32_t *)bias;
                    acc0 += b[k + 0];
                    acc1 += b[k + 1];
                    acc2 += b[k + 2];
                    acc3 += b[k + 3];
                }

#if CONV2D_INT8_FIXEDPOINT_REQUANT
                int32_t q0 = mul_shift_round_nearest_even(acc0, requant_mul, requant_shift);
                int32_t q1 = mul_shift_round_nearest_even(acc1, requant_mul, requant_shift);
                int32_t q2 = mul_shift_round_nearest_even(acc2, requant_mul, requant_shift);
                int32_t q3 = mul_shift_round_nearest_even(acc3, requant_mul, requant_shift);
#else
                int32_t q0 = qround((float)acc0 * combined_scale);
                int32_t q1 = qround((float)acc1 * combined_scale);
                int32_t q2 = qround((float)acc2 * combined_scale);
                int32_t q3 = qround((float)acc3 * combined_scale);
#endif
                if (q0 > 127) q0 = 127; if (q0 < -128) q0 = -128;
                if (q1 > 127) q1 = 127; if (q1 < -128) q1 = -128;
                if (q2 > 127) q2 = 127; if (q2 < -128) q2 = -128;
                if (q3 > 127) q3 = 127; if (q3 < -128) q3 = -128;

                output[(k + 0) * out_plane_size + out_idx] = (int8_t)q0;
                output[(k + 1) * out_plane_size + out_idx] = (int8_t)q1;
                output[(k + 2) * out_plane_size + out_idx] = (int8_t)q2;
                output[(k + 3) * out_plane_size + out_idx] = (int8_t)q3;
            }
#endif

            for (; k < out_ch; k++) {
                int32_t acc = 0;
                const int8_t *w_row = weights + k * w_stride;
                const v4s *pB = (const v4s *)w_row;

                for (int j = 0; j < simd_count; j++) {
                    acc = SumDotpSS(pA[j], pB[j], acc);
                }

                if (remainder > 0) {
                    const int8_t *pA_rem = col_buffer + (simd_count << 2);
                    const int8_t *pB_rem = w_row + (simd_count << 2);
#if CONV2D_IM2COL_SIMD_TAIL_DOTP
                    if (remainder == 1) {
                        acc += (int32_t)pA_rem[0] * (int32_t)pB_rem[0];
                    } else {
                        const v4s a_tail = {
                            pA_rem[0],
                            (remainder > 1) ? pA_rem[1] : 0,
                            (remainder > 2) ? pA_rem[2] : 0,
                            0
                        };
                        const v4s b_tail = {
                            pB_rem[0],
                            (remainder > 1) ? pB_rem[1] : 0,
                            (remainder > 2) ? pB_rem[2] : 0,
                            0
                        };
                        acc = SumDotpSS(a_tail, b_tail, acc);
                    }
#else
                    for (int j = 0; j < remainder; j++) {
                        acc += (int32_t)pA_rem[j] * (int32_t)pB_rem[j];
                    }
#endif
                }

                if (bias) acc += ((int32_t*)bias)[k];

#if CONV2D_INT8_FIXEDPOINT_REQUANT
                int32_t q = mul_shift_round_nearest_even(acc, requant_mul, requant_shift);
#else
                int32_t q = qround((float)acc * combined_scale);
#endif
                if (q > 127) q = 127;
                if (q < -128) q = -128;

                output[k * out_plane_size + out_idx] = (int8_t)q;
            }
        }
        }
    }
}

// ---
// HWC CONV2D KERNEL - Optimized for Height-Width-Channel layout
// ---
// For networks with small channel counts, HWC layout enables efficient SIMD
// because channels are contiguous at each spatial position.
//
// HWC Layout: input[h, w, c] = input[(h * W + w) * C + c]
// - At each (h, w), C consecutive bytes can be loaded with SIMD
//
// Weights: [out_ch, kernel_h * kernel_w * in_ch] (same as CHW, row-major)
//
// Optimized for 1xK and Kx1 kernels common in 1D-style conv networks.

#ifndef CONV2D_HWC_OUTCH_UNROLL
#define CONV2D_HWC_OUTCH_UNROLL 4
#endif

// PULP clip to INT8 range [-128, 127]
#ifndef clip8_hwc
#define clip8_hwc(x) __builtin_pulp_clip_r((x), 127)
#endif

void network_conv2d_int8_hwc(
    const int8_t *input,          // INT8 input [H, W, C] in HWC layout
    const int8_t *weights,        // INT8 weights [out_ch, kernel_h * kernel_w * in_ch]
    const void *bias,             // INT32 bias [out_ch]
    int8_t *output,               // INT8 output [H_out, W_out, out_ch] in HWC layout
    uint16_t in_h, uint16_t in_w, uint16_t in_ch,
    uint16_t out_h, uint16_t out_w, uint16_t out_ch,
    uint16_t kernel_h, uint16_t kernel_w,
    uint16_t weight_row_stride,   // Weight stride per output channel (0 = use col_size)
    uint16_t stride_h, uint16_t stride_w,
    uint16_t pad_h, uint16_t pad_w,
    float scale_input, float scale_weight, float scale_output,
    uint16_t out_ch_stride,       // Output channel stride (0 = use out_ch, for Ko-tiling use total_out_ch)
    uint16_t out_ch_offset        // Output channel offset (for Ko-tiling)
) {
    const int core_id = pi_core_id();
    const int col_size = in_ch * kernel_h * kernel_w;
    // Weight stride: use provided value or default to col_size (for SIMD-padded weights)
    const int w_stride = (weight_row_stride > 0) ? (int)weight_row_stride : col_size;

    // For Ko-tiling: use out_ch_stride for output indexing, out_ch_offset for channel position
    // If out_ch_stride == 0, use out_ch (non-tiled case)
    const int eff_out_ch_stride = (out_ch_stride > 0) ? out_ch_stride : out_ch;

    // Fixed-point requantization
    const float combined_scale = (scale_input * scale_weight) / scale_output;
    const int requant_shift = 24;
    const int32_t requant_mul = qround(combined_scale * (float)(1 << requant_shift));

    // SIMD counts for col_size (total weights per output channel)
    const int simd_count = col_size >> 2;
    const int remainder = col_size & 0x3;

    // Specialize for common kernel patterns
    const int is_1xK = (kernel_h == 1);
    const int is_Kx1 = (kernel_w == 1);

    // Parallelize across output spatial positions (better load balance)
    const int total_out_pos = out_h * out_w;
    const int chunk = (total_out_pos + CL_NUM_CORES - 1) / CL_NUM_CORES;
    const int start_pos = core_id * chunk;
    const int end_pos = (start_pos + chunk > total_out_pos) ? total_out_pos : start_pos + chunk;

    // -------------------------------------------------------------------------
    // Fast path: 1xK kernel (horizontal convolution) - very common in 1D-style nets
    // -------------------------------------------------------------------------
    if (is_1xK && pad_h == 0 && pad_w == 0 && stride_h == 1) {
        // For 1xK kernels without padding, input access is simple
        // Input pattern: at output (y, x_out), read input at (y, x_out*stride..x_out*stride+K-1)

        // Special fast path for in_ch=1, kernel_w=4: SIMD over kernel positions
        // This is common for 1D-style networks processing single-channel time series
        if (in_ch == 1 && kernel_w == 4) {
            for (int pos = start_pos; pos < end_pos; pos++) {
                const int out_y = pos / out_w;
                const int out_x = pos - out_y * out_w;
                const int in_x_start = out_x * stride_w;

                int8_t *out_ptr = output + (out_y * out_w + out_x) * eff_out_ch_stride + out_ch_offset;

                // Load 4 consecutive input bytes (kernel_w=4, in_ch=1)
                const int8_t *in_ptr = input + out_y * in_w + in_x_start;
                const v4s input_vec = *((const v4s *)in_ptr);

                // Process 4 output channels at a time
                int oc = 0;
                for (; oc + 3 < out_ch; oc += 4) {
                    // Each output channel has 4 weight bytes (kernel_w=4)
                    const v4s w0 = *((const v4s *)(weights + (oc + 0) * 4));
                    const v4s w1 = *((const v4s *)(weights + (oc + 1) * 4));
                    const v4s w2 = *((const v4s *)(weights + (oc + 2) * 4));
                    const v4s w3 = *((const v4s *)(weights + (oc + 3) * 4));

                    int32_t acc0 = SumDotpSS(input_vec, w0, 0);
                    int32_t acc1 = SumDotpSS(input_vec, w1, 0);
                    int32_t acc2 = SumDotpSS(input_vec, w2, 0);
                    int32_t acc3 = SumDotpSS(input_vec, w3, 0);

                    if (bias) {
                        acc0 += ((int32_t *)bias)[oc + 0];
                        acc1 += ((int32_t *)bias)[oc + 1];
                        acc2 += ((int32_t *)bias)[oc + 2];
                        acc3 += ((int32_t *)bias)[oc + 3];
                    }

                    int32_t q0 = mul_shift_round_nearest_even(acc0, requant_mul, requant_shift);
                    int32_t q1 = mul_shift_round_nearest_even(acc1, requant_mul, requant_shift);
                    int32_t q2 = mul_shift_round_nearest_even(acc2, requant_mul, requant_shift);
                    int32_t q3 = mul_shift_round_nearest_even(acc3, requant_mul, requant_shift);

                    q0 = clip8_hwc(q0); q1 = clip8_hwc(q1);
                    q2 = clip8_hwc(q2); q3 = clip8_hwc(q3);

                    out_ptr[oc + 0] = (int8_t)q0;
                    out_ptr[oc + 1] = (int8_t)q1;
                    out_ptr[oc + 2] = (int8_t)q2;
                    out_ptr[oc + 3] = (int8_t)q3;
                }
                // Handle remaining output channels
                for (; oc < out_ch; oc++) {
                    const v4s w = *((const v4s *)(weights + oc * 4));
                    int32_t acc = SumDotpSS(input_vec, w, 0);
                    if (bias) acc += ((int32_t *)bias)[oc];
                    int32_t q = mul_shift_round_nearest_even(acc, requant_mul, requant_shift);
                    q = clip8_hwc(q);
                    out_ptr[oc] = (int8_t)q;
                }
            }
            return;
        }

        // Special fast path for in_ch=32, kernel_w=16: used by ppg_conv5 (2.23M MACs)
        // col_size = 512 bytes = 128 SIMDs per output channel
        // Process 2 output channels at a time with inner channel loop unrolled
        if (in_ch == 32 && kernel_w == 16) {
            for (int pos = start_pos; pos < end_pos; pos++) {
                const int out_y = pos / out_w;
                const int out_x = pos - out_y * out_w;
                const int in_x_start = out_x * stride_w;

                int8_t *out_ptr = output + (out_y * out_w + out_x) * eff_out_ch_stride + out_ch_offset;
                const int in_row_offset = out_y * in_w * 32;

                // Process 2 output channels at a time
                int oc = 0;
                for (; oc + 1 < out_ch; oc += 2) {
                    const v4s *w0 = (const v4s *)(weights + (oc + 0) * 512);
                    const v4s *w1 = (const v4s *)(weights + (oc + 1) * 512);

                    int32_t acc0 = 0, acc1 = 0;

                    // Loop over 16 kernel positions
                    for (int kx = 0; kx < 16; kx++) {
                        const int8_t *in_ptr = input + in_row_offset + (in_x_start + kx) * 32;
                        const v4s in0 = ((const v4s *)in_ptr)[0];
                        const v4s in1 = ((const v4s *)in_ptr)[1];
                        const v4s in2 = ((const v4s *)in_ptr)[2];
                        const v4s in3 = ((const v4s *)in_ptr)[3];
                        const v4s in4 = ((const v4s *)in_ptr)[4];
                        const v4s in5 = ((const v4s *)in_ptr)[5];
                        const v4s in6 = ((const v4s *)in_ptr)[6];
                        const v4s in7 = ((const v4s *)in_ptr)[7];

                        const int w_off = kx * 8;
                        acc0 = SumDotpSS(in0, w0[w_off+0], acc0);
                        acc0 = SumDotpSS(in1, w0[w_off+1], acc0);
                        acc0 = SumDotpSS(in2, w0[w_off+2], acc0);
                        acc0 = SumDotpSS(in3, w0[w_off+3], acc0);
                        acc0 = SumDotpSS(in4, w0[w_off+4], acc0);
                        acc0 = SumDotpSS(in5, w0[w_off+5], acc0);
                        acc0 = SumDotpSS(in6, w0[w_off+6], acc0);
                        acc0 = SumDotpSS(in7, w0[w_off+7], acc0);

                        acc1 = SumDotpSS(in0, w1[w_off+0], acc1);
                        acc1 = SumDotpSS(in1, w1[w_off+1], acc1);
                        acc1 = SumDotpSS(in2, w1[w_off+2], acc1);
                        acc1 = SumDotpSS(in3, w1[w_off+3], acc1);
                        acc1 = SumDotpSS(in4, w1[w_off+4], acc1);
                        acc1 = SumDotpSS(in5, w1[w_off+5], acc1);
                        acc1 = SumDotpSS(in6, w1[w_off+6], acc1);
                        acc1 = SumDotpSS(in7, w1[w_off+7], acc1);
                    }

                    if (bias) {
                        acc0 += ((int32_t *)bias)[oc + 0];
                        acc1 += ((int32_t *)bias)[oc + 1];
                    }

                    int32_t q0 = mul_shift_round_nearest_even(acc0, requant_mul, requant_shift);
                    int32_t q1 = mul_shift_round_nearest_even(acc1, requant_mul, requant_shift);
                    q0 = clip8_hwc(q0); q1 = clip8_hwc(q1);

                    out_ptr[oc + 0] = (int8_t)q0;
                    out_ptr[oc + 1] = (int8_t)q1;
                }
                // Handle last output channel if odd count
                for (; oc < out_ch; oc++) {
                    const v4s *w = (const v4s *)(weights + oc * 512);
                    int32_t acc = 0;
                    for (int kx = 0; kx < 16; kx++) {
                        const int8_t *in_ptr = input + in_row_offset + (in_x_start + kx) * 32;
                        const v4s in0 = ((const v4s *)in_ptr)[0];
                        const v4s in1 = ((const v4s *)in_ptr)[1];
                        const v4s in2 = ((const v4s *)in_ptr)[2];
                        const v4s in3 = ((const v4s *)in_ptr)[3];
                        const v4s in4 = ((const v4s *)in_ptr)[4];
                        const v4s in5 = ((const v4s *)in_ptr)[5];
                        const v4s in6 = ((const v4s *)in_ptr)[6];
                        const v4s in7 = ((const v4s *)in_ptr)[7];
                        const int w_off = kx * 8;
                        acc = SumDotpSS(in0, w[w_off+0], acc);
                        acc = SumDotpSS(in1, w[w_off+1], acc);
                        acc = SumDotpSS(in2, w[w_off+2], acc);
                        acc = SumDotpSS(in3, w[w_off+3], acc);
                        acc = SumDotpSS(in4, w[w_off+4], acc);
                        acc = SumDotpSS(in5, w[w_off+5], acc);
                        acc = SumDotpSS(in6, w[w_off+6], acc);
                        acc = SumDotpSS(in7, w[w_off+7], acc);
                    }
                    if (bias) acc += ((int32_t *)bias)[oc];
                    int32_t q = mul_shift_round_nearest_even(acc, requant_mul, requant_shift);
                    q = clip8_hwc(q);
                    out_ptr[oc] = (int8_t)q;
                }
            }
            return;
        }

        // Special fast path for in_ch=16, kernel_w=8: used by eeg_conv3, ppg_conv4
        // col_size = 128 bytes = 32 SIMDs per output channel
        // Process 4 output channels at a time with inner channel loop unrolled
        if (in_ch == 16 && kernel_w == 8) {
            for (int pos = start_pos; pos < end_pos; pos++) {
                const int out_y = pos / out_w;
                const int out_x = pos - out_y * out_w;
                const int in_x_start = out_x * stride_w;

                int8_t *out_ptr = output + (out_y * out_w + out_x) * eff_out_ch_stride + out_ch_offset;
                const int in_row_offset = out_y * in_w * 16;

                // Process 4 output channels at a time
                int oc = 0;
                for (; oc + 3 < out_ch; oc += 4) {
                    const v4s *w0 = (const v4s *)(weights + (oc + 0) * 128);
                    const v4s *w1 = (const v4s *)(weights + (oc + 1) * 128);
                    const v4s *w2 = (const v4s *)(weights + (oc + 2) * 128);
                    const v4s *w3 = (const v4s *)(weights + (oc + 3) * 128);

                    int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

                    // Loop over 8 kernel positions
                    for (int kx = 0; kx < 8; kx++) {
                        const int8_t *in_ptr = input + in_row_offset + (in_x_start + kx) * 16;
                        const v4s in0 = ((const v4s *)in_ptr)[0];
                        const v4s in1 = ((const v4s *)in_ptr)[1];
                        const v4s in2 = ((const v4s *)in_ptr)[2];
                        const v4s in3 = ((const v4s *)in_ptr)[3];

                        const int w_off = kx * 4;
                        acc0 = SumDotpSS(in0, w0[w_off+0], acc0);
                        acc0 = SumDotpSS(in1, w0[w_off+1], acc0);
                        acc0 = SumDotpSS(in2, w0[w_off+2], acc0);
                        acc0 = SumDotpSS(in3, w0[w_off+3], acc0);

                        acc1 = SumDotpSS(in0, w1[w_off+0], acc1);
                        acc1 = SumDotpSS(in1, w1[w_off+1], acc1);
                        acc1 = SumDotpSS(in2, w1[w_off+2], acc1);
                        acc1 = SumDotpSS(in3, w1[w_off+3], acc1);

                        acc2 = SumDotpSS(in0, w2[w_off+0], acc2);
                        acc2 = SumDotpSS(in1, w2[w_off+1], acc2);
                        acc2 = SumDotpSS(in2, w2[w_off+2], acc2);
                        acc2 = SumDotpSS(in3, w2[w_off+3], acc2);

                        acc3 = SumDotpSS(in0, w3[w_off+0], acc3);
                        acc3 = SumDotpSS(in1, w3[w_off+1], acc3);
                        acc3 = SumDotpSS(in2, w3[w_off+2], acc3);
                        acc3 = SumDotpSS(in3, w3[w_off+3], acc3);
                    }

                    if (bias) {
                        acc0 += ((int32_t *)bias)[oc + 0];
                        acc1 += ((int32_t *)bias)[oc + 1];
                        acc2 += ((int32_t *)bias)[oc + 2];
                        acc3 += ((int32_t *)bias)[oc + 3];
                    }

                    int32_t q0 = mul_shift_round_nearest_even(acc0, requant_mul, requant_shift);
                    int32_t q1 = mul_shift_round_nearest_even(acc1, requant_mul, requant_shift);
                    int32_t q2 = mul_shift_round_nearest_even(acc2, requant_mul, requant_shift);
                    int32_t q3 = mul_shift_round_nearest_even(acc3, requant_mul, requant_shift);

                    q0 = clip8_hwc(q0); q1 = clip8_hwc(q1);
                    q2 = clip8_hwc(q2); q3 = clip8_hwc(q3);

                    out_ptr[oc + 0] = (int8_t)q0;
                    out_ptr[oc + 1] = (int8_t)q1;
                    out_ptr[oc + 2] = (int8_t)q2;
                    out_ptr[oc + 3] = (int8_t)q3;
                }
                // Handle remaining output channels
                for (; oc < out_ch; oc++) {
                    const v4s *w = (const v4s *)(weights + oc * 128);
                    int32_t acc = 0;
                    for (int kx = 0; kx < 8; kx++) {
                        const int8_t *in_ptr = input + in_row_offset + (in_x_start + kx) * 16;
                        const v4s in0 = ((const v4s *)in_ptr)[0];
                        const v4s in1 = ((const v4s *)in_ptr)[1];
                        const v4s in2 = ((const v4s *)in_ptr)[2];
                        const v4s in3 = ((const v4s *)in_ptr)[3];
                        const int w_off = kx * 4;
                        acc = SumDotpSS(in0, w[w_off+0], acc);
                        acc = SumDotpSS(in1, w[w_off+1], acc);
                        acc = SumDotpSS(in2, w[w_off+2], acc);
                        acc = SumDotpSS(in3, w[w_off+3], acc);
                    }
                    if (bias) acc += ((int32_t *)bias)[oc];
                    int32_t q = mul_shift_round_nearest_even(acc, requant_mul, requant_shift);
                    q = clip8_hwc(q);
                    out_ptr[oc] = (int8_t)q;
                }
            }
            return;
        }

        // Special fast path for in_ch=4, kernel_w=4: fully unrolled SIMD
        // This is common for the second conv layer after expanding from 1 channel
        // col_size = 16 bytes = 4 SIMD ops per output channel
        if (in_ch == 4 && kernel_w == 4) {
            for (int pos = start_pos; pos < end_pos; pos++) {
                const int out_y = pos / out_w;
                const int out_x = pos - out_y * out_w;
                const int in_x_start = out_x * stride_w;

                int8_t *out_ptr = output + (out_y * out_w + out_x) * eff_out_ch_stride + out_ch_offset;

                // Load 4 input vectors upfront (one per kernel position, 4 channels each)
                const v4s in0 = *((const v4s *)(input + (out_y * in_w + in_x_start + 0 * stride_w) * 4));
                const v4s in1 = *((const v4s *)(input + (out_y * in_w + in_x_start + 1) * 4));
                const v4s in2 = *((const v4s *)(input + (out_y * in_w + in_x_start + 2) * 4));
                const v4s in3 = *((const v4s *)(input + (out_y * in_w + in_x_start + 3) * 4));

                // Process 4 output channels at a time
                int oc = 0;
                for (; oc + 3 < out_ch; oc += 4) {
                    // Each output channel has 16 weight bytes (4 positions * 4 channels)
                    const v4s *w0 = (const v4s *)(weights + (oc + 0) * 16);
                    const v4s *w1 = (const v4s *)(weights + (oc + 1) * 16);
                    const v4s *w2 = (const v4s *)(weights + (oc + 2) * 16);
                    const v4s *w3 = (const v4s *)(weights + (oc + 3) * 16);

                    // Fully unrolled: 4 SIMDs per output channel
                    int32_t acc0 = SumDotpSS(in0, w0[0], 0);
                    acc0 = SumDotpSS(in1, w0[1], acc0);
                    acc0 = SumDotpSS(in2, w0[2], acc0);
                    acc0 = SumDotpSS(in3, w0[3], acc0);

                    int32_t acc1 = SumDotpSS(in0, w1[0], 0);
                    acc1 = SumDotpSS(in1, w1[1], acc1);
                    acc1 = SumDotpSS(in2, w1[2], acc1);
                    acc1 = SumDotpSS(in3, w1[3], acc1);

                    int32_t acc2 = SumDotpSS(in0, w2[0], 0);
                    acc2 = SumDotpSS(in1, w2[1], acc2);
                    acc2 = SumDotpSS(in2, w2[2], acc2);
                    acc2 = SumDotpSS(in3, w2[3], acc2);

                    int32_t acc3 = SumDotpSS(in0, w3[0], 0);
                    acc3 = SumDotpSS(in1, w3[1], acc3);
                    acc3 = SumDotpSS(in2, w3[2], acc3);
                    acc3 = SumDotpSS(in3, w3[3], acc3);

                    if (bias) {
                        acc0 += ((int32_t *)bias)[oc + 0];
                        acc1 += ((int32_t *)bias)[oc + 1];
                        acc2 += ((int32_t *)bias)[oc + 2];
                        acc3 += ((int32_t *)bias)[oc + 3];
                    }

                    int32_t q0 = mul_shift_round_nearest_even(acc0, requant_mul, requant_shift);
                    int32_t q1 = mul_shift_round_nearest_even(acc1, requant_mul, requant_shift);
                    int32_t q2 = mul_shift_round_nearest_even(acc2, requant_mul, requant_shift);
                    int32_t q3 = mul_shift_round_nearest_even(acc3, requant_mul, requant_shift);

                    q0 = clip8_hwc(q0); q1 = clip8_hwc(q1);
                    q2 = clip8_hwc(q2); q3 = clip8_hwc(q3);

                    out_ptr[oc + 0] = (int8_t)q0;
                    out_ptr[oc + 1] = (int8_t)q1;
                    out_ptr[oc + 2] = (int8_t)q2;
                    out_ptr[oc + 3] = (int8_t)q3;
                }
                // Handle remaining output channels
                for (; oc < out_ch; oc++) {
                    const v4s *w = (const v4s *)(weights + oc * 16);
                    int32_t acc = SumDotpSS(in0, w[0], 0);
                    acc = SumDotpSS(in1, w[1], acc);
                    acc = SumDotpSS(in2, w[2], acc);
                    acc = SumDotpSS(in3, w[3], acc);
                    if (bias) acc += ((int32_t *)bias)[oc];
                    int32_t q = mul_shift_round_nearest_even(acc, requant_mul, requant_shift);
                    q = clip8_hwc(q);
                    out_ptr[oc] = (int8_t)q;
                }
            }
            return;
        }

        // Special fast path for in_ch=4, kernel_w=16: fully unrolled SIMD (eeg_conv2)
        // col_size = 64 bytes = 16 SIMD ops per output channel
        // This path handles the largest MAC layer (2.25M MACs, 31% of network)
        if (in_ch == 4 && kernel_w == 16) {
            for (int pos = start_pos; pos < end_pos; pos++) {
                const int out_y = pos / out_w;
                const int out_x = pos - out_y * out_w;
                const int in_x_start = out_x * stride_w;

                int8_t *out_ptr = output + (out_y * out_w + out_x) * eff_out_ch_stride + out_ch_offset;
                const int in_row_offset = out_y * in_w * 4;

                // Load 16 input vectors (one per kernel position, 4 channels each)
                const v4s in0  = *((const v4s *)(input + in_row_offset + (in_x_start +  0) * 4));
                const v4s in1  = *((const v4s *)(input + in_row_offset + (in_x_start +  1) * 4));
                const v4s in2  = *((const v4s *)(input + in_row_offset + (in_x_start +  2) * 4));
                const v4s in3  = *((const v4s *)(input + in_row_offset + (in_x_start +  3) * 4));
                const v4s in4  = *((const v4s *)(input + in_row_offset + (in_x_start +  4) * 4));
                const v4s in5  = *((const v4s *)(input + in_row_offset + (in_x_start +  5) * 4));
                const v4s in6  = *((const v4s *)(input + in_row_offset + (in_x_start +  6) * 4));
                const v4s in7  = *((const v4s *)(input + in_row_offset + (in_x_start +  7) * 4));
                const v4s in8  = *((const v4s *)(input + in_row_offset + (in_x_start +  8) * 4));
                const v4s in9  = *((const v4s *)(input + in_row_offset + (in_x_start +  9) * 4));
                const v4s in10 = *((const v4s *)(input + in_row_offset + (in_x_start + 10) * 4));
                const v4s in11 = *((const v4s *)(input + in_row_offset + (in_x_start + 11) * 4));
                const v4s in12 = *((const v4s *)(input + in_row_offset + (in_x_start + 12) * 4));
                const v4s in13 = *((const v4s *)(input + in_row_offset + (in_x_start + 13) * 4));
                const v4s in14 = *((const v4s *)(input + in_row_offset + (in_x_start + 14) * 4));
                const v4s in15 = *((const v4s *)(input + in_row_offset + (in_x_start + 15) * 4));

                // Process 2 output channels at a time (16 SIMDs each = 32 SIMDs total)
                int oc = 0;
                for (; oc + 1 < out_ch; oc += 2) {
                    const v4s *w0 = (const v4s *)(weights + (oc + 0) * 64);
                    const v4s *w1 = (const v4s *)(weights + (oc + 1) * 64);

                    int32_t acc0 = SumDotpSS(in0, w0[0], 0);
                    int32_t acc1 = SumDotpSS(in0, w1[0], 0);
                    acc0 = SumDotpSS(in1, w0[1], acc0);   acc1 = SumDotpSS(in1, w1[1], acc1);
                    acc0 = SumDotpSS(in2, w0[2], acc0);   acc1 = SumDotpSS(in2, w1[2], acc1);
                    acc0 = SumDotpSS(in3, w0[3], acc0);   acc1 = SumDotpSS(in3, w1[3], acc1);
                    acc0 = SumDotpSS(in4, w0[4], acc0);   acc1 = SumDotpSS(in4, w1[4], acc1);
                    acc0 = SumDotpSS(in5, w0[5], acc0);   acc1 = SumDotpSS(in5, w1[5], acc1);
                    acc0 = SumDotpSS(in6, w0[6], acc0);   acc1 = SumDotpSS(in6, w1[6], acc1);
                    acc0 = SumDotpSS(in7, w0[7], acc0);   acc1 = SumDotpSS(in7, w1[7], acc1);
                    acc0 = SumDotpSS(in8, w0[8], acc0);   acc1 = SumDotpSS(in8, w1[8], acc1);
                    acc0 = SumDotpSS(in9, w0[9], acc0);   acc1 = SumDotpSS(in9, w1[9], acc1);
                    acc0 = SumDotpSS(in10, w0[10], acc0); acc1 = SumDotpSS(in10, w1[10], acc1);
                    acc0 = SumDotpSS(in11, w0[11], acc0); acc1 = SumDotpSS(in11, w1[11], acc1);
                    acc0 = SumDotpSS(in12, w0[12], acc0); acc1 = SumDotpSS(in12, w1[12], acc1);
                    acc0 = SumDotpSS(in13, w0[13], acc0); acc1 = SumDotpSS(in13, w1[13], acc1);
                    acc0 = SumDotpSS(in14, w0[14], acc0); acc1 = SumDotpSS(in14, w1[14], acc1);
                    acc0 = SumDotpSS(in15, w0[15], acc0); acc1 = SumDotpSS(in15, w1[15], acc1);

                    if (bias) {
                        acc0 += ((int32_t *)bias)[oc + 0];
                        acc1 += ((int32_t *)bias)[oc + 1];
                    }

                    int32_t q0 = mul_shift_round_nearest_even(acc0, requant_mul, requant_shift);
                    int32_t q1 = mul_shift_round_nearest_even(acc1, requant_mul, requant_shift);
                    q0 = clip8_hwc(q0); q1 = clip8_hwc(q1);

                    out_ptr[oc + 0] = (int8_t)q0;
                    out_ptr[oc + 1] = (int8_t)q1;
                }
                // Handle last output channel if odd count
                for (; oc < out_ch; oc++) {
                    const v4s *w = (const v4s *)(weights + oc * 64);
                    int32_t acc = SumDotpSS(in0, w[0], 0);
                    acc = SumDotpSS(in1, w[1], acc);   acc = SumDotpSS(in2, w[2], acc);
                    acc = SumDotpSS(in3, w[3], acc);   acc = SumDotpSS(in4, w[4], acc);
                    acc = SumDotpSS(in5, w[5], acc);   acc = SumDotpSS(in6, w[6], acc);
                    acc = SumDotpSS(in7, w[7], acc);   acc = SumDotpSS(in8, w[8], acc);
                    acc = SumDotpSS(in9, w[9], acc);   acc = SumDotpSS(in10, w[10], acc);
                    acc = SumDotpSS(in11, w[11], acc); acc = SumDotpSS(in12, w[12], acc);
                    acc = SumDotpSS(in13, w[13], acc); acc = SumDotpSS(in14, w[14], acc);
                    acc = SumDotpSS(in15, w[15], acc);
                    if (bias) acc += ((int32_t *)bias)[oc];
                    int32_t q = mul_shift_round_nearest_even(acc, requant_mul, requant_shift);
                    q = clip8_hwc(q);
                    out_ptr[oc] = (int8_t)q;
                }
            }
            return;
        }

        // Special fast path for in_ch=8, kernel_w=4: fully unrolled SIMD
        // col_size = 32 bytes = 8 SIMD ops per output channel
        if (in_ch == 8 && kernel_w == 4) {
            for (int pos = start_pos; pos < end_pos; pos++) {
                const int out_y = pos / out_w;
                const int out_x = pos - out_y * out_w;
                const int in_x_start = out_x * stride_w;

                int8_t *out_ptr = output + (out_y * out_w + out_x) * eff_out_ch_stride + out_ch_offset;

                // Load 8 input vectors (2 per kernel position, 4 channels each = 8 channels)
                const v4s in0a = *((const v4s *)(input + (out_y * in_w + in_x_start + 0) * 8 + 0));
                const v4s in0b = *((const v4s *)(input + (out_y * in_w + in_x_start + 0) * 8 + 4));
                const v4s in1a = *((const v4s *)(input + (out_y * in_w + in_x_start + 1) * 8 + 0));
                const v4s in1b = *((const v4s *)(input + (out_y * in_w + in_x_start + 1) * 8 + 4));
                const v4s in2a = *((const v4s *)(input + (out_y * in_w + in_x_start + 2) * 8 + 0));
                const v4s in2b = *((const v4s *)(input + (out_y * in_w + in_x_start + 2) * 8 + 4));
                const v4s in3a = *((const v4s *)(input + (out_y * in_w + in_x_start + 3) * 8 + 0));
                const v4s in3b = *((const v4s *)(input + (out_y * in_w + in_x_start + 3) * 8 + 4));

                // Process 4 output channels at a time
                int oc = 0;
                for (; oc + 3 < out_ch; oc += 4) {
                    // Each output channel has 32 weight bytes (4 positions * 8 channels)
                    const v4s *w0 = (const v4s *)(weights + (oc + 0) * 32);
                    const v4s *w1 = (const v4s *)(weights + (oc + 1) * 32);
                    const v4s *w2 = (const v4s *)(weights + (oc + 2) * 32);
                    const v4s *w3 = (const v4s *)(weights + (oc + 3) * 32);

                    // Fully unrolled: 8 SIMDs per output channel
                    int32_t acc0 = SumDotpSS(in0a, w0[0], 0);
                    acc0 = SumDotpSS(in0b, w0[1], acc0);
                    acc0 = SumDotpSS(in1a, w0[2], acc0);
                    acc0 = SumDotpSS(in1b, w0[3], acc0);
                    acc0 = SumDotpSS(in2a, w0[4], acc0);
                    acc0 = SumDotpSS(in2b, w0[5], acc0);
                    acc0 = SumDotpSS(in3a, w0[6], acc0);
                    acc0 = SumDotpSS(in3b, w0[7], acc0);

                    int32_t acc1 = SumDotpSS(in0a, w1[0], 0);
                    acc1 = SumDotpSS(in0b, w1[1], acc1);
                    acc1 = SumDotpSS(in1a, w1[2], acc1);
                    acc1 = SumDotpSS(in1b, w1[3], acc1);
                    acc1 = SumDotpSS(in2a, w1[4], acc1);
                    acc1 = SumDotpSS(in2b, w1[5], acc1);
                    acc1 = SumDotpSS(in3a, w1[6], acc1);
                    acc1 = SumDotpSS(in3b, w1[7], acc1);

                    int32_t acc2 = SumDotpSS(in0a, w2[0], 0);
                    acc2 = SumDotpSS(in0b, w2[1], acc2);
                    acc2 = SumDotpSS(in1a, w2[2], acc2);
                    acc2 = SumDotpSS(in1b, w2[3], acc2);
                    acc2 = SumDotpSS(in2a, w2[4], acc2);
                    acc2 = SumDotpSS(in2b, w2[5], acc2);
                    acc2 = SumDotpSS(in3a, w2[6], acc2);
                    acc2 = SumDotpSS(in3b, w2[7], acc2);

                    int32_t acc3 = SumDotpSS(in0a, w3[0], 0);
                    acc3 = SumDotpSS(in0b, w3[1], acc3);
                    acc3 = SumDotpSS(in1a, w3[2], acc3);
                    acc3 = SumDotpSS(in1b, w3[3], acc3);
                    acc3 = SumDotpSS(in2a, w3[4], acc3);
                    acc3 = SumDotpSS(in2b, w3[5], acc3);
                    acc3 = SumDotpSS(in3a, w3[6], acc3);
                    acc3 = SumDotpSS(in3b, w3[7], acc3);

                    if (bias) {
                        acc0 += ((int32_t *)bias)[oc + 0];
                        acc1 += ((int32_t *)bias)[oc + 1];
                        acc2 += ((int32_t *)bias)[oc + 2];
                        acc3 += ((int32_t *)bias)[oc + 3];
                    }

                    int32_t q0 = mul_shift_round_nearest_even(acc0, requant_mul, requant_shift);
                    int32_t q1 = mul_shift_round_nearest_even(acc1, requant_mul, requant_shift);
                    int32_t q2 = mul_shift_round_nearest_even(acc2, requant_mul, requant_shift);
                    int32_t q3 = mul_shift_round_nearest_even(acc3, requant_mul, requant_shift);

                    q0 = clip8_hwc(q0); q1 = clip8_hwc(q1);
                    q2 = clip8_hwc(q2); q3 = clip8_hwc(q3);

                    out_ptr[oc + 0] = (int8_t)q0;
                    out_ptr[oc + 1] = (int8_t)q1;
                    out_ptr[oc + 2] = (int8_t)q2;
                    out_ptr[oc + 3] = (int8_t)q3;
                }
                // Handle remaining output channels
                for (; oc < out_ch; oc++) {
                    const v4s *w = (const v4s *)(weights + oc * 32);
                    int32_t acc = SumDotpSS(in0a, w[0], 0);
                    acc = SumDotpSS(in0b, w[1], acc);
                    acc = SumDotpSS(in1a, w[2], acc);
                    acc = SumDotpSS(in1b, w[3], acc);
                    acc = SumDotpSS(in2a, w[4], acc);
                    acc = SumDotpSS(in2b, w[5], acc);
                    acc = SumDotpSS(in3a, w[6], acc);
                    acc = SumDotpSS(in3b, w[7], acc);
                    if (bias) acc += ((int32_t *)bias)[oc];
                    int32_t q = mul_shift_round_nearest_even(acc, requant_mul, requant_shift);
                    q = clip8_hwc(q);
                    out_ptr[oc] = (int8_t)q;
                }
            }
            return;
        }

        for (int pos = start_pos; pos < end_pos; pos++) {
            const int out_y = pos / out_w;
            const int out_x = pos - out_y * out_w;
            const int in_x_start = out_x * stride_w;

            // Base pointer for this output position
            int8_t *out_ptr = output + (out_y * out_w + out_x) * eff_out_ch_stride + out_ch_offset;

            // Process output channels in groups of 4 for better register usage
            int oc = 0;

#if CONV2D_HWC_OUTCH_UNROLL >= 4
            for (; oc + 3 < out_ch; oc += 4) {
                int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

                const int8_t *w0 = weights + (oc + 0) * w_stride;
                const int8_t *w1 = weights + (oc + 1) * w_stride;
                const int8_t *w2 = weights + (oc + 2) * w_stride;
                const int8_t *w3 = weights + (oc + 3) * w_stride;

                // Iterate over K spatial positions
                for (int kx = 0; kx < kernel_w; kx++) {
                    const int in_x = in_x_start + kx;
                    // HWC: in_ch contiguous values at (out_y, in_x)
                    const int8_t *in_ptr = input + (out_y * in_w + in_x) * in_ch;
                    const int w_offset = kx * in_ch;

                    // SIMD over channels
                    const v4s *pA = (const v4s *)in_ptr;
                    const v4s *pB0 = (const v4s *)(w0 + w_offset);
                    const v4s *pB1 = (const v4s *)(w1 + w_offset);
                    const v4s *pB2 = (const v4s *)(w2 + w_offset);
                    const v4s *pB3 = (const v4s *)(w3 + w_offset);

                    const int ch_simd = in_ch >> 2;
                    for (int j = 0; j < ch_simd; j++) {
                        const v4s a = pA[j];
                        acc0 = SumDotpSS(a, pB0[j], acc0);
                        acc1 = SumDotpSS(a, pB1[j], acc1);
                        acc2 = SumDotpSS(a, pB2[j], acc2);
                        acc3 = SumDotpSS(a, pB3[j], acc3);
                    }

                    // Handle remainder channels
                    const int ch_rem = in_ch & 0x3;
                    if (ch_rem > 0) {
                        const int8_t *pA_rem = in_ptr + (ch_simd << 2);
                        const int8_t *pB0_rem = w0 + w_offset + (ch_simd << 2);
                        const int8_t *pB1_rem = w1 + w_offset + (ch_simd << 2);
                        const int8_t *pB2_rem = w2 + w_offset + (ch_simd << 2);
                        const int8_t *pB3_rem = w3 + w_offset + (ch_simd << 2);
                        for (int j = 0; j < ch_rem; j++) {
                            const int32_t a_val = (int32_t)pA_rem[j];
                            acc0 += a_val * (int32_t)pB0_rem[j];
                            acc1 += a_val * (int32_t)pB1_rem[j];
                            acc2 += a_val * (int32_t)pB2_rem[j];
                            acc3 += a_val * (int32_t)pB3_rem[j];
                        }
                    }
                }

                // Add bias and requantize
                if (bias) {
                    acc0 += ((int32_t *)bias)[oc + 0];
                    acc1 += ((int32_t *)bias)[oc + 1];
                    acc2 += ((int32_t *)bias)[oc + 2];
                    acc3 += ((int32_t *)bias)[oc + 3];
                }

                int32_t q0 = mul_shift_round_nearest_even(acc0, requant_mul, requant_shift);
                int32_t q1 = mul_shift_round_nearest_even(acc1, requant_mul, requant_shift);
                int32_t q2 = mul_shift_round_nearest_even(acc2, requant_mul, requant_shift);
                int32_t q3 = mul_shift_round_nearest_even(acc3, requant_mul, requant_shift);

                if (q0 > 127) q0 = 127; if (q0 < -128) q0 = -128;
                if (q1 > 127) q1 = 127; if (q1 < -128) q1 = -128;
                if (q2 > 127) q2 = 127; if (q2 < -128) q2 = -128;
                if (q3 > 127) q3 = 127; if (q3 < -128) q3 = -128;

                out_ptr[oc + 0] = (int8_t)q0;
                out_ptr[oc + 1] = (int8_t)q1;
                out_ptr[oc + 2] = (int8_t)q2;
                out_ptr[oc + 3] = (int8_t)q3;
            }
#endif

            // Handle remaining output channels
            for (; oc < out_ch; oc++) {
                int32_t acc = 0;
                const int8_t *w_row = weights + oc * w_stride;

                for (int kx = 0; kx < kernel_w; kx++) {
                    const int in_x = in_x_start + kx;
                    const int8_t *in_ptr = input + (out_y * in_w + in_x) * in_ch;
                    const int w_offset = kx * in_ch;

                    const v4s *pA = (const v4s *)in_ptr;
                    const v4s *pB = (const v4s *)(w_row + w_offset);

                    const int ch_simd = in_ch >> 2;
                    for (int j = 0; j < ch_simd; j++) {
                        acc = SumDotpSS(pA[j], pB[j], acc);
                    }

                    const int ch_rem = in_ch & 0x3;
                    if (ch_rem > 0) {
                        const int8_t *pA_rem = in_ptr + (ch_simd << 2);
                        const int8_t *pB_rem = w_row + w_offset + (ch_simd << 2);
                        for (int j = 0; j < ch_rem; j++) {
                            acc += (int32_t)pA_rem[j] * (int32_t)pB_rem[j];
                        }
                    }
                }

                if (bias) acc += ((int32_t *)bias)[oc];

                int32_t q = mul_shift_round_nearest_even(acc, requant_mul, requant_shift);
                if (q > 127) q = 127;
                if (q < -128) q = -128;

                out_ptr[oc] = (int8_t)q;
            }
        }
        return;
    }

    // -------------------------------------------------------------------------
    // Fast path: Kx1 kernel (vertical convolution)
    // -------------------------------------------------------------------------
    if (is_Kx1 && pad_h == 0 && pad_w == 0 && stride_w == 1) {
        for (int pos = start_pos; pos < end_pos; pos++) {
            const int out_y = pos / out_w;
            const int out_x = pos - out_y * out_w;
            const int in_y_start = out_y * stride_h;

            int8_t *out_ptr = output + (out_y * out_w + out_x) * eff_out_ch_stride + out_ch_offset;

            int oc = 0;

#if CONV2D_HWC_OUTCH_UNROLL >= 4
            for (; oc + 3 < out_ch; oc += 4) {
                int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

                const int8_t *w0 = weights + (oc + 0) * w_stride;
                const int8_t *w1 = weights + (oc + 1) * w_stride;
                const int8_t *w2 = weights + (oc + 2) * w_stride;
                const int8_t *w3 = weights + (oc + 3) * w_stride;

                for (int ky = 0; ky < kernel_h; ky++) {
                    const int in_y = in_y_start + ky;
                    const int8_t *in_ptr = input + (in_y * in_w + out_x) * in_ch;
                    const int w_offset = ky * in_ch;

                    const v4s *pA = (const v4s *)in_ptr;
                    const v4s *pB0 = (const v4s *)(w0 + w_offset);
                    const v4s *pB1 = (const v4s *)(w1 + w_offset);
                    const v4s *pB2 = (const v4s *)(w2 + w_offset);
                    const v4s *pB3 = (const v4s *)(w3 + w_offset);

                    const int ch_simd = in_ch >> 2;
                    for (int j = 0; j < ch_simd; j++) {
                        const v4s a = pA[j];
                        acc0 = SumDotpSS(a, pB0[j], acc0);
                        acc1 = SumDotpSS(a, pB1[j], acc1);
                        acc2 = SumDotpSS(a, pB2[j], acc2);
                        acc3 = SumDotpSS(a, pB3[j], acc3);
                    }

                    const int ch_rem = in_ch & 0x3;
                    if (ch_rem > 0) {
                        const int8_t *pA_rem = in_ptr + (ch_simd << 2);
                        const int8_t *pB0_rem = w0 + w_offset + (ch_simd << 2);
                        const int8_t *pB1_rem = w1 + w_offset + (ch_simd << 2);
                        const int8_t *pB2_rem = w2 + w_offset + (ch_simd << 2);
                        const int8_t *pB3_rem = w3 + w_offset + (ch_simd << 2);
                        for (int j = 0; j < ch_rem; j++) {
                            const int32_t a_val = (int32_t)pA_rem[j];
                            acc0 += a_val * (int32_t)pB0_rem[j];
                            acc1 += a_val * (int32_t)pB1_rem[j];
                            acc2 += a_val * (int32_t)pB2_rem[j];
                            acc3 += a_val * (int32_t)pB3_rem[j];
                        }
                    }
                }

                if (bias) {
                    acc0 += ((int32_t *)bias)[oc + 0];
                    acc1 += ((int32_t *)bias)[oc + 1];
                    acc2 += ((int32_t *)bias)[oc + 2];
                    acc3 += ((int32_t *)bias)[oc + 3];
                }

                int32_t q0 = mul_shift_round_nearest_even(acc0, requant_mul, requant_shift);
                int32_t q1 = mul_shift_round_nearest_even(acc1, requant_mul, requant_shift);
                int32_t q2 = mul_shift_round_nearest_even(acc2, requant_mul, requant_shift);
                int32_t q3 = mul_shift_round_nearest_even(acc3, requant_mul, requant_shift);

                if (q0 > 127) q0 = 127; if (q0 < -128) q0 = -128;
                if (q1 > 127) q1 = 127; if (q1 < -128) q1 = -128;
                if (q2 > 127) q2 = 127; if (q2 < -128) q2 = -128;
                if (q3 > 127) q3 = 127; if (q3 < -128) q3 = -128;

                out_ptr[oc + 0] = (int8_t)q0;
                out_ptr[oc + 1] = (int8_t)q1;
                out_ptr[oc + 2] = (int8_t)q2;
                out_ptr[oc + 3] = (int8_t)q3;
            }
#endif

            for (; oc < out_ch; oc++) {
                int32_t acc = 0;
                const int8_t *w_row = weights + oc * w_stride;

                for (int ky = 0; ky < kernel_h; ky++) {
                    const int in_y = in_y_start + ky;
                    const int8_t *in_ptr = input + (in_y * in_w + out_x) * in_ch;
                    const int w_offset = ky * in_ch;

                    const v4s *pA = (const v4s *)in_ptr;
                    const v4s *pB = (const v4s *)(w_row + w_offset);

                    const int ch_simd = in_ch >> 2;
                    for (int j = 0; j < ch_simd; j++) {
                        acc = SumDotpSS(pA[j], pB[j], acc);
                    }

                    const int ch_rem = in_ch & 0x3;
                    if (ch_rem > 0) {
                        const int8_t *pA_rem = in_ptr + (ch_simd << 2);
                        const int8_t *pB_rem = w_row + w_offset + (ch_simd << 2);
                        for (int j = 0; j < ch_rem; j++) {
                            acc += (int32_t)pA_rem[j] * (int32_t)pB_rem[j];
                        }
                    }
                }

                if (bias) acc += ((int32_t *)bias)[oc];

                int32_t q = mul_shift_round_nearest_even(acc, requant_mul, requant_shift);
                if (q > 127) q = 127;
                if (q < -128) q = -128;

                out_ptr[oc] = (int8_t)q;
            }
        }
        return;
    }

    // -------------------------------------------------------------------------
    // Fast path: 3x3 kernel with stride=1 (54% of conv layers in typical CNNs)
    // -------------------------------------------------------------------------
    // Weight layout for HWC: [out_ch, kernel_h * kernel_w * in_ch]
    // For 3x3: weights[oc, (ky*3 + kx) * in_ch + c]
    // Optimized for in_ch divisible by 4 (SIMD processing of channels)
    if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1) {
        const int ch_simd = in_ch >> 2;
        const int ch_rem = in_ch & 0x3;
        const int in_w_ch = in_w * in_ch;  // Row stride in bytes

        for (int pos = start_pos; pos < end_pos; pos++) {
            const int out_y = pos / out_w;
            const int out_x = pos - out_y * out_w;

            // Calculate input coordinates for the 3x3 patch center
            const int in_y0 = out_y - pad_h;  // Top-left input y
            const int in_x0 = out_x - pad_w;  // Top-left input x

            int8_t *out_ptr = output + (out_y * out_w + out_x) * eff_out_ch_stride + out_ch_offset;

            // Fast interior path: no bounds checking needed
            if (in_y0 >= 0 && in_y0 + 2 < in_h && in_x0 >= 0 && in_x0 + 2 < in_w) {
                // Process 2 output channels at a time (better register usage)
                int oc = 0;
                for (; oc + 1 < out_ch; oc += 2) {
                    const int8_t *w0 = weights + (oc + 0) * w_stride;
                    const int8_t *w1 = weights + (oc + 1) * w_stride;
                    int32_t acc0 = 0, acc1 = 0;

                    // Unrolled 3x3 loop: 9 positions
                    for (int ky = 0; ky < 3; ky++) {
                        const int8_t *row_in = input + (in_y0 + ky) * in_w_ch + in_x0 * in_ch;
                        const int w_ky_off = ky * 3 * in_ch;

                        for (int kx = 0; kx < 3; kx++) {
                            const int8_t *in_ptr = row_in + kx * in_ch;
                            const int w_off = w_ky_off + kx * in_ch;
                            const v4s *pA = (const v4s *)in_ptr;
                            const v4s *pB0 = (const v4s *)(w0 + w_off);
                            const v4s *pB1 = (const v4s *)(w1 + w_off);

                            for (int j = 0; j < ch_simd; j++) {
                                const v4s a = pA[j];
                                acc0 = SumDotpSS(a, pB0[j], acc0);
                                acc1 = SumDotpSS(a, pB1[j], acc1);
                            }
                            if (ch_rem > 0) {
                                const int8_t *pA_rem = in_ptr + (ch_simd << 2);
                                const int8_t *pB0_rem = w0 + w_off + (ch_simd << 2);
                                const int8_t *pB1_rem = w1 + w_off + (ch_simd << 2);
                                for (int j = 0; j < ch_rem; j++) {
                                    const int32_t a_val = (int32_t)pA_rem[j];
                                    acc0 += a_val * (int32_t)pB0_rem[j];
                                    acc1 += a_val * (int32_t)pB1_rem[j];
                                }
                            }
                        }
                    }

                    if (bias) {
                        acc0 += ((int32_t *)bias)[oc + 0];
                        acc1 += ((int32_t *)bias)[oc + 1];
                    }

                    int32_t q0 = mul_shift_round_nearest_even(acc0, requant_mul, requant_shift);
                    int32_t q1 = mul_shift_round_nearest_even(acc1, requant_mul, requant_shift);
                    q0 = clip8_hwc(q0); q1 = clip8_hwc(q1);
                    out_ptr[oc + 0] = (int8_t)q0;
                    out_ptr[oc + 1] = (int8_t)q1;
                }

                // Handle remaining output channel if odd count
                for (; oc < out_ch; oc++) {
                    const int8_t *w = weights + oc * w_stride;
                    int32_t acc = 0;

                    for (int ky = 0; ky < 3; ky++) {
                        const int8_t *row_in = input + (in_y0 + ky) * in_w_ch + in_x0 * in_ch;
                        const int w_ky_off = ky * 3 * in_ch;

                        for (int kx = 0; kx < 3; kx++) {
                            const int8_t *in_ptr = row_in + kx * in_ch;
                            const int w_off = w_ky_off + kx * in_ch;
                            const v4s *pA = (const v4s *)in_ptr;
                            const v4s *pB = (const v4s *)(w + w_off);

                            for (int j = 0; j < ch_simd; j++) {
                                acc = SumDotpSS(pA[j], pB[j], acc);
                            }
                            if (ch_rem > 0) {
                                const int8_t *pA_rem = in_ptr + (ch_simd << 2);
                                const int8_t *pB_rem = w + w_off + (ch_simd << 2);
                                for (int j = 0; j < ch_rem; j++) {
                                    acc += (int32_t)pA_rem[j] * (int32_t)pB_rem[j];
                                }
                            }
                        }
                    }

                    if (bias) acc += ((int32_t *)bias)[oc];
                    int32_t q = mul_shift_round_nearest_even(acc, requant_mul, requant_shift);
                    q = clip8_hwc(q);
                    out_ptr[oc] = (int8_t)q;
                }
            } else {
                // Border path: with bounds checking
                for (int oc = 0; oc < out_ch; oc++) {
                    int32_t acc = 0;
                    const int8_t *w_row = weights + oc * w_stride;

                    for (int ky = 0; ky < 3; ky++) {
                        for (int kx = 0; kx < 3; kx++) {
                            const int in_y = in_y0 + ky;
                            const int in_x = in_x0 + kx;

                            if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                                const int8_t *in_ptr = input + (in_y * in_w + in_x) * in_ch;
                                const int w_off = (ky * 3 + kx) * in_ch;
                                const v4s *pA = (const v4s *)in_ptr;
                                const v4s *pB = (const v4s *)(w_row + w_off);

                                for (int j = 0; j < ch_simd; j++) {
                                    acc = SumDotpSS(pA[j], pB[j], acc);
                                }
                                if (ch_rem > 0) {
                                    const int8_t *pA_rem = in_ptr + (ch_simd << 2);
                                    const int8_t *pB_rem = w_row + w_off + (ch_simd << 2);
                                    for (int j = 0; j < ch_rem; j++) {
                                        acc += (int32_t)pA_rem[j] * (int32_t)pB_rem[j];
                                    }
                                }
                            }
                        }
                    }

                    if (bias) acc += ((int32_t *)bias)[oc];
                    int32_t q = mul_shift_round_nearest_even(acc, requant_mul, requant_shift);
                    q = clip8_hwc(q);
                    out_ptr[oc] = (int8_t)q;
                }
            }
        }
        return;
    }

    // -------------------------------------------------------------------------
    // General path: handles padding and non-unit strides
    // -------------------------------------------------------------------------
    for (int pos = start_pos; pos < end_pos; pos++) {
        const int out_y = pos / out_w;
        const int out_x = pos - out_y * out_w;

        int8_t *out_ptr = output + (out_y * out_w + out_x) * eff_out_ch_stride + out_ch_offset;

        for (int oc = 0; oc < out_ch; oc++) {
            int32_t acc = 0;
            const int8_t *w_row = weights + oc * w_stride;

            for (int ky = 0; ky < kernel_h; ky++) {
                for (int kx = 0; kx < kernel_w; kx++) {
                    const int in_y = out_y * stride_h + ky - pad_h;
                    const int in_x = out_x * stride_w + kx - pad_w;

                    if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                        // HWC input access
                        const int8_t *in_ptr = input + (in_y * in_w + in_x) * in_ch;
                        const int w_offset = (ky * kernel_w + kx) * in_ch;

                        const v4s *pA = (const v4s *)in_ptr;
                        const v4s *pB = (const v4s *)(w_row + w_offset);

                        const int ch_simd = in_ch >> 2;
                        for (int j = 0; j < ch_simd; j++) {
                            acc = SumDotpSS(pA[j], pB[j], acc);
                        }

                        const int ch_rem = in_ch & 0x3;
                        if (ch_rem > 0) {
                            const int8_t *pA_rem = in_ptr + (ch_simd << 2);
                            const int8_t *pB_rem = w_row + w_offset + (ch_simd << 2);
                            for (int j = 0; j < ch_rem; j++) {
                                acc += (int32_t)pA_rem[j] * (int32_t)pB_rem[j];
                            }
                        }
                    }
                }
            }

            if (bias) acc += ((int32_t *)bias)[oc];

            int32_t q = mul_shift_round_nearest_even(acc, requant_mul, requant_shift);
            if (q > 127) q = 127;
            if (q < -128) q = -128;

            out_ptr[oc] = (int8_t)q;
        }
    }
}

// ---
// DEPTHWISE CONV2D KERNEL (groups = channels)
// ---
// For depthwise convolution, each output channel only convolves with one input channel.
// This is much simpler than standard conv: no cross-channel accumulation.
// Input: [H, W, C] in HWC layout
// Weights: [C, kernel_h * kernel_w] (each channel has its own kernel)
// Output: [H_out, W_out, C] in HWC layout
//
// Ko-tiling support (weight tiling over output channels):
// - channels: number of channels to process in this tile
// - total_channels: total channels in input/output (for stride calculation)
// - ch_offset: starting channel offset for this tile

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
    uint16_t ch_offset)           // Starting channel offset for Ko-tiling (0 = no offset)
{
    const float combined_scale = (scale_input * scale_weight) / scale_output;

    const int kernel_size = kernel_h * kernel_w;
    const int core_id = pi_core_id();
    const int num_cores = NUM_CORES;

    // For Ko-tiling: use total_channels for stride, channels for loop count
    const int ch_stride = (total_channels > 0) ? total_channels : channels;

    // Parallelize over output pixels (each pixel processes tile channels)
    const int total_pixels = out_h * out_w;
    const int pixels_per_core = (total_pixels + num_cores - 1) / num_cores;
    const int pixel_start = core_id * pixels_per_core;
    const int pixel_end = (pixel_start + pixels_per_core > total_pixels) ? total_pixels : pixel_start + pixels_per_core;

    for (int pixel = pixel_start; pixel < pixel_end; pixel++) {
        const int out_y = pixel / out_w;
        const int out_x = pixel % out_w;

        // Output pointer at this pixel, offset to starting channel
        int8_t *out_ptr = output + pixel * ch_stride + ch_offset;

        // Process tile channels for this output pixel
        for (int c = 0; c < channels; c++) {
            int32_t acc = 0;

            // Convolve with kernel (depthwise: each channel uses its own kernel)
            // weights are already offset to current tile, so use c directly
            const int8_t *w_ptr = weights + c * kernel_size;

            for (int ky = 0; ky < kernel_h; ky++) {
                for (int kx = 0; kx < kernel_w; kx++) {
                    const int in_y = out_y * stride_h + ky - pad_h;
                    const int in_x = out_x * stride_w + kx - pad_w;

                    if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                        // HWC input: access channel (ch_offset + c) at position (in_y, in_x)
                        const int8_t in_val = input[(in_y * in_w + in_x) * ch_stride + ch_offset + c];
                        const int8_t w_val = w_ptr[ky * kernel_w + kx];
                        acc += (int32_t)in_val * (int32_t)w_val;
                    }
                }
            }

            // Add bias (bias already offset to current tile)
            if (bias) {
                acc += ((int32_t *)bias)[c];
            }

            // Requantize using floating-point rescale (like other kernels)
            float val_fp = (float)acc * combined_scale;
            int32_t q = (int32_t)roundf(val_fp);
            if (q > 127) q = 127;
            if (q < -128) q = -128;

            out_ptr[c] = (int8_t)q;
        }
    }

    pi_cl_team_barrier();
}

