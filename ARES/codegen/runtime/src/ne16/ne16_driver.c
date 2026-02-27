/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * NE16 Neural Accelerator Driver Implementation
 *
 * Low-level register interface for the GAP9 NE16 accelerator.
 * Provides 1x1 and 3x3 convolution operations with INT8 inputs/outputs.
 *
 * Reference: GAP SDK NE16 documentation and pulp-nnx library.
 */

#include "ne16/ne16_driver.h"

#ifdef ARES_USE_NE16

#include "pmsis.h"
/* Event-unit wait API differs between GAP SDK and PULP-SDK. */
#if defined(__has_include)
#  if __has_include("pmsis/implem/hal/event_unit/event_unit.h")
#    include "pmsis/implem/hal/event_unit/event_unit.h"
#  elif __has_include("hal/eu/eu_v3.h")
#    include "hal/eu/eu_v3.h"
#    ifndef hal_eu_evt_mask_wait_and_clr
#      define hal_eu_evt_mask_wait_and_clr(mask) ((void)eu_evt_maskWaitAndClr(mask))
#    endif
#  else
#    error "No compatible event-unit HAL header found for NE16 driver"
#  endif
#else
#  include "pmsis/implem/hal/event_unit/event_unit.h"
#endif

/* --- NE16 Register Definitions --- */

#define NE16_ADDR_BASE 0x00201000u
#define CLUS_CTRL_ADDR_BASE 0x00200000u

/* Cluster controller HWPE config register */
#define CLUS_CTRL_HWPE_OFFS 0x18u
#define CLUS_CTRL_HWPE_CG_EN_MASK 0x800u
#define CLUS_CTRL_HWPE_HCI_PRIO_MASK 0x100u
#define CLUS_CTRL_HWPE_HCI_MAXSTALL_MASK 0xffu

/* NE16 Commands */
#define NE16_CMD_TRIGGER 0x00u
#define NE16_CMD_ACQUIRE 0x04u
#define NE16_CMD_STATUS 0x0cu
#define NE16_CMD_SOFT_CLEAR 0x14u

/* Register file base offset (after commands) */
#define NE16_REGISTER_OFFS 0x20u

/* Register offsets (relative to NE16_REGISTER_OFFS) */
#define NE16_REG_WEIGHTS_PTR 0x00u
#define NE16_REG_INFEAT_PTR 0x04u
#define NE16_REG_OUTFEAT_PTR 0x08u
#define NE16_REG_SCALE_PTR 0x0cu
#define NE16_REG_SCALE_SHIFT_PTR 0x10u
#define NE16_REG_SCALE_BIAS_PTR 0x14u

#define NE16_REG_INFEAT_D0_STRIDE 0x18u
#define NE16_REG_INFEAT_D1_STRIDE 0x1cu
#define NE16_REG_INFEAT_D2_STRIDE 0x20u
#define NE16_REG_OUTFEAT_D0_STRIDE 0x24u
#define NE16_REG_OUTFEAT_D1_STRIDE 0x28u
#define NE16_REG_OUTFEAT_D2_STRIDE 0x2cu
#define NE16_REG_WEIGHTS_D0_STRIDE 0x30u
#define NE16_REG_WEIGHTS_D1_STRIDE 0x34u
#define NE16_REG_WEIGHTS_D2_STRIDE 0x38u

#define NE16_REG_REM_KO_KI 0x3cu
#define NE16_SHIFT_REM_KI 0
#define NE16_MASK_REM_KI 0xffff
#define NE16_SHIFT_REM_KO 16
#define NE16_MASK_REM_KO 0xffff

#define NE16_REG_REM_HO_WO 0x40u
#define NE16_SHIFT_REM_WO 0
#define NE16_MASK_REM_WO 0xffff
#define NE16_SHIFT_REM_HO 16
#define NE16_MASK_REM_HO 0xffff

#define NE16_REG_REM_HI_WI 0x44u
#define NE16_SHIFT_REM_WI 0
#define NE16_MASK_REM_WI 0xffff
#define NE16_SHIFT_REM_HI 16
#define NE16_MASK_REM_HI 0xffff

#define NE16_REG_NB_KO_KI 0x48u
#define NE16_SHIFT_NB_KI 0
#define NE16_MASK_NB_KI 0xffff
#define NE16_SHIFT_NB_KO 16
#define NE16_MASK_NB_KO 0xffff

#define NE16_REG_NB_HO_WO 0x4cu
#define NE16_SHIFT_NB_WO 0
#define NE16_MASK_NB_WO 0xffff
#define NE16_SHIFT_NB_HO 16
#define NE16_MASK_NB_HO 0xffff

#define NE16_REG_PADDING 0x50u
#define NE16_REG_WEIGHT_OFFSET 0x54u
#define NE16_REG_FILTER_MASK 0x58u

#define NE16_REG_CONFIG 0x5cu
#define NE16_SHIFT_WBITS_M1 0
#define NE16_MASK_WBITS_M1 0x7
#define NE16_SHIFT_MODE16 3
#define NE16_MASK_MODE16 0x1
#define NE16_SHIFT_OUTQUANT 4
#define NE16_MASK_OUTQUANT 0x1
#define NE16_SHIFT_FILTER_MODE 5
#define NE16_MASK_FILTER_MODE 0x3
#define NE16_SHIFT_LINEAR_MODE 7
#define NE16_MASK_LINEAR_MODE 0x1
#define NE16_SHIFT_STRIDED_MODE 8
#define NE16_MASK_STRIDED_MODE 0x1
#define NE16_SHIFT_NORM_BITS 12
#define NE16_MASK_NORM_BITS 0x3
#define NE16_SHIFT_STREAMIN 14
#define NE16_MASK_STREAMIN 0x1
#define NE16_SHIFT_WEIGHT_OFFSET_CFG 15
#define NE16_MASK_WEIGHT_OFFSET_CFG 0x1
#define NE16_SHIFT_QUANT_RIGHT_SHIFT 16
#define NE16_MASK_QUANT_RIGHT_SHIFT 0xf
#define NE16_SHIFT_QUANT_BITS 21
#define NE16_MASK_QUANT_BITS 0x3
#define NE16_SHIFT_QUANT_NORECT 23
#define NE16_MASK_QUANT_NORECT 0x1
#define NE16_SHIFT_NORM_SHIFT 24
#define NE16_MASK_NORM_SHIFT 0x1
#define NE16_SHIFT_NORM_BIAS 25
#define NE16_MASK_NORM_BIAS 0x1

/* NE16 hardware constants */
#define NE16_OUTPUT_BANDWIDTH_BYTES 32  /* Output burst width in bytes */
#define NE16_INPUT_CHANNEL_SUBTILE 16   /* Ki subtile size */
#define NE16_OUTPUT_CHANNEL_SUBTILE 32  /* Ko subtile size */
#define NE16_WEIGHT_D0_STRIDE 2         /* Weight d0 stride base */
#define NE16_WEIGHT_BITS 8              /* Weight bit width */

/* --- Helper Functions --- */

static inline void ne16_write_cmd(uint32_t offs, uint32_t value)
{
    *(volatile uint32_t *)(NE16_ADDR_BASE + offs) = value;
}

static inline int32_t ne16_read_cmd(uint32_t offs)
{
    return *(volatile int32_t *)(NE16_ADDR_BASE + offs);
}

static inline void ne16_write_reg(uint32_t offs, uint32_t value)
{
    *(volatile uint32_t *)(NE16_ADDR_BASE + NE16_REGISTER_OFFS + offs) = value;
}

void ne16_soft_clear_all(void)
{
    /* Write soft clear command multiple times to ensure it takes effect */
    ne16_write_cmd(NE16_CMD_SOFT_CLEAR, 0);
    ne16_write_cmd(NE16_CMD_SOFT_CLEAR, 0);

    /* Longer delay to let reset take effect */
    for (volatile int i = 0; i < 100; i++) {
        __asm__ __volatile__("nop");
    }

    /* Full memory barrier to ensure all operations complete */
    asm volatile("fence iorw, iorw" ::: "memory");
}

static inline void ne16_enable_clock_and_priority(void)
{
    volatile uint32_t *hwpe = (volatile uint32_t *)(CLUS_CTRL_ADDR_BASE + CLUS_CTRL_HWPE_OFFS);
    /* Enable NE16 clock gate */
    *hwpe |= CLUS_CTRL_HWPE_CG_EN_MASK;
    /* Give priority to NE16 over cores/DMA */
    *hwpe |= CLUS_CTRL_HWPE_HCI_PRIO_MASK;
    /* Reset maxstall then set to 8 (same as SDK defaults) */
    *hwpe &= ~CLUS_CTRL_HWPE_HCI_MAXSTALL_MASK;
    *hwpe |= (8u & CLUS_CTRL_HWPE_HCI_MAXSTALL_MASK);
}

static inline int ne16_acquire_job(void)
{
    int job_id = ne16_read_cmd(NE16_CMD_ACQUIRE);
    while (job_id < 0) {
        job_id = ne16_read_cmd(NE16_CMD_ACQUIRE);
    }
    return job_id;
}

static inline void ne16_wait_job_done(int job_id)
{
    (void)job_id;

    /* Use event-based waiting to avoid polling deadlock on short jobs.
     * Event IDs: NE16_EVT0=12, NE16_EVT1=13 (see GAP SDK hal_ne16.h). */
    const uint32_t evt_mask = 1u << 12;

    /* Wait for at least one NE16 event, then ensure all jobs are done. */
    hal_eu_evt_mask_wait_and_clr(evt_mask);
    while ((uint32_t)ne16_read_cmd(NE16_CMD_STATUS) != 0u) {
        hal_eu_evt_mask_wait_and_clr(evt_mask);
    }
}

/* Public wrapper for waiting - exposed in header */
void ne16_wait_job(int job_id)
{
    ne16_wait_job_done(job_id);
}

static inline int ceil_div(int a, int b)
{
    return (a + b - 1) / b;
}

/* --- Public API --- */

void ne16_init(void)
{
    static int s_inited = 0;
    if (s_inited) return;  /* Match tinymyo: only init once */
    ne16_enable_clock_and_priority();
    ne16_soft_clear_all();
    s_inited = 1;
}

void ne16_conv1x1_u8_u8_to_s32(
    const uint8_t *infeat,
    const uint8_t *weights_packed,
    int32_t *outfeat,
    int in_w,
    int in_h,
    int in_feat,
    int out_feat,
    int8_t weight_offset
) {

    const int job_id = ne16_acquire_job();

    /* Stride setup (token-major HWC mapping) */
    const int q_w_bits = 8;
    const int tp_in = 16;
    const int nb_ki = ceil_div(in_feat, tp_in);

    /* InFeat strides (HWC) */
    const uint32_t in_d0 = (uint32_t)in_feat;
    const uint32_t in_d1 = (uint32_t)in_feat * (uint32_t)in_w;
    const uint32_t in_d2 = 0;

    /* OutFeat strides (HWC, 32-bit streamout)
     *
     * d0 = NE16 output bandwidth (fixed at 32, burst write size)
     * d1 = stride between consecutive W positions = out_channels * bytes_per_element
     * d2 = stride between H rows
     *
     * For 32-bit mode with 64 output channels: d1 = 64 * 4 = 256
     * This matches pulp-nnx: w_out_stride = OUTPUT_CHANNEL * OUTPUT_BITS / 8
     */
    const uint32_t out_bytes = 4;
    const uint32_t out_d0 = NE16_OUTPUT_BANDWIDTH_BYTES;  /* 32, fixed */
    const uint32_t out_d1 = out_bytes * (uint32_t)out_feat;
    const uint32_t out_d2 = out_bytes * (uint32_t)out_feat * (uint32_t)in_w;

    /* Weights strides for packed 1x1 layout
     *
     * w_d0 = stride between bit planes within a Ki subtile (2 * weight_bits = 16 for 8-bit)
     * w_d1 = size of packed weights per output channel = nb_ki * 16
     * w_d2 = unused for 1x1 mode (matches tinymyo driver which also uses 0)
     */
    const uint32_t w_d0 = 2u * (uint32_t)q_w_bits;
    const uint32_t w_d1 = (uint32_t)nb_ki * 16u * (uint32_t)q_w_bits / 8u;
    const uint32_t w_d2 = 0;  /* Matches tinymyo driver */

    /* Debug: verify NE16 register values after programming */
#if defined(ARES_NE16_DEBUG) && !defined(MINIMAL_OUTPUT)
    printf("CL: NE16 DEBUG STRIDES: w_d0=%u w_d1=%u w_d2=%u expected_d1=%u\n",
           w_d0, w_d1, w_d2, nb_ki * 16);
#endif

    /* NB / REM fields */
    const int rem_ki = (in_feat % 16) ? (in_feat % 16) : 16;
    const int nb_ko = ceil_div(out_feat, 32);
    const int rem_ko = (out_feat % 32) ? (out_feat % 32) : 32;

    const int rem_wo = in_w % 3;
    const int nb_wo = in_w / 3 + (rem_wo ? 1 : 0);
    const int rem_ho = in_h % 3;
    const int nb_ho = in_h / 3 + (rem_ho ? 1 : 0);

    const int rem_wi = rem_wo;
    const int rem_hi = rem_ho;

    /* Program pointers */
    ne16_write_reg(NE16_REG_WEIGHTS_PTR, (uint32_t)weights_packed);
    ne16_write_reg(NE16_REG_INFEAT_PTR, (uint32_t)infeat);
    ne16_write_reg(NE16_REG_OUTFEAT_PTR, (uint32_t)outfeat);
    ne16_write_reg(NE16_REG_SCALE_PTR, 0);
    ne16_write_reg(NE16_REG_SCALE_SHIFT_PTR, 0);
    ne16_write_reg(NE16_REG_SCALE_BIAS_PTR, 0);

    /* Program strides */
    ne16_write_reg(NE16_REG_INFEAT_D0_STRIDE, in_d0);
    ne16_write_reg(NE16_REG_INFEAT_D1_STRIDE, in_d1);
    ne16_write_reg(NE16_REG_INFEAT_D2_STRIDE, in_d2);
    ne16_write_reg(NE16_REG_OUTFEAT_D0_STRIDE, out_d0);
    ne16_write_reg(NE16_REG_OUTFEAT_D1_STRIDE, out_d1);
    ne16_write_reg(NE16_REG_OUTFEAT_D2_STRIDE, out_d2);
    ne16_write_reg(NE16_REG_WEIGHTS_D0_STRIDE, w_d0);
    ne16_write_reg(NE16_REG_WEIGHTS_D1_STRIDE, w_d1);
    ne16_write_reg(NE16_REG_WEIGHTS_D2_STRIDE, w_d2);

    /* Reminders */
    ne16_write_reg(NE16_REG_REM_KO_KI,
                   ((uint32_t)(rem_ki & NE16_MASK_REM_KI) << NE16_SHIFT_REM_KI) |
                       ((uint32_t)(rem_ko & NE16_MASK_REM_KO) << NE16_SHIFT_REM_KO));
    ne16_write_reg(NE16_REG_REM_HO_WO,
                   ((uint32_t)(rem_wo & NE16_MASK_REM_WO) << NE16_SHIFT_REM_WO) |
                       ((uint32_t)(rem_ho & NE16_MASK_REM_HO) << NE16_SHIFT_REM_HO));
    ne16_write_reg(NE16_REG_REM_HI_WI,
                   ((uint32_t)(rem_wi & NE16_MASK_REM_WI) << NE16_SHIFT_REM_WI) |
                       ((uint32_t)(rem_hi & NE16_MASK_REM_HI) << NE16_SHIFT_REM_HI));

    /* Dimensions */
    ne16_write_reg(NE16_REG_NB_KO_KI,
                   ((uint32_t)(nb_ki & NE16_MASK_NB_KI) << NE16_SHIFT_NB_KI) |
                       ((uint32_t)(nb_ko & NE16_MASK_NB_KO) << NE16_SHIFT_NB_KO));
    ne16_write_reg(NE16_REG_NB_HO_WO,
                   ((uint32_t)(nb_wo & NE16_MASK_NB_WO) << NE16_SHIFT_NB_WO) |
                       ((uint32_t)(nb_ho & NE16_MASK_NB_HO) << NE16_SHIFT_NB_HO));

    /* Padding and filter mask: unused for 1x1 */
    ne16_write_reg(NE16_REG_PADDING, 0);
    ne16_write_reg(NE16_REG_FILTER_MASK, 0);

    /* Weight offset (signed 8-bit encoded in 32-bit register write) */
    ne16_write_reg(NE16_REG_WEIGHT_OFFSET, (uint32_t)(int32_t)weight_offset);

    /* Config: 1x1 mode, 8-bit weights, 32-bit streamout, no quantization */
    uint32_t cfg = 0;
    cfg |= ((uint32_t)(7 & NE16_MASK_WBITS_M1) << NE16_SHIFT_WBITS_M1);               /* Qw = 8 */
    cfg |= ((uint32_t)(0 & NE16_MASK_MODE16) << NE16_SHIFT_MODE16);                   /* mode16 = 0 */
    cfg |= ((uint32_t)(0 & NE16_MASK_OUTQUANT) << NE16_SHIFT_OUTQUANT);               /* streamout only */
    cfg |= ((uint32_t)(2 & NE16_MASK_FILTER_MODE) << NE16_SHIFT_FILTER_MODE);         /* 1x1 mode */
    cfg |= ((uint32_t)(0 & NE16_MASK_LINEAR_MODE) << NE16_SHIFT_LINEAR_MODE);         /* not linear mode */
    cfg |= ((uint32_t)(0 & NE16_MASK_STRIDED_MODE) << NE16_SHIFT_STRIDED_MODE);       /* stride1 */
    cfg |= ((uint32_t)(0 & NE16_MASK_NORM_BITS) << NE16_SHIFT_NORM_BITS);
    cfg |= ((uint32_t)(0 & NE16_MASK_STREAMIN) << NE16_SHIFT_STREAMIN);
    cfg |= ((uint32_t)(1 & NE16_MASK_WEIGHT_OFFSET_CFG) << NE16_SHIFT_WEIGHT_OFFSET_CFG);
    cfg |= ((uint32_t)(0 & NE16_MASK_QUANT_RIGHT_SHIFT) << NE16_SHIFT_QUANT_RIGHT_SHIFT);
    cfg |= ((uint32_t)(2 & NE16_MASK_QUANT_BITS) << NE16_SHIFT_QUANT_BITS);           /* 32-bit output */
    cfg |= ((uint32_t)(1 & NE16_MASK_QUANT_NORECT) << NE16_SHIFT_QUANT_NORECT);       /* keep sign */
    cfg |= ((uint32_t)(0 & NE16_MASK_NORM_SHIFT) << NE16_SHIFT_NORM_SHIFT);
    cfg |= ((uint32_t)(0 & NE16_MASK_NORM_BIAS) << NE16_SHIFT_NORM_BIAS);

#if defined(ARES_NE16_DEBUG) && !defined(MINIMAL_OUTPUT)
    printf("CL: NE16 1x1 config: in_w=%d in_h=%d in_feat=%d out_feat=%d\n",
           in_w, in_h, in_feat, out_feat);
    printf("CL: NE16 nb_ki=%d nb_ko=%d rem_ki=%d rem_ko=%d\n",
           nb_ki, nb_ko, rem_ki, rem_ko);
    printf("CL: NE16 nb_wo=%d nb_ho=%d rem_wo=%d rem_ho=%d\n",
           nb_wo, nb_ho, rem_wo, rem_ho);

    /* Compute expected SW accumulator for channels 0 and 1 using unpacked weights */
    /* This helps diagnose if the issue is weight packing or NE16 config */
    {
        int32_t sw_acc_0 = 0, sw_acc_1 = 0;
        for (int kimaj = 0; kimaj < nb_ki; kimaj++) {
            /* Unpack 16 weights for channel 0 from bit-sliced format */
            uint8_t w0_u8[16] = {0};
            uint8_t w1_u8[16] = {0};
            const int base0 = 0 * (int)w_d1 + kimaj * 16;  /* Channel 0 */
            const int base1 = 1 * (int)w_d1 + kimaj * 16;  /* Channel 1 */
            for (int bit = 0; bit < 8; bit++) {
                uint8_t b0_ch0 = weights_packed[base0 + bit * 2];
                uint8_t b1_ch0 = weights_packed[base0 + bit * 2 + 1];
                uint8_t b0_ch1 = weights_packed[base1 + bit * 2];
                uint8_t b1_ch1 = weights_packed[base1 + bit * 2 + 1];
                for (int i = 0; i < 8; i++) {
                    w0_u8[i]     |= ((b0_ch0 >> i) & 0x1) << bit;
                    w0_u8[i + 8] |= ((b1_ch0 >> i) & 0x1) << bit;
                    w1_u8[i]     |= ((b0_ch1 >> i) & 0x1) << bit;
                    w1_u8[i + 8] |= ((b1_ch1 >> i) & 0x1) << bit;
                }
            }
            /* Compute partial dot product with first spatial position */
            for (int kimin = 0; kimin < 16; kimin++) {
                int idx = kimaj * 16 + kimin;
                if (idx < in_feat) {
                    /* Proper signed conversion: uint8 + signed offset */
                    int w0_s = (int)w0_u8[kimin] + (int)weight_offset;
                    int w1_s = (int)w1_u8[kimin] + (int)weight_offset;
                    sw_acc_0 += (int32_t)infeat[idx] * (int32_t)w0_s;
                    sw_acc_1 += (int32_t)infeat[idx] * (int32_t)w1_s;
                }
            }
        }
        printf("CL: NE16 SW expected acc[0]=%ld acc[1]=%ld (unpacked weights x input_u8)\n",
               (long)sw_acc_0, (long)sw_acc_1);

        /* DEEP DEBUG: Print first few unpacked weights for channel 0 */
        printf("CL: NE16 DEEP: First 8 unpacked weights (ch0): ");
        {
            uint8_t w0_check[8] = {0};
            for (int bit = 0; bit < 8; bit++) {
                uint8_t b0 = weights_packed[bit * 2];
                for (int i = 0; i < 8; i++) {
                    w0_check[i] |= ((b0 >> i) & 0x1) << bit;
                }
            }
            for (int i = 0; i < 8; i++) {
                printf("%d ", (int)w0_check[i] + (int)weight_offset);
            }
            printf("\n");
        }
        /* DEEP DEBUG: Print first few unpacked weights for channel 1 */
        printf("CL: NE16 DEEP: First 8 unpacked weights (ch1): ");
        {
            uint8_t w1_check[8] = {0};
            for (int bit = 0; bit < 8; bit++) {
                uint8_t b0 = weights_packed[w_d1 + bit * 2];
                for (int i = 0; i < 8; i++) {
                    w1_check[i] |= ((b0 >> i) & 0x1) << bit;
                }
            }
            for (int i = 0; i < 8; i++) {
                printf("%d ", (int)w1_check[i] + (int)weight_offset);
            }
            printf("\n");
        }
        /* Print actual input values for verification */
        printf("CL: NE16 DEEP: First 8 input_u8: ");
        for (int i = 0; i < 8; i++) {
            printf("%u ", (unsigned)infeat[i]);
        }
        printf("\n");

        /* Also print the CONFIG register value for comparison */
        printf("CL: NE16 CFG register will be: 0x%08x\n", cfg);
    }
    printf("CL: NE16 in_d: %u %u %u  out_d: %u %u %u  w_d: %u %u %u\n",
           in_d0, in_d1, in_d2, out_d0, out_d1, out_d2, w_d0, w_d1, w_d2);
    printf("CL: NE16 cfg=0x%08x weight_offset=%d\n", cfg, (int)weight_offset);
    // Debug: Print weight pointer and first few packed weights
    printf("CL: NE16 weights_packed=%p first 16 bytes: ", weights_packed);
    for (int i = 0; i < 16; i++) {
        printf("%02x ", weights_packed[i]);
    }
    printf("\n");
    // Print weights at row 1 (second output channel, offset = w_d1 = 64)
    printf("CL: NE16 weights row 1 (offset %u): ", w_d1);
    for (int i = 0; i < 16; i++) {
        printf("%02x ", weights_packed[w_d1 + i]);
    }
    printf("\n");

    /* Additional debug: verify total weight size and check boundary */
    printf("CL: NE16 DEBUG: total weight bytes expected = %u\n",
           (uint32_t)out_feat * w_d1);

    /* CRITICAL DEBUG: Verify weights at channel 0 and channel 1 offsets */
    printf("CL: NE16 DEBUG: ch0 packed bytes at offset 0: ");
    for (int i = 0; i < 8; i++) printf("%02x ", weights_packed[i]);
    printf("\n");
    printf("CL: NE16 DEBUG: ch1 packed bytes at offset %u: ", w_d1);
    for (int i = 0; i < 8; i++) printf("%02x ", weights_packed[w_d1 + i]);
    printf("\n");
    /* Verify ch0 != ch1 */
    int ch0_ch1_same = 1;
    for (int i = 0; i < 16; i++) {
        if (weights_packed[i] != weights_packed[w_d1 + i]) ch0_ch1_same = 0;
    }
    printf("CL: NE16 DEBUG: ch0 == ch1? %s\n", ch0_ch1_same ? "YES (BUG!)" : "NO (good)");

    /* Check weight bytes at Ko subtile boundary (channel 32) if applicable */
    if (out_feat > 32) {
        printf("CL: NE16 weights row 32 (offset %u): ", 32 * w_d1);
        for (int i = 0; i < 16; i++) {
            printf("%02x ", weights_packed[32 * w_d1 + i]);
        }
        printf("\n");
    }

    /* Verify weight bytes are different for adjacent output channels */
    printf("CL: NE16 VERIFY: ch0 byte sum = ");
    uint32_t sum0 = 0, sum1 = 0;
    for (int i = 0; i < (int)w_d1; i++) {
        sum0 += weights_packed[0 * w_d1 + i];
        sum1 += weights_packed[1 * w_d1 + i];
    }
    printf("%u, ch1 byte sum = %u (should be different!)\n", sum0, sum1);

    /* Print byte-by-byte comparison for first 8 bytes */
    printf("CL: NE16 VERIFY: ch0[0..7] = %02x %02x %02x %02x %02x %02x %02x %02x\n",
           weights_packed[0], weights_packed[1], weights_packed[2], weights_packed[3],
           weights_packed[4], weights_packed[5], weights_packed[6], weights_packed[7]);
    printf("CL: NE16 VERIFY: ch1[0..7] = %02x %02x %02x %02x %02x %02x %02x %02x\n",
           weights_packed[w_d1], weights_packed[w_d1+1], weights_packed[w_d1+2], weights_packed[w_d1+3],
           weights_packed[w_d1+4], weights_packed[w_d1+5], weights_packed[w_d1+6], weights_packed[w_d1+7]);
#endif

    ne16_write_reg(NE16_REG_CONFIG, 0);
    ne16_write_reg(NE16_REG_CONFIG, cfg);

#if defined(ARES_NE16_DEBUG) && !defined(MINIMAL_OUTPUT)
    /* DEBUG: Read back critical registers to verify programming */
    volatile uint32_t *reg_base = (volatile uint32_t *)(NE16_ADDR_BASE + NE16_REGISTER_OFFS);
    printf("CL: NE16 register readback:\n");
    printf("  WEIGHTS_PTR (0): 0x%08x\n", reg_base[0]);
    printf("  WEIGHTS_D0_STRIDE (12): %u\n", reg_base[12]);
    printf("  WEIGHTS_D1_STRIDE (13): %u (expected %u)\n", reg_base[13], w_d1);
    printf("  CONFIG (23): 0x%08x\n", reg_base[23]);
#endif

    /* Full memory barrier to ensure all weight/input data is visible to NE16 DMA */
    asm volatile("fence iorw, iorw" ::: "memory");

    /* Commit + trigger */
    ne16_write_cmd(NE16_CMD_TRIGGER, 0);
    ne16_wait_job_done(job_id);
}

int ne16_conv1x1_s32_submit_async(
    const uint8_t *infeat,
    const uint8_t *weights_packed,
    int32_t *outfeat,
    int in_w,
    int in_h,
    int in_feat,
    int out_feat,
    int8_t weight_offset
) {
    const int job_id = ne16_acquire_job();

    const int q_w_bits = 8;
    const int tp_in = 16;
    const int nb_ki = ceil_div(in_feat, tp_in);

    /* InFeat strides (HWC) */
    const uint32_t in_d0 = (uint32_t)in_feat;
    const uint32_t in_d1 = (uint32_t)in_feat * (uint32_t)in_w;
    const uint32_t in_d2 = 0;

    /* OutFeat strides (HWC, 32-bit streamout) */
    const uint32_t out_bytes = 4;
    const uint32_t out_d0 = NE16_OUTPUT_BANDWIDTH_BYTES;  /* 32, fixed */
    const uint32_t out_d1 = out_bytes * (uint32_t)out_feat;
    const uint32_t out_d2 = out_bytes * (uint32_t)out_feat * (uint32_t)in_w;

    /* Weights strides */
    const uint32_t w_d0 = 2u * (uint32_t)q_w_bits;
    const uint32_t w_d1 = (uint32_t)nb_ki * 16u * (uint32_t)q_w_bits / 8u;
    const uint32_t w_d2 = 0;

    /* NB / REM fields */
    const int rem_ki = (in_feat % 16) ? (in_feat % 16) : 16;
    const int nb_ko = ceil_div(out_feat, 32);
    const int rem_ko = (out_feat % 32) ? (out_feat % 32) : 32;

    const int rem_wo = in_w % 3;
    const int nb_wo = in_w / 3 + (rem_wo ? 1 : 0);
    const int rem_ho = in_h % 3;
    const int nb_ho = in_h / 3 + (rem_ho ? 1 : 0);

    const int rem_wi = rem_wo;
    const int rem_hi = rem_ho;

    /* Program pointers */
    ne16_write_reg(NE16_REG_WEIGHTS_PTR, (uint32_t)weights_packed);
    ne16_write_reg(NE16_REG_INFEAT_PTR, (uint32_t)infeat);
    ne16_write_reg(NE16_REG_OUTFEAT_PTR, (uint32_t)outfeat);
    ne16_write_reg(NE16_REG_SCALE_PTR, 0);
    ne16_write_reg(NE16_REG_SCALE_SHIFT_PTR, 0);
    ne16_write_reg(NE16_REG_SCALE_BIAS_PTR, 0);

    /* Program strides */
    ne16_write_reg(NE16_REG_INFEAT_D0_STRIDE, in_d0);
    ne16_write_reg(NE16_REG_INFEAT_D1_STRIDE, in_d1);
    ne16_write_reg(NE16_REG_INFEAT_D2_STRIDE, in_d2);
    ne16_write_reg(NE16_REG_OUTFEAT_D0_STRIDE, out_d0);
    ne16_write_reg(NE16_REG_OUTFEAT_D1_STRIDE, out_d1);
    ne16_write_reg(NE16_REG_OUTFEAT_D2_STRIDE, out_d2);
    ne16_write_reg(NE16_REG_WEIGHTS_D0_STRIDE, w_d0);
    ne16_write_reg(NE16_REG_WEIGHTS_D1_STRIDE, w_d1);
    ne16_write_reg(NE16_REG_WEIGHTS_D2_STRIDE, w_d2);

    /* Reminders */
    ne16_write_reg(NE16_REG_REM_KO_KI,
                   ((uint32_t)(rem_ki & NE16_MASK_REM_KI) << NE16_SHIFT_REM_KI) |
                       ((uint32_t)(rem_ko & NE16_MASK_REM_KO) << NE16_SHIFT_REM_KO));
    ne16_write_reg(NE16_REG_REM_HO_WO,
                   ((uint32_t)(rem_wo & NE16_MASK_REM_WO) << NE16_SHIFT_REM_WO) |
                       ((uint32_t)(rem_ho & NE16_MASK_REM_HO) << NE16_SHIFT_REM_HO));
    ne16_write_reg(NE16_REG_REM_HI_WI,
                   ((uint32_t)(rem_wi & NE16_MASK_REM_WI) << NE16_SHIFT_REM_WI) |
                       ((uint32_t)(rem_hi & NE16_MASK_REM_HI) << NE16_SHIFT_REM_HI));

    /* Dimensions */
    ne16_write_reg(NE16_REG_NB_KO_KI,
                   ((uint32_t)(nb_ki & NE16_MASK_NB_KI) << NE16_SHIFT_NB_KI) |
                       ((uint32_t)(nb_ko & NE16_MASK_NB_KO) << NE16_SHIFT_NB_KO));
    ne16_write_reg(NE16_REG_NB_HO_WO,
                   ((uint32_t)(nb_wo & NE16_MASK_NB_WO) << NE16_SHIFT_NB_WO) |
                       ((uint32_t)(nb_ho & NE16_MASK_NB_HO) << NE16_SHIFT_NB_HO));

    /* Padding and filter mask */
    ne16_write_reg(NE16_REG_PADDING, 0);
    ne16_write_reg(NE16_REG_FILTER_MASK, 0);

    /* Weight offset */
    ne16_write_reg(NE16_REG_WEIGHT_OFFSET, (uint32_t)(int32_t)weight_offset);

    /* Config: 1x1 mode, 8-bit weights, 32-bit streamout, no quantization */
    uint32_t cfg = 0;
    cfg |= ((uint32_t)(7 & NE16_MASK_WBITS_M1) << NE16_SHIFT_WBITS_M1);               /* Qw = 8 */
    cfg |= ((uint32_t)(0 & NE16_MASK_MODE16) << NE16_SHIFT_MODE16);                   /* mode16 = 0 */
    cfg |= ((uint32_t)(0 & NE16_MASK_OUTQUANT) << NE16_SHIFT_OUTQUANT);               /* streamout only */
    cfg |= ((uint32_t)(2 & NE16_MASK_FILTER_MODE) << NE16_SHIFT_FILTER_MODE);         /* 1x1 mode */
    cfg |= ((uint32_t)(0 & NE16_MASK_LINEAR_MODE) << NE16_SHIFT_LINEAR_MODE);
    cfg |= ((uint32_t)(0 & NE16_MASK_STRIDED_MODE) << NE16_SHIFT_STRIDED_MODE);
    cfg |= ((uint32_t)(0 & NE16_MASK_NORM_BITS) << NE16_SHIFT_NORM_BITS);
    cfg |= ((uint32_t)(0 & NE16_MASK_STREAMIN) << NE16_SHIFT_STREAMIN);
    cfg |= ((uint32_t)(1 & NE16_MASK_WEIGHT_OFFSET_CFG) << NE16_SHIFT_WEIGHT_OFFSET_CFG);
    cfg |= ((uint32_t)(0 & NE16_MASK_QUANT_RIGHT_SHIFT) << NE16_SHIFT_QUANT_RIGHT_SHIFT);
    cfg |= ((uint32_t)(2 & NE16_MASK_QUANT_BITS) << NE16_SHIFT_QUANT_BITS);           /* 32-bit output */
    cfg |= ((uint32_t)(1 & NE16_MASK_QUANT_NORECT) << NE16_SHIFT_QUANT_NORECT);       /* keep sign */
    cfg |= ((uint32_t)(0 & NE16_MASK_NORM_SHIFT) << NE16_SHIFT_NORM_SHIFT);
    cfg |= ((uint32_t)(0 & NE16_MASK_NORM_BIAS) << NE16_SHIFT_NORM_BIAS);

    ne16_write_reg(NE16_REG_CONFIG, 0);
    ne16_write_reg(NE16_REG_CONFIG, cfg);

    /* Full memory barrier */
    asm volatile("fence iorw, iorw" ::: "memory");

    /* Trigger job but don't wait */
    ne16_write_cmd(NE16_CMD_TRIGGER, 0);

    return job_id;
}

int ne16_conv1x1_submit_async(
    const uint8_t *infeat,
    const uint8_t *weights_packed,
    const int32_t *bias,
    const uint8_t *scale,
    const uint8_t *scale_shift,
    int8_t *outfeat,
    int in_w,
    int in_h,
    int in_feat,
    int out_feat,
    int8_t weight_offset
) {
    const int job_id = ne16_acquire_job();

    const int q_w_bits = 8;
    const int tp_in = 16;
    const int nb_ki = ceil_div(in_feat, tp_in);

    /* InFeat strides (HWC) */
    const uint32_t in_d0 = (uint32_t)in_feat;
    const uint32_t in_d1 = (uint32_t)in_feat * (uint32_t)in_w;
    const uint32_t in_d2 = 0;

    /* OutFeat strides (HWC, 8-bit) */
    const uint32_t out_bytes = 1;
    const uint32_t out_d0 = 0; /* GAP SDK sets d0 stride to 0 for 8-bit streamout */
    const uint32_t out_d1 = out_bytes * (uint32_t)out_feat;
    const uint32_t out_d2 = out_bytes * (uint32_t)out_feat * (uint32_t)in_w;

    /* Weights strides */
    const uint32_t w_d0 = 2u * (uint32_t)q_w_bits;
    const uint32_t w_d1 = (uint32_t)nb_ki * 16u * (uint32_t)q_w_bits / 8u;
    const uint32_t w_d2 = 0;

    /* NB / REM fields */
    const int rem_ki = (in_feat % 16) ? (in_feat % 16) : 16;
    const int nb_ko = ceil_div(out_feat, 32);
    const int rem_ko = (out_feat % 32) ? (out_feat % 32) : 32;

    const int rem_wo = in_w % 3;
    const int nb_wo = in_w / 3 + (rem_wo ? 1 : 0);
    const int rem_ho = in_h % 3;
    const int nb_ho = in_h / 3 + (rem_ho ? 1 : 0);

    const int rem_wi = rem_wo;
    const int rem_hi = rem_ho;

    /* Program pointers */
    ne16_write_reg(NE16_REG_WEIGHTS_PTR, (uint32_t)weights_packed);
    ne16_write_reg(NE16_REG_INFEAT_PTR, (uint32_t)infeat);
    ne16_write_reg(NE16_REG_OUTFEAT_PTR, (uint32_t)outfeat);
    ne16_write_reg(NE16_REG_SCALE_PTR, (uint32_t)scale);
    ne16_write_reg(NE16_REG_SCALE_SHIFT_PTR, (uint32_t)scale_shift);
    ne16_write_reg(NE16_REG_SCALE_BIAS_PTR, (uint32_t)bias);

    /* Program strides */
    ne16_write_reg(NE16_REG_INFEAT_D0_STRIDE, in_d0);
    ne16_write_reg(NE16_REG_INFEAT_D1_STRIDE, in_d1);
    ne16_write_reg(NE16_REG_INFEAT_D2_STRIDE, in_d2);
    ne16_write_reg(NE16_REG_OUTFEAT_D0_STRIDE, out_d0);
    ne16_write_reg(NE16_REG_OUTFEAT_D1_STRIDE, out_d1);
    ne16_write_reg(NE16_REG_OUTFEAT_D2_STRIDE, out_d2);
    ne16_write_reg(NE16_REG_WEIGHTS_D0_STRIDE, w_d0);
    ne16_write_reg(NE16_REG_WEIGHTS_D1_STRIDE, w_d1);
    ne16_write_reg(NE16_REG_WEIGHTS_D2_STRIDE, w_d2);

    /* Reminders */
    ne16_write_reg(NE16_REG_REM_KO_KI,
                   ((uint32_t)(rem_ki & NE16_MASK_REM_KI) << NE16_SHIFT_REM_KI) |
                       ((uint32_t)(rem_ko & NE16_MASK_REM_KO) << NE16_SHIFT_REM_KO));
    ne16_write_reg(NE16_REG_REM_HO_WO,
                   ((uint32_t)(rem_wo & NE16_MASK_REM_WO) << NE16_SHIFT_REM_WO) |
                       ((uint32_t)(rem_ho & NE16_MASK_REM_HO) << NE16_SHIFT_REM_HO));
    ne16_write_reg(NE16_REG_REM_HI_WI,
                   ((uint32_t)(rem_wi & NE16_MASK_REM_WI) << NE16_SHIFT_REM_WI) |
                       ((uint32_t)(rem_hi & NE16_MASK_REM_HI) << NE16_SHIFT_REM_HI));

    /* Dimensions */
    ne16_write_reg(NE16_REG_NB_KO_KI,
                   ((uint32_t)(nb_ki & NE16_MASK_NB_KI) << NE16_SHIFT_NB_KI) |
                       ((uint32_t)(nb_ko & NE16_MASK_NB_KO) << NE16_SHIFT_NB_KO));
    ne16_write_reg(NE16_REG_NB_HO_WO,
                   ((uint32_t)(nb_wo & NE16_MASK_NB_WO) << NE16_SHIFT_NB_WO) |
                       ((uint32_t)(nb_ho & NE16_MASK_NB_HO) << NE16_SHIFT_NB_HO));

    /* Padding and filter mask */
    ne16_write_reg(NE16_REG_PADDING, 0);
    ne16_write_reg(NE16_REG_FILTER_MASK, 0);

    /* Weight offset */
    ne16_write_reg(NE16_REG_WEIGHT_OFFSET, (uint32_t)(int32_t)weight_offset);

    /* Config: 1x1 mode, 8-bit weights, quantization+streamout enabled, signed 8-bit output */
    uint32_t cfg = 0;
    cfg |= ((uint32_t)(7 & NE16_MASK_WBITS_M1) << NE16_SHIFT_WBITS_M1);               /* Qw = 8 */
    cfg |= ((uint32_t)(0 & NE16_MASK_MODE16) << NE16_SHIFT_MODE16);                   /* mode16 = 0 */
    cfg |= ((uint32_t)(1 & NE16_MASK_OUTQUANT) << NE16_SHIFT_OUTQUANT);               /* quant + streamout */
    cfg |= ((uint32_t)(2 & NE16_MASK_FILTER_MODE) << NE16_SHIFT_FILTER_MODE);         /* 1x1 mode */
    cfg |= ((uint32_t)(0 & NE16_MASK_LINEAR_MODE) << NE16_SHIFT_LINEAR_MODE);
    cfg |= ((uint32_t)(0 & NE16_MASK_STRIDED_MODE) << NE16_SHIFT_STRIDED_MODE);
    cfg |= ((uint32_t)(0 & NE16_MASK_NORM_BITS) << NE16_SHIFT_NORM_BITS);             /* 8-bit norm params */
    cfg |= ((uint32_t)(0 & NE16_MASK_STREAMIN) << NE16_SHIFT_STREAMIN);
    cfg |= ((uint32_t)(1 & NE16_MASK_WEIGHT_OFFSET_CFG) << NE16_SHIFT_WEIGHT_OFFSET_CFG);
    cfg |= ((uint32_t)(0 & NE16_MASK_QUANT_RIGHT_SHIFT) << NE16_SHIFT_QUANT_RIGHT_SHIFT);
    cfg |= ((uint32_t)(0 & NE16_MASK_QUANT_BITS) << NE16_SHIFT_QUANT_BITS);           /* 8-bit output */
    cfg |= ((uint32_t)(1 & NE16_MASK_QUANT_NORECT) << NE16_SHIFT_QUANT_NORECT);       /* keep sign */
    cfg |= ((uint32_t)(1 & NE16_MASK_NORM_SHIFT) << NE16_SHIFT_NORM_SHIFT);           /* load per-channel shift */
    cfg |= ((uint32_t)(1 & NE16_MASK_NORM_BIAS) << NE16_SHIFT_NORM_BIAS);             /* load bias */

    ne16_write_reg(NE16_REG_CONFIG, 0);
    ne16_write_reg(NE16_REG_CONFIG, cfg);

    /* Full memory barrier to ensure all data is visible to NE16 DMA */
    asm volatile("fence iorw, iorw" ::: "memory");

    /* Trigger job but don't wait */
    ne16_write_cmd(NE16_CMD_TRIGGER, 0);

    return job_id;
}

void ne16_conv1x1_u8_u8_to_s8(
    const uint8_t *infeat,
    const uint8_t *weights_packed,
    const int32_t *bias,
    const uint8_t *scale,
    const uint8_t *scale_shift,
    int8_t *outfeat,
    int in_w,
    int in_h,
    int in_feat,
    int out_feat,
    int8_t weight_offset
) {
    const int job_id = ne16_acquire_job();

    const int q_w_bits = 8;
    const int tp_in = 16;
    const int nb_ki = ceil_div(in_feat, tp_in);

    /* InFeat strides (HWC) */
    const uint32_t in_d0 = (uint32_t)in_feat;
    const uint32_t in_d1 = (uint32_t)in_feat * (uint32_t)in_w;
    const uint32_t in_d2 = 0;

    /* OutFeat strides (HWC, 8-bit) */
    const uint32_t out_bytes = 1;
    const uint32_t out_d0 = 0; /* GAP SDK sets d0 stride to 0 for 8-bit streamout */
    const uint32_t out_d1 = out_bytes * (uint32_t)out_feat;
    const uint32_t out_d2 = out_bytes * (uint32_t)out_feat * (uint32_t)in_w;

    /* Weights strides */
    const uint32_t w_d0 = 2u * (uint32_t)q_w_bits;
    const uint32_t w_d1 = (uint32_t)nb_ki * 16u * (uint32_t)q_w_bits / 8u;
    const uint32_t w_d2 = 0;

    /* NB / REM fields */
    const int rem_ki = (in_feat % 16) ? (in_feat % 16) : 16;
    const int nb_ko = ceil_div(out_feat, 32);
    const int rem_ko = (out_feat % 32) ? (out_feat % 32) : 32;

    const int rem_wo = in_w % 3;
    const int nb_wo = in_w / 3 + (rem_wo ? 1 : 0);
    const int rem_ho = in_h % 3;
    const int nb_ho = in_h / 3 + (rem_ho ? 1 : 0);

    const int rem_wi = rem_wo;
    const int rem_hi = rem_ho;

    /* Program pointers */
    ne16_write_reg(NE16_REG_WEIGHTS_PTR, (uint32_t)weights_packed);
    ne16_write_reg(NE16_REG_INFEAT_PTR, (uint32_t)infeat);
    ne16_write_reg(NE16_REG_OUTFEAT_PTR, (uint32_t)outfeat);
    ne16_write_reg(NE16_REG_SCALE_PTR, (uint32_t)scale);
    ne16_write_reg(NE16_REG_SCALE_SHIFT_PTR, (uint32_t)scale_shift);
    ne16_write_reg(NE16_REG_SCALE_BIAS_PTR, (uint32_t)bias);

    /* Program strides */
    ne16_write_reg(NE16_REG_INFEAT_D0_STRIDE, in_d0);
    ne16_write_reg(NE16_REG_INFEAT_D1_STRIDE, in_d1);
    ne16_write_reg(NE16_REG_INFEAT_D2_STRIDE, in_d2);
    ne16_write_reg(NE16_REG_OUTFEAT_D0_STRIDE, out_d0);
    ne16_write_reg(NE16_REG_OUTFEAT_D1_STRIDE, out_d1);
    ne16_write_reg(NE16_REG_OUTFEAT_D2_STRIDE, out_d2);
    ne16_write_reg(NE16_REG_WEIGHTS_D0_STRIDE, w_d0);
    ne16_write_reg(NE16_REG_WEIGHTS_D1_STRIDE, w_d1);
    ne16_write_reg(NE16_REG_WEIGHTS_D2_STRIDE, w_d2);

    /* Reminders */
    ne16_write_reg(NE16_REG_REM_KO_KI,
                   ((uint32_t)(rem_ki & NE16_MASK_REM_KI) << NE16_SHIFT_REM_KI) |
                       ((uint32_t)(rem_ko & NE16_MASK_REM_KO) << NE16_SHIFT_REM_KO));
    ne16_write_reg(NE16_REG_REM_HO_WO,
                   ((uint32_t)(rem_wo & NE16_MASK_REM_WO) << NE16_SHIFT_REM_WO) |
                       ((uint32_t)(rem_ho & NE16_MASK_REM_HO) << NE16_SHIFT_REM_HO));
    ne16_write_reg(NE16_REG_REM_HI_WI,
                   ((uint32_t)(rem_wi & NE16_MASK_REM_WI) << NE16_SHIFT_REM_WI) |
                       ((uint32_t)(rem_hi & NE16_MASK_REM_HI) << NE16_SHIFT_REM_HI));

    /* Dimensions */
    ne16_write_reg(NE16_REG_NB_KO_KI,
                   ((uint32_t)(nb_ki & NE16_MASK_NB_KI) << NE16_SHIFT_NB_KI) |
                       ((uint32_t)(nb_ko & NE16_MASK_NB_KO) << NE16_SHIFT_NB_KO));
    ne16_write_reg(NE16_REG_NB_HO_WO,
                   ((uint32_t)(nb_wo & NE16_MASK_NB_WO) << NE16_SHIFT_NB_WO) |
                       ((uint32_t)(nb_ho & NE16_MASK_NB_HO) << NE16_SHIFT_NB_HO));

    /* Padding and filter mask */
    ne16_write_reg(NE16_REG_PADDING, 0);
    ne16_write_reg(NE16_REG_FILTER_MASK, 0);

    /* Weight offset */
    ne16_write_reg(NE16_REG_WEIGHT_OFFSET, (uint32_t)(int32_t)weight_offset);

    /* Config: 1x1 mode, 8-bit weights, quantization+streamout enabled, signed 8-bit output */
    uint32_t cfg = 0;
    cfg |= ((uint32_t)(7 & NE16_MASK_WBITS_M1) << NE16_SHIFT_WBITS_M1);               /* Qw = 8 */
    cfg |= ((uint32_t)(0 & NE16_MASK_MODE16) << NE16_SHIFT_MODE16);                   /* mode16 = 0 */
    cfg |= ((uint32_t)(1 & NE16_MASK_OUTQUANT) << NE16_SHIFT_OUTQUANT);               /* quant + streamout */
    cfg |= ((uint32_t)(2 & NE16_MASK_FILTER_MODE) << NE16_SHIFT_FILTER_MODE);         /* 1x1 mode */
    cfg |= ((uint32_t)(0 & NE16_MASK_LINEAR_MODE) << NE16_SHIFT_LINEAR_MODE);
    cfg |= ((uint32_t)(0 & NE16_MASK_STRIDED_MODE) << NE16_SHIFT_STRIDED_MODE);
    cfg |= ((uint32_t)(0 & NE16_MASK_NORM_BITS) << NE16_SHIFT_NORM_BITS);             /* 8-bit norm params */
    cfg |= ((uint32_t)(0 & NE16_MASK_STREAMIN) << NE16_SHIFT_STREAMIN);
    cfg |= ((uint32_t)(1 & NE16_MASK_WEIGHT_OFFSET_CFG) << NE16_SHIFT_WEIGHT_OFFSET_CFG);
    cfg |= ((uint32_t)(0 & NE16_MASK_QUANT_RIGHT_SHIFT) << NE16_SHIFT_QUANT_RIGHT_SHIFT);
    cfg |= ((uint32_t)(0 & NE16_MASK_QUANT_BITS) << NE16_SHIFT_QUANT_BITS);           /* 8-bit output */
    cfg |= ((uint32_t)(1 & NE16_MASK_QUANT_NORECT) << NE16_SHIFT_QUANT_NORECT);       /* keep sign */
    cfg |= ((uint32_t)(1 & NE16_MASK_NORM_SHIFT) << NE16_SHIFT_NORM_SHIFT);           /* load per-channel shift */
    cfg |= ((uint32_t)(1 & NE16_MASK_NORM_BIAS) << NE16_SHIFT_NORM_BIAS);             /* load bias */

    ne16_write_reg(NE16_REG_CONFIG, 0);
    ne16_write_reg(NE16_REG_CONFIG, cfg);

    /* Commit + trigger */
    ne16_write_cmd(NE16_CMD_TRIGGER, 0);
    ne16_wait_job_done(job_id);
}

void ne16_conv3x3_u8_u8_to_s32(
    const uint8_t *infeat,
    const uint8_t *weights_packed,
    int32_t *outfeat,
    int in_w,
    int in_h,
    int in_feat,
    int out_feat,
    int pad_h,
    int pad_w,
    int8_t weight_offset
) {

    const int job_id = ne16_acquire_job();

    const int q_w_bits = 8;
    const int tp_in = 16;
    const int nb_ki = ceil_div(in_feat, tp_in);

    /* Output dimensions */
    const int out_h = in_h + 2 * pad_h - 2;  /* 3x3 kernel, stride 1 */
    const int out_w = in_w + 2 * pad_w - 2;

    /* InFeat strides (HWC) */
    const uint32_t in_d0 = (uint32_t)in_feat;
    const uint32_t in_d1 = (uint32_t)in_feat * (uint32_t)in_w;
    const uint32_t in_d2 = 0;

    /* OutFeat strides (HWC, 32-bit) */
    const uint32_t out_bytes = 4;
    const uint32_t out_d0 = 32;
    const uint32_t out_d1 = out_bytes * (uint32_t)out_feat;
    const uint32_t out_d2 = out_bytes * (uint32_t)out_feat * (uint32_t)out_w;

    /* Weights strides for packed 3x3 layout
     *
     * From GAP SDK CNN_BasicKernels_NE16.c:
     *   W_D0 = (Mode16?1:2)*3*3 = 2*3*3 = 18 for 8-bit mode
     *   W_D1 = (Mode16?1:2)*3*3*Qw*Nb_KI = 2*3*3*8*Nb_KI = 144*Nb_KI
     *
     * Layout: [ko][ki_group][bit][h][w][2bytes]
     * - w_d0 is stride between bit planes = 3*3*2 = 18
     * - w_d1 is stride per output channel = nb_ki * 8 * 3 * 3 * 2 = nb_ki * 144
     */
    const uint32_t w_d0 = 2u * 3u * 3u;  /* = 18, stride per bit plane */
    const uint32_t w_d1 = (uint32_t)nb_ki * 144u;  /* = nb_ki * 8 * 3 * 3 * 2 */
    const uint32_t w_d2 = 0;

    /* NB / REM fields */
    const int rem_ki = (in_feat % 16) ? (in_feat % 16) : 16;
    const int nb_ko = ceil_div(out_feat, 32);
    const int rem_ko = (out_feat % 32) ? (out_feat % 32) : 32;

    const int rem_wo = out_w % 3;
    const int nb_wo = out_w / 3 + (rem_wo ? 1 : 0);
    const int rem_ho = out_h % 3;
    const int nb_ho = out_h / 3 + (rem_ho ? 1 : 0);

    /* For 3x3, input remainder includes halo.
     * From GAP SDK: Rem_WI = Rem_WO ? (Rem_WO+2) : 0
     * When output is exactly divisible by 3, rem_wi/rem_hi should be 0.
     */
    const int rem_wi = rem_wo ? (rem_wo + 2) : 0;
    const int rem_hi = rem_ho ? (rem_ho + 2) : 0;

    /* Program pointers
     *
     */
    ne16_write_reg(NE16_REG_WEIGHTS_PTR, (uint32_t)weights_packed);
    ne16_write_reg(NE16_REG_INFEAT_PTR, (uint32_t)infeat);
    ne16_write_reg(NE16_REG_OUTFEAT_PTR, (uint32_t)outfeat);
    ne16_write_reg(NE16_REG_SCALE_PTR, 0);
    ne16_write_reg(NE16_REG_SCALE_SHIFT_PTR, 0);
    ne16_write_reg(NE16_REG_SCALE_BIAS_PTR, 0);

    /* Program strides */
    ne16_write_reg(NE16_REG_INFEAT_D0_STRIDE, in_d0);
    ne16_write_reg(NE16_REG_INFEAT_D1_STRIDE, in_d1);
    ne16_write_reg(NE16_REG_INFEAT_D2_STRIDE, in_d2);
    ne16_write_reg(NE16_REG_OUTFEAT_D0_STRIDE, out_d0);
    ne16_write_reg(NE16_REG_OUTFEAT_D1_STRIDE, out_d1);
    ne16_write_reg(NE16_REG_OUTFEAT_D2_STRIDE, out_d2);
    ne16_write_reg(NE16_REG_WEIGHTS_D0_STRIDE, w_d0);
    ne16_write_reg(NE16_REG_WEIGHTS_D1_STRIDE, w_d1);
    ne16_write_reg(NE16_REG_WEIGHTS_D2_STRIDE, w_d2);

    /* Reminders */
    ne16_write_reg(NE16_REG_REM_KO_KI,
                   ((uint32_t)(rem_ki & NE16_MASK_REM_KI) << NE16_SHIFT_REM_KI) |
                       ((uint32_t)(rem_ko & NE16_MASK_REM_KO) << NE16_SHIFT_REM_KO));
    ne16_write_reg(NE16_REG_REM_HO_WO,
                   ((uint32_t)(rem_wo & NE16_MASK_REM_WO) << NE16_SHIFT_REM_WO) |
                       ((uint32_t)(rem_ho & NE16_MASK_REM_HO) << NE16_SHIFT_REM_HO));
    ne16_write_reg(NE16_REG_REM_HI_WI,
                   ((uint32_t)(rem_wi & NE16_MASK_REM_WI) << NE16_SHIFT_REM_WI) |
                       ((uint32_t)(rem_hi & NE16_MASK_REM_HI) << NE16_SHIFT_REM_HI));

    /* Dimensions */
    ne16_write_reg(NE16_REG_NB_KO_KI,
                   ((uint32_t)(nb_ki & NE16_MASK_NB_KI) << NE16_SHIFT_NB_KI) |
                       ((uint32_t)(nb_ko & NE16_MASK_NB_KO) << NE16_SHIFT_NB_KO));
    ne16_write_reg(NE16_REG_NB_HO_WO,
                   ((uint32_t)(nb_wo & NE16_MASK_NB_WO) << NE16_SHIFT_NB_WO) |
                       ((uint32_t)(nb_ho & NE16_MASK_NB_HO) << NE16_SHIFT_NB_HO));

    /* Padding register layout (from GAP SDK):
     *   Bits 0-15:  PADDING_VALUE (value to use for padded positions)
     *   Bits 16-19: PADDING_LEFT (0-2)
     *   Bits 20-23: PADDING_BOTTOM (0-2)
     *   Bits 24-27: PADDING_RIGHT (0-2)
     *   Bits 28-31: PADDING_TOP (0-2)
    */
    const uint16_t pad_val = 128;  /* 0 in signed domain */
    uint32_t padding = 0;
    padding |= (pad_val & 0xFFFF);           /* padding value in lower 16 bits */
    padding |= ((pad_w & 0xF) << 16);        /* left */
    padding |= ((pad_h & 0xF) << 20);        /* bottom */
    padding |= ((pad_w & 0xF) << 24);        /* right */
    padding |= ((pad_h & 0xF) << 28);        /* top */
    ne16_write_reg(NE16_REG_PADDING, padding);

    /* Filter mask: all 9 positions active for standard 3x3 */
    ne16_write_reg(NE16_REG_FILTER_MASK, 0);

    /* Weight offset */
    ne16_write_reg(NE16_REG_WEIGHT_OFFSET, (uint32_t)(int32_t)weight_offset);

    /* Config: 3x3 mode, 8-bit weights, 32-bit streamout */
    uint32_t cfg = 0;
    cfg |= ((uint32_t)(7 & NE16_MASK_WBITS_M1) << NE16_SHIFT_WBITS_M1);               /* Qw = 8 */
    cfg |= ((uint32_t)(0 & NE16_MASK_MODE16) << NE16_SHIFT_MODE16);
    cfg |= ((uint32_t)(0 & NE16_MASK_OUTQUANT) << NE16_SHIFT_OUTQUANT);               /* streamout only */
    cfg |= ((uint32_t)(0 & NE16_MASK_FILTER_MODE) << NE16_SHIFT_FILTER_MODE);         /* 3x3 mode */
    cfg |= ((uint32_t)(0 & NE16_MASK_LINEAR_MODE) << NE16_SHIFT_LINEAR_MODE);
    cfg |= ((uint32_t)(0 & NE16_MASK_STRIDED_MODE) << NE16_SHIFT_STRIDED_MODE);
    cfg |= ((uint32_t)(0 & NE16_MASK_NORM_BITS) << NE16_SHIFT_NORM_BITS);
    cfg |= ((uint32_t)(0 & NE16_MASK_STREAMIN) << NE16_SHIFT_STREAMIN);
    cfg |= ((uint32_t)(1 & NE16_MASK_WEIGHT_OFFSET_CFG) << NE16_SHIFT_WEIGHT_OFFSET_CFG);
    cfg |= ((uint32_t)(0 & NE16_MASK_QUANT_RIGHT_SHIFT) << NE16_SHIFT_QUANT_RIGHT_SHIFT);
    cfg |= ((uint32_t)(2 & NE16_MASK_QUANT_BITS) << NE16_SHIFT_QUANT_BITS);           /* 32-bit output */
    cfg |= ((uint32_t)(1 & NE16_MASK_QUANT_NORECT) << NE16_SHIFT_QUANT_NORECT);
    cfg |= ((uint32_t)(0 & NE16_MASK_NORM_SHIFT) << NE16_SHIFT_NORM_SHIFT);
    cfg |= ((uint32_t)(0 & NE16_MASK_NORM_BIAS) << NE16_SHIFT_NORM_BIAS);

    ne16_write_reg(NE16_REG_CONFIG, 0);
    ne16_write_reg(NE16_REG_CONFIG, cfg);

#if defined(ARES_NE16_DEBUG) && !defined(MINIMAL_OUTPUT)
    /* Debug: Print NE16 3x3 configuration */
    printf("CL: NE16 3x3 DEBUG: in=%dx%dx%d out=%dx%dx%d pad=%d,%d\n",
           in_h, in_w, in_feat, out_h, out_w, out_feat, pad_h, pad_w);
    printf("CL: NE16 3x3 DEBUG: nb_ki=%d nb_ko=%d rem_ki=%d rem_ko=%d\n",
           nb_ki, nb_ko, rem_ki, rem_ko);
    printf("CL: NE16 3x3 DEBUG: nb_ho=%d nb_wo=%d rem_ho=%d rem_wo=%d\n",
           nb_ho, nb_wo, rem_ho, rem_wo);
    printf("CL: NE16 3x3 DEBUG: rem_hi=%d rem_wi=%d\n", rem_hi, rem_wi);
    printf("CL: NE16 3x3 DEBUG: w_d0=%u w_d1=%u w_d2=%u\n", w_d0, w_d1, w_d2);
    printf("CL: NE16 3x3 DEBUG: in_d0=%u in_d1=%u in_d2=%u\n", in_d0, in_d1, in_d2);
    printf("CL: NE16 3x3 DEBUG: out_d0=%u out_d1=%u out_d2=%u\n", out_d0, out_d1, out_d2);
    printf("CL: NE16 3x3 DEBUG: cfg=0x%08x weight_offset=%d\n", cfg, (int)weight_offset);
    printf("CL: NE16 3x3 DEBUG: padding=0x%08x\n", padding);

    /* Print first 32 raw packed weight bytes for verification */
    printf("CL: NE16 3x3 DEBUG: packed_weights[0..31]: ");
    for (int i = 0; i < 32; i++) {
        printf("%02x ", weights_packed[i]);
    }
    printf("\n");

    /* Also print weights at offset 144 (second ki_group for ko=0) */
    printf("CL: NE16 3x3 DEBUG: packed_weights[144..159]: ");
    for (int i = 144; i < 160; i++) {
        printf("%02x ", weights_packed[i]);
    }
    printf("\n");

    /* Compute expected SW accumulator for output(0,0) channel 0 */
    /* This uses the unpacked weights from bit-sliced format */
    {
        int64_t sw_acc_ch0 = 0;
        int64_t input_sum = 0;

        /* For each input channel group */
        for (int ki_group = 0; ki_group < nb_ki; ki_group++) {
            /* Unpack 16 weights for channel 0 from 3x3 bit-sliced format */
            /* Layout: [ko][ki_group][bit][h][w][2bytes] */
            int w_base = 0 * (int)w_d1 + ki_group * 8 * 18;  /* Channel 0, this ki_group */

            for (int fh = 0; fh < 3; fh++) {
                for (int fw = 0; fw < 3; fw++) {
                    /* Get input at this 3x3 position (for output 0,0) */
                    /* Input is HWC: input[fh][fw][ki_group*16 + ki_local] */
                    int in_base = fh * in_w * in_feat + fw * in_feat + ki_group * 16;

                    /* Unpack weights for this spatial position */
                    uint8_t w_u8[16] = {0};
                    for (int bit = 0; bit < 8; bit++) {
                        int bit_offset = w_base + bit * 18 + fh * 6 + fw * 2;
                        uint8_t b0 = weights_packed[bit_offset];
                        uint8_t b1 = weights_packed[bit_offset + 1];
                        for (int i = 0; i < 8; i++) {
                            w_u8[i]     |= ((b0 >> i) & 0x1) << bit;
                            w_u8[i + 8] |= ((b1 >> i) & 0x1) << bit;
                        }
                    }

                    /* Compute partial dot product */
                    for (int ki_local = 0; ki_local < 16; ki_local++) {
                        int ki = ki_group * 16 + ki_local;
                        if (ki < in_feat) {
                            uint8_t in_u8 = infeat[in_base + ki_local];
                            int8_t w_s8 = (int8_t)((int)w_u8[ki_local] + (int)weight_offset);
                            sw_acc_ch0 += (int64_t)in_u8 * (int64_t)w_s8;
                            input_sum += in_u8;
                        }
                    }
                }
            }
        }
        printf("CL: NE16 3x3 SW_ACC: expected acc[0]=%lld input_sum=%lld\n",
               (long long)sw_acc_ch0, (long long)input_sum);

        /* Also print first few input values and unpacked weights */
        printf("CL: NE16 3x3 DEBUG: infeat[0..9] = ");
        for (int i = 0; i < 10 && i < in_feat; i++) {
            printf("%u ", (unsigned)infeat[i]);
        }
        printf("\n");

        /* Unpack and print first few weights for channel 0 */
        printf("CL: NE16 3x3 DEBUG: w_s8 ch0 (first 16, spatial 0): ");
        int w_base = 0;
        uint8_t w_u8[16] = {0};
        for (int bit = 0; bit < 8; bit++) {
            int bit_offset = w_base + bit * 18;  /* h=0, w=0 */
            uint8_t b0 = weights_packed[bit_offset];
            uint8_t b1 = weights_packed[bit_offset + 1];
            for (int i = 0; i < 8; i++) {
                w_u8[i]     |= ((b0 >> i) & 0x1) << bit;
                w_u8[i + 8] |= ((b1 >> i) & 0x1) << bit;
            }
        }
        for (int i = 0; i < 16; i++) {
            printf("%d ", (int)w_u8[i] + (int)weight_offset);
        }
        printf("\n");
    }
#endif

    /* Commit + trigger */
    ne16_write_cmd(NE16_CMD_TRIGGER, 0);
    ne16_wait_job_done(job_id);
}

/* --- Depthwise 3x3 Convolution (FILTER_MODE = 1) --- */

void ne16_conv3x3_dw_u8_u8_to_s32(
    const uint8_t *infeat,
    const uint8_t *weights_packed,
    int32_t *outfeat,
    int in_w,
    int in_h,
    int channels,
    int pad_h,
    int pad_w,
    int8_t weight_offset
) {
    const int job_id = ne16_acquire_job();

    /* Output dimensions (stride 1, 3x3 kernel) */
    const int out_h = in_h + 2 * pad_h - 2;  /* out_h = in_h - 3 + 1 + 2*pad */
    const int out_w = in_w + 2 * pad_w - 2;

    /* NE16 depthwise: channels are both input and output channels.
     *
     * In depthwise mode, NE16 uses symmetric channel tiling:
     * - Ko subtile size is 16 (same as Ki subtile size), not 32.
     * - The HW requires nb_ko == nb_ki and rem_ko == rem_ki.
     */
    const int nb_ko = ceil_div(channels, 16);
    const int rem_ko = (channels % 16) ? (channels % 16) : 16;

    /* For depthwise: NE16 requires nb_ki == nb_ko and rem_ki == rem_ko
     * (each Ki group maps to exactly one Ko group in depthwise mode) */
    const int nb_ki = nb_ko;
    const int rem_ki = rem_ko;

    /* Spatial tiling */
    const int rem_wo = out_w % 3;
    const int nb_wo = out_w / 3 + (rem_wo ? 1 : 0);
    const int rem_ho = out_h % 3;
    const int nb_ho = out_h / 3 + (rem_ho ? 1 : 0);

    /* For 3x3, input remainder includes halo (same as regular 3x3). */
    const int rem_wi = rem_wo ? (rem_wo + 2) : 0;
    const int rem_hi = rem_ho ? (rem_ho + 2) : 0;

    /* InFeat strides (HWC) */
    const uint32_t in_d0 = (uint32_t)channels;
    const uint32_t in_d1 = (uint32_t)channels * (uint32_t)in_w;
    const uint32_t in_d2 = 0;

    /* OutFeat strides (HWC, 32-bit output) */
    const uint32_t out_bytes = 4;
    const uint32_t out_d0 = NE16_OUTPUT_BANDWIDTH_BYTES;
    const uint32_t out_d1 = out_bytes * (uint32_t)channels;
    const uint32_t out_d2 = out_bytes * (uint32_t)channels * (uint32_t)out_w;

    /* Weight strides for depthwise 3x3 (pulp-nnx / reference-compatible):
     * Layout: [cin_major][bit][h][w][2 bytes] with cin_minor = 16 channels.
     * - w_d0 = 3*3*2 = 18 bytes (stride per bit plane)
     * - w_d1 = 0 (depthwise mode ignores the second weight stride)
     * - w_d2 = 0
     */
    const uint32_t w_d0 = 2u * 3u * 3u;  /* 18 */
    const uint32_t w_d1 = 0;
    const uint32_t w_d2 = 0;

    /* Program pointers */
    ne16_write_reg(NE16_REG_WEIGHTS_PTR, (uint32_t)weights_packed);
    ne16_write_reg(NE16_REG_INFEAT_PTR, (uint32_t)infeat);
    ne16_write_reg(NE16_REG_OUTFEAT_PTR, (uint32_t)outfeat);
    ne16_write_reg(NE16_REG_SCALE_PTR, 0);
    ne16_write_reg(NE16_REG_SCALE_SHIFT_PTR, 0);
    ne16_write_reg(NE16_REG_SCALE_BIAS_PTR, 0);

    /* Program strides */
    ne16_write_reg(NE16_REG_INFEAT_D0_STRIDE, in_d0);
    ne16_write_reg(NE16_REG_INFEAT_D1_STRIDE, in_d1);
    ne16_write_reg(NE16_REG_INFEAT_D2_STRIDE, in_d2);
    ne16_write_reg(NE16_REG_OUTFEAT_D0_STRIDE, out_d0);
    ne16_write_reg(NE16_REG_OUTFEAT_D1_STRIDE, out_d1);
    ne16_write_reg(NE16_REG_OUTFEAT_D2_STRIDE, out_d2);
    ne16_write_reg(NE16_REG_WEIGHTS_D0_STRIDE, w_d0);
    ne16_write_reg(NE16_REG_WEIGHTS_D1_STRIDE, w_d1);
    ne16_write_reg(NE16_REG_WEIGHTS_D2_STRIDE, w_d2);

    /* Reminders */
    ne16_write_reg(NE16_REG_REM_KO_KI,
                   ((uint32_t)(rem_ki & NE16_MASK_REM_KI) << NE16_SHIFT_REM_KI) |
                       ((uint32_t)(rem_ko & NE16_MASK_REM_KO) << NE16_SHIFT_REM_KO));
    ne16_write_reg(NE16_REG_REM_HO_WO,
                   ((uint32_t)(rem_wo & NE16_MASK_REM_WO) << NE16_SHIFT_REM_WO) |
                       ((uint32_t)(rem_ho & NE16_MASK_REM_HO) << NE16_SHIFT_REM_HO));
    ne16_write_reg(NE16_REG_REM_HI_WI,
                   ((uint32_t)(rem_wi & NE16_MASK_REM_WI) << NE16_SHIFT_REM_WI) |
                       ((uint32_t)(rem_hi & NE16_MASK_REM_HI) << NE16_SHIFT_REM_HI));

    /* Dimensions */
    ne16_write_reg(NE16_REG_NB_KO_KI,
                   ((uint32_t)(nb_ki & NE16_MASK_NB_KI) << NE16_SHIFT_NB_KI) |
                       ((uint32_t)(nb_ko & NE16_MASK_NB_KO) << NE16_SHIFT_NB_KO));
    ne16_write_reg(NE16_REG_NB_HO_WO,
                   ((uint32_t)(nb_wo & NE16_MASK_NB_WO) << NE16_SHIFT_NB_WO) |
                       ((uint32_t)(nb_ho & NE16_MASK_NB_HO) << NE16_SHIFT_NB_HO));

    /* Padding register */
    const uint16_t pad_val = 128;
    uint32_t padding = 0;
    padding |= (pad_val & 0xFFFF);
    padding |= ((pad_w & 0xF) << 16);  /* left */
    padding |= ((pad_h & 0xF) << 20);  /* bottom */
    padding |= ((pad_w & 0xF) << 24);  /* right */
    padding |= ((pad_h & 0xF) << 28);  /* top */
    ne16_write_reg(NE16_REG_PADDING, padding);

    /* Filter mask: all 9 positions active */
    ne16_write_reg(NE16_REG_FILTER_MASK, 0);

    /* Weight offset */
    ne16_write_reg(NE16_REG_WEIGHT_OFFSET, (uint32_t)(int32_t)weight_offset);

    /* Config: depthwise 3x3 mode (FILTER_MODE = 1), 8-bit weights, 32-bit output */
    uint32_t cfg = 0;
    cfg |= ((uint32_t)(7 & NE16_MASK_WBITS_M1) << NE16_SHIFT_WBITS_M1);               /* Qw = 8 */
    cfg |= ((uint32_t)(0 & NE16_MASK_MODE16) << NE16_SHIFT_MODE16);
    cfg |= ((uint32_t)(0 & NE16_MASK_OUTQUANT) << NE16_SHIFT_OUTQUANT);               /* streamout only */
    cfg |= ((uint32_t)(1 & NE16_MASK_FILTER_MODE) << NE16_SHIFT_FILTER_MODE);         /* depthwise 3x3 mode */
    cfg |= ((uint32_t)(0 & NE16_MASK_LINEAR_MODE) << NE16_SHIFT_LINEAR_MODE);
    cfg |= ((uint32_t)(0 & NE16_MASK_STRIDED_MODE) << NE16_SHIFT_STRIDED_MODE);
    cfg |= ((uint32_t)(0 & NE16_MASK_NORM_BITS) << NE16_SHIFT_NORM_BITS);
    cfg |= ((uint32_t)(0 & NE16_MASK_STREAMIN) << NE16_SHIFT_STREAMIN);
    cfg |= ((uint32_t)(1 & NE16_MASK_WEIGHT_OFFSET_CFG) << NE16_SHIFT_WEIGHT_OFFSET_CFG);
    cfg |= ((uint32_t)(0 & NE16_MASK_QUANT_RIGHT_SHIFT) << NE16_SHIFT_QUANT_RIGHT_SHIFT);
    cfg |= ((uint32_t)(2 & NE16_MASK_QUANT_BITS) << NE16_SHIFT_QUANT_BITS);           /* 32-bit output */
    cfg |= ((uint32_t)(1 & NE16_MASK_QUANT_NORECT) << NE16_SHIFT_QUANT_NORECT);
    cfg |= ((uint32_t)(0 & NE16_MASK_NORM_SHIFT) << NE16_SHIFT_NORM_SHIFT);
    cfg |= ((uint32_t)(0 & NE16_MASK_NORM_BIAS) << NE16_SHIFT_NORM_BIAS);

    ne16_write_reg(NE16_REG_CONFIG, 0);
    ne16_write_reg(NE16_REG_CONFIG, cfg);

#if defined(ARES_NE16_DEBUG) && !defined(MINIMAL_OUTPUT)
    printf("CL: NE16 DW 3x3 DEBUG: in=%dx%dx%d out=%dx%dx%d pad=%d,%d\n",
           in_h, in_w, channels, out_h, out_w, channels, pad_h, pad_w);
    printf("CL: NE16 DW 3x3 DEBUG: nb_ko=%d rem_ko=%d\n", nb_ko, rem_ko);
    printf("CL: NE16 DW 3x3 DEBUG: cfg=0x%08x weight_offset=%d\n", cfg, (int)weight_offset);
#endif

    /* Commit + trigger */
    ne16_write_cmd(NE16_CMD_TRIGGER, 0);
    ne16_wait_job_done(job_id);
}

void ne16_conv3x3_dw_u8_u8_to_s8(
    const uint8_t *infeat,
    const uint8_t *weights_packed,
    const int32_t *bias,
    const uint8_t *scale,
    const uint8_t *scale_shift,
    int8_t *outfeat,
    int in_w,
    int in_h,
    int channels,
    int pad_h,
    int pad_w,
    int8_t weight_offset
) {
    const int job_id = ne16_acquire_job();

    /* Output dimensions (stride 1, 3x3 kernel) */
    const int out_h = in_h + 2 * pad_h - 2;
    const int out_w = in_w + 2 * pad_w - 2;

    /* NE16 depthwise channel processing (Ko/Ki are both 16-wide subtiles). */
    const int nb_ko = ceil_div(channels, 16);
    const int rem_ko = (channels % 16) ? (channels % 16) : 16;
    const int nb_ki = nb_ko;
    const int rem_ki = rem_ko;

    /* Spatial tiling */
    const int rem_wo = out_w % 3;
    const int nb_wo = out_w / 3 + (rem_wo ? 1 : 0);
    const int rem_ho = out_h % 3;
    const int nb_ho = out_h / 3 + (rem_ho ? 1 : 0);
    const int rem_wi = rem_wo ? (rem_wo + 2) : 0;
    const int rem_hi = rem_ho ? (rem_ho + 2) : 0;

    /* Strides */
    const uint32_t in_d0 = (uint32_t)channels;
    const uint32_t in_d1 = (uint32_t)channels * (uint32_t)in_w;
    const uint32_t in_d2 = 0;

    /* Output is INT8 now */
    const uint32_t out_bytes = 1;
    const uint32_t out_d0 = NE16_OUTPUT_BANDWIDTH_BYTES;
    const uint32_t out_d1 = out_bytes * (uint32_t)channels;
    const uint32_t out_d2 = out_bytes * (uint32_t)channels * (uint32_t)out_w;

    const uint32_t w_d0 = 2u * 3u * 3u;  /* 18 */
    const uint32_t w_d1 = 0;
    const uint32_t w_d2 = 0;

    /* Program pointers */
    ne16_write_reg(NE16_REG_WEIGHTS_PTR, (uint32_t)weights_packed);
    ne16_write_reg(NE16_REG_INFEAT_PTR, (uint32_t)infeat);
    ne16_write_reg(NE16_REG_OUTFEAT_PTR, (uint32_t)outfeat);
    ne16_write_reg(NE16_REG_SCALE_PTR, (uint32_t)scale);
    ne16_write_reg(NE16_REG_SCALE_SHIFT_PTR, (uint32_t)scale_shift);
    ne16_write_reg(NE16_REG_SCALE_BIAS_PTR, (uint32_t)bias);

    /* Program strides */
    ne16_write_reg(NE16_REG_INFEAT_D0_STRIDE, in_d0);
    ne16_write_reg(NE16_REG_INFEAT_D1_STRIDE, in_d1);
    ne16_write_reg(NE16_REG_INFEAT_D2_STRIDE, in_d2);
    ne16_write_reg(NE16_REG_OUTFEAT_D0_STRIDE, out_d0);
    ne16_write_reg(NE16_REG_OUTFEAT_D1_STRIDE, out_d1);
    ne16_write_reg(NE16_REG_OUTFEAT_D2_STRIDE, out_d2);
    ne16_write_reg(NE16_REG_WEIGHTS_D0_STRIDE, w_d0);
    ne16_write_reg(NE16_REG_WEIGHTS_D1_STRIDE, w_d1);
    ne16_write_reg(NE16_REG_WEIGHTS_D2_STRIDE, w_d2);

    /* Reminders and dimensions */
    ne16_write_reg(NE16_REG_REM_KO_KI,
                   ((uint32_t)(rem_ki & NE16_MASK_REM_KI) << NE16_SHIFT_REM_KI) |
                       ((uint32_t)(rem_ko & NE16_MASK_REM_KO) << NE16_SHIFT_REM_KO));
    ne16_write_reg(NE16_REG_REM_HO_WO,
                   ((uint32_t)(rem_wo & NE16_MASK_REM_WO) << NE16_SHIFT_REM_WO) |
                       ((uint32_t)(rem_ho & NE16_MASK_REM_HO) << NE16_SHIFT_REM_HO));
    ne16_write_reg(NE16_REG_REM_HI_WI,
                   ((uint32_t)(rem_wi & NE16_MASK_REM_WI) << NE16_SHIFT_REM_WI) |
                       ((uint32_t)(rem_hi & NE16_MASK_REM_HI) << NE16_SHIFT_REM_HI));
    ne16_write_reg(NE16_REG_NB_KO_KI,
                   ((uint32_t)(nb_ki & NE16_MASK_NB_KI) << NE16_SHIFT_NB_KI) |
                       ((uint32_t)(nb_ko & NE16_MASK_NB_KO) << NE16_SHIFT_NB_KO));
    ne16_write_reg(NE16_REG_NB_HO_WO,
                   ((uint32_t)(nb_wo & NE16_MASK_NB_WO) << NE16_SHIFT_NB_WO) |
                       ((uint32_t)(nb_ho & NE16_MASK_NB_HO) << NE16_SHIFT_NB_HO));

    /* Padding */
    const uint16_t pad_val = 128;
    uint32_t padding_reg = 0;
    padding_reg |= (pad_val & 0xFFFF);
    padding_reg |= ((pad_w & 0xF) << 16);
    padding_reg |= ((pad_h & 0xF) << 20);
    padding_reg |= ((pad_w & 0xF) << 24);
    padding_reg |= ((pad_h & 0xF) << 28);
    ne16_write_reg(NE16_REG_PADDING, padding_reg);

    ne16_write_reg(NE16_REG_FILTER_MASK, 0);
    ne16_write_reg(NE16_REG_WEIGHT_OFFSET, (uint32_t)(int32_t)weight_offset);

    /* Config: depthwise 3x3 mode, 8-bit weights, HW requantization to INT8 */
    uint32_t cfg = 0;
    cfg |= ((uint32_t)(7 & NE16_MASK_WBITS_M1) << NE16_SHIFT_WBITS_M1);
    cfg |= ((uint32_t)(0 & NE16_MASK_MODE16) << NE16_SHIFT_MODE16);
    cfg |= ((uint32_t)(1 & NE16_MASK_OUTQUANT) << NE16_SHIFT_OUTQUANT);               /* HW quant enabled */
    cfg |= ((uint32_t)(1 & NE16_MASK_FILTER_MODE) << NE16_SHIFT_FILTER_MODE);         /* depthwise 3x3 */
    cfg |= ((uint32_t)(0 & NE16_MASK_LINEAR_MODE) << NE16_SHIFT_LINEAR_MODE);
    cfg |= ((uint32_t)(0 & NE16_MASK_STRIDED_MODE) << NE16_SHIFT_STRIDED_MODE);
    cfg |= ((uint32_t)(0 & NE16_MASK_NORM_BITS) << NE16_SHIFT_NORM_BITS);
    cfg |= ((uint32_t)(0 & NE16_MASK_STREAMIN) << NE16_SHIFT_STREAMIN);
    cfg |= ((uint32_t)(1 & NE16_MASK_WEIGHT_OFFSET_CFG) << NE16_SHIFT_WEIGHT_OFFSET_CFG);
    cfg |= ((uint32_t)(0 & NE16_MASK_QUANT_RIGHT_SHIFT) << NE16_SHIFT_QUANT_RIGHT_SHIFT);
    cfg |= ((uint32_t)(0 & NE16_MASK_QUANT_BITS) << NE16_SHIFT_QUANT_BITS);           /* 8-bit output */
    cfg |= ((uint32_t)(1 & NE16_MASK_QUANT_NORECT) << NE16_SHIFT_QUANT_NORECT);       /* ReLU disabled (NORECT=1) */
    cfg |= ((uint32_t)(1 & NE16_MASK_NORM_SHIFT) << NE16_SHIFT_NORM_SHIFT);
    cfg |= ((uint32_t)(1 & NE16_MASK_NORM_BIAS) << NE16_SHIFT_NORM_BIAS);

    ne16_write_reg(NE16_REG_CONFIG, 0);
    ne16_write_reg(NE16_REG_CONFIG, cfg);

    /* Commit + trigger */
    ne16_write_cmd(NE16_CMD_TRIGGER, 0);
    ne16_wait_job_done(job_id);
}

#endif /* ARES_USE_NE16 */
