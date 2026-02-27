/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Network DMA Pipeline - Tiled Execution with Double Buffering
 *
 * This file implements DMA-based tiled execution for neural network layers
 * on the GAP9 platform. Functions are organized into logical sections.
 *
 * MODULAR STRUCTURE (for future extraction into separate files):
 *   - dma_transfer.c (lines ~27-140): Generic DMA dispatch utilities
 *   - pipeline_conv2d.c (lines ~187-1036): Conv2D tiling pipeline
 *   - pipeline_linear.c (lines ~1063-1920): Linear tiling pipeline
 *   - pipeline_mhsa.c (lines ~2099-4619): MHSA tiling pipeline
 *   - pipeline_helpers.c (scattered): L2 fallback, helper utilities
 *

 */

#include "network_dma_pipeline.h"
#include "dma_async_contract.h"
#include "network_kernels.h"
#include "tile_buffer_manager.h"
#include <stdio.h>
#include <string.h>
#include <pmsis.h>


// Maximum DMA descriptors per pipeline (see `codegen/runtime/inc/ares_config.h`)

// Global DMA descriptor arrays (shared across layers)
extern pi_cl_dma_copy_t g_load_dma_descs[MAX_DMA_DESCRIPTORS];
extern pi_cl_dma_copy_t g_store_dma_descs[MAX_DMA_DESCRIPTORS];

// CL-DMA descriptor size field is 16-bit on Siracusa/GAP SDK.
// Large transfers must be split explicitly, otherwise size truncation corrupts data.
#define CL_DMA_MAX_COPY_BYTES 65535u

static inline void cl_dma_memcpy_1d_sync_chunked(
    pi_cl_dma_dir_e dir,
    uint32_t ext_addr,
    uint32_t loc_addr,
    size_t bytes
) {
    while (bytes > 0) {
        uint16_t chunk = (bytes > CL_DMA_MAX_COPY_BYTES) ? (uint16_t)CL_DMA_MAX_COPY_BYTES : (uint16_t)bytes;
        pi_cl_dma_copy_t copy = {
            .ext = ext_addr,
            .loc = loc_addr,
            .id = 0,
            .size = chunk,
            .dir = dir,
            .merge = 0,
            .stride = 0,
            .length = 0
        };
        pi_cl_dma_memcpy(&copy);
        pi_cl_dma_wait(&copy);
        ext_addr += chunk;
        loc_addr += chunk;
        bytes -= chunk;
    }
}

// ---
// Generic Tile Mover Implementation
// ---

void execute_dma_transfer(dma_layout_t *src, dma_layout_t *dst) {
    if (!src || !dst) return;

    // Determine direction and type
    int is_l2_l1 = (src->loc == MEM_LOC_L2 && dst->loc == MEM_LOC_L1) ||
                   (src->loc == MEM_LOC_L1 && dst->loc == MEM_LOC_L2);
    int is_l3_l2 = (src->loc == MEM_LOC_L3 && dst->loc == MEM_LOC_L2) ||
                   (src->loc == MEM_LOC_L2 && dst->loc == MEM_LOC_L3);
    int is_l2_l2 = (src->loc == MEM_LOC_L2 && dst->loc == MEM_LOC_L2);

    if (src->width != dst->width || src->height != dst->height || src->channels != dst->channels) {
        return;
    }

    if (is_l2_l2) {
        // L2→L2 copy (simple memcpy, both in cluster shared memory)
        for (int c = 0; c < src->channels; c++) {
            uint32_t src_addr = (uint32_t)src->base_addr + c * src->stride_channel;
            uint32_t dst_addr = (uint32_t)dst->base_addr + c * dst->stride_channel;

            if (src->height > 1) {
                for (int h = 0; h < src->height; h++) {
                    uint32_t s = src_addr + h * src->stride_row;
                    uint32_t d = dst_addr + h * dst->stride_row;
                    memcpy((void*)d, (void*)s, src->width);
                }
            } else {
                memcpy((void*)dst_addr, (void*)src_addr, src->width);
            }
        }
    }
    else if (is_l2_l1) {
        // Cluster DMA (L2 <-> L1)
        pi_cl_dma_dir_e dir = (src->loc == MEM_LOC_L2) ? PI_CL_DMA_DIR_EXT2LOC : PI_CL_DMA_DIR_LOC2EXT;
        for (int c = 0; c < src->channels; c++) {
            uint32_t src_addr = (uint32_t)src->base_addr + c * src->stride_channel;
            uint32_t dst_addr = (uint32_t)dst->base_addr + c * dst->stride_channel;
            pi_cl_dma_copy_t *cmd = src->dma_cmd ? src->dma_cmd : &g_load_dma_descs[0];
            
            cmd->dir = dir;
            cmd->merge = 0;
            cmd->size = src->width;
            cmd->id = 0; 
            
            if (src->height > 1) {
                for (int h = 0; h < src->height; h++) {
                    uint32_t s = src_addr + h * src->stride_row;
                    uint32_t d = dst_addr + h * dst->stride_row;
                    cmd->ext = (dir == PI_CL_DMA_DIR_EXT2LOC) ? s : d;
                    cmd->loc = (dir == PI_CL_DMA_DIR_EXT2LOC) ? d : s;
                    pi_cl_dma_memcpy(cmd);
                    if (!src->is_async) pi_cl_dma_wait(cmd);
                }
            } else {
                cmd->ext = (dir == PI_CL_DMA_DIR_EXT2LOC) ? src_addr : dst_addr;
                cmd->loc = (dir == PI_CL_DMA_DIR_EXT2LOC) ? dst_addr : src_addr;
                pi_cl_dma_memcpy(cmd);
                if (!src->is_async) pi_cl_dma_wait(cmd);
            }
        }
    } 
    else if (is_l3_l2) {
        // RAM Copy (L3 <-> L2)
        int is_read = (src->loc == MEM_LOC_L3);

        // Use valid RAM device handle from the struct
        struct pi_device *ram_dev = src->ram_dev ? src->ram_dev : dst->ram_dev;
        if (!ram_dev) {
            return;
        }

        for (int c = 0; c < src->channels; c++) {
            uint32_t l3_addr = is_read ? ((uint32_t)src->base_addr + c * src->stride_channel) 
                                       : ((uint32_t)dst->base_addr + c * dst->stride_channel);
            uint32_t l2_addr = is_read ? ((uint32_t)dst->base_addr + c * dst->stride_channel)
                                       : ((uint32_t)src->base_addr + c * src->stride_channel);
            
            //  Check BOTH src and dst for the command handle
            pi_cl_ram_req_t *req = src->ram_cmd ? src->ram_cmd : dst->ram_cmd;
            pi_cl_ram_req_t stack_req;
            if (!req) req = &stack_req;

            if (src->height > 1) {
                 for (int h = 0; h < src->height; h++) {
                     uint32_t s_l3 = l3_addr + h * (is_read ? src->stride_row : dst->stride_row);
                     uint32_t s_l2 = l2_addr + h * (is_read ? dst->stride_row : src->stride_row);

                     if (is_read) {
                         pi_cl_ram_read(ram_dev, (uint32_t)s_l3, (void*)s_l2, src->width, req);
                         if (!src->is_async) pi_cl_ram_read_wait(req);
                     } else {
                         pi_cl_ram_write(ram_dev, (uint32_t)s_l3, (void*)s_l2, src->width, req);
                         //  Use correct write wait
                         if (!src->is_async) pi_cl_ram_write_wait(req);
                     }
                 }
            } else {
                 if (is_read) {
                     pi_cl_ram_read(ram_dev, (uint32_t)l3_addr, (void*)l2_addr, src->width, req);
                     if (!src->is_async) pi_cl_ram_read_wait(req);
                 } else {
                     pi_cl_ram_write(ram_dev, (uint32_t)l3_addr, (void*)l2_addr, src->width, req);
                     if (!src->is_async) pi_cl_ram_write_wait(req);
                 }
            }
        }
    }
}

// --- Helper Kernels ---
static inline void relu_int8_inplace_l1(int8_t *data, size_t size) {
    size_t i = pi_core_id();
    for (; i < size; i += NUM_CORES) if (data[i] < 0) data[i] = 0;
}
typedef struct { int8_t *data; size_t size; } relu_l1_args_t;
static void relu_l1_worker(void *arg) { relu_l1_args_t *a = (relu_l1_args_t *)arg; relu_int8_inplace_l1(a->data, a->size); }

static inline void requantize_int8_inplace_l1(int8_t *data, size_t size, float scale_in, float scale_out) {
    if (fabsf(scale_in - scale_out) < 1e-12f || pi_core_id() != 0) return;
    for (size_t i = 0; i < size; i++) {
        float val_fp32 = (float)data[i] * scale_in;
        int32_t val_int32 = (int32_t)lrintf(val_fp32 / scale_out);
        if (val_int32 > 127) val_int32 = 127; else if (val_int32 < -128) val_int32 = -128;
        data[i] = (int8_t)val_int32;
    }
}
// ---
// MCHAN Helper for Multi-Channel Tile Transfers (EXPERIMENTAL - DISABLED)
// ---
// STATUS: This path has persistent out-of-bounds addressing issues and is
//         disabled by default. The row-batched baseline path (below) works
//         correctly and should be used for production.
//
// INTENDED BEHAVIOR: When USE_MCHAN_LOC_STRIDE is defined, use MCHAN v7's
// local 2D transfer support to bundle all channel rows for a tile under one
// transaction ID, avoiding per-row waits.
//
// KNOWN ISSUES: MCHAN hardware setup causes out-of-bounds L1 accesses at
// address 0x2ccc. Likely issues with register write order, stride encoding,
// or dual 2D flag interpretation. Needs hardware documentation review.
//
// INFRASTRUCTURE: All #ifdef branches are in place. Once the helper function
// is corrected, the MCHAN path should work immediately.
// ---

#ifdef USE_MCHAN_LOC_STRIDE

// Include HAL for low-level DMA register access
#include "pmsis/implem/hal/hal.h"

// DMA command bit definitions (from SDK cluster_dma.h)
#define DMA_CMD_LEN_SHIFT      0
#define DMA_CMD_TYPE_SHIFT     17   // 0=L1->L2, 1=L2->L1
#define DMA_CMD_INC_SHIFT      18   // 1=incremental
#define DMA_CMD_2D_EXT_SHIFT   19   // 1=2D on EXT (L2) side
#define DMA_CMD_ELE_SHIFT      20   // 1=event enable
#define DMA_CMD_ILE_SHIFT      21   // interrupt enable
#define DMA_CMD_BLE_SHIFT      22   // broadcast enable
#define DMA_CMD_2D_TCDM_SHIFT  23   // 1=2D on TCDM (L1) side

// Structure to track 2D DMA command for waiting
typedef struct {
    int tid;
} dma_2d_cmd_t;

// Low-level dual 2D DMA transfer using HAL
// Transfers a rectangle with stride on BOTH L1 and L2 sides
static inline void dma_cmd_2d_dual(
    uint32_t ext,           // L2 address
    uint32_t loc,           // L1 address
    uint32_t total_size,    // Total bytes = rows * row_length
    uint32_t ext_stride,    // Stride between rows in L2
    uint32_t ext_length,    // Bytes per row (same as row_length)
    uint32_t tcdm_stride,   // Stride between rows in L1
    uint32_t tcdm_length,   // Bytes per row in L1 (usually same as ext_length)
    pi_cl_dma_dir_e dir,
    dma_2d_cmd_t *cmd)
{
    // Get transfer ID
    cmd->tid = hal_cl_dma_cmd_get(CL_DMA_ID(0)) & 0xFFFF;

    // Build command with both 2D flags enabled
    uint32_t dma_cmd = (total_size << DMA_CMD_LEN_SHIFT) |
                       (dir << DMA_CMD_TYPE_SHIFT) |
                       (1 << DMA_CMD_INC_SHIFT) |       // incremental
                       (1 << DMA_CMD_2D_EXT_SHIFT) |    // 2D on L2
                       (1 << DMA_CMD_ELE_SHIFT) |       // event enable
                       (0 << DMA_CMD_ILE_SHIFT) |       // no interrupt
                       (1 << DMA_CMD_BLE_SHIFT) |       // broadcast
                       (1 << DMA_CMD_2D_TCDM_SHIFT);    // 2D on L1

    // Push command sequence: CMD, LOC, EXT, EXT_LENGTH, EXT_STRIDE, TCDM_LENGTH, TCDM_STRIDE
    hal_cl_dma_cmd_set(CL_DMA_ID(0), dma_cmd);
    hal_cl_dma_cmd_set(CL_DMA_ID(0), loc);
    hal_cl_dma_cmd_set(CL_DMA_ID(0), ext);
    hal_cl_dma_cmd_set(CL_DMA_ID(0), ext_length);
    hal_cl_dma_cmd_set(CL_DMA_ID(0), ext_stride);
    hal_cl_dma_cmd_set(CL_DMA_ID(0), tcdm_length);
    hal_cl_dma_cmd_set(CL_DMA_ID(0), tcdm_stride);
}

// Wait for dual 2D DMA command to complete
static inline void dma_cmd_2d_dual_wait(dma_2d_cmd_t *cmd)
{
    while (hal_cl_dma_status_get(CL_DMA_ID(0)) & (1 << cmd->tid)) {
        hal_eu_evt_mask_wait_and_clr(1 << CL_IRQ_DMA0);
    }
    hal_cl_dma_status_set(CL_DMA_ID(0), 1 << cmd->tid);  // Free TID
}

// Transfer a 2D tile for a single channel using dual 2D DMA
// Returns immediately; caller must wait using dma_cmd_2d_dual_wait
static void dma_tile_transfer_2d_channel(
    int8_t *l1_base, int8_t *l2_base,
    int valid_h, int valid_w,
    int l1_row_stride, int l2_row_stride,
    pi_cl_dma_dir_e dir,
    dma_2d_cmd_t *cmd)
{
    uint32_t total_size = valid_h * valid_w;
    dma_cmd_2d_dual(
        (uint32_t)l2_base,    // ext
        (uint32_t)l1_base,    // loc
        total_size,           // total bytes
        l2_row_stride,        // L2 stride
        valid_w,              // L2 row length
        l1_row_stride,        // L1 stride
        valid_w,              // L1 row length
        dir,
        cmd
    );
}

#endif // USE_MCHAN_LOC_STRIDE

// ---
// Conv2D Inner Loop (L2 -> L1) - Look-Ahead/Look-Behind Pattern
// ---
// Pattern: Load next tile, store prev tile, fork compute with fusion, wait DMAs, swap
// DEFAULT: Uses row-batched pi_cl_dma_memcpy (supports both L2 and L1 strides)
// OPTIONAL: With USE_MCHAN_LOC_STRIDE, uses MCHAN v7 local 2D transfers
// Main benefit: Fusion inside worker eliminates separate fork overhead
// ---

static void conv2d_tiled_l1_inner_loop(conv2d_pipeline_config_t *cfg) {
    tile_buffer_mgr_t buf_mgr;
    tile_buffer_init(&buf_mgr, cfg->l1_buffer, cfg->l1_input_size, cfg->l1_output_size);

#ifndef USE_MCHAN_LOC_STRIDE
    // Row-batched path: DMA descriptor arrays for batched transfers
    pi_cl_dma_copy_t load_dmas[MAX_DMA_BATCH];
    pi_cl_dma_copy_t store_dmas[MAX_DMA_BATCH];
#endif
    // Track previous tile's output buffer and geometry for look-behind store
    int8_t *prev_output_l1 = NULL;
    int prev_out_y_start = 0, prev_out_x_start = 0;
    int prev_actual_out_h = 0, prev_actual_out_w = 0;

    // PROLOGUE: Load tile 0 (synchronous)
    {
        int8_t *tile0_input_l1 = tile_buffer_get_input(&buf_mgr);
        memset(tile0_input_l1, 0, cfg->l1_input_size);

        // Calculate tile 0 input geometry
        int out_y_end = (cfg->out_tile_h > cfg->out_h) ? cfg->out_h : cfg->out_tile_h;
        int out_x_end = (cfg->out_tile_w > cfg->out_w) ? cfg->out_w : cfg->out_tile_w;
        int in_y_start = -cfg->pad_h;
        int in_x_start = -cfg->pad_w;
        int in_y_end = in_y_start + cfg->tile_h_halo;
        int in_x_end = in_x_start + cfg->tile_w_halo;
        int valid_in_y_start = (in_y_start < 0) ? 0 : in_y_start;
        int valid_in_x_start = (in_x_start < 0) ? 0 : in_x_start;
        int valid_in_y_end = (in_y_end > cfg->in_h) ? cfg->in_h : in_y_end;
        int valid_in_x_end = (in_x_end > cfg->in_w) ? cfg->in_w : in_x_end;

#ifdef USE_MCHAN_LOC_STRIDE
        // Dual 2D DMA path: Each channel as a single 2D transfer
        int valid_h = valid_in_y_end - valid_in_y_start;
        int valid_w = valid_in_x_end - valid_in_x_start;
        if (valid_h > 0 && valid_w > 0) {
            int l1_y_offset = valid_in_y_start - in_y_start;
            int l1_x_offset = valid_in_x_start - in_x_start;

            // Transfer each channel using dual 2D DMA (2D on both L1 and L2)
            for (int c = 0; c < cfg->in_ch; c++) {
                int8_t *l1_chan_base = tile0_input_l1 + c * cfg->tile_h_halo * cfg->tile_w_halo
                                       + l1_y_offset * cfg->tile_w_halo + l1_x_offset;
                int8_t *l2_chan_base = cfg->input_buffer_l2 + c * cfg->in_h * cfg->in_w
                                       + valid_in_y_start * cfg->in_w + valid_in_x_start;

                dma_2d_cmd_t cmd;
                dma_tile_transfer_2d_channel(
                    l1_chan_base, l2_chan_base,
                    valid_h, valid_w,
                    cfg->tile_w_halo,  // L1 row stride
                    cfg->in_w,         // L2 row stride
                    PI_CL_DMA_DIR_EXT2LOC,
                    &cmd
                );
                dma_cmd_2d_dual_wait(&cmd);
            }
        }
#else
        // Row-batched path: Load tile 0 using row-batched DMA (synchronous)
        for (int c = 0; c < cfg->in_ch; c++) {
            int valid_h = valid_in_y_end - valid_in_y_start;
            int valid_w = valid_in_x_end - valid_in_x_start;
            if (valid_h > 0 && valid_w > 0) {
                int l1_y_offset = valid_in_y_start - in_y_start;
                int l1_x_offset = valid_in_x_start - in_x_start;
                int8_t *l1_chan_base = tile0_input_l1 + c * cfg->tile_h_halo * cfg->tile_w_halo + l1_y_offset * cfg->tile_w_halo + l1_x_offset;
                int8_t *l2_chan_base = cfg->input_buffer_l2 + c * cfg->in_h * cfg->in_w + valid_in_y_start * cfg->in_w + valid_in_x_start;

                // Row-batched transfers (respects 16-counter limit)
                for (int row_start = 0; row_start < valid_h; row_start += MAX_DMA_BATCH) {
                    int batch_size = (row_start + MAX_DMA_BATCH > valid_h) ? (valid_h - row_start) : MAX_DMA_BATCH;

                    // Queue batch
                    for (int i = 0; i < batch_size; i++) {
                        int row = row_start + i;
                        load_dmas[i].dir = PI_CL_DMA_DIR_EXT2LOC;
                        load_dmas[i].size = valid_w;
                        load_dmas[i].ext = (uint32_t)(l2_chan_base + row * cfg->in_w);
                        load_dmas[i].loc = (uint32_t)(l1_chan_base + row * cfg->tile_w_halo);
                        load_dmas[i].merge = 0;
                        load_dmas[i].id = 0;
                        pi_cl_dma_memcpy(&load_dmas[i]);
                    }

                    // Wait batch
                    for (int i = 0; i < batch_size; i++) {
                        pi_cl_dma_wait(&load_dmas[i]);
                    }
                }
            }
        }
#endif // USE_MCHAN_LOC_STRIDE
    }

    // STEADY-STATE LOOP: Look-Ahead/Look-Behind Pattern
    for (int i = 0; i < cfg->num_tiles; i++) {
        int has_load = 0, has_store = 0;

        // Get current tile buffers
        int8_t *curr_input_l1 = tile_buffer_get_input(&buf_mgr);
        int8_t *curr_output_l1 = tile_buffer_get_output(&buf_mgr);

        // Calculate current tile geometry
        int tile_y = i / cfg->num_tiles_w;
        int tile_x = i % cfg->num_tiles_w;
        int out_y_start = tile_y * cfg->out_tile_h;
        int out_x_start = tile_x * cfg->out_tile_w;
        int out_y_end = out_y_start + cfg->out_tile_h;
        int out_x_end = out_x_start + cfg->out_tile_w;
        if (out_y_end > cfg->out_h) out_y_end = cfg->out_h;
        if (out_x_end > cfg->out_w) out_x_end = cfg->out_w;
        int actual_out_tile_h = out_y_end - out_y_start;
        int actual_out_tile_w = out_x_end - out_x_start;

        // --- STEP 1: LOAD NEXT TILE (i+1) ---
        if (i < cfg->num_tiles - 1) {
            int8_t *next_input_l1 = tile_buffer_get_next_input(&buf_mgr);
            memset(next_input_l1, 0, cfg->l1_input_size);

            // Calculate next tile input geometry
            int next_tile_idx = i + 1;
            int next_tile_y = next_tile_idx / cfg->num_tiles_w;
            int next_tile_x = next_tile_idx % cfg->num_tiles_w;
            int next_out_y_start = next_tile_y * cfg->out_tile_h;
            int next_out_x_start = next_tile_x * cfg->out_tile_w;
            int next_in_y_start = next_out_y_start * cfg->stride_h - cfg->pad_h;
            int next_in_x_start = next_out_x_start * cfg->stride_w - cfg->pad_w;
            int next_in_y_end = next_in_y_start + cfg->tile_h_halo;
            int next_in_x_end = next_in_x_start + cfg->tile_w_halo;
            int next_valid_in_y_start = (next_in_y_start < 0) ? 0 : next_in_y_start;
            int next_valid_in_x_start = (next_in_x_start < 0) ? 0 : next_in_x_start;
            int next_valid_in_y_end = (next_in_y_end > cfg->in_h) ? cfg->in_h : next_in_y_end;
            int next_valid_in_x_end = (next_in_x_end > cfg->in_w) ? cfg->in_w : next_in_x_end;

#ifdef USE_MCHAN_LOC_STRIDE
            // Dual 2D DMA path: Each channel as a single 2D transfer
            int valid_h = next_valid_in_y_end - next_valid_in_y_start;
            int valid_w = next_valid_in_x_end - next_valid_in_x_start;
            if (valid_h > 0 && valid_w > 0) {
                int l1_y_offset = next_valid_in_y_start - next_in_y_start;
                int l1_x_offset = next_valid_in_x_start - next_in_x_start;

                // Transfer each channel using dual 2D DMA
                for (int c = 0; c < cfg->in_ch; c++) {
                    int8_t *l1_chan_base = next_input_l1 + c * cfg->tile_h_halo * cfg->tile_w_halo
                                           + l1_y_offset * cfg->tile_w_halo + l1_x_offset;
                    int8_t *l2_chan_base = cfg->input_buffer_l2 + c * cfg->in_h * cfg->in_w
                                           + next_valid_in_y_start * cfg->in_w + next_valid_in_x_start;

                    dma_2d_cmd_t cmd;
                    dma_tile_transfer_2d_channel(
                        l1_chan_base, l2_chan_base,
                        valid_h, valid_w,
                        cfg->tile_w_halo,  // L1 row stride
                        cfg->in_w,         // L2 row stride
                        PI_CL_DMA_DIR_EXT2LOC,
                        &cmd
                    );
                    dma_cmd_2d_dual_wait(&cmd);
                }
                has_load = 1;
            }
#else
            // Row-batched path: Load all channels using row-batched DMA
            for (int c = 0; c < cfg->in_ch; c++) {
                int valid_h = next_valid_in_y_end - next_valid_in_y_start;
                int valid_w = next_valid_in_x_end - next_valid_in_x_start;
                if (valid_h > 0 && valid_w > 0) {
                    int l1_y_offset = next_valid_in_y_start - next_in_y_start;
                    int l1_x_offset = next_valid_in_x_start - next_in_x_start;
                    int8_t *l1_chan_base = next_input_l1 + c * cfg->tile_h_halo * cfg->tile_w_halo + l1_y_offset * cfg->tile_w_halo + l1_x_offset;
                    int8_t *l2_chan_base = cfg->input_buffer_l2 + c * cfg->in_h * cfg->in_w + next_valid_in_y_start * cfg->in_w + next_valid_in_x_start;

                    // Row-batched transfers
                    for (int row_start = 0; row_start < valid_h; row_start += MAX_DMA_BATCH) {
                        int batch_size = (row_start + MAX_DMA_BATCH > valid_h) ? (valid_h - row_start) : MAX_DMA_BATCH;
                        for (int bi = 0; bi < batch_size; bi++) {
                            int row = row_start + bi;
                            load_dmas[bi].dir = PI_CL_DMA_DIR_EXT2LOC;
                            load_dmas[bi].size = valid_w;
                            load_dmas[bi].ext = (uint32_t)(l2_chan_base + row * cfg->in_w);
                            load_dmas[bi].loc = (uint32_t)(l1_chan_base + row * cfg->tile_w_halo);
                            load_dmas[bi].merge = 0;
                            load_dmas[bi].id = 0;
                            pi_cl_dma_memcpy(&load_dmas[bi]);
                        }
                        for (int bi = 0; bi < batch_size; bi++) {
                            pi_cl_dma_wait(&load_dmas[bi]);
                        }
                    }
                }
            }
#endif // USE_MCHAN_LOC_STRIDE
        }

        // --- STEP 2: STORE PREV TILE (i-1) ---
        if (i > 0 && prev_actual_out_h > 0 && prev_actual_out_w > 0) {
#ifdef USE_MCHAN_LOC_STRIDE
            // Dual 2D DMA path: Each channel as a single 2D transfer
            const int prev_l1_plane_size = prev_actual_out_h * prev_actual_out_w;
            for (int c = 0; c < cfg->out_ch; c++) {
                int8_t *l1_chan_base = prev_output_l1 + c * prev_l1_plane_size;
                int8_t *l2_chan_base = cfg->output_buffer_l2 + c * cfg->out_h * cfg->out_w
                                       + prev_out_y_start * cfg->out_w + prev_out_x_start;

                dma_2d_cmd_t cmd;
                dma_tile_transfer_2d_channel(
                    l1_chan_base, l2_chan_base,
                    prev_actual_out_h, prev_actual_out_w,
                    prev_actual_out_w,  // L1 row stride (packed)
                    cfg->out_w,         // L2 row stride
                    PI_CL_DMA_DIR_LOC2EXT,
                    &cmd
                );
                dma_cmd_2d_dual_wait(&cmd);
            }
            has_store = 1;
#else
            // Row-batched path: Store all channels using row-batched DMA
            const int prev_l1_plane_size = prev_actual_out_h * prev_actual_out_w;
            for (int c = 0; c < cfg->out_ch; c++) {
                int8_t *l1_chan_base = prev_output_l1 + c * prev_l1_plane_size;
                int8_t *l2_chan_base = cfg->output_buffer_l2 + c * cfg->out_h * cfg->out_w + prev_out_y_start * cfg->out_w + prev_out_x_start;

                // Row-batched transfers
                for (int row_start = 0; row_start < prev_actual_out_h; row_start += MAX_DMA_BATCH) {
                    int batch_size = (row_start + MAX_DMA_BATCH > prev_actual_out_h) ? (prev_actual_out_h - row_start) : MAX_DMA_BATCH;
                    for (int bi = 0; bi < batch_size; bi++) {
                        int row = row_start + bi;
                        store_dmas[bi].dir = PI_CL_DMA_DIR_LOC2EXT;
                        store_dmas[bi].size = prev_actual_out_w;
                        store_dmas[bi].ext = (uint32_t)(l2_chan_base + row * cfg->out_w);
                        store_dmas[bi].loc = (uint32_t)(l1_chan_base + row * prev_actual_out_w);
                        store_dmas[bi].merge = 0;
                        store_dmas[bi].id = 0;
                        pi_cl_dma_memcpy(&store_dmas[bi]);
                    }
                    for (int bi = 0; bi < batch_size; bi++) {
                        pi_cl_dma_wait(&store_dmas[bi]);
                    }
                }
            }
#endif // USE_MCHAN_LOC_STRIDE
        }

        // --- STEP 3: COMPUTE CURRENT TILE (i) with fusion ---
        conv2d_tile_args_t curr_args = {
            .tile_input_l1 = curr_input_l1, .tile_output_l1 = curr_output_l1,
            .weights_l2 = cfg->weight_l2, .bias_l2 = cfg->bias_l2,
            .tile_in_h = cfg->tile_h_halo, .tile_in_w = cfg->tile_w_halo,
            .tile_out_h = actual_out_tile_h, .tile_out_w = actual_out_tile_w,
            .in_ch = cfg->in_ch, .out_ch = cfg->out_ch,
            .groups = cfg->groups,
            .kernel_h = cfg->kernel_h, .kernel_w = cfg->kernel_w,
            .stride_h = cfg->stride_h, .stride_w = cfg->stride_w,
            .pad_h = 0, .pad_w = 0,  // Halo already included
            .scale_input = cfg->scale_input, .scale_weight = cfg->scale_weight, .scale_output = cfg->scale_output,
            .cluster_dev = cfg->cluster_dev,
            .fusion_relu = cfg->fusion_relu, .fusion_quant = cfg->fusion_quant,
            .quant_scale_in = cfg->quant_scale_in, .quant_scale_out = cfg->quant_scale_out,
            .layout = (uint8_t)cfg->layout
        };

        // Fork compute with fusion inside worker
#ifdef ENABLE_PERF_COUNTERS
        perf_compute_start();
#endif
        pi_cl_team_fork(NUM_CORES, conv2d_tile_worker_with_fusion, &curr_args);
#ifdef ENABLE_PERF_COUNTERS
        if (cfg->perf_counter) {
            cfg->perf_counter->compute_cycles += perf_compute_end();
        }
#endif

        // --- STEP 4: WAIT ON ASYNC DMAs ---
        // Both paths now wait synchronously, so no additional wait needed
        (void)has_load;
        (void)has_store;

        // --- STEP 5: SWAP BUFFERS & UPDATE STATE ---
        // Save current tile's output info for next iteration's store
        prev_output_l1 = curr_output_l1;
        prev_out_y_start = out_y_start;
        prev_out_x_start = out_x_start;
        prev_actual_out_h = actual_out_tile_h;
        prev_actual_out_w = actual_out_tile_w;

        // Swap buffers for next iteration
        if (i < cfg->num_tiles - 1) {
            tile_buffer_swap(&buf_mgr);
        }
    }

    // EPILOGUE: Store last tile
    if (prev_actual_out_h > 0 && prev_actual_out_w > 0) {
#ifdef USE_MCHAN_LOC_STRIDE
        // Dual 2D DMA path: Each channel as a single 2D transfer
        const int prev_l1_plane_size = prev_actual_out_h * prev_actual_out_w;
        for (int c = 0; c < cfg->out_ch; c++) {
            int8_t *l1_chan_base = prev_output_l1 + c * prev_l1_plane_size;
            int8_t *l2_chan_base = cfg->output_buffer_l2 + c * cfg->out_h * cfg->out_w
                                   + prev_out_y_start * cfg->out_w + prev_out_x_start;

            dma_2d_cmd_t cmd;
            dma_tile_transfer_2d_channel(
                l1_chan_base, l2_chan_base,
                prev_actual_out_h, prev_actual_out_w,
                prev_actual_out_w,  // L1 row stride (packed)
                cfg->out_w,         // L2 row stride
                PI_CL_DMA_DIR_LOC2EXT,
                &cmd
            );
            dma_cmd_2d_dual_wait(&cmd);
        }
#else
        // Row-batched path: Store all channels using row-batched DMA
        const int prev_l1_plane_size = prev_actual_out_h * prev_actual_out_w;
        for (int c = 0; c < cfg->out_ch; c++) {
            int8_t *l1_chan_base = prev_output_l1 + c * prev_l1_plane_size;
            int8_t *l2_chan_base = cfg->output_buffer_l2 + c * cfg->out_h * cfg->out_w + prev_out_y_start * cfg->out_w + prev_out_x_start;

            // Row-batched transfers
            for (int row_start = 0; row_start < prev_actual_out_h; row_start += MAX_DMA_BATCH) {
                int batch_size = (row_start + MAX_DMA_BATCH > prev_actual_out_h) ? (prev_actual_out_h - row_start) : MAX_DMA_BATCH;
                for (int bi = 0; bi < batch_size; bi++) {
                    int row = row_start + bi;
                    store_dmas[bi].dir = PI_CL_DMA_DIR_LOC2EXT;
                    store_dmas[bi].size = prev_actual_out_w;
                    store_dmas[bi].ext = (uint32_t)(l2_chan_base + row * cfg->out_w);
                    store_dmas[bi].loc = (uint32_t)(l1_chan_base + row * prev_actual_out_w);
                    store_dmas[bi].merge = 0;
                    store_dmas[bi].id = 0;
                    pi_cl_dma_memcpy(&store_dmas[bi]);
                }
                for (int bi = 0; bi < batch_size; bi++) {
                    pi_cl_dma_wait(&store_dmas[bi]);
                }
            }
        }
#endif // USE_MCHAN_LOC_STRIDE
    }
}

// ---
// Conv2D inner loop with L1 weight caching
// ---
// This version caches weights in L1 alongside input/output tiles.
// Loop structure:
//   FOR each spatial tile:
//     Load input tile to L1
//     Load first weight tile to L1 (sync)
//     FOR each output channel tile:
//       ASYNC: Load next weight tile to L1 buffer B
//       COMPUTE: Process current channel tile with buffer A
//       WAIT: for async weight load
//       SWAP: weight buffers A <-> B
//     Store output tile from L1

static void conv2d_tiled_l1_inner_loop_with_weights(conv2d_pipeline_config_t *cfg) {
#ifndef MINIMAL_OUTPUT
    printf("CL: inner_loop_with_weights: layer=%s in_ch=%d out_ch=%d tile_out_ch=%d num_out_ch_tiles=%d\n",
           cfg->layer_name ? cfg->layer_name : "?", cfg->in_ch, cfg->out_ch, cfg->tile_out_ch, cfg->num_out_ch_tiles);
    printf("CL: inner_loop_with_weights: l1_input_size=%zu l1_output_size=%zu l1_weight_size=%zu\n",
           cfg->l1_input_size, cfg->l1_output_size, cfg->l1_weight_size);
#endif

    tile_buffer_conv2d_weight_mgr_t buf_mgr;
    tile_buffer_conv2d_weight_init(&buf_mgr, cfg->l1_buffer,
                                   cfg->l1_input_size, cfg->l1_output_size, cfg->l1_weight_size);

    pi_cl_dma_copy_t load_dmas[MAX_DMA_BATCH];
    pi_cl_dma_copy_t store_dmas[MAX_DMA_BATCH];
    pi_cl_dma_copy_t weight_copy;

    // Track previous tile's output buffer and geometry for look-behind store
    int8_t *prev_output_l1 = NULL;
    int prev_out_y_start = 0, prev_out_x_start = 0;
    int prev_actual_out_h = 0, prev_actual_out_w = 0;

    // Calculate output dimensions for the slab
    int out_slab_h = (cfg->l3_tiling_enabled) ? cfg->out_h : cfg->out_h;

    // PROLOGUE: Load first spatial tile (tile 0)
    {
        int8_t *tile0_input_l1 = tile_buffer_conv2d_weight_get_input(&buf_mgr);
        memset(tile0_input_l1, 0, cfg->l1_input_size);

        // Calculate tile 0 input geometry
        int out_y_end = (cfg->out_tile_h > cfg->out_h) ? cfg->out_h : cfg->out_tile_h;
        int out_x_end = (cfg->out_tile_w > cfg->out_w) ? cfg->out_w : cfg->out_tile_w;
        int in_y_start = -cfg->pad_h;
        int in_x_start = -cfg->pad_w;
        int in_y_end = in_y_start + cfg->tile_h_halo;
        int in_x_end = in_x_start + cfg->tile_w_halo;
        int valid_in_y_start = (in_y_start < 0) ? 0 : in_y_start;
        int valid_in_x_start = (in_x_start < 0) ? 0 : in_x_start;
        int valid_in_y_end = (in_y_end > cfg->in_h) ? cfg->in_h : in_y_end;
        int valid_in_x_end = (in_x_end > cfg->in_w) ? cfg->in_w : in_x_end;

        // Load tile 0 input using row-batched DMA
        int valid_h = valid_in_y_end - valid_in_y_start;
        int valid_w = valid_in_x_end - valid_in_x_start;
        if (valid_h > 0 && valid_w > 0) {
            int l1_y_offset = valid_in_y_start - in_y_start;
            int l1_x_offset = valid_in_x_start - in_x_start;

            if (cfg->layout == 1) {  // LAYOUT_HWC
                // HWC: rows of (W * C) bytes contiguous
                // L2 HWC: input[y * in_w * in_ch + x * in_ch + c]
                // L1 HWC: tile[y * tile_w_halo * in_ch + x * in_ch + c]
                int8_t *l1_base = tile0_input_l1 + l1_y_offset * cfg->tile_w_halo * cfg->in_ch + l1_x_offset * cfg->in_ch;
                int8_t *l2_base = cfg->input_buffer_l2 + valid_in_y_start * cfg->in_w * cfg->in_ch + valid_in_x_start * cfg->in_ch;

                for (int row_start = 0; row_start < valid_h; row_start += MAX_DMA_BATCH) {
                    int batch_size = (row_start + MAX_DMA_BATCH > valid_h) ? (valid_h - row_start) : MAX_DMA_BATCH;
                    for (int i = 0; i < batch_size; i++) {
                        int row = row_start + i;
                        load_dmas[i].dir = PI_CL_DMA_DIR_EXT2LOC;
                        load_dmas[i].size = valid_w * cfg->in_ch;  // Copy all channels at once
                        load_dmas[i].ext = (uint32_t)(l2_base + row * cfg->in_w * cfg->in_ch);
                        load_dmas[i].loc = (uint32_t)(l1_base + row * cfg->tile_w_halo * cfg->in_ch);
                        load_dmas[i].merge = 0;
                        load_dmas[i].id = 0;
                        pi_cl_dma_memcpy(&load_dmas[i]);
                    }
                    for (int i = 0; i < batch_size; i++) {
                        pi_cl_dma_wait(&load_dmas[i]);
                    }
                }
            } else {  // LAYOUT_CHW (default)
                // CHW: channel planes are contiguous
                for (int c = 0; c < cfg->in_ch; c++) {
                    int8_t *l1_chan_base = tile0_input_l1 + c * cfg->tile_h_halo * cfg->tile_w_halo + l1_y_offset * cfg->tile_w_halo + l1_x_offset;
                    int8_t *l2_chan_base = cfg->input_buffer_l2 + c * cfg->in_h * cfg->in_w + valid_in_y_start * cfg->in_w + valid_in_x_start;

                    for (int row_start = 0; row_start < valid_h; row_start += MAX_DMA_BATCH) {
                        int batch_size = (row_start + MAX_DMA_BATCH > valid_h) ? (valid_h - row_start) : MAX_DMA_BATCH;
                        for (int i = 0; i < batch_size; i++) {
                            int row = row_start + i;
                            load_dmas[i].dir = PI_CL_DMA_DIR_EXT2LOC;
                            load_dmas[i].size = valid_w;
                            load_dmas[i].ext = (uint32_t)(l2_chan_base + row * cfg->in_w);
                            load_dmas[i].loc = (uint32_t)(l1_chan_base + row * cfg->tile_w_halo);
                            load_dmas[i].merge = 0;
                            load_dmas[i].id = 0;
                            pi_cl_dma_memcpy(&load_dmas[i]);
                        }
                        for (int i = 0; i < batch_size; i++) {
                            pi_cl_dma_wait(&load_dmas[i]);
                        }
                    }
                }
            }
        }
    }

    // Debug: check loaded tile 0 input
#ifndef MINIMAL_OUTPUT
    {
        int8_t *tile0_input_l1 = tile_buffer_conv2d_weight_get_input(&buf_mgr);
        // First 5 values are in padding area (zeros expected)
        // Valid data starts at l1_y_offset*tile_w_halo + l1_x_offset
        // For pad_h=pad_w=1: offset = 1*tile_w_halo + 1
        int data_offset = 1 * cfg->tile_w_halo + 1;  // Approximate valid data start
        printf("CL: inner_loop tile0 input L1 [pad](first 5): %d %d %d %d %d\n",
               tile0_input_l1[0], tile0_input_l1[1], tile0_input_l1[2],
               tile0_input_l1[3], tile0_input_l1[4]);
        printf("CL: inner_loop tile0 input L1 [data@%d](first 5): %d %d %d %d %d\n",
               data_offset,
               tile0_input_l1[data_offset], tile0_input_l1[data_offset+1],
               tile0_input_l1[data_offset+2], tile0_input_l1[data_offset+3],
               tile0_input_l1[data_offset+4]);
        // Also print L2 source for comparison
        printf("CL: inner_loop L2 input (first 5): %d %d %d %d %d\n",
               cfg->input_buffer_l2[0], cfg->input_buffer_l2[1],
               cfg->input_buffer_l2[2], cfg->input_buffer_l2[3],
               cfg->input_buffer_l2[4]);
    }
#endif

    // MAIN LOOP: Iterate over spatial tiles
    for (int spatial_tile = 0; spatial_tile < cfg->num_tiles; spatial_tile++) {
        // Get current tile buffers
        int8_t *curr_input_l1 = tile_buffer_conv2d_weight_get_input(&buf_mgr);
        int8_t *curr_output_l1 = tile_buffer_conv2d_weight_get_output(&buf_mgr);

        // Calculate current tile geometry
        int tile_y = spatial_tile / cfg->num_tiles_w;
        int tile_x = spatial_tile % cfg->num_tiles_w;
        int out_y_start = tile_y * cfg->out_tile_h;
        int out_x_start = tile_x * cfg->out_tile_w;
        int out_y_end = out_y_start + cfg->out_tile_h;
        int out_x_end = out_x_start + cfg->out_tile_w;
        if (out_y_end > cfg->out_h) out_y_end = cfg->out_h;
        if (out_x_end > cfg->out_w) out_x_end = cfg->out_w;
        int actual_out_tile_h = out_y_end - out_y_start;
        int actual_out_tile_w = out_x_end - out_x_start;

        // Zero output buffer before accumulation
        memset(curr_output_l1, 0, cfg->l1_output_size);

        // LOAD FIRST WEIGHT TILE (synchronous)
        // Use separate weight buffer index for channel tiling (independent of spatial tile buffer index)
        int weight_buf_idx = 0;  // Track weight buffer ping-pong separately
        int8_t *curr_weights_l1 = (weight_buf_idx == 0) ? buf_mgr.weights_a : buf_mgr.weights_b;
        // For depthwise convolution (groups == in_ch), each output channel has kernel_h * kernel_w weights
        // For standard convolution, each output channel has in_ch * kernel_h * kernel_w weights
        const int is_depthwise = (cfg->groups > 1 && cfg->groups == cfg->in_ch);
        const int weight_per_out_ch = is_depthwise ? (cfg->kernel_h * cfg->kernel_w) : (cfg->in_ch * cfg->kernel_h * cfg->kernel_w);
#ifndef CONV2D_WEIGHT_ROW_STRIDE_PAD
#define CONV2D_WEIGHT_ROW_STRIDE_PAD 0
#endif
#if CONV2D_WEIGHT_ROW_STRIDE_PAD
        const int weight_per_out_ch_padded = (weight_per_out_ch + 3) & ~3;
#else
        const int weight_per_out_ch_padded = weight_per_out_ch;
#endif
        int first_tile_out_ch = (cfg->tile_out_ch > cfg->out_ch) ? cfg->out_ch : cfg->tile_out_ch;
        size_t first_weight_size = first_tile_out_ch * weight_per_out_ch;

        weight_copy.dir = PI_CL_DMA_DIR_EXT2LOC;
        weight_copy.size = first_weight_size;
        weight_copy.ext = (uint32_t)(cfg->weight_l2);
        weight_copy.loc = (uint32_t)(curr_weights_l1);
        weight_copy.merge = 0;
        dma_contract_l2_copy_sync(&weight_copy);

#ifndef MINIMAL_OUTPUT
        printf("CL: inner_loop weights L1 (first 5): %d %d %d %d %d\n",
               curr_weights_l1[0], curr_weights_l1[1], curr_weights_l1[2],
               curr_weights_l1[3], curr_weights_l1[4]);
#endif

        // Pack weights in-place to a 4-byte stride (helps SIMD path when col_size % 4 != 0)
        if (weight_per_out_ch_padded != weight_per_out_ch) {
            for (int oc = first_tile_out_ch - 1; oc >= 0; oc--) {
                int8_t *dst = curr_weights_l1 + oc * weight_per_out_ch_padded;
                int8_t *src = curr_weights_l1 + oc * weight_per_out_ch;
                if (dst != src) {
                    for (int i = weight_per_out_ch - 1; i >= 0; i--) {
                        dst[i] = src[i];
                    }
                }
                memset(dst + weight_per_out_ch, 0, (size_t)(weight_per_out_ch_padded - weight_per_out_ch));
            }
        }

        // INNER LOOP: Iterate over output channel tiles
        for (int ch_tile = 0; ch_tile < cfg->num_out_ch_tiles; ch_tile++) {
            int out_ch_start = ch_tile * cfg->tile_out_ch;
            int out_ch_end = out_ch_start + cfg->tile_out_ch;
            if (out_ch_end > cfg->out_ch) out_ch_end = cfg->out_ch;
            int actual_out_ch = out_ch_end - out_ch_start;

            // --- ASYNC LOAD NEXT WEIGHT TILE ---
            pi_cl_dma_copy_t next_weight_copy;
            dma_async_future_t next_weight_future;
            int has_next_weight = 0;
            int8_t *next_weights_l1 = NULL;
            int next_actual_out_ch = 0;
            if (ch_tile < cfg->num_out_ch_tiles - 1) {
                // Get next weight buffer (opposite of current)
                int next_weight_buf_idx = 1 - weight_buf_idx;
                next_weights_l1 = (next_weight_buf_idx == 0) ? buf_mgr.weights_a : buf_mgr.weights_b;
                int next_out_ch_start = (ch_tile + 1) * cfg->tile_out_ch;
                int next_out_ch_end = next_out_ch_start + cfg->tile_out_ch;
                if (next_out_ch_end > cfg->out_ch) next_out_ch_end = cfg->out_ch;
                next_actual_out_ch = next_out_ch_end - next_out_ch_start;
                size_t next_weight_size = next_actual_out_ch * weight_per_out_ch;

                next_weight_copy.dir = PI_CL_DMA_DIR_EXT2LOC;
                next_weight_copy.size = next_weight_size;
                next_weight_copy.ext = (uint32_t)(cfg->weight_l2 + next_out_ch_start * weight_per_out_ch);
                next_weight_copy.loc = (uint32_t)(next_weights_l1);
                next_weight_copy.merge = 0;
                has_next_weight = (dma_contract_l2_copy_start(&next_weight_future, &next_weight_copy) == 0);
            }

            // --- COMPUTE CURRENT OUTPUT CHANNEL TILE ---
            // The kernel reads weights from L1 and writes partial outputs
            // For HWC layout with Ko-tiling, we pass total_out_ch and out_ch_offset so the kernel
            // can write to correct interleaved positions in the L1 output buffer.
            conv2d_tile_args_t curr_args = {
                .tile_input_l1 = curr_input_l1,
                .tile_output_l1 = (cfg->layout == 1)
                    ? curr_output_l1  // HWC: no offset, kernel writes to interleaved positions
                    : curr_output_l1 + out_ch_start * actual_out_tile_h * actual_out_tile_w,  // CHW: stack planes
                .weights_l2 = curr_weights_l1,  // Actually L1 now!
                .weight_row_stride = weight_per_out_ch_padded,
                .bias_l2 = cfg->bias_l2 + out_ch_start,
                .tile_in_h = cfg->tile_h_halo, .tile_in_w = cfg->tile_w_halo,
                .tile_out_h = actual_out_tile_h, .tile_out_w = actual_out_tile_w,
                .in_ch = cfg->in_ch, .out_ch = actual_out_ch,
                .groups = cfg->groups,
                .kernel_h = cfg->kernel_h, .kernel_w = cfg->kernel_w,
                .stride_h = cfg->stride_h, .stride_w = cfg->stride_w,
                .pad_h = 0, .pad_w = 0,  // Halo already included
                .scale_input = cfg->scale_input, .scale_weight = cfg->scale_weight, .scale_output = cfg->scale_output,
                .cluster_dev = cfg->cluster_dev,
                .fusion_relu = cfg->fusion_relu,  // Apply ReLU to EVERY channel tile (each tile owns its channels)
                .fusion_quant = cfg->fusion_quant,  // Apply requant to EVERY channel tile
                .quant_scale_in = cfg->quant_scale_in, .quant_scale_out = cfg->quant_scale_out,
                .layout = (uint8_t)cfg->layout,
                .total_out_ch = (cfg->layout == 1) ? cfg->out_ch : 0,  // For HWC Ko-tiling
                .out_ch_offset = (cfg->layout == 1) ? out_ch_start : 0  // For HWC Ko-tiling
            };

#ifndef MINIMAL_OUTPUT
            printf("CL: inner_loop worker_args: tile_in_h=%d tile_in_w=%d tile_out_h=%d tile_out_w=%d\n",
                   curr_args.tile_in_h, curr_args.tile_in_w, curr_args.tile_out_h, curr_args.tile_out_w);
            printf("CL: inner_loop worker_args: in_ch=%d out_ch=%d kernel=%dx%d stride=%dx%d pad=%dx%d\n",
                   curr_args.in_ch, curr_args.out_ch, curr_args.kernel_h, curr_args.kernel_w,
                   curr_args.stride_h, curr_args.stride_w, curr_args.pad_h, curr_args.pad_w);
            printf("CL: inner_loop worker_args: scale_in=%f scale_w=%f scale_out=%f\n",
                   curr_args.scale_input, curr_args.scale_weight, curr_args.scale_output);
            printf("CL: inner_loop worker_args: fusion_relu=%d fusion_quant=%d\n",
                   curr_args.fusion_relu, curr_args.fusion_quant);
#endif

#ifdef ENABLE_PERF_COUNTERS
            perf_compute_start();
#endif
            pi_cl_team_fork(NUM_CORES, conv2d_tile_worker_with_fusion, &curr_args);
#ifdef ENABLE_PERF_COUNTERS
            if (cfg->perf_counter) {
                cfg->perf_counter->compute_cycles += perf_compute_end();
            }
#endif

#ifndef MINIMAL_OUTPUT
            // Debug: Print first few values of output after compute
            printf("CL: inner_loop after compute, L1 output[0..4]: %d %d %d %d %d\n",
                   curr_args.tile_output_l1[0], curr_args.tile_output_l1[1],
                   curr_args.tile_output_l1[2], curr_args.tile_output_l1[3],
                   curr_args.tile_output_l1[4]);
#endif

            // --- WAIT FOR ASYNC WEIGHT LOAD ---
            if (has_next_weight) {
                dma_contract_l2_copy_wait(&next_weight_future);
                if (weight_per_out_ch_padded != weight_per_out_ch) {
                    for (int oc = next_actual_out_ch - 1; oc >= 0; oc--) {
                        int8_t *dst = next_weights_l1 + oc * weight_per_out_ch_padded;
                        int8_t *src = next_weights_l1 + oc * weight_per_out_ch;
                        if (dst != src) {
                            for (int i = weight_per_out_ch - 1; i >= 0; i--) {
                                dst[i] = src[i];
                            }
                        }
                        memset(dst + weight_per_out_ch, 0, (size_t)(weight_per_out_ch_padded - weight_per_out_ch));
                    }
                }
            }

            // --- SWAP WEIGHT BUFFERS ---
            // Toggle weight buffer index for next channel tile iteration
            weight_buf_idx = 1 - weight_buf_idx;
            curr_weights_l1 = (weight_buf_idx == 0) ? buf_mgr.weights_a : buf_mgr.weights_b;
        }

        // --- STORE PREVIOUS TILE (look-behind) ---
        if (prev_actual_out_h > 0 && prev_actual_out_w > 0) {
            if (cfg->layout == 1) {  // LAYOUT_HWC
                // HWC: L1 has full HWC output (kernel wrote to interleaved positions with Ko-tiling)
                // Transfer rows of (W * C) bytes directly
                int8_t *l1_base = prev_output_l1;
                int8_t *l2_base = cfg->output_buffer_l2 + prev_out_y_start * cfg->out_w * cfg->out_ch + prev_out_x_start * cfg->out_ch;

                for (int row_start = 0; row_start < prev_actual_out_h; row_start += MAX_DMA_BATCH) {
                    int batch_size = (row_start + MAX_DMA_BATCH > prev_actual_out_h) ? (prev_actual_out_h - row_start) : MAX_DMA_BATCH;
                    for (int bi = 0; bi < batch_size; bi++) {
                        int row = row_start + bi;
                        store_dmas[bi].dir = PI_CL_DMA_DIR_LOC2EXT;
                        store_dmas[bi].size = prev_actual_out_w * cfg->out_ch;  // All channels at once
                        store_dmas[bi].ext = (uint32_t)(l2_base + row * cfg->out_w * cfg->out_ch);
                        store_dmas[bi].loc = (uint32_t)(l1_base + row * prev_actual_out_w * cfg->out_ch);
                        store_dmas[bi].merge = 0;
                        store_dmas[bi].id = 0;
                        pi_cl_dma_memcpy(&store_dmas[bi]);
                    }
                    for (int bi = 0; bi < batch_size; bi++) {
                        pi_cl_dma_wait(&store_dmas[bi]);
                    }
                }
            } else {  // LAYOUT_CHW (default)
                const int prev_l1_plane_size = prev_actual_out_h * prev_actual_out_w;
                for (int c = 0; c < cfg->out_ch; c++) {
                    int8_t *l1_chan_base = prev_output_l1 + c * prev_l1_plane_size;
                    int8_t *l2_chan_base = cfg->output_buffer_l2 + c * cfg->out_h * cfg->out_w + prev_out_y_start * cfg->out_w + prev_out_x_start;

                    for (int row_start = 0; row_start < prev_actual_out_h; row_start += MAX_DMA_BATCH) {
                        int batch_size = (row_start + MAX_DMA_BATCH > prev_actual_out_h) ? (prev_actual_out_h - row_start) : MAX_DMA_BATCH;
                        for (int bi = 0; bi < batch_size; bi++) {
                            int row = row_start + bi;
                            store_dmas[bi].dir = PI_CL_DMA_DIR_LOC2EXT;
                            store_dmas[bi].size = prev_actual_out_w;
                            store_dmas[bi].ext = (uint32_t)(l2_chan_base + row * cfg->out_w);
                            store_dmas[bi].loc = (uint32_t)(l1_chan_base + row * prev_actual_out_w);
                            store_dmas[bi].merge = 0;
                            store_dmas[bi].id = 0;
                            pi_cl_dma_memcpy(&store_dmas[bi]);
                        }
                        for (int bi = 0; bi < batch_size; bi++) {
                            pi_cl_dma_wait(&store_dmas[bi]);
                        }
                    }
                }
            }
        }

        // --- LOAD NEXT SPATIAL TILE (look-ahead) ---
        if (spatial_tile < cfg->num_tiles - 1) {
            int8_t *next_input_l1 = tile_buffer_conv2d_weight_get_next_input(&buf_mgr);
            memset(next_input_l1, 0, cfg->l1_input_size);

            int next_tile_idx = spatial_tile + 1;
            int next_tile_y = next_tile_idx / cfg->num_tiles_w;
            int next_tile_x = next_tile_idx % cfg->num_tiles_w;
            int next_out_y_start = next_tile_y * cfg->out_tile_h;
            int next_out_x_start = next_tile_x * cfg->out_tile_w;
            int next_in_y_start = next_out_y_start * cfg->stride_h - cfg->pad_h;
            int next_in_x_start = next_out_x_start * cfg->stride_w - cfg->pad_w;
            int next_in_y_end = next_in_y_start + cfg->tile_h_halo;
            int next_in_x_end = next_in_x_start + cfg->tile_w_halo;
            int next_valid_in_y_start = (next_in_y_start < 0) ? 0 : next_in_y_start;
            int next_valid_in_x_start = (next_in_x_start < 0) ? 0 : next_in_x_start;
            int next_valid_in_y_end = (next_in_y_end > cfg->in_h) ? cfg->in_h : next_in_y_end;
            int next_valid_in_x_end = (next_in_x_end > cfg->in_w) ? cfg->in_w : next_in_x_end;

            int valid_h = next_valid_in_y_end - next_valid_in_y_start;
            int valid_w = next_valid_in_x_end - next_valid_in_x_start;
            if (valid_h > 0 && valid_w > 0) {
                int l1_y_offset = next_valid_in_y_start - next_in_y_start;
                int l1_x_offset = next_valid_in_x_start - next_in_x_start;

                if (cfg->layout == 1) {  // LAYOUT_HWC
                    // HWC: rows of (W * C) bytes contiguous
                    int8_t *l1_base = next_input_l1 + l1_y_offset * cfg->tile_w_halo * cfg->in_ch + l1_x_offset * cfg->in_ch;
                    int8_t *l2_base = cfg->input_buffer_l2 + next_valid_in_y_start * cfg->in_w * cfg->in_ch + next_valid_in_x_start * cfg->in_ch;

                    for (int row_start = 0; row_start < valid_h; row_start += MAX_DMA_BATCH) {
                        int batch_size = (row_start + MAX_DMA_BATCH > valid_h) ? (valid_h - row_start) : MAX_DMA_BATCH;
                        for (int i = 0; i < batch_size; i++) {
                            int row = row_start + i;
                            load_dmas[i].dir = PI_CL_DMA_DIR_EXT2LOC;
                            load_dmas[i].size = valid_w * cfg->in_ch;  // All channels at once
                            load_dmas[i].ext = (uint32_t)(l2_base + row * cfg->in_w * cfg->in_ch);
                            load_dmas[i].loc = (uint32_t)(l1_base + row * cfg->tile_w_halo * cfg->in_ch);
                            load_dmas[i].merge = 0;
                            load_dmas[i].id = 0;
                            pi_cl_dma_memcpy(&load_dmas[i]);
                        }
                        for (int i = 0; i < batch_size; i++) {
                            pi_cl_dma_wait(&load_dmas[i]);
                        }
                    }
                } else {  // LAYOUT_CHW (default)
                    for (int c = 0; c < cfg->in_ch; c++) {
                        int8_t *l1_chan_base = next_input_l1 + c * cfg->tile_h_halo * cfg->tile_w_halo + l1_y_offset * cfg->tile_w_halo + l1_x_offset;
                        int8_t *l2_chan_base = cfg->input_buffer_l2 + c * cfg->in_h * cfg->in_w + next_valid_in_y_start * cfg->in_w + next_valid_in_x_start;

                        for (int row_start = 0; row_start < valid_h; row_start += MAX_DMA_BATCH) {
                            int batch_size = (row_start + MAX_DMA_BATCH > valid_h) ? (valid_h - row_start) : MAX_DMA_BATCH;
                            for (int i = 0; i < batch_size; i++) {
                                int row = row_start + i;
                                load_dmas[i].dir = PI_CL_DMA_DIR_EXT2LOC;
                                load_dmas[i].size = valid_w;
                                load_dmas[i].ext = (uint32_t)(l2_chan_base + row * cfg->in_w);
                                load_dmas[i].loc = (uint32_t)(l1_chan_base + row * cfg->tile_w_halo);
                                load_dmas[i].merge = 0;
                                load_dmas[i].id = 0;
                                pi_cl_dma_memcpy(&load_dmas[i]);
                            }
                            for (int i = 0; i < batch_size; i++) {
                                pi_cl_dma_wait(&load_dmas[i]);
                            }
                        }
                    }
                }
            }
        }

        // --- UPDATE STATE FOR NEXT ITERATION ---
        prev_output_l1 = curr_output_l1;
        prev_out_y_start = out_y_start;
        prev_out_x_start = out_x_start;
        prev_actual_out_h = actual_out_tile_h;
        prev_actual_out_w = actual_out_tile_w;

        // Swap spatial tile buffers (input/output, weights handled separately)
        if (spatial_tile < cfg->num_tiles - 1) {
            tile_buffer_conv2d_weight_swap(&buf_mgr);
        }
    }

    // EPILOGUE: Store last tile
    if (prev_actual_out_h > 0 && prev_actual_out_w > 0) {
        if (cfg->layout == 1) {  // LAYOUT_HWC
            // HWC: L1 has full HWC output (kernel wrote to interleaved positions with Ko-tiling)
            int8_t *l1_base = prev_output_l1;
            int8_t *l2_base = cfg->output_buffer_l2 + prev_out_y_start * cfg->out_w * cfg->out_ch + prev_out_x_start * cfg->out_ch;

            for (int row_start = 0; row_start < prev_actual_out_h; row_start += MAX_DMA_BATCH) {
                int batch_size = (row_start + MAX_DMA_BATCH > prev_actual_out_h) ? (prev_actual_out_h - row_start) : MAX_DMA_BATCH;
                for (int bi = 0; bi < batch_size; bi++) {
                    int row = row_start + bi;
                    store_dmas[bi].dir = PI_CL_DMA_DIR_LOC2EXT;
                    store_dmas[bi].size = prev_actual_out_w * cfg->out_ch;
                    store_dmas[bi].ext = (uint32_t)(l2_base + row * cfg->out_w * cfg->out_ch);
                    store_dmas[bi].loc = (uint32_t)(l1_base + row * prev_actual_out_w * cfg->out_ch);
                    store_dmas[bi].merge = 0;
                    store_dmas[bi].id = 0;
                    pi_cl_dma_memcpy(&store_dmas[bi]);
                }
                for (int bi = 0; bi < batch_size; bi++) {
                    pi_cl_dma_wait(&store_dmas[bi]);
                }
            }
        } else {  // LAYOUT_CHW (default)
            const int prev_l1_plane_size = prev_actual_out_h * prev_actual_out_w;
            for (int c = 0; c < cfg->out_ch; c++) {
                int8_t *l1_chan_base = prev_output_l1 + c * prev_l1_plane_size;
                int8_t *l2_chan_base = cfg->output_buffer_l2 + c * cfg->out_h * cfg->out_w + prev_out_y_start * cfg->out_w + prev_out_x_start;

                for (int row_start = 0; row_start < prev_actual_out_h; row_start += MAX_DMA_BATCH) {
                    int batch_size = (row_start + MAX_DMA_BATCH > prev_actual_out_h) ? (prev_actual_out_h - row_start) : MAX_DMA_BATCH;
                    for (int bi = 0; bi < batch_size; bi++) {
                        int row = row_start + bi;
                        store_dmas[bi].dir = PI_CL_DMA_DIR_LOC2EXT;
                        store_dmas[bi].size = prev_actual_out_w;
                        store_dmas[bi].ext = (uint32_t)(l2_chan_base + row * cfg->out_w);
                        store_dmas[bi].loc = (uint32_t)(l1_chan_base + row * prev_actual_out_w);
                        store_dmas[bi].merge = 0;
                        store_dmas[bi].id = 0;
                        pi_cl_dma_memcpy(&store_dmas[bi]);
                    }
                    for (int bi = 0; bi < batch_size; bi++) {
                        pi_cl_dma_wait(&store_dmas[bi]);
                    }
                }
            }
        }
    }
}

// ---
// Conv2D Inner Loop: Triple-Buffered Weight Pipeline
// ---
//
// Eliminates blocking wait on first weight load by using 3 weight buffers:
//   - Buffer A: Currently computing
//   - Buffer B: DMA completed (ready for next compute)
//   - Buffer C: DMA in progress (prefetch for iteration+2)
//
// Timeline comparison:
//   Double-buffer:  LOAD W0 (BLOCK) -> COMPUTE W0 + LOAD W1 -> COMPUTE W1 + LOAD W2
//   Triple-buffer:  LOAD W0+W1 (async) -> wait W0 -> COMPUTE W0 + LOAD W2 -> ...
//
// Only enabled when num_out_ch_tiles >= 3 (enough tiles to fill the pipeline)

static void conv2d_tiled_l1_inner_loop_triple_weight(conv2d_pipeline_config_t *cfg) {
#ifndef MINIMAL_OUTPUT
    printf("CL: inner_loop_triple_weight: layer=%s in_ch=%d out_ch=%d tile_out_ch=%d num_out_ch_tiles=%d\n",
           cfg->layer_name ? cfg->layer_name : "?", cfg->in_ch, cfg->out_ch, cfg->tile_out_ch, cfg->num_out_ch_tiles);
    printf("CL: inner_loop_triple_weight: l1_input_size=%zu l1_output_size=%zu l1_weight_size=%zu\n",
           cfg->l1_input_size, cfg->l1_output_size, cfg->l1_weight_size);
#endif

    tile_buffer_conv2d_triple_weight_mgr_t buf_mgr;
    tile_buffer_conv2d_triple_weight_init(&buf_mgr, cfg->l1_buffer,
                                          cfg->l1_input_size, cfg->l1_output_size, cfg->l1_weight_size);

    pi_cl_dma_copy_t load_dmas[MAX_DMA_BATCH];
    pi_cl_dma_copy_t store_dmas[MAX_DMA_BATCH];
    pi_cl_dma_copy_t weight_dma[3];  // DMA handles for triple-buffered weights
    dma_async_future_t weight_dma_future[3];

    // Track previous tile's output buffer and geometry for look-behind store
    int8_t *prev_output_l1 = NULL;
    int prev_out_y_start = 0, prev_out_x_start = 0;
    int prev_actual_out_h = 0, prev_actual_out_w = 0;

    // Weight geometry
    // For depthwise convolution (groups == in_ch), each output channel has kernel_h * kernel_w weights
    // For standard convolution, each output channel has in_ch * kernel_h * kernel_w weights
    const int is_depthwise = (cfg->groups > 1 && cfg->groups == cfg->in_ch);
    const int weight_per_out_ch = is_depthwise ? (cfg->kernel_h * cfg->kernel_w) : (cfg->in_ch * cfg->kernel_h * cfg->kernel_w);
#ifndef CONV2D_WEIGHT_ROW_STRIDE_PAD
#define CONV2D_WEIGHT_ROW_STRIDE_PAD 0
#endif
#if CONV2D_WEIGHT_ROW_STRIDE_PAD
    const int weight_per_out_ch_padded = (weight_per_out_ch + 3) & ~3;
#else
    const int weight_per_out_ch_padded = weight_per_out_ch;
#endif

    // PROLOGUE: Load first spatial tile (tile 0)
    {
        int8_t *tile0_input_l1 = tile_buffer_conv2d_triple_weight_get_input(&buf_mgr);
        memset(tile0_input_l1, 0, cfg->l1_input_size);

        // Calculate tile 0 input geometry
        int out_y_end = (cfg->out_tile_h > cfg->out_h) ? cfg->out_h : cfg->out_tile_h;
        int out_x_end = (cfg->out_tile_w > cfg->out_w) ? cfg->out_w : cfg->out_tile_w;
        int in_y_start = -cfg->pad_h;
        int in_x_start = -cfg->pad_w;
        int in_y_end = in_y_start + cfg->tile_h_halo;
        int in_x_end = in_x_start + cfg->tile_w_halo;
        int valid_in_y_start = (in_y_start < 0) ? 0 : in_y_start;
        int valid_in_x_start = (in_x_start < 0) ? 0 : in_x_start;
        int valid_in_y_end = (in_y_end > cfg->in_h) ? cfg->in_h : in_y_end;
        int valid_in_x_end = (in_x_end > cfg->in_w) ? cfg->in_w : in_x_end;

        // Load tile 0 input using row-batched DMA
        for (int c = 0; c < cfg->in_ch; c++) {
            int valid_h = valid_in_y_end - valid_in_y_start;
            int valid_w = valid_in_x_end - valid_in_x_start;
            if (valid_h > 0 && valid_w > 0) {
                int l1_y_offset = valid_in_y_start - in_y_start;
                int l1_x_offset = valid_in_x_start - in_x_start;
                int8_t *l1_chan_base = tile0_input_l1 + c * cfg->tile_h_halo * cfg->tile_w_halo + l1_y_offset * cfg->tile_w_halo + l1_x_offset;
                int8_t *l2_chan_base = cfg->input_buffer_l2 + c * cfg->in_h * cfg->in_w + valid_in_y_start * cfg->in_w + valid_in_x_start;

                for (int row_start = 0; row_start < valid_h; row_start += MAX_DMA_BATCH) {
                    int batch_size = (row_start + MAX_DMA_BATCH > valid_h) ? (valid_h - row_start) : MAX_DMA_BATCH;
                    for (int i = 0; i < batch_size; i++) {
                        int row = row_start + i;
                        load_dmas[i].dir = PI_CL_DMA_DIR_EXT2LOC;
                        load_dmas[i].size = valid_w;
                        load_dmas[i].ext = (uint32_t)(l2_chan_base + row * cfg->in_w);
                        load_dmas[i].loc = (uint32_t)(l1_chan_base + row * cfg->tile_w_halo);
                        load_dmas[i].merge = 0;
                        load_dmas[i].id = 0;
                        pi_cl_dma_memcpy(&load_dmas[i]);
                    }
                    for (int i = 0; i < batch_size; i++) {
                        pi_cl_dma_wait(&load_dmas[i]);
                    }
                }
            }
        }
    }

    // MAIN LOOP: Iterate over spatial tiles
    for (int spatial_tile = 0; spatial_tile < cfg->num_tiles; spatial_tile++) {
        // Get current tile buffers
        int8_t *curr_input_l1 = tile_buffer_conv2d_triple_weight_get_input(&buf_mgr);
        int8_t *curr_output_l1 = tile_buffer_conv2d_triple_weight_get_output(&buf_mgr);

        // Calculate current tile geometry
        int tile_y = spatial_tile / cfg->num_tiles_w;
        int tile_x = spatial_tile % cfg->num_tiles_w;
        int out_y_start = tile_y * cfg->out_tile_h;
        int out_x_start = tile_x * cfg->out_tile_w;
        int out_y_end = out_y_start + cfg->out_tile_h;
        int out_x_end = out_x_start + cfg->out_tile_w;
        if (out_y_end > cfg->out_h) out_y_end = cfg->out_h;
        if (out_x_end > cfg->out_w) out_x_end = cfg->out_w;
        int actual_out_tile_h = out_y_end - out_y_start;
        int actual_out_tile_w = out_x_end - out_x_start;

        // Zero output buffer before accumulation
        memset(curr_output_l1, 0, cfg->l1_output_size);

        // Reset weight buffer rotation for this spatial tile
        buf_mgr.compute_idx = 0;

        // TRIPLE-BUFFER WEIGHT PROLOGUE
        // Start ASYNC loads for weight tiles 0 AND 1 simultaneously
        // This eliminates the blocking wait on the first tile!
        int num_ch_tiles = cfg->num_out_ch_tiles;
        int tiles_to_prefetch = (num_ch_tiles >= 2) ? 2 : num_ch_tiles;

        for (int prefetch_idx = 0; prefetch_idx < tiles_to_prefetch; prefetch_idx++) {
            int out_ch_start = prefetch_idx * cfg->tile_out_ch;
            int out_ch_end = out_ch_start + cfg->tile_out_ch;
            if (out_ch_end > cfg->out_ch) out_ch_end = cfg->out_ch;
            int actual_out_ch = out_ch_end - out_ch_start;
            size_t weight_size = actual_out_ch * weight_per_out_ch;

            int8_t *weight_buf = buf_mgr.weights[prefetch_idx];
            weight_dma[prefetch_idx].dir = PI_CL_DMA_DIR_EXT2LOC;
            weight_dma[prefetch_idx].size = weight_size;
            weight_dma[prefetch_idx].ext = (uint32_t)(cfg->weight_l2 + out_ch_start * weight_per_out_ch);
            weight_dma[prefetch_idx].loc = (uint32_t)weight_buf;
            weight_dma[prefetch_idx].merge = 0;
            dma_contract_l2_copy_start(&weight_dma_future[prefetch_idx], &weight_dma[prefetch_idx]);  // Non-blocking
        }

        // Wait for tile 0 to complete (tile 1 continues loading in parallel!)
        dma_contract_l2_copy_wait(&weight_dma_future[0]);

        // Pack weights in-place for tile 0
        if (weight_per_out_ch_padded != weight_per_out_ch) {
            int first_tile_out_ch = (cfg->tile_out_ch > cfg->out_ch) ? cfg->out_ch : cfg->tile_out_ch;
            int8_t *weights_buf = buf_mgr.weights[0];
            for (int oc = first_tile_out_ch - 1; oc >= 0; oc--) {
                int8_t *dst = weights_buf + oc * weight_per_out_ch_padded;
                int8_t *src = weights_buf + oc * weight_per_out_ch;
                if (dst != src) {
                    for (int i = weight_per_out_ch - 1; i >= 0; i--) {
                        dst[i] = src[i];
                    }
                }
                memset(dst + weight_per_out_ch, 0, (size_t)(weight_per_out_ch_padded - weight_per_out_ch));
            }
        }

        // INNER LOOP: Iterate over output channel tiles
        for (int ch_tile = 0; ch_tile < num_ch_tiles; ch_tile++) {
            int out_ch_start = ch_tile * cfg->tile_out_ch;
            int out_ch_end = out_ch_start + cfg->tile_out_ch;
            if (out_ch_end > cfg->out_ch) out_ch_end = cfg->out_ch;
            int actual_out_ch = out_ch_end - out_ch_start;

            // Get current weight buffer (compute_idx)
            int8_t *curr_weights_l1 = tile_buffer_conv2d_triple_weight_get_compute(&buf_mgr);

            // --- START PREFETCH FOR TILE ch_tile+2 (into loading buffer) ---
            int prefetch_tile = ch_tile + 2;
            if (prefetch_tile < num_ch_tiles) {
                int prefetch_out_ch_start = prefetch_tile * cfg->tile_out_ch;
                int prefetch_out_ch_end = prefetch_out_ch_start + cfg->tile_out_ch;
                if (prefetch_out_ch_end > cfg->out_ch) prefetch_out_ch_end = cfg->out_ch;
                int prefetch_actual_out_ch = prefetch_out_ch_end - prefetch_out_ch_start;
                size_t prefetch_weight_size = prefetch_actual_out_ch * weight_per_out_ch;

                int8_t *loading_buf = tile_buffer_conv2d_triple_weight_get_loading(&buf_mgr);
                int loading_dma_idx = prefetch_tile % 3;  // Reuse DMA handle
                weight_dma[loading_dma_idx].dir = PI_CL_DMA_DIR_EXT2LOC;
                weight_dma[loading_dma_idx].size = prefetch_weight_size;
                weight_dma[loading_dma_idx].ext = (uint32_t)(cfg->weight_l2 + prefetch_out_ch_start * weight_per_out_ch);
                weight_dma[loading_dma_idx].loc = (uint32_t)loading_buf;
                weight_dma[loading_dma_idx].merge = 0;
                dma_contract_l2_copy_start(&weight_dma_future[loading_dma_idx], &weight_dma[loading_dma_idx]);  // Non-blocking
            }

            // --- COMPUTE CURRENT OUTPUT CHANNEL TILE ---
            // For HWC layout with Ko-tiling: kernel writes to interleaved positions using
            // total_out_ch and out_ch_offset, so output pointer stays at buffer start.
            // For CHW layout: offset the output pointer by out_ch_start * spatial_size.
            int8_t *tile_output_ptr = curr_output_l1;
            if (cfg->layout == 0) {  // CHW layout
                tile_output_ptr = curr_output_l1 + out_ch_start * actual_out_tile_h * actual_out_tile_w;
            }

            conv2d_tile_args_t curr_args = {
                .tile_input_l1 = curr_input_l1,
                .tile_output_l1 = tile_output_ptr,
                .weights_l2 = curr_weights_l1,  // Actually L1 now!
                .weight_row_stride = weight_per_out_ch_padded,
                .bias_l2 = cfg->bias_l2 + out_ch_start,
                .tile_in_h = cfg->tile_h_halo, .tile_in_w = cfg->tile_w_halo,
                .tile_out_h = actual_out_tile_h, .tile_out_w = actual_out_tile_w,
                .in_ch = cfg->in_ch, .out_ch = actual_out_ch,
                .groups = cfg->groups,
                .kernel_h = cfg->kernel_h, .kernel_w = cfg->kernel_w,
                .stride_h = cfg->stride_h, .stride_w = cfg->stride_w,
                .pad_h = 0, .pad_w = 0,  // Halo already included
                .scale_input = cfg->scale_input, .scale_weight = cfg->scale_weight, .scale_output = cfg->scale_output,
                .cluster_dev = cfg->cluster_dev,
                .fusion_relu = cfg->fusion_relu,
                .fusion_quant = cfg->fusion_quant,
                .quant_scale_in = cfg->quant_scale_in, .quant_scale_out = cfg->quant_scale_out,
                .layout = (uint8_t)cfg->layout,
                .total_out_ch = (cfg->layout == 1) ? cfg->out_ch : 0,  // For HWC Ko-tiling
                .out_ch_offset = (cfg->layout == 1) ? out_ch_start : 0  // For HWC Ko-tiling
            };

#ifdef ENABLE_PERF_COUNTERS
            perf_compute_start();
#endif
            pi_cl_team_fork(NUM_CORES, conv2d_tile_worker_with_fusion, &curr_args);
#ifdef ENABLE_PERF_COUNTERS
            if (cfg->perf_counter) {
                cfg->perf_counter->compute_cycles += perf_compute_end();
            }
#endif

            // --- WAIT FOR READY BUFFER (tile ch_tile+1) ---
            int ready_tile = ch_tile + 1;
            if (ready_tile < num_ch_tiles) {
                int ready_dma_idx = ready_tile % 3;
                dma_contract_l2_copy_wait(&weight_dma_future[ready_dma_idx]);

                // Pack weights in-place for ready buffer
                if (weight_per_out_ch_padded != weight_per_out_ch) {
                    int8_t *ready_buf = tile_buffer_conv2d_triple_weight_get_ready(&buf_mgr);
                    int ready_out_ch_end = (ready_tile + 1) * cfg->tile_out_ch;
                    if (ready_out_ch_end > cfg->out_ch) ready_out_ch_end = cfg->out_ch;
                    int ready_actual_out_ch = ready_out_ch_end - ready_tile * cfg->tile_out_ch;

                    for (int oc = ready_actual_out_ch - 1; oc >= 0; oc--) {
                        int8_t *dst = ready_buf + oc * weight_per_out_ch_padded;
                        int8_t *src = ready_buf + oc * weight_per_out_ch;
                        if (dst != src) {
                            for (int i = weight_per_out_ch - 1; i >= 0; i--) {
                                dst[i] = src[i];
                            }
                        }
                        memset(dst + weight_per_out_ch, 0, (size_t)(weight_per_out_ch_padded - weight_per_out_ch));
                    }
                }
            }

            // --- ADVANCE WEIGHT BUFFERS: compute -> ready -> loading -> compute ---
            tile_buffer_conv2d_triple_weight_advance(&buf_mgr);
        }

        // --- STORE PREVIOUS TILE (look-behind) ---
        if (prev_actual_out_h > 0 && prev_actual_out_w > 0) {
            if (cfg->layout == 1) {  // LAYOUT_HWC
                // HWC: L1 has full HWC output (kernel wrote to interleaved positions with Ko-tiling)
                int8_t *l1_base = prev_output_l1;
                int8_t *l2_base = cfg->output_buffer_l2 + prev_out_y_start * cfg->out_w * cfg->out_ch + prev_out_x_start * cfg->out_ch;

                for (int row_start = 0; row_start < prev_actual_out_h; row_start += MAX_DMA_BATCH) {
                    int batch_size = (row_start + MAX_DMA_BATCH > prev_actual_out_h) ? (prev_actual_out_h - row_start) : MAX_DMA_BATCH;
                    for (int bi = 0; bi < batch_size; bi++) {
                        int row = row_start + bi;
                        store_dmas[bi].dir = PI_CL_DMA_DIR_LOC2EXT;
                        store_dmas[bi].size = prev_actual_out_w * cfg->out_ch;
                        store_dmas[bi].ext = (uint32_t)(l2_base + row * cfg->out_w * cfg->out_ch);
                        store_dmas[bi].loc = (uint32_t)(l1_base + row * prev_actual_out_w * cfg->out_ch);
                        store_dmas[bi].merge = 0;
                        store_dmas[bi].id = 0;
                        pi_cl_dma_memcpy(&store_dmas[bi]);
                    }
                    for (int bi = 0; bi < batch_size; bi++) {
                        pi_cl_dma_wait(&store_dmas[bi]);
                    }
                }
            } else {  // LAYOUT_CHW (default)
                const int prev_l1_plane_size = prev_actual_out_h * prev_actual_out_w;
                for (int c = 0; c < cfg->out_ch; c++) {
                    int8_t *l1_chan_base = prev_output_l1 + c * prev_l1_plane_size;
                    int8_t *l2_chan_base = cfg->output_buffer_l2 + c * cfg->out_h * cfg->out_w + prev_out_y_start * cfg->out_w + prev_out_x_start;

                    for (int row_start = 0; row_start < prev_actual_out_h; row_start += MAX_DMA_BATCH) {
                        int batch_size = (row_start + MAX_DMA_BATCH > prev_actual_out_h) ? (prev_actual_out_h - row_start) : MAX_DMA_BATCH;
                        for (int bi = 0; bi < batch_size; bi++) {
                            int row = row_start + bi;
                            store_dmas[bi].dir = PI_CL_DMA_DIR_LOC2EXT;
                            store_dmas[bi].size = prev_actual_out_w;
                            store_dmas[bi].ext = (uint32_t)(l2_chan_base + row * cfg->out_w);
                            store_dmas[bi].loc = (uint32_t)(l1_chan_base + row * prev_actual_out_w);
                            store_dmas[bi].merge = 0;
                            store_dmas[bi].id = 0;
                            pi_cl_dma_memcpy(&store_dmas[bi]);
                        }
                        for (int bi = 0; bi < batch_size; bi++) {
                            pi_cl_dma_wait(&store_dmas[bi]);
                        }
                    }
                }
            }
        }

        // --- LOAD NEXT SPATIAL TILE (look-ahead) ---
        if (spatial_tile < cfg->num_tiles - 1) {
            int8_t *next_input_l1 = tile_buffer_conv2d_triple_weight_get_next_input(&buf_mgr);
            memset(next_input_l1, 0, cfg->l1_input_size);

            int next_tile_idx = spatial_tile + 1;
            int next_tile_y = next_tile_idx / cfg->num_tiles_w;
            int next_tile_x = next_tile_idx % cfg->num_tiles_w;
            int next_out_y_start = next_tile_y * cfg->out_tile_h;
            int next_out_x_start = next_tile_x * cfg->out_tile_w;
            int next_in_y_start = next_out_y_start * cfg->stride_h - cfg->pad_h;
            int next_in_x_start = next_out_x_start * cfg->stride_w - cfg->pad_w;
            int next_in_y_end = next_in_y_start + cfg->tile_h_halo;
            int next_in_x_end = next_in_x_start + cfg->tile_w_halo;
            int next_valid_in_y_start = (next_in_y_start < 0) ? 0 : next_in_y_start;
            int next_valid_in_x_start = (next_in_x_start < 0) ? 0 : next_in_x_start;
            int next_valid_in_y_end = (next_in_y_end > cfg->in_h) ? cfg->in_h : next_in_y_end;
            int next_valid_in_x_end = (next_in_x_end > cfg->in_w) ? cfg->in_w : next_in_x_end;

            int valid_h = next_valid_in_y_end - next_valid_in_y_start;
            int valid_w = next_valid_in_x_end - next_valid_in_x_start;
            if (valid_h > 0 && valid_w > 0) {
                int l1_y_offset = next_valid_in_y_start - next_in_y_start;
                int l1_x_offset = next_valid_in_x_start - next_in_x_start;

                if (cfg->layout == 1) {  // LAYOUT_HWC
                    // HWC: rows of (W * C) bytes contiguous
                    int8_t *l1_base = next_input_l1 + l1_y_offset * cfg->tile_w_halo * cfg->in_ch + l1_x_offset * cfg->in_ch;
                    int8_t *l2_base = cfg->input_buffer_l2 + next_valid_in_y_start * cfg->in_w * cfg->in_ch + next_valid_in_x_start * cfg->in_ch;

                    for (int row_start = 0; row_start < valid_h; row_start += MAX_DMA_BATCH) {
                        int batch_size = (row_start + MAX_DMA_BATCH > valid_h) ? (valid_h - row_start) : MAX_DMA_BATCH;
                        for (int i = 0; i < batch_size; i++) {
                            int row = row_start + i;
                            load_dmas[i].dir = PI_CL_DMA_DIR_EXT2LOC;
                            load_dmas[i].size = valid_w * cfg->in_ch;  // All channels at once
                            load_dmas[i].ext = (uint32_t)(l2_base + row * cfg->in_w * cfg->in_ch);
                            load_dmas[i].loc = (uint32_t)(l1_base + row * cfg->tile_w_halo * cfg->in_ch);
                            load_dmas[i].merge = 0;
                            load_dmas[i].id = 0;
                            pi_cl_dma_memcpy(&load_dmas[i]);
                        }
                        for (int i = 0; i < batch_size; i++) {
                            pi_cl_dma_wait(&load_dmas[i]);
                        }
                    }
                } else {  // LAYOUT_CHW (default)
                    for (int c = 0; c < cfg->in_ch; c++) {
                        int8_t *l1_chan_base = next_input_l1 + c * cfg->tile_h_halo * cfg->tile_w_halo + l1_y_offset * cfg->tile_w_halo + l1_x_offset;
                        int8_t *l2_chan_base = cfg->input_buffer_l2 + c * cfg->in_h * cfg->in_w + next_valid_in_y_start * cfg->in_w + next_valid_in_x_start;

                        for (int row_start = 0; row_start < valid_h; row_start += MAX_DMA_BATCH) {
                            int batch_size = (row_start + MAX_DMA_BATCH > valid_h) ? (valid_h - row_start) : MAX_DMA_BATCH;
                            for (int i = 0; i < batch_size; i++) {
                                int row = row_start + i;
                                load_dmas[i].dir = PI_CL_DMA_DIR_EXT2LOC;
                                load_dmas[i].size = valid_w;
                                load_dmas[i].ext = (uint32_t)(l2_chan_base + row * cfg->in_w);
                                load_dmas[i].loc = (uint32_t)(l1_chan_base + row * cfg->tile_w_halo);
                                load_dmas[i].merge = 0;
                                load_dmas[i].id = 0;
                                pi_cl_dma_memcpy(&load_dmas[i]);
                            }
                            for (int i = 0; i < batch_size; i++) {
                                pi_cl_dma_wait(&load_dmas[i]);
                            }
                        }
                    }
                }
            }
        }

        // --- UPDATE STATE FOR NEXT ITERATION ---
        prev_output_l1 = curr_output_l1;
        prev_out_y_start = out_y_start;
        prev_out_x_start = out_x_start;
        prev_actual_out_h = actual_out_tile_h;
        prev_actual_out_w = actual_out_tile_w;

        // Swap spatial tile I/O buffers
        if (spatial_tile < cfg->num_tiles - 1) {
            tile_buffer_conv2d_triple_weight_swap_io(&buf_mgr);
        }
    }

    // EPILOGUE: Store last tile
    if (prev_actual_out_h > 0 && prev_actual_out_w > 0) {
        if (cfg->layout == 1) {  // LAYOUT_HWC
            // HWC: L1 has full HWC output (kernel wrote to interleaved positions with Ko-tiling)
            int8_t *l1_base = prev_output_l1;
            int8_t *l2_base = cfg->output_buffer_l2 + prev_out_y_start * cfg->out_w * cfg->out_ch + prev_out_x_start * cfg->out_ch;

            for (int row_start = 0; row_start < prev_actual_out_h; row_start += MAX_DMA_BATCH) {
                int batch_size = (row_start + MAX_DMA_BATCH > prev_actual_out_h) ? (prev_actual_out_h - row_start) : MAX_DMA_BATCH;
                for (int bi = 0; bi < batch_size; bi++) {
                    int row = row_start + bi;
                    store_dmas[bi].dir = PI_CL_DMA_DIR_LOC2EXT;
                    store_dmas[bi].size = prev_actual_out_w * cfg->out_ch;
                    store_dmas[bi].ext = (uint32_t)(l2_base + row * cfg->out_w * cfg->out_ch);
                    store_dmas[bi].loc = (uint32_t)(l1_base + row * prev_actual_out_w * cfg->out_ch);
                    store_dmas[bi].merge = 0;
                    store_dmas[bi].id = 0;
                    pi_cl_dma_memcpy(&store_dmas[bi]);
                }
                for (int bi = 0; bi < batch_size; bi++) {
                    pi_cl_dma_wait(&store_dmas[bi]);
                }
            }
        } else {  // LAYOUT_CHW (default)
            const int prev_l1_plane_size = prev_actual_out_h * prev_actual_out_w;
            for (int c = 0; c < cfg->out_ch; c++) {
                int8_t *l1_chan_base = prev_output_l1 + c * prev_l1_plane_size;
                int8_t *l2_chan_base = cfg->output_buffer_l2 + c * cfg->out_h * cfg->out_w + prev_out_y_start * cfg->out_w + prev_out_x_start;

                for (int row_start = 0; row_start < prev_actual_out_h; row_start += MAX_DMA_BATCH) {
                    int batch_size = (row_start + MAX_DMA_BATCH > prev_actual_out_h) ? (prev_actual_out_h - row_start) : MAX_DMA_BATCH;
                    for (int bi = 0; bi < batch_size; bi++) {
                        int row = row_start + bi;
                        store_dmas[bi].dir = PI_CL_DMA_DIR_LOC2EXT;
                        store_dmas[bi].size = prev_actual_out_w;
                        store_dmas[bi].ext = (uint32_t)(l2_chan_base + row * cfg->out_w);
                        store_dmas[bi].loc = (uint32_t)(l1_chan_base + row * prev_actual_out_w);
                        store_dmas[bi].merge = 0;
                        store_dmas[bi].id = 0;
                        pi_cl_dma_memcpy(&store_dmas[bi]);
                    }
                    for (int bi = 0; bi < batch_size; bi++) {
                        pi_cl_dma_wait(&store_dmas[bi]);
                    }
                }
            }
        }
    }
}

// ---
// Conv2D Main Pipeline (Outer Loop: L3 -> L2)
// ---

void conv2d_tiled_l1_pipeline(conv2d_pipeline_config_t *cfg) {
    if (cfg->l3_tiling_enabled) {
        for (int slab_idx = 0; slab_idx < cfg->num_l3_tiles; slab_idx++) {
            int slab_y_start = slab_idx * cfg->l3_tile_h;
            int slab_y_end   = slab_y_start + cfg->l3_tile_h;
            if (slab_y_end > cfg->out_h) slab_y_end = cfg->out_h;
            int actual_slab_h = slab_y_end - slab_y_start;
            
            int slab_in_y_start = slab_y_start * cfg->stride_h - cfg->pad_h;
            int slab_in_y_end   = slab_in_y_start + cfg->l3_tile_h_halo; 
            if (slab_in_y_start < 0) slab_in_y_start = 0;
            if (slab_in_y_end > cfg->in_h) slab_in_y_end = cfg->in_h;
            int actual_slab_in_h = slab_in_y_end - slab_in_y_start;
            
            dma_layout_t l3_src = {
                .base_addr = (int8_t*)cfg->l3_input_addr + (slab_in_y_start * cfg->in_w * cfg->in_ch),
                .width = cfg->in_w * cfg->in_ch, .height = actual_slab_in_h, .channels = 1, .stride_row = cfg->in_w * cfg->in_ch, 
                .loc = MEM_LOC_L3, .is_async = 0, .ram_dev = cfg->ram_dev
            };
            dma_layout_t l2_dst = {
                .base_addr = cfg->input_buffer_l2, .width = cfg->in_w * cfg->in_ch, .height = actual_slab_in_h, .channels = 1, .stride_row = cfg->in_w * cfg->in_ch,
                .loc = MEM_LOC_L2, .ram_dev = cfg->ram_dev
            };
            execute_dma_transfer(&l3_src, &l2_dst);
            
            int original_out_h = cfg->out_h;
            int original_in_h = cfg->in_h;
            cfg->out_h = actual_slab_h;
            cfg->in_h = actual_slab_in_h;

            // Dispatch to weight-cached or standard inner loop
            if (cfg->triple_buffer_weights) {
                conv2d_tiled_l1_inner_loop_triple_weight(cfg);
            } else if (cfg->weight_tiling_enabled) {
                conv2d_tiled_l1_inner_loop_with_weights(cfg);
            } else {
                conv2d_tiled_l1_inner_loop(cfg);
            }

            cfg->out_h = original_out_h;
            cfg->in_h = original_in_h;

            dma_layout_t l2_src = {
                .base_addr = cfg->output_buffer_l2, .width = cfg->out_w * cfg->out_ch, .height = actual_slab_h, .channels = 1, .stride_row = cfg->out_w * cfg->out_ch,
                .loc = MEM_LOC_L2, .is_async = 0, .ram_dev = cfg->ram_dev
            };
            dma_layout_t l3_dst = {
                .base_addr = (int8_t*)cfg->l3_output_addr + (slab_y_start * cfg->out_w * cfg->out_ch),
                .width = cfg->out_w * cfg->out_ch, .height = actual_slab_h, .channels = 1, .stride_row = cfg->out_w * cfg->out_ch,
                .loc = MEM_LOC_L3, .ram_dev = cfg->ram_dev
            };
            execute_dma_transfer(&l2_src, &l3_dst);
        }
    } else {
        // Dispatch to weight-cached or standard inner loop
        if (cfg->triple_buffer_weights) {
            conv2d_tiled_l1_inner_loop_triple_weight(cfg);
        } else if (cfg->weight_tiling_enabled) {
            conv2d_tiled_l1_inner_loop_with_weights(cfg);
        } else {
            conv2d_tiled_l1_inner_loop(cfg);
        }
    }
}

// ---
// K-tiling helper workers (baseline implementation)
// ---

// Arguments for K-tiling accumulate worker
typedef struct {
    const int8_t *input_l1;    // Input tile in L1
    const int8_t *weight_l1;   // Weight tile in L1
    int32_t *acc_l1;           // INT32 accumulator buffer in L1
    int k_features;            // Number of input features in this K-tile
    int n_features;            // Number of output features in this N-tile
    int in_features_full;      // Full input feature dimension (for weight striding)
} linear_nk_accumulate_args_t;

// Arguments for K-tiling finalize worker
typedef struct {
    const int32_t *acc_l1;     // INT32 accumulator buffer in L1
    const int32_t *bias_l2;    // Bias in L2 (or NULL)
    int8_t *output_l1;         // Output tile in L1
    int n_features;            // Number of output features in this N-tile
    float scale_input;
    float scale_weight;
    float scale_output;
} linear_nk_finalize_args_t;

// Worker: Accumulate INT32 results without applying bias/scale
// acc[n] += sum(input[k] * weight[n,k]) for this K-tile
static void linear_nk_accumulate_worker(void *arg) {
    linear_nk_accumulate_args_t *t = (linear_nk_accumulate_args_t *)arg;

    int core_id = pi_core_id();
    int chunk = (t->n_features + NUM_CORES - 1) / NUM_CORES;
    int start_n = core_id * chunk;
    int end_n = (start_n + chunk > t->n_features) ? t->n_features : (start_n + chunk);

    // SIMD setup
    const int simd_count = t->k_features >> 2;
    const int remainder = t->k_features & 0x3;
    const v4s *pA = (const v4s *)t->input_l1;
    const int8_t *pA_rem = t->input_l1 + (simd_count << 2);

    // Accumulate for assigned output features
    for (int n = start_n; n < end_n; n++) {
        int32_t acc = t->acc_l1[n];  // Load existing accumulator value

        // Weight row starts at n * in_features_full (full weight matrix stride)
        const int8_t *w = t->weight_l1 + n * t->in_features_full;
        const v4s *pB = (const v4s *)w;
        const int8_t *pB_rem = w + (simd_count << 2);

        // SIMD accumulation
        for (int j = 0; j < simd_count; j++) {
            acc = SumDotpSS(pA[j], pB[j], acc);
        }

        // Remainder elements
        for (int j = 0; j < remainder; j++) {
            acc += pA_rem[j] * pB_rem[j];
        }

        t->acc_l1[n] = acc;  // Store back
    }
}

// Worker: Apply bias and scale, convert INT32 -> INT8
// output[n] = qround((acc[n] + bias[n]) * scale)
static void linear_nk_finalize_worker(void *arg) {
    linear_nk_finalize_args_t *t = (linear_nk_finalize_args_t *)arg;

    int core_id = pi_core_id();
    int chunk = (t->n_features + NUM_CORES - 1) / NUM_CORES;
    int start_n = core_id * chunk;
    int end_n = (start_n + chunk > t->n_features) ? t->n_features : (start_n + chunk);

    const float combined_scale = t->scale_input * t->scale_weight / t->scale_output;
    const int32_t *bias = t->bias_l2;

    for (int n = start_n; n < end_n; n++) {
        int32_t acc = t->acc_l1[n];

        // Apply bias BEFORE scaling (critical for accuracy)
        if (bias) {
            acc += bias[n];
        }

        // Scale and quantize to INT8
        float val = (float)acc * combined_scale;
        t->output_l1[n] = qround(val);
    }
}

// ---
// K-tiling main loop (baseline N+K tiling)
// ---

static void linear_int8_nk_tiled_loop(linear_int8_pipeline_config_t *cfg) {
#ifndef MINIMAL_OUTPUT
    printf("CL: %s using N+K tiling: %d N-tiles x %d K-tiles (tile_n=%d, tile_k=%d)\n",
           cfg->layer_name, cfg->num_tiles, cfg->num_k_tiles,
           cfg->tile_out_features, cfg->tile_in_features);
#endif

    // L1 buffer layout:
    // [INT32 accumulator (tile_n * 4 bytes)] [input tile (tile_k)] [weight tile (tile_n * tile_k)] [output tile (tile_n)]
    const size_t acc_size = cfg->tile_out_features * sizeof(int32_t);
    const size_t input_size = cfg->tile_in_features;
    const size_t weight_size = cfg->tile_out_features * cfg->tile_in_features;
    const size_t output_size = cfg->tile_out_features;

    int32_t *acc_l1 = (int32_t *)cfg->l1_buffer;
    int8_t *input_l1 = (int8_t *)(acc_l1 + cfg->tile_out_features);
    int8_t *weight_l1 = input_l1 + input_size;
    int8_t *output_l1 = weight_l1 + weight_size;

    // Outer loop: Tile over output features (N-dimension)
    for (int n_tile = 0; n_tile < cfg->num_tiles; n_tile++) {
        int n_start = n_tile * cfg->tile_out_features;
        int n_end = n_start + cfg->tile_out_features;
        if (n_end > cfg->out_features) n_end = cfg->out_features;
        int n_actual = n_end - n_start;

        // Zero accumulator buffer for this N-tile
        memset(acc_l1, 0, cfg->tile_out_features * sizeof(int32_t));

        // Inner loop: Tile over input features (K-dimension)
        for (int k_tile = 0; k_tile < cfg->num_k_tiles; k_tile++) {
            int k_start = k_tile * cfg->tile_in_features;
            int k_end = k_start + cfg->tile_in_features;
            if (k_end > cfg->in_features) k_end = cfg->in_features;
            int k_actual = k_end - k_start;

            // Load input slice from L2 to L1
            pi_cl_dma_copy_t input_copy;
            dma_async_future_t input_future;
            input_copy.dir = PI_CL_DMA_DIR_EXT2LOC;
            input_copy.size = k_actual;
            input_copy.ext = (uint32_t)(cfg->input_buffer_l2 + k_start);
            input_copy.loc = (uint32_t)input_l1;
            input_copy.merge = 0;
            dma_contract_l2_copy_start(&input_future, &input_copy);

            // Load weight slice from L2 to L1 (row-by-row, strided access)
            // Weight matrix is [out_features x in_features], need slice [n_start:n_end, k_start:k_end]
            // Dense packing in L1: each row is k_actual bytes (not full in_features stride)
            for (int n = 0; n < n_actual; n++) {
                const int8_t *weight_row_src = cfg->weight_l2 + (n_start + n) * cfg->in_features + k_start;
                int8_t *weight_row_dst = weight_l1 + n * k_actual;  // Dense packing: k_actual bytes per row
                memcpy(weight_row_dst, weight_row_src, k_actual);
            }

            // Wait for input DMA
            dma_contract_l2_copy_wait(&input_future);

            // Compute and accumulate (no bias, no requant)
            linear_nk_accumulate_args_t acc_args = {
                .input_l1 = input_l1,
                .weight_l1 = weight_l1,
                .acc_l1 = acc_l1,
                .k_features = k_actual,
                .n_features = n_actual,
                .in_features_full = k_actual  // Use K-tile stride (dense packing) for weight indexing
            };

#ifdef ENABLE_PERF_COUNTERS
            if (cfg->perf_counter) perf_compute_start();
#endif
            pi_cl_team_fork(NUM_CORES, linear_nk_accumulate_worker, &acc_args);
#ifdef ENABLE_PERF_COUNTERS
            if (cfg->perf_counter) cfg->perf_counter->compute_cycles += perf_compute_end();
#endif
        }

        // After K-loop completes: apply bias and scale
        const int32_t *tile_bias = cfg->bias_l2 ? (cfg->bias_l2 + n_start) : NULL;
        linear_nk_finalize_args_t fin_args = {
            .acc_l1 = acc_l1,
            .bias_l2 = tile_bias,
            .output_l1 = output_l1,
            .n_features = n_actual,
            .scale_input = cfg->scale_input,
            .scale_weight = cfg->scale_weight,
            .scale_output = cfg->scale_output
        };

#ifdef ENABLE_PERF_COUNTERS
        if (cfg->perf_counter) perf_compute_start();
#endif
        pi_cl_team_fork(NUM_CORES, linear_nk_finalize_worker, &fin_args);
#ifdef ENABLE_PERF_COUNTERS
        if (cfg->perf_counter) cfg->perf_counter->compute_cycles += perf_compute_end();
#endif

        // Apply fused operations on L1 tile output (before storing to L2)
        if (cfg->fusion_relu || cfg->fusion_quant) {
            if (cfg->fusion_relu) {
                relu_l1_args_t a = {.data = output_l1, .size = n_actual};
                pi_cl_team_fork(NUM_CORES, relu_l1_worker, &a);
            }
            if (cfg->relu_output_scale != 0.0f && fabsf(cfg->relu_output_scale - cfg->scale_output) > 1e-5f) {
                requantize_l1_args_t a = {
                    .data = output_l1, .size = n_actual,
                    .scale_in = cfg->scale_output, .scale_out = cfg->relu_output_scale
                };
                pi_cl_team_fork(NUM_CORES, requantize_l1_worker, &a);
            }
            if (cfg->fusion_quant) {
                requantize_l1_args_t a = {
                    .data = output_l1, .size = n_actual,
                    .scale_in = cfg->quant_scale_in, .scale_out = cfg->quant_scale_out
                };
                pi_cl_team_fork(NUM_CORES, requantize_l1_worker, &a);
            }
        }

        // Store output slice from L1 to L2
        pi_cl_dma_copy_t output_copy;
        output_copy.dir = PI_CL_DMA_DIR_LOC2EXT;
        output_copy.size = n_actual;
        output_copy.loc = (uint32_t)output_l1;
        output_copy.ext = (uint32_t)(cfg->output_buffer_l2 + n_start);
        output_copy.merge = 0;
        pi_cl_dma_memcpy(&output_copy);
        pi_cl_dma_wait(&output_copy);
    }
}

// ---
// Linear INT8 Tiled L1 Inner Loop (Renamed)
// ---

static void linear_int8_tiled_l1_inner_loop(linear_int8_pipeline_config_t *cfg) {
    // Get batch_tokens (defaults to 1 for 2D inputs, >1 for 3D transformer MLP)
    int batch_tokens = cfg->batch_tokens > 0 ? cfg->batch_tokens : 1;

    // ------------------------------------------------------------------------
    // 3D Linear (batch_tokens > 1): Weight-tiled L1 path (reuse weights across tokens)
    // ------------------------------------------------------------------------
    if (batch_tokens > 1) {
        // Choose tile_out_features:
        // - Prefer codegen-provided cfg->tile_out_features
        // - Allow compile-time override for tuning
        // - Otherwise compute a safe value from available L1 (double-buffered weights)
        int tile_out_features = cfg->tile_out_features;
#ifdef LINEAR_3D_TILE_OUT_FEATURES
        tile_out_features = LINEAR_3D_TILE_OUT_FEATURES;
#endif

        if (tile_out_features <= 0) {
            if (cfg->l1_buffer && cfg->in_features > 0) {
                tile_out_features = (int)(cfg->l1_buffer_size / (2 * (size_t)cfg->in_features));
            } else {
                tile_out_features = 0;
            }
        }

        // Align for SIMD kernel (groups of 4 outputs)
        tile_out_features &= ~3;
        if (tile_out_features > cfg->out_features) tile_out_features = cfg->out_features & ~3;
        if (tile_out_features < 4) tile_out_features = 4;

        const size_t weight_tile_bytes = (size_t)tile_out_features * (size_t)cfg->in_features;
        const size_t required_l1_size = 2 * weight_tile_bytes;  // Double-buffer weights

        if (!cfg->l1_buffer || cfg->l1_buffer_size < required_l1_size) {
            // L1 unavailable - fall back to L2 parallel kernel (with optional M-tiling)
            const int num_m_tiles_l2 = cfg->m_tiling_enabled ? cfg->num_m_tiles : 1;
            const int tile_m_l2 = cfg->m_tiling_enabled ? cfg->tile_batch_tokens : batch_tokens;

            if (((cfg->in_features & 3) == 0) && ((cfg->out_features & 3) == 0)) {
                for (int m_tile = 0; m_tile < num_m_tiles_l2; m_tile++) {
                    const int token_start = m_tile * tile_m_l2;
                    int tokens_this_tile = tile_m_l2;
                    if (token_start + tokens_this_tile > batch_tokens) {
                        tokens_this_tile = batch_tokens - token_start;
                    }
                    if (tokens_this_tile <= 0) break;

#ifdef ENABLE_PERF_COUNTERS
                    if (cfg->perf_counter) perf_compute_start();
#endif
                    network_linear_int8_parallel_tokens(
                        cfg->input_buffer_l2 + token_start * cfg->in_features,
                        cfg->weight_l2,
                        cfg->bias_l2,
                        cfg->output_buffer_l2 + token_start * cfg->out_features,
                        tokens_this_tile,
                        cfg->in_features,
                        cfg->out_features,
                        cfg->scale_input,
                        cfg->scale_weight,
                        cfg->scale_output
                    );
#ifdef ENABLE_PERF_COUNTERS
                    if (cfg->perf_counter) cfg->perf_counter->compute_cycles += perf_compute_end();
#endif
                }
            } else {
                // Generic fallback: per-token (rare in transformer models)
                for (int t = 0; t < batch_tokens; t++) {
                    int8_t *token_input = cfg->input_buffer_l2 + t * cfg->in_features;
                    int8_t *token_output = cfg->output_buffer_l2 + t * cfg->out_features;

                    linear_tile_args_t linear_args = {
                        .input_l1 = token_input, .weights_l2 = cfg->weight_l2, .bias_l2 = cfg->bias_l2, .output_l1 = token_output,
                        .dim_in = cfg->in_features, .dim_out = cfg->out_features,
                        .scale_input = cfg->scale_input, .scale_weight = cfg->scale_weight, .scale_output = cfg->scale_output
                    };
#ifdef ENABLE_PERF_COUNTERS
                    if (cfg->perf_counter) perf_compute_start();
#endif
                    pi_cl_team_fork(NUM_CORES, linear_tile_worker, &linear_args);
#ifdef ENABLE_PERF_COUNTERS
                    if (cfg->perf_counter) cfg->perf_counter->compute_cycles += perf_compute_end();
#endif
                }
            }
        } else {
            const int num_tiles_to_process = (cfg->out_features + tile_out_features - 1) / tile_out_features;

#ifndef MINIMAL_OUTPUT
#ifdef LINEAR_3D_DEBUG_USE_L2_WEIGHTS
            printf("CL: %s using 3D output-tiling (DEBUG: weights in L2): %d tiles (tile_out=%d)\n",
                   cfg->layer_name, num_tiles_to_process, tile_out_features);
#else
            printf("CL: %s using 3D weight-tiling: %d tiles (tile_out=%d, weight_tile=%zu bytes)\n",
                   cfg->layer_name, num_tiles_to_process, tile_out_features, weight_tile_bytes);
#endif
#endif

#ifdef LINEAR_3D_DEBUG_USE_L2_WEIGHTS
            // Debug/diagnostics path: tile output features but keep weights in L2 (no DMA to L1).
            // This helps isolate correctness issues between the strided-output kernel and weight DMA/cache.
            for (int tile_idx = 0; tile_idx < num_tiles_to_process; tile_idx++) {
                const int out_start = tile_idx * tile_out_features;
                int actual_tile_out = tile_out_features;
                if (out_start + actual_tile_out > cfg->out_features) {
                    actual_tile_out = cfg->out_features - out_start;
                }
                const int8_t *w_tile_l2 = cfg->weight_l2 + out_start * cfg->in_features;

#ifdef ENABLE_PERF_COUNTERS
                if (cfg->perf_counter) perf_compute_start();
#endif
                network_linear_int8_parallel_tokens_strided_out(
                    cfg->input_buffer_l2,
                    w_tile_l2,
                    cfg->bias_l2 ? (cfg->bias_l2 + out_start) : NULL,
                    cfg->output_buffer_l2 + out_start,
                    batch_tokens,
                    cfg->in_features,
                    actual_tile_out,
                    cfg->out_features,
                    cfg->scale_input,
                    cfg->scale_weight,
                    cfg->scale_output
                );
#ifdef ENABLE_PERF_COUNTERS
                if (cfg->perf_counter) cfg->perf_counter->compute_cycles += perf_compute_end();
#endif
            }
#else
            // L1 weight-tiled path: double-buffer weight tiles, output stays in L2
            int8_t *w_l1_a = cfg->l1_buffer;
            int8_t *w_l1_b = cfg->l1_buffer + weight_tile_bytes;
            int buf_idx = 0;

            // Prologue: load tile 0 weights
            {
                const int out_start = 0;
                int actual_tile_out = tile_out_features;
                if (out_start + actual_tile_out > cfg->out_features) {
                    actual_tile_out = cfg->out_features - out_start;
                }
                size_t tile_bytes = (size_t)actual_tile_out * (size_t)cfg->in_features;
                cl_dma_memcpy_1d_sync_chunked(
                    PI_CL_DMA_DIR_EXT2LOC,
                    (uint32_t)(cfg->weight_l2 + out_start * cfg->in_features),
                    (uint32_t)w_l1_a,
                    tile_bytes
                );
            }

            for (int tile_idx = 0; tile_idx < num_tiles_to_process; tile_idx++) {
                const int out_start = tile_idx * tile_out_features;
                int actual_tile_out = tile_out_features;
                if (out_start + actual_tile_out > cfg->out_features) {
                    actual_tile_out = cfg->out_features - out_start;
                }

                int8_t *w_curr = (buf_idx == 0) ? w_l1_a : w_l1_b;
                int8_t *w_next = (buf_idx == 0) ? w_l1_b : w_l1_a;

                // Prefetch next tile weights
                pi_cl_dma_copy_t w_next_copy;
                int has_next = 0;
                if (tile_idx < num_tiles_to_process - 1) {
                    const int next_out_start = (tile_idx + 1) * tile_out_features;
                    int next_actual_out = tile_out_features;
                    if (next_out_start + next_actual_out > cfg->out_features) {
                        next_actual_out = cfg->out_features - next_out_start;
                    }
                    size_t next_tile_bytes = (size_t)next_actual_out * (size_t)cfg->in_features;
                    if (next_tile_bytes <= CL_DMA_MAX_COPY_BYTES) {
                        w_next_copy.dir = PI_CL_DMA_DIR_EXT2LOC;
                        w_next_copy.size = (uint16_t)next_tile_bytes;
                        w_next_copy.ext = (uint32_t)(cfg->weight_l2 + next_out_start * cfg->in_features);
                        w_next_copy.loc = (uint32_t)w_next;
                        w_next_copy.merge = 0;
                        pi_cl_dma_memcpy(&w_next_copy);
                        has_next = 1;
                    } else {
                        // Fallback for oversized CL-DMA: complete synchronously in chunks.
                        cl_dma_memcpy_1d_sync_chunked(
                            PI_CL_DMA_DIR_EXT2LOC,
                            (uint32_t)(cfg->weight_l2 + next_out_start * cfg->in_features),
                            (uint32_t)w_next,
                            next_tile_bytes
                        );
                        has_next = 0;
                    }
                }

                // Compute this output feature tile for all tokens (with optional M-tiling)
                // M-tiling: process token subsets while reusing weights already in L1
                const int num_m_tiles = cfg->m_tiling_enabled ? cfg->num_m_tiles : 1;
                const int tile_m = cfg->m_tiling_enabled ? cfg->tile_batch_tokens : batch_tokens;

                for (int m_tile_idx = 0; m_tile_idx < num_m_tiles; m_tile_idx++) {
                    const int token_start = m_tile_idx * tile_m;
                    int tokens_this_tile = tile_m;
                    if (token_start + tokens_this_tile > batch_tokens) {
                        tokens_this_tile = batch_tokens - token_start;
                    }
                    if (tokens_this_tile <= 0) break;

                    // Adjust input/output pointers for this M-tile
                    const int8_t *input_m_tile = cfg->input_buffer_l2 + token_start * cfg->in_features;
                    int8_t *output_m_tile = cfg->output_buffer_l2 + token_start * cfg->out_features + out_start;

#ifdef ENABLE_PERF_COUNTERS
                    if (cfg->perf_counter) perf_compute_start();
#endif
                    network_linear_int8_parallel_tokens_strided_out(
                        input_m_tile,
                        w_curr,
                        cfg->bias_l2 ? (cfg->bias_l2 + out_start) : NULL,
                        output_m_tile,
                        tokens_this_tile,
                        cfg->in_features,
                        actual_tile_out,
                        cfg->out_features,
                        cfg->scale_input,
                        cfg->scale_weight,
                        cfg->scale_output
                    );
#ifdef ENABLE_PERF_COUNTERS
                    if (cfg->perf_counter) cfg->perf_counter->compute_cycles += perf_compute_end();
#endif
                }

                if (has_next) {
                    pi_cl_dma_wait(&w_next_copy);
                }
                buf_idx = 1 - buf_idx;
            }
#endif  // LINEAR_3D_DEBUG_USE_L2_WEIGHTS
        }

linear_3d_apply_fusions:
        // Apply fused operations on full L2 output (after all tiles)
        if (cfg->fusion_relu || cfg->fusion_quant) {
            size_t output_size = (size_t)batch_tokens * (size_t)cfg->out_features;
            if (cfg->fusion_relu) { relu_l1_args_t a = {.data=cfg->output_buffer_l2, .size=output_size}; pi_cl_team_fork(NUM_CORES, relu_l1_worker, &a); }
            if (cfg->relu_output_scale != 0.0f && fabsf(cfg->relu_output_scale - cfg->scale_output) > 1e-5f) {
                requantize_l1_args_t a = {.data=cfg->output_buffer_l2, .size=output_size, .scale_in=cfg->scale_output, .scale_out=cfg->relu_output_scale};
                pi_cl_team_fork(NUM_CORES, requantize_l1_worker, &a);
            }
            if (cfg->fusion_quant) { requantize_l1_args_t a = {.data=cfg->output_buffer_l2, .size=output_size, .scale_in=cfg->quant_scale_in, .scale_out=cfg->quant_scale_out}; pi_cl_team_fork(NUM_CORES, requantize_l1_worker, &a); }
        }
        return;
    }

    const size_t single_tile_size = cfg->l1_output_size + cfg->l1_weight_size;
    const size_t required_l1_size = cfg->l1_input_size + 2 * single_tile_size;

    // ------------------------------------------------------------------------
    // 2D Linear (batch_tokens == 1): K-tiling or N-only tiling
    // ------------------------------------------------------------------------
    // Check if K-tiling is enabled
    if (cfg->k_tiling_enabled && cfg->num_k_tiles > 1 && cfg->l1_buffer &&
        cfg->tile_in_features > 0 && cfg->tile_out_features > 0 &&
        cfg->tile_in_features < cfg->in_features) {
        const size_t acc_size = (size_t)cfg->tile_out_features * sizeof(int32_t);
        const size_t input_size = (size_t)cfg->tile_in_features;
        const size_t weight_size = (size_t)cfg->tile_out_features * (size_t)cfg->tile_in_features;
        const size_t output_size = (size_t)cfg->tile_out_features;
        const size_t k_required_l1 = acc_size + input_size + weight_size + output_size;

        if (cfg->l1_buffer_size >= k_required_l1) {
            // K-dimension tiling path (tile both output and input features)
            linear_int8_nk_tiled_loop(cfg);
            return;
        }
    }

    // Fall through to existing N-only tiling or L2-only paths
    if (!cfg->l1_buffer || cfg->l1_buffer_size < required_l1_size) {
        // Standard 2D linear: single batch
        linear_tile_args_t linear_args = {
            .input_l1 = cfg->input_buffer_l2, .weights_l2 = cfg->weight_l2, .bias_l2 = cfg->bias_l2, .output_l1 = cfg->output_buffer_l2,
            .dim_in = cfg->in_features, .dim_out = cfg->out_features,
            .scale_input = cfg->scale_input, .scale_weight = cfg->scale_weight, .scale_output = cfg->scale_output
        };
#ifdef ENABLE_PERF_COUNTERS
        if (cfg->perf_counter) perf_compute_start();
#endif
        pi_cl_team_fork(NUM_CORES, linear_tile_worker, &linear_args);
#ifdef ENABLE_PERF_COUNTERS
        if (cfg->perf_counter) cfg->perf_counter->compute_cycles += perf_compute_end();
#endif

        // Apply fused operations on L2 output
        if (cfg->fusion_relu || cfg->fusion_quant) {
            size_t output_size = (size_t)cfg->out_features;
            if (cfg->fusion_relu) { relu_l1_args_t a = {.data=cfg->output_buffer_l2, .size=output_size}; pi_cl_team_fork(NUM_CORES, relu_l1_worker, &a); }
            if (cfg->relu_output_scale != 0.0f && fabsf(cfg->relu_output_scale - cfg->scale_output) > 1e-5f) {
                requantize_l1_args_t a = {.data=cfg->output_buffer_l2, .size=output_size, .scale_in=cfg->scale_output, .scale_out=cfg->relu_output_scale};
                pi_cl_team_fork(NUM_CORES, requantize_l1_worker, &a);
            }
            if (cfg->fusion_quant) { requantize_l1_args_t a = {.data=cfg->output_buffer_l2, .size=output_size, .scale_in=cfg->quant_scale_in, .scale_out=cfg->quant_scale_out}; pi_cl_team_fork(NUM_CORES, requantize_l1_worker, &a); }
        }
        return;
    }

    tile_buffer_linear_mgr_t buf_mgr;
    tile_buffer_linear_init(&buf_mgr, cfg->l1_buffer, cfg->l1_input_size, cfg->l1_output_size, cfg->l1_weight_size);

    // Load input once (shared across all tiles)
    pi_cl_dma_copy_t input_copy;
    input_copy.dir = PI_CL_DMA_DIR_EXT2LOC;
    input_copy.size = cfg->in_features;
    input_copy.ext = (uint32_t)(cfg->input_buffer_l2);
    input_copy.loc = (uint32_t)(buf_mgr.input);
    input_copy.merge = 0;
    dma_contract_l2_copy_sync(&input_copy);

    // PROLOGUE: Load weights for tile 0
    int8_t *tile0_weights_l1 = tile_buffer_linear_get_weights(&buf_mgr);
    int tile0_out_start = 0;
    int tile0_out_end = cfg->tile_out_features;
    if (tile0_out_end > cfg->out_features) tile0_out_end = cfg->out_features;
    int tile0_actual_out = tile0_out_end - tile0_out_start;

    size_t tile0_weight_bytes = (size_t)tile0_actual_out * (size_t)cfg->in_features;
    cl_dma_memcpy_1d_sync_chunked(
        PI_CL_DMA_DIR_EXT2LOC,
        (uint32_t)(cfg->weight_l2),
        (uint32_t)(tile0_weights_l1),
        tile0_weight_bytes
    );

    // Track previous tile for look-behind store
    int8_t *prev_output_l1 = NULL;
    int prev_out_start = 0;
    int prev_actual_out = 0;

    // STEADY-STATE LOOP: Look-Ahead/Look-Behind Pattern
    //  Calculate num_tiles dynamically for L3 tiling (cfg->out_features is slab size, not full size)
    int num_tiles_to_process = (cfg->out_features + cfg->tile_out_features - 1) / cfg->tile_out_features;
    for (int i = 0; i < num_tiles_to_process; i++) {
        // Get current tile buffers
        int8_t *curr_output_l1 = tile_buffer_linear_get_output(&buf_mgr);
        int8_t *curr_weights_l1 = tile_buffer_linear_get_weights(&buf_mgr);

        // Calculate current tile geometry
        int out_start = i * cfg->tile_out_features;
        int out_end = out_start + cfg->tile_out_features;
        if (out_end > cfg->out_features) out_end = cfg->out_features;
        int actual_tile_out = out_end - out_start;

        // --- STEP 1: LOAD NEXT WEIGHTS (i+1) async ---
        pi_cl_dma_copy_t next_weight_copy;
        dma_async_future_t next_weight_future;
        int has_next_weight = 0;
        if (i < num_tiles_to_process - 1) {
            int8_t *next_weights_l1 = tile_buffer_linear_get_next_weights(&buf_mgr);
            int next_out_start = (i + 1) * cfg->tile_out_features;
            int next_out_end = next_out_start + cfg->tile_out_features;
            if (next_out_end > cfg->out_features) next_out_end = cfg->out_features;
            int next_actual_out = next_out_end - next_out_start;
            size_t next_weight_bytes = (size_t)next_actual_out * (size_t)cfg->in_features;

            if (next_weight_bytes <= CL_DMA_MAX_COPY_BYTES) {
                next_weight_copy.dir = PI_CL_DMA_DIR_EXT2LOC;
                next_weight_copy.size = (uint16_t)next_weight_bytes;
                next_weight_copy.ext = (uint32_t)(cfg->weight_l2 + next_out_start * cfg->in_features);
                next_weight_copy.loc = (uint32_t)(next_weights_l1);
                next_weight_copy.merge = 0;
                has_next_weight = (dma_contract_l2_copy_start(&next_weight_future, &next_weight_copy) == 0);
            } else {
                // Oversized CL-DMA copy (size field is uint16_t): perform safe chunked sync copy.
                cl_dma_memcpy_1d_sync_chunked(
                    PI_CL_DMA_DIR_EXT2LOC,
                    (uint32_t)(cfg->weight_l2 + next_out_start * cfg->in_features),
                    (uint32_t)(next_weights_l1),
                    next_weight_bytes
                );
                has_next_weight = 0;
            }
        }

        // --- STEP 2: STORE PREV OUTPUT (i-1) async ---
        pi_cl_dma_copy_t prev_output_copy;
        dma_async_future_t prev_output_future;
        int has_prev_output = 0;
        if (i > 0 && prev_actual_out > 0) {
            prev_output_copy.dir = PI_CL_DMA_DIR_LOC2EXT;
            prev_output_copy.size = prev_actual_out;
            prev_output_copy.loc = (uint32_t)(prev_output_l1);
            prev_output_copy.ext = (uint32_t)(cfg->output_buffer_l2 + prev_out_start);
            prev_output_copy.merge = 0;
            has_prev_output = (dma_contract_l2_copy_start(&prev_output_future, &prev_output_copy) == 0);
        }

        // --- STEP 3: COMPUTE CURRENT TILE (i) with fusion ---
        int32_t *tile_bias = NULL;
        if (cfg->bias_l2) {
            tile_bias = cfg->bias_l2 + out_start;
        }
        linear_tile_args_t tile_args = {
            .input_l1 = buf_mgr.input,
            .weights_l2 = curr_weights_l1,
            .bias_l2 = tile_bias,
            .output_l1 = curr_output_l1,
            .dim_in = cfg->in_features,
            .dim_out = actual_tile_out,
            .scale_input = cfg->scale_input,
            .scale_weight = cfg->scale_weight,
            .scale_output = cfg->scale_output
        };
#ifdef ENABLE_PERF_COUNTERS
        if (cfg->perf_counter) {
            perf_compute_start();
        }
#endif
        pi_cl_team_fork(NUM_CORES, linear_tile_worker, &tile_args);
#ifdef ENABLE_PERF_COUNTERS
        if (cfg->perf_counter) {
            cfg->perf_counter->compute_cycles += perf_compute_end();
        }
#endif

        // Apply fused operations on L1 tile output
        if (cfg->fusion_relu || cfg->fusion_quant) {
            if (cfg->fusion_relu) {
                relu_l1_args_t a = {.data=curr_output_l1, .size=actual_tile_out};
                pi_cl_team_fork(NUM_CORES, relu_l1_worker, &a);
            }
            if (cfg->relu_output_scale != 0.0f && fabsf(cfg->relu_output_scale - cfg->scale_output) > 1e-5f) {
                requantize_l1_args_t a = {.data=curr_output_l1, .size=actual_tile_out,
                                          .scale_in=cfg->scale_output, .scale_out=cfg->relu_output_scale};
                pi_cl_team_fork(NUM_CORES, requantize_l1_worker, &a);
            }
            if (cfg->fusion_quant) {
                requantize_l1_args_t a = {.data=curr_output_l1, .size=actual_tile_out,
                                          .scale_in=cfg->quant_scale_in, .scale_out=cfg->quant_scale_out};
                pi_cl_team_fork(NUM_CORES, requantize_l1_worker, &a);
            }
        }

        // --- STEP 4: WAIT ON ASYNC DMAs ---
        if (has_next_weight) dma_contract_l2_copy_wait(&next_weight_future);
        if (has_prev_output) dma_contract_l2_copy_wait(&prev_output_future);

        // --- STEP 5: UPDATE STATE & SWAP BUFFERS ---
        prev_output_l1 = curr_output_l1;
        prev_out_start = out_start;
        prev_actual_out = actual_tile_out;

        if (i < cfg->num_tiles - 1) {
            tile_buffer_linear_swap(&buf_mgr);
        }
    }

    // EPILOGUE: Store last tile
    if (prev_actual_out > 0) {
        pi_cl_dma_copy_t final_output_copy;
        final_output_copy.dir = PI_CL_DMA_DIR_LOC2EXT;
        final_output_copy.size = prev_actual_out;
        final_output_copy.loc = (uint32_t)(prev_output_l1);
        final_output_copy.ext = (uint32_t)(cfg->output_buffer_l2 + prev_out_start);
        final_output_copy.merge = 0;
        dma_contract_l2_copy_sync(&final_output_copy);
    }
}

// ---
// Linear INT8 Main Pipeline (Outer Loop: L3 -> L2 Weight Streaming)
// ---

void linear_int8_tiled_l1_pipeline(linear_int8_pipeline_config_t *cfg) {
    if (cfg->l3_tiling_enabled) {
        int slab_idx = 0;
        int current_buff = 0; 
        
        int slab_out_feats = cfg->l3_tile_out_features;
        size_t weight_slab_bytes = slab_out_feats * cfg->in_features;
        size_t bias_slab_bytes = slab_out_feats * 4; 
        size_t out_slab_bytes = slab_out_feats; 

        //  Persistent async request for L3 streaming
        pi_cl_ram_req_t async_req_w;

        // --- PROLOGUE: Load Slab 0 ---
        {
            int out_start = 0;
            int out_end = slab_out_feats;
            if (out_end > cfg->out_features) out_end = cfg->out_features;
            int actual_slab = out_end - out_start;

            // Load Weights 0 (Sync)
            dma_layout_t l3_w = { .base_addr = (int8_t*)cfg->l3_weight_addr, .width = actual_slab * cfg->in_features, .height = 1, .channels = 1, .loc = MEM_LOC_L3, .ram_dev = cfg->ram_dev };
            dma_layout_t l2_w = { .base_addr = cfg->weight_l2, .width = actual_slab * cfg->in_features, .height = 1, .channels = 1, .loc = MEM_LOC_L2, .ram_dev = cfg->ram_dev };
            execute_dma_transfer(&l3_w, &l2_w);

            if (cfg->l3_bias_addr) {
                dma_layout_t l3_b = { .base_addr = (float*)cfg->l3_bias_addr, .width = actual_slab * 4, .height = 1, .channels = 1, .loc = MEM_LOC_L3, .ram_dev = cfg->ram_dev };
                dma_layout_t l2_b = { .base_addr = cfg->bias_l2, .width = actual_slab * 4, .height = 1, .channels = 1, .loc = MEM_LOC_L2, .ram_dev = cfg->ram_dev };
                execute_dma_transfer(&l3_b, &l2_b);
            }

        }

        // --- STEADY STATE ---
        for (slab_idx = 0; slab_idx < cfg->num_l3_tiles; slab_idx++) {
            int out_start = slab_idx * slab_out_feats;
            int out_end = out_start + slab_out_feats;
            if (out_end > cfg->out_features) out_end = cfg->out_features;
            int actual_slab = out_end - out_start;

            int8_t *curr_w = cfg->weight_l2 + (current_buff * weight_slab_bytes);
            int32_t *curr_b = (int32_t*)((int8_t*)cfg->bias_l2 + (current_buff * bias_slab_bytes));
            int8_t *curr_o = cfg->output_buffer_l2 + (current_buff * out_slab_bytes);

            //  Save original out_features BEFORE async load calculation
            // The async load block needs the FULL output size, not the slab size
            int original_out_features = cfg->out_features;

            int next_idx = slab_idx + 1;
            int next_buff = 1 - current_buff;

            if (next_idx < cfg->num_l3_tiles) {
                //  Use original_out_features for async load calculation
                int next_out_start = next_idx * slab_out_feats;
                int next_out_end = next_out_start + slab_out_feats;
                if (next_out_end > original_out_features) next_out_end = original_out_features;
                int next_actual = next_out_end - next_out_start;

                int8_t *next_w_ptr = cfg->weight_l2 + (next_buff * weight_slab_bytes);

                // Start Async Load for Next Slab (threaded request)
                int width_bytes = next_actual * cfg->in_features;
                dma_layout_t l3_w = { .base_addr = (int8_t*)cfg->l3_weight_addr + (next_out_start * cfg->in_features),
                                      .width = width_bytes, .height = 1, .channels = 1, .loc = MEM_LOC_L3,
                                      .is_async = 1, .ram_dev = cfg->ram_dev, .ram_cmd = &async_req_w }; // [ENABLED: Async L3 prefetch]
                dma_layout_t l2_w = { .base_addr = next_w_ptr, .width = width_bytes, .height = 1, .channels = 1, .loc = MEM_LOC_L2, .ram_dev = cfg->ram_dev };

                execute_dma_transfer(&l3_w, &l2_w);
            }
            
            int original_out = cfg->out_features;
            int8_t *original_w = cfg->weight_l2;
            int32_t *original_b = cfg->bias_l2;
            int8_t *original_o = cfg->output_buffer_l2;
            
            cfg->out_features = actual_slab;
            cfg->weight_l2 = curr_w;
            cfg->bias_l2 = curr_b;
            cfg->output_buffer_l2 = curr_o;

            linear_int8_tiled_l1_inner_loop(cfg);

            cfg->out_features = original_out;
            cfg->weight_l2 = original_w;
            cfg->bias_l2 = original_b;
            cfg->output_buffer_l2 = original_o;
            
            if (next_idx < cfg->num_l3_tiles) {
                 pi_cl_ram_read_wait(&async_req_w);

                 // Use original_out_features here (this block runs after cfg->out_features was modified)
                 int next_out_start = next_idx * slab_out_feats;
                 int next_out_end = next_out_start + slab_out_feats;
                 if (next_out_end > original_out_features) next_out_end = original_out_features;
                 int next_actual = next_out_end - next_out_start;
                 int8_t *next_b_ptr = (int8_t*)cfg->bias_l2 + (next_buff * bias_slab_bytes);

                 if (cfg->l3_bias_addr) {
                     dma_layout_t l3_b = { .base_addr = (float*)cfg->l3_bias_addr + next_out_start, .width = next_actual * 4, .height = 1, .channels = 1, .loc = MEM_LOC_L3, .ram_dev = cfg->ram_dev };
                     dma_layout_t l2_b = { .base_addr = next_b_ptr, .width = next_actual * 4, .height = 1, .channels = 1, .loc = MEM_LOC_L2, .ram_dev = cfg->ram_dev };
                     execute_dma_transfer(&l3_b, &l2_b);
                 }
            }

            // Assemble output slabs into final output buffer (L2)
            dma_layout_t l2_src = { .base_addr = curr_o, .width = actual_slab, .height = 1, .channels = 1, .loc = MEM_LOC_L2, .ram_dev = cfg->ram_dev };
            dma_layout_t l2_dst = { .base_addr = (int8_t*)cfg->l3_output_addr + out_start, .width = actual_slab, .height = 1, .channels = 1, .loc = MEM_LOC_L2, .ram_dev = cfg->ram_dev };
            execute_dma_transfer(&l2_src, &l2_dst);

            current_buff = 1 - current_buff;
        }
    } else {
        linear_int8_tiled_l1_inner_loop(cfg);
    }
}

// ---
// Linear FP32 Tiled L1 Pipeline (Final classifier layer)
// ---

/**
 * Linear FP32 Tiled L1 Pipeline for final classifier layer.
 *
 * Implements output feature tiling with INT8 → FP32 conversion:
 * - Loads input once to L1 (reused across all tiles)
 * - Processes output features in tiles (e.g., 10 classes → 1 tile)
 * - Each tile: Load weights → Compute → Store output
 * - INT8 weights x INT8 input → FP32 logits (no requantization)
 *
 * TILING STRATEGY:
 * - Input features: No tiling (full input loaded once)
 * - Output features: Tiled to fit weight matrix in L1
 * - Each tile loads slice of weight matrix (in_features x tile_out_features)
 *
 * MEMORY LAYOUT (L1):
 
 * - Input buffer: Shared across all tiles (loaded once, INT8)
 * - Weight buffer A: Weights for tile N (INT8)
 * - Output buffer A: Output for tile N (FP32)
 * - Weight buffer B: Weights for tile N+1 (double-buffering)
 * - Output buffer B: Output for tile N+1 (FP32)
 *
 * INT8 → FP32 CONVERSION:
 * - Computes: output_fp32 = (input_int8 x weight_int8) * (scale_input * scale_weight) + bias_fp32
 * - No output quantization (final layer produces FP32 logits for softmax/argmax)
 *
 * FALLBACK BEHAVIOR:
 * - If L1 buffer unavailable or too small: Falls back to L2-only 
execution (no tiling)
 * - L2 execution loses L1 latency benefits but maintains correctness
 *
 * @param cfg Pipeline configuration containing:
 * - input L2 buffer (INT8)
 * - output L2 buffer (FP32)
 * - weight/bias parameters (INT8 weights, FP32 bias)
 * - tiling dimensions (num_tiles, tile_out_features)
 * - quantization scales (input and weight only, no output scale)
 * - optional performance counters
 */
void linear_fp32_tiled_l1_pipeline(linear_fp32_pipeline_config_t *cfg) {
    // Check L1 buffer availability
    const size_t single_tile_size = cfg->l1_output_size + cfg->l1_weight_size;
    const size_t required_l1_size = cfg->l1_input_size + 2 * single_tile_size;

    if (!cfg->l1_buffer || cfg->l1_buffer_size < required_l1_size) {
        // L1 buffer not available or too small - fallback to L2
#ifndef MINIMAL_OUTPUT
        if (cfg->l1_buffer) {
            printf("CL: L1 buffer too small for %s (need %zu, have %zu), using L2 fallback\n",
                   cfg->layer_name, required_l1_size, cfg->l1_buffer_size);
        } else {
            printf("CL: L1 buffer not available for %s, using L2 fallback\n", cfg->layer_name);
        }
#endif

        // Execute on L2 (no tiling)
        linear_to_fp32_tile_args_t linear_args = {
            .input_l1 = cfg->input_buffer_l2,
            .weights_l2 = cfg->weight_l2,
            .bias_l2 = cfg->bias_l2,
            .output_l1 = cfg->output_buffer_l2,
            .dim_in = cfg->in_features,
      
       .dim_out = cfg->out_features,
            .scale_input = cfg->scale_input,
            .scale_weight = cfg->scale_weight
        };
#ifdef ENABLE_PERF_COUNTERS
        perf_compute_start();
#endif
        pi_cl_team_fork(NUM_CORES, linear_to_fp32_tile_worker, &linear_args);
#ifdef ENABLE_PERF_COUNTERS
        if (cfg->perf_counter) {
            cfg->perf_counter->compute_cycles += perf_compute_end();
}
#endif
        return;
    }

    // L1 buffer available - use double-buffering
    tile_buffer_linear_mgr_t buf_mgr;
    tile_buffer_linear_init(&buf_mgr, cfg->l1_buffer, cfg->l1_input_size,
                            cfg->l1_output_size, cfg->l1_weight_size);
#ifndef MINIMAL_OUTPUT
    printf("CL: %s using L1 double-buffer tiling: %d tiles (tile_out=%d)\n",
           cfg->layer_name, cfg->num_tiles, cfg->tile_out_features);
#endif

    // Load input once (L2 → L1) - reused across all output tiles
    pi_cl_dma_copy_t input_copy;
    input_copy.dir = PI_CL_DMA_DIR_EXT2LOC;
    input_copy.size = cfg->in_features;
    input_copy.ext = (uint32_t)(cfg->input_buffer_l2);
    input_copy.loc = (uint32_t)(buf_mgr.input);
    input_copy.merge = 0;
    dma_contract_l2_copy_sync(&input_copy);

    // PROLOGUE: Load weights for tile 0
    int8_t *tile0_weights_l1 = tile_buffer_linear_get_weights(&buf_mgr);
    int tile0_out_start = 0;
    int tile0_out_end = cfg->tile_out_features;
    if (tile0_out_end > cfg->out_features) tile0_out_end = cfg->out_features;
    int tile0_actual_out = tile0_out_end - tile0_out_start;

    pi_cl_dma_copy_t weight_copy;
    weight_copy.dir = PI_CL_DMA_DIR_EXT2LOC;
    weight_copy.size = tile0_actual_out * cfg->in_features;
    weight_copy.ext = (uint32_t)(cfg->weight_l2);
    weight_copy.loc = (uint32_t)(tile0_weights_l1);
    weight_copy.merge = 0;
    dma_contract_l2_copy_sync(&weight_copy);

    // Track previous tile for look-behind store
    float *prev_output_l1 = NULL;
    int prev_out_start = 0;
    int prev_actual_out = 0;

    // STEADY-STATE LOOP: Look-Ahead/Look-Behind Pattern
    //  Calculate num_tiles dynamically for L3 tiling (cfg->out_features is slab size, not full size)
    int num_tiles_to_process = (cfg->out_features + cfg->tile_out_features - 1) / cfg->tile_out_features;
    for (int i = 0; i < num_tiles_to_process; i++) {
        // Get current tile buffers
        float *curr_output_l1 = (float *)tile_buffer_linear_get_output(&buf_mgr);
        int8_t *curr_weights_l1 = tile_buffer_linear_get_weights(&buf_mgr);

        // Calculate current tile geometry
        int out_start = i * cfg->tile_out_features;
        int out_end = out_start + cfg->tile_out_features;
        if (out_end > cfg->out_features) out_end = cfg->out_features;
        int actual_tile_out = out_end - out_start;

        // --- STEP 1: LOAD NEXT WEIGHTS (i+1) async ---
        pi_cl_dma_copy_t next_weight_copy;
        dma_async_future_t next_weight_future;
        int has_next_weight = 0;
        if (i < num_tiles_to_process - 1) {
            int8_t *next_weights_l1 = tile_buffer_linear_get_next_weights(&buf_mgr);
            int next_out_start = (i + 1) * cfg->tile_out_features;
            int next_out_end = next_out_start + cfg->tile_out_features;
            if (next_out_end > cfg->out_features) next_out_end = cfg->out_features;
            int next_actual_out = next_out_end - next_out_start;

            next_weight_copy.dir = PI_CL_DMA_DIR_EXT2LOC;
            next_weight_copy.size = next_actual_out * cfg->in_features;
            next_weight_copy.ext = (uint32_t)(cfg->weight_l2 + next_out_start * cfg->in_features);
            next_weight_copy.loc = (uint32_t)(next_weights_l1);
            next_weight_copy.merge = 0;
            has_next_weight = (dma_contract_l2_copy_start(&next_weight_future, &next_weight_copy) == 0);
        }

        // --- STEP 2: STORE PREV OUTPUT (i-1) async ---
        pi_cl_dma_copy_t prev_output_copy;
        dma_async_future_t prev_output_future;
        int has_prev_output = 0;
        if (i > 0 && prev_actual_out > 0) {
            prev_output_copy.dir = PI_CL_DMA_DIR_LOC2EXT;
            prev_output_copy.size = prev_actual_out * sizeof(float);
            prev_output_copy.loc = (uint32_t)(prev_output_l1);
            prev_output_copy.ext = (uint32_t)(cfg->output_buffer_l2 + prev_out_start);
            prev_output_copy.merge = 0;
            has_prev_output = (dma_contract_l2_copy_start(&prev_output_future, &prev_output_copy) == 0);
        }

        // --- STEP 3: COMPUTE CURRENT TILE (i) ---
        linear_to_fp32_tile_args_t tile_args = {
            .input_l1 = buf_mgr.input,
            .weights_l2 = curr_weights_l1,
            .bias_l2 = cfg->bias_l2 ? (cfg->bias_l2 + out_start) : NULL,
            .output_l1 = curr_output_l1,
            .dim_in = cfg->in_features,
            .dim_out = actual_tile_out,
            .scale_input = cfg->scale_input,
            .scale_weight = cfg->scale_weight
        };
#ifdef ENABLE_PERF_COUNTERS
        if (cfg->perf_counter) {
            perf_compute_start();
        }
#endif
        pi_cl_team_fork(NUM_CORES, linear_to_fp32_tile_worker, &tile_args);
#ifdef ENABLE_PERF_COUNTERS
        if (cfg->perf_counter) {
            cfg->perf_counter->compute_cycles += perf_compute_end();
        }
#endif

        // --- STEP 4: WAIT ON ASYNC DMAs ---
        if (has_next_weight) dma_contract_l2_copy_wait(&next_weight_future);
        if (has_prev_output) dma_contract_l2_copy_wait(&prev_output_future);

        // --- STEP 5: UPDATE STATE & SWAP BUFFERS ---
        prev_output_l1 = curr_output_l1;
        prev_out_start = out_start;
        prev_actual_out = actual_tile_out;

        if (i < cfg->num_tiles - 1) {
            tile_buffer_linear_swap(&buf_mgr);
        }
    }

    // EPILOGUE: Store last tile
    if (prev_actual_out > 0) {
        pi_cl_dma_copy_t final_output_copy;
        final_output_copy.dir = PI_CL_DMA_DIR_LOC2EXT;
        final_output_copy.size = prev_actual_out * sizeof(float);
        final_output_copy.loc = (uint32_t)(prev_output_l1);
        final_output_copy.ext = (uint32_t)(cfg->output_buffer_l2 + prev_out_start);
        final_output_copy.merge = 0;
        dma_contract_l2_copy_sync(&final_output_copy);
    }
}

// ---
// MHSA Tiled L1 Pipeline (Depth-First Tiling)
// ---

/**
 * Worker kernel arguments for MHSA tile operations.
 * Shared across all three worker kernels (QK, softmax, AV).
 */
	typedef struct {
	    // Input tensors (INT8 or FP32 depending on kernel)
	    const int8_t *q_tile_l1;    // Query tile [tile_q x head_dim] INT8
	    const int8_t *k_l1;         // Key (full) [seq_len x head_dim] INT8
	    const int8_t *v_l1;         // Value (full) [seq_len x head_dim] INT8
	    const int8_t *v_t_l1;       // Value transposed [head_dim x seq_len] INT8 (optional)
	    float *scores_l1;           // Attention scores [tile_q x seq_len] FP32 (reference path)
	    int32_t *scores_int32_l1;   // Attention scores [tile_q x seq_len] INT32 (optimized)
	    int8_t *scores_int8_l1;     // Requantized scores [tile_q x seq_len] INT8 (for polynomial iSoftmax)
	    uint8_t *attn_uint8_l1;     // Attention weights [tile_q x seq_len] UINT8 (polynomial iSoftmax output)
	    int16_t *attn_weights_l1;   // Attention weights [tile_q x seq_len] INT16 (after softmax)
    int8_t *m_tile_l1;          // Output tile [tile_q x head_dim] INT8

    // Dimensions
    int tile_q;                 // Query tile size (number of rows)
    int seq_len;                // Sequence length
    int head_dim;               // Head dimension

    // Scales
    float scale_q;              // Query scale
    float scale_k;              // Key scale
    float scale_v;              // Value scale
    float scale_output;         // Output scale
    float softmax_scale;        // 1/sqrt(head_dim)

    // Requantization for fully-integer softmax (INT32 diff -> INT8 for LUT)
    // INT8 = (diff_int32 * requant_mul + round) >> requant_shift
    // where diff_int32 = score - max_score (always <= 0)
    int32_t requant_mul;        // Requantization multiplier (can be large with shift=24)
    int32_t requant_shift;      // Requantization shift (right shift amount)

    // Polynomial i-Softmax coefficients (from reference: iSoftmax.c)
    // Formula: y = ((coeffA*(p+coeffB)^2 + coeffC) >> z) where z = -(x / log2), p = x + z*log2
    int32_t isoftmax_coeffA;    // Quadratic coefficient
    int32_t isoftmax_coeffB;    // Linear offset
    int32_t isoftmax_coeffC;    // Constant term
    int32_t isoftmax_log2;      // ln(2) in fixed point for range reduction
    uint32_t isoftmax_n_levels; // Output levels (256 for UINT8)

    // Mode selector
    int use_integer_softmax;    // 1 = polynomial iSoftmax (fully integer), 0 = FP32 fast_exp

    // i-Softmax LUT for bit-exact matching (NULL = use fast_exp FP32 path)
    const int16_t *softmax_lut;
} mhsa_tile_args_t;

// SIMD types for optimized MHSA kernels
typedef int8_t v4s __attribute__((vector_size(4)));
#ifndef SumDotpSS
#define SumDotpSS(a, b, c) __builtin_pulp_sdotsp4(a, b, c)
#endif

// i-Softmax LUT parameters (must match network_kernels.c constants)
#ifndef I_SOFTMAX_INPUT_MIN
#define I_SOFTMAX_INPUT_MIN (-8.0f)
#define I_SOFTMAX_INPUT_STEP (8.0f / 1024.0f)  // 0.0078125
#define I_SOFTMAX_NUM_ENTRIES 1024
#define I_SOFTMAX_OUTPUT_SCALE 32767
#endif

// SIMD intrinsic for unsigned x signed dot product (UINT8 attn x INT8 V)
#ifndef SumDotpUS
#define SumDotpUS(a, b, c) __builtin_pulp_sdotusp4(a, b, c)
#endif
typedef uint8_t v4u __attribute__((vector_size(4)));

// Clipping helper
#ifndef clip8
#define clip8(x) __builtin_pulp_clip_r(x, 127)
#endif

// MHSA tuning macros are centralized in `codegen/runtime/inc/ares_config.h`.

/**
 * Worker kernel: Compute Q_tile x K^T with optional INT8 requantization
 * SIMD-optimized version: Uses SumDotpSS for 4 MACs per cycle
 *
 * Output modes:
 * - Always outputs INT32 scores to scores_int32_l1
 * - If use_integer_softmax: also outputs requantized INT8 to scores_int8_l1
 */
static void mhsa_tile_qk_worker(void *arg) {
    mhsa_tile_args_t *a = (mhsa_tile_args_t *)arg;
    const int core_id = pi_core_id();
    const int head_dim = a->head_dim;
    const int simd_iters = head_dim >> 2;
    const int tail_start = simd_iters << 2;
    const int16_t requant_mul = a->requant_mul;
    const int16_t requant_shift = a->requant_shift;
    const int use_int_softmax = a->use_integer_softmax;
    const int seq_len = a->seq_len;

    for (int i = core_id; i < a->tile_q; i += NUM_CORES) {
        const int8_t *q_row = a->q_tile_l1 + i * head_dim;
        int32_t *score_row_int32 = a->scores_int32_l1 + i * seq_len;
        int8_t *score_row_int8 =
            (use_int_softmax && MHSA_TILED_QK_STORE_INT8_SCORES && a->scores_int8_l1)
                ? (a->scores_int8_l1 + i * seq_len)
                : NULL;

#if MHSA_QK_UNROLL_HEAD_DIM_64
        if (head_dim == 64) {
            // Unroll d loop by 2 for head_dim=64 (simd_iters=16).
            int j = 0;
            for (; j + 4 <= seq_len; j += 4) {
                const int8_t *k_row0 = a->k_l1 + (j + 0) * head_dim;
                const int8_t *k_row1 = a->k_l1 + (j + 1) * head_dim;
                const int8_t *k_row2 = a->k_l1 + (j + 2) * head_dim;
                const int8_t *k_row3 = a->k_l1 + (j + 3) * head_dim;

                int32_t dot0 = 0, dot1 = 0, dot2 = 0, dot3 = 0;

                for (int d = 0; d < 16; d += 2) {
                    int off0 = d << 2;
                    int off1 = (d + 1) << 2;
                    v4s q0 = *((v4s *)(q_row + off0));
                    v4s q1 = *((v4s *)(q_row + off1));

                    v4s k00 = *((v4s *)(k_row0 + off0));
                    v4s k01 = *((v4s *)(k_row0 + off1));
                    v4s k10 = *((v4s *)(k_row1 + off0));
                    v4s k11 = *((v4s *)(k_row1 + off1));
                    v4s k20 = *((v4s *)(k_row2 + off0));
                    v4s k21 = *((v4s *)(k_row2 + off1));
                    v4s k30 = *((v4s *)(k_row3 + off0));
                    v4s k31 = *((v4s *)(k_row3 + off1));

                    dot0 = SumDotpSS(q0, k00, dot0);
                    dot0 = SumDotpSS(q1, k01, dot0);
                    dot1 = SumDotpSS(q0, k10, dot1);
                    dot1 = SumDotpSS(q1, k11, dot1);
                    dot2 = SumDotpSS(q0, k20, dot2);
                    dot2 = SumDotpSS(q1, k21, dot2);
                    dot3 = SumDotpSS(q0, k30, dot3);
                    dot3 = SumDotpSS(q1, k31, dot3);
                }

                score_row_int32[j + 0] = dot0;
                score_row_int32[j + 1] = dot1;
                score_row_int32[j + 2] = dot2;
                score_row_int32[j + 3] = dot3;

                if (score_row_int8) {
                    score_row_int8[j + 0] = clip8((dot0 * requant_mul) >> requant_shift);
                    score_row_int8[j + 1] = clip8((dot1 * requant_mul) >> requant_shift);
                    score_row_int8[j + 2] = clip8((dot2 * requant_mul) >> requant_shift);
                    score_row_int8[j + 3] = clip8((dot3 * requant_mul) >> requant_shift);
                }
            }
            for (; j < seq_len; j++) {
                const int8_t *k_row = a->k_l1 + j * head_dim;
                int32_t dot = 0;

                for (int d = 0; d < 16; d += 2) {
                    int off0 = d << 2;
                    int off1 = (d + 1) << 2;
                    v4s q0 = *((v4s *)(q_row + off0));
                    v4s q1 = *((v4s *)(q_row + off1));
                    v4s k0 = *((v4s *)(k_row + off0));
                    v4s k1 = *((v4s *)(k_row + off1));
                    dot = SumDotpSS(q0, k0, dot);
                    dot = SumDotpSS(q1, k1, dot);
                }

                score_row_int32[j] = dot;
                if (score_row_int8) {
                    score_row_int8[j] = clip8((dot * requant_mul) >> requant_shift);
                }
            }
        } else
#endif
        {
            // Unroll over keys to reuse q_vec loads across multiple K rows.
            int j = 0;
            for (; j + 4 <= seq_len; j += 4) {
                const int8_t *k_row0 = a->k_l1 + (j + 0) * head_dim;
                const int8_t *k_row1 = a->k_l1 + (j + 1) * head_dim;
                const int8_t *k_row2 = a->k_l1 + (j + 2) * head_dim;
                const int8_t *k_row3 = a->k_l1 + (j + 3) * head_dim;

                int32_t dot0 = 0, dot1 = 0, dot2 = 0, dot3 = 0;

                for (int d = 0; d < simd_iters; d++) {
                    v4s q_vec = *((v4s *)(q_row + (d << 2)));
                    v4s k0 = *((v4s *)(k_row0 + (d << 2)));
                    v4s k1 = *((v4s *)(k_row1 + (d << 2)));
                    v4s k2 = *((v4s *)(k_row2 + (d << 2)));
                    v4s k3 = *((v4s *)(k_row3 + (d << 2)));
                    dot0 = SumDotpSS(q_vec, k0, dot0);
                    dot1 = SumDotpSS(q_vec, k1, dot1);
                    dot2 = SumDotpSS(q_vec, k2, dot2);
                    dot3 = SumDotpSS(q_vec, k3, dot3);
                }
                for (int d = tail_start; d < head_dim; d++) {
                    const int32_t q = (int32_t)q_row[d];
                    dot0 += q * (int32_t)k_row0[d];
                    dot1 += q * (int32_t)k_row1[d];
                    dot2 += q * (int32_t)k_row2[d];
                    dot3 += q * (int32_t)k_row3[d];
                }

                score_row_int32[j + 0] = dot0;
                score_row_int32[j + 1] = dot1;
                score_row_int32[j + 2] = dot2;
                score_row_int32[j + 3] = dot3;

                if (score_row_int8) {
                    score_row_int8[j + 0] = clip8((dot0 * requant_mul) >> requant_shift);
                    score_row_int8[j + 1] = clip8((dot1 * requant_mul) >> requant_shift);
                    score_row_int8[j + 2] = clip8((dot2 * requant_mul) >> requant_shift);
                    score_row_int8[j + 3] = clip8((dot3 * requant_mul) >> requant_shift);
                }
            }
            for (; j < seq_len; j++) {
                const int8_t *k_row = a->k_l1 + j * head_dim;
                int32_t dot = 0;

                for (int d = 0; d < simd_iters; d++) {
                    v4s q_vec = *((v4s *)(q_row + (d << 2)));
                    v4s k_vec = *((v4s *)(k_row + (d << 2)));
                    dot = SumDotpSS(q_vec, k_vec, dot);
                }
                for (int d = tail_start; d < head_dim; d++) {
                    dot += (int32_t)q_row[d] * (int32_t)k_row[d];
                }

                score_row_int32[j] = dot;
                if (score_row_int8) {
                    score_row_int8[j] = clip8((dot * requant_mul) >> requant_shift);
                }
            }
        }
    }
}

// Pre-computed LUT for integer softmax: exp(x) * 2^24 for x in [-128, 0]
// Index: x + 128 (so index 0 = exp(-128), index 128 = exp(0))
// This gives exact exp() values in fixed-point for perfect softmax matching
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

/**
 * LUT-based integer softmax for one row.
 *
 * Algorithm:
 * 1. Find max value x_max for numerical stability
 * 2. First pass: compute sum of exp() values via LUT lookup
 * 3. Second pass: normalize each value = lut_val * 255 / sum
 *
 * This gives BIT-EXACT results compared to FP32 softmax.
 * Two-pass implementation avoids large stack arrays on cluster cores.
 */
static void isoftmax_row(
    const int8_t *input,
    uint8_t *output,
    int length,
    int32_t coeffA,     // Unused; kept for API compatibility.
    int32_t coeffB,     // Unused
    int32_t coeffC,     // Unused
    int32_t log2_val,   // Unused
    uint32_t n_levels
) {
    (void)coeffA; (void)coeffB; (void)coeffC; (void)log2_val;  // Suppress warnings

    // Step 1: Find max
    int8_t x_max = -128;
    for (int i = 0; i < length; i++) {
        if (input[i] > x_max) x_max = input[i];
    }

    // Step 2: First pass - compute sum of LUT values
    uint64_t y_sum = 0;
    for (int i = 0; i < length; i++) {
        int16_t diff = (int16_t)input[i] - (int16_t)x_max;  // Always <= 0
        int idx = diff + 128;  // Map to LUT index [0, 128]
        if (idx < 0) idx = 0;  // Clamp for safety
        if (idx > 128) idx = 128;

        y_sum += i_softmax_lut_int8[idx];
    }

    // Step 3: Second pass - normalize to n_levels (typically 256)
    if (y_sum > 0) {
        for (int i = 0; i < length; i++) {
            int16_t diff = (int16_t)input[i] - (int16_t)x_max;
            int idx = diff + 128;
            if (idx < 0) idx = 0;
            if (idx > 128) idx = 128;

            uint32_t y_i = i_softmax_lut_int8[idx];
            output[i] = (uint8_t)((y_i * (n_levels - 1)) / y_sum);
        }
    } else {
        // Uniform distribution if sum is zero (shouldn't happen with proper input)
        uint8_t uniform = (n_levels - 1) / length;
        for (int i = 0; i < length; i++) {
            output[i] = uniform;
        }
    }
}

/**
 * Worker kernel: Apply row-wise softmax (INT32 → FP32) - FP32 path
 * Converts INT32 scores to FP32, applies softmax, outputs FP32 attention weights.
 */
static void mhsa_tile_softmax_fp32_worker(void *arg) {
    mhsa_tile_args_t *a = (mhsa_tile_args_t *)arg;
    const int core_id = pi_core_id();
    const int seq_len = a->seq_len;
    const float combined_scale = a->scale_q * a->scale_k * a->softmax_scale;

    for (int i = core_id; i < a->tile_q; i += NUM_CORES) {
        const int32_t *score_row_int32 = a->scores_int32_l1 + i * seq_len;
        float *attn_row_fp32 = a->scores_l1 + i * seq_len;

        // Convert INT32 to FP32 and find max
        float max_val = (float)score_row_int32[0] * combined_scale;
        for (int j = 0; j < seq_len; j++) {
            float val = (float)score_row_int32[j] * combined_scale;
            attn_row_fp32[j] = val;
            if (val > max_val) max_val = val;
        }

        // Compute exp(x - max) and sum
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            attn_row_fp32[j] = fast_exp(attn_row_fp32[j] - max_val);
            sum += attn_row_fp32[j];
        }

        // Normalize
        float inv_sum = (sum > 1e-8f) ? (1.0f / sum) : 0.0f;
        for (int j = 0; j < seq_len; j++) {
            attn_row_fp32[j] *= inv_sum;
        }
    }
}

/**
 * Worker kernel: Apply LUT-based i-Softmax (INT32 → UINT8) - FULLY INTEGER path
 *
 * This kernel computes softmax entirely in integer arithmetic:
 * 1. Find max INT32 score (no FP32 needed - monotonic)
 * 2. Compute INT32 differences: diff = score - max (always <= 0)
 * 3. Requantize to INT8: x_int8 = (diff * requant_mul) >> requant_shift
 * 4. LUT lookup: exp_val = LUT[x_int8 + 128]
 * 5. Normalize: attn = exp_val * 255 / sum(exp_vals) (integer division)
 *
 * The requantization parameters are pre-computed to match Python:
 *   requant_scale = scale_q * scale_k * softmax_scale * 16.0
 * This is converted to fixed-point: (diff * requant_mul) >> requant_shift
 *
 * Output: UINT8 attention weights in [0, 255]
 */
static void mhsa_tile_softmax_int_worker(void *arg) {
    mhsa_tile_args_t *a = (mhsa_tile_args_t *)arg;
    const int core_id = pi_core_id();
    const int seq_len = a->seq_len;

    // Pre-computed requantization: converts INT32 diff to INT8 for LUT
    // requant_scale = scale_q * scale_k * softmax_scale * 16.0
    // We use fixed-point: (diff * requant_mul + round) >> requant_shift
    const int32_t requant_mul = a->requant_mul;
    const int32_t requant_shift = a->requant_shift;
    const int64_t round_val = ((requant_shift > 0) && (requant_shift < 63)) ? (1LL << (requant_shift - 1)) : 0;

    for (int i = core_id; i < a->tile_q; i += NUM_CORES) {
        const int32_t *score_row_int32 = a->scores_int32_l1 + i * seq_len;
        uint8_t *attn_row_uint8 = a->attn_uint8_l1 + i * seq_len;

        // OPTIMIZED: Loop unrolling + cached LUT indices
        //
        // Optimizations applied:
        // 1. 4x loop unrolling for find-max (better ILP, reduced loop overhead)
        // 2. Cache LUT index instead of value (avoid duplicate 64-bit requantization)
        // 3. 4x loop unrolling for normalize (better ILP)

        // Step 1: Find max INT32 score - UNROLLED 4x for better ILP
        int32_t max0 = score_row_int32[0];
        int32_t max1 = max0, max2 = max0, max3 = max0;
        int j;
        const int unroll_limit = seq_len - 3;
        for (j = 1; j < unroll_limit; j += 4) {
            int32_t s0 = score_row_int32[j];
            int32_t s1 = score_row_int32[j + 1];
            int32_t s2 = score_row_int32[j + 2];
            int32_t s3 = score_row_int32[j + 3];
            if (s0 > max0) max0 = s0;
            if (s1 > max1) max1 = s1;
            if (s2 > max2) max2 = s2;
            if (s3 > max3) max3 = s3;
        }
        // Handle tail elements
        for (; j < seq_len; j++) {
            if (score_row_int32[j] > max0) max0 = score_row_int32[j];
        }
        // Reduce 4 maxes to 1
        if (max1 > max0) max0 = max1;
        if (max2 > max0) max0 = max2;
        if (max3 > max0) max0 = max3;
        const int32_t max_score = max0;

        // Step 2: FUSED - Requantize + cache LUT index + accumulate sum
        uint64_t y_sum = 0;
        for (j = 0; j < seq_len; j++) {
            int32_t diff = score_row_int32[j] - max_score;  // Always <= 0

            // Requantize: (diff * requant_mul + round) >> requant_shift
            // This maps INT32 diff to INT8 range [-128, 0]
            int32_t x_int = ((int64_t)diff * (int64_t)requant_mul + round_val) >> requant_shift;
            if (x_int < -128) x_int = -128;
            if (x_int > 0) x_int = 0;
            uint8_t idx = (uint8_t)(x_int + 128);
            attn_row_uint8[j] = idx;  // Cache index in output buffer (fits in uint8_t)
            y_sum += i_softmax_lut_int8[idx];
        }

        // Step 3: Normalize using cached indices - UNROLLED 4x
        if (y_sum > 0) {
            // Compute fixed-point inverse (single division per row)
            const uint64_t inv_sum = (255ULL << 24) / y_sum;
            const uint64_t round_norm = (1ULL << 23);

            // Unrolled normalization loop
            for (j = 0; j + 4 <= seq_len; j += 4) {
                uint8_t idx0 = attn_row_uint8[j];
                uint8_t idx1 = attn_row_uint8[j + 1];
                uint8_t idx2 = attn_row_uint8[j + 2];
                uint8_t idx3 = attn_row_uint8[j + 3];
                uint32_t y0 = i_softmax_lut_int8[idx0];
                uint32_t y1 = i_softmax_lut_int8[idx1];
                uint32_t y2 = i_softmax_lut_int8[idx2];
                uint32_t y3 = i_softmax_lut_int8[idx3];
                attn_row_uint8[j]     = (uint8_t)(((uint64_t)y0 * inv_sum + round_norm) >> 24);
                attn_row_uint8[j + 1] = (uint8_t)(((uint64_t)y1 * inv_sum + round_norm) >> 24);
                attn_row_uint8[j + 2] = (uint8_t)(((uint64_t)y2 * inv_sum + round_norm) >> 24);
                attn_row_uint8[j + 3] = (uint8_t)(((uint64_t)y3 * inv_sum + round_norm) >> 24);
            }
            // Handle tail
            for (; j < seq_len; j++) {
                uint8_t idx = attn_row_uint8[j];
                uint32_t y_i = i_softmax_lut_int8[idx];
                attn_row_uint8[j] = (uint8_t)(((uint64_t)y_i * inv_sum + round_norm) >> 24);
            }
        } else {
            // Uniform distribution (shouldn't happen with proper input)
            uint8_t uniform = 255 / seq_len;
            for (j = 0; j < seq_len; j++) {
                attn_row_uint8[j] = uniform;
            }
        }
    }
}

/**
 * Worker kernel: Dispatcher for softmax (selects FP32 or integer path)
 */
static void mhsa_tile_softmax_worker(void *arg) {
    mhsa_tile_args_t *a = (mhsa_tile_args_t *)arg;
    if (a->use_integer_softmax) {
        mhsa_tile_softmax_int_worker(arg);
    } else {
        mhsa_tile_softmax_fp32_worker(arg);
    }
}

/**
 * Worker kernel: Fused integer softmax + AV using transposed V (v_t_l1)
 * This removes one fork/barrier per tile for the integer path.
 */
static void mhsa_tile_softmax_av_int_vt_worker(void *arg) {
    mhsa_tile_args_t *a = (mhsa_tile_args_t *)arg;
    const int core_id = pi_core_id();
    const int seq_len = a->seq_len;
    const int head_dim = a->head_dim;
    const int simd_iters = seq_len >> 2;
    const int tail_start = simd_iters << 2;
    const int32_t requant_mul = a->requant_mul;
    const int32_t requant_shift = a->requant_shift;
    const int64_t round_val = ((requant_shift > 0) && (requant_shift < 63)) ? (1LL << (requant_shift - 1)) : 0;

    for (int i = core_id; i < a->tile_q; i += NUM_CORES) {
        const int32_t *score_row_int32 = a->scores_int32_l1 + i * seq_len;
        uint8_t *attn_row_uint8 = a->attn_uint8_l1 + i * seq_len;
        int8_t *out_row = a->m_tile_l1 + i * head_dim;

        // STEP 1: Integer softmax (bit-exact with mhsa_tile_softmax_int_worker)
        int32_t max0 = score_row_int32[0];
        int32_t max1 = max0, max2 = max0, max3 = max0;
        int j;
        const int unroll_limit = seq_len - 3;
        for (j = 1; j < unroll_limit; j += 4) {
            int32_t s0 = score_row_int32[j];
            int32_t s1 = score_row_int32[j + 1];
            int32_t s2 = score_row_int32[j + 2];
            int32_t s3 = score_row_int32[j + 3];
            if (s0 > max0) max0 = s0;
            if (s1 > max1) max1 = s1;
            if (s2 > max2) max2 = s2;
            if (s3 > max3) max3 = s3;
        }
        for (; j < seq_len; j++) {
            if (score_row_int32[j] > max0) max0 = score_row_int32[j];
        }
        if (max1 > max0) max0 = max1;
        if (max2 > max0) max0 = max2;
        if (max3 > max0) max0 = max3;
        const int32_t max_score = max0;

        uint64_t y_sum = 0;
        for (j = 0; j < seq_len; j++) {
            int32_t diff = score_row_int32[j] - max_score;
            int32_t x_int = ((int64_t)diff * (int64_t)requant_mul + round_val) >> requant_shift;
            if (x_int < -128) x_int = -128;
            if (x_int > 0) x_int = 0;
            uint8_t idx = (uint8_t)(x_int + 128);
            attn_row_uint8[j] = idx;
            y_sum += i_softmax_lut_int8[idx];
        }

        if (y_sum > 0) {
            const uint64_t inv_sum = (255ULL << 24) / y_sum;
            const uint64_t round_norm = (1ULL << 23);
            for (j = 0; j + 4 <= seq_len; j += 4) {
                uint8_t idx0 = attn_row_uint8[j];
                uint8_t idx1 = attn_row_uint8[j + 1];
                uint8_t idx2 = attn_row_uint8[j + 2];
                uint8_t idx3 = attn_row_uint8[j + 3];
                uint32_t y0 = i_softmax_lut_int8[idx0];
                uint32_t y1 = i_softmax_lut_int8[idx1];
                uint32_t y2 = i_softmax_lut_int8[idx2];
                uint32_t y3 = i_softmax_lut_int8[idx3];
                attn_row_uint8[j]     = (uint8_t)(((uint64_t)y0 * inv_sum + round_norm) >> 24);
                attn_row_uint8[j + 1] = (uint8_t)(((uint64_t)y1 * inv_sum + round_norm) >> 24);
                attn_row_uint8[j + 2] = (uint8_t)(((uint64_t)y2 * inv_sum + round_norm) >> 24);
                attn_row_uint8[j + 3] = (uint8_t)(((uint64_t)y3 * inv_sum + round_norm) >> 24);
            }
            for (; j < seq_len; j++) {
                uint8_t idx = attn_row_uint8[j];
                uint32_t y_i = i_softmax_lut_int8[idx];
                attn_row_uint8[j] = (uint8_t)(((uint64_t)y_i * inv_sum + round_norm) >> 24);
            }
        } else {
            uint8_t uniform = 255 / seq_len;
            for (j = 0; j < seq_len; j++) {
                attn_row_uint8[j] = uniform;
            }
        }

        // STEP 2: AV (use transposed V for contiguous loads)
        int d = 0;
        for (; d + 4 <= head_dim; d += 4) {
            const int8_t *v_col0 = a->v_t_l1 + (d + 0) * seq_len;
            const int8_t *v_col1 = a->v_t_l1 + (d + 1) * seq_len;
            const int8_t *v_col2 = a->v_t_l1 + (d + 2) * seq_len;
            const int8_t *v_col3 = a->v_t_l1 + (d + 3) * seq_len;

            int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

            for (int js = 0; js < simd_iters; js++) {
                v4u attn_vec = *((v4u *)(attn_row_uint8 + (js << 2)));
                v4s v_vec0 = *((v4s *)(v_col0 + (js << 2)));
                v4s v_vec1 = *((v4s *)(v_col1 + (js << 2)));
                v4s v_vec2 = *((v4s *)(v_col2 + (js << 2)));
                v4s v_vec3 = *((v4s *)(v_col3 + (js << 2)));
                acc0 = SumDotpUS(attn_vec, v_vec0, acc0);
                acc1 = SumDotpUS(attn_vec, v_vec1, acc1);
                acc2 = SumDotpUS(attn_vec, v_vec2, acc2);
                acc3 = SumDotpUS(attn_vec, v_vec3, acc3);
            }

            for (int jj = tail_start; jj < seq_len; jj++) {
                const int32_t w = (int32_t)attn_row_uint8[jj];
                acc0 += w * (int32_t)v_col0[jj];
                acc1 += w * (int32_t)v_col1[jj];
                acc2 += w * (int32_t)v_col2[jj];
                acc3 += w * (int32_t)v_col3[jj];
            }

            int32_t q0 = (acc0 + 128) >> 8;
            int32_t q1 = (acc1 + 128) >> 8;
            int32_t q2 = (acc2 + 128) >> 8;
            int32_t q3 = (acc3 + 128) >> 8;

            if (q0 > 127) q0 = 127; if (q0 < -128) q0 = -128;
            if (q1 > 127) q1 = 127; if (q1 < -128) q1 = -128;
            if (q2 > 127) q2 = 127; if (q2 < -128) q2 = -128;
            if (q3 > 127) q3 = 127; if (q3 < -128) q3 = -128;

            out_row[d + 0] = (int8_t)q0;
            out_row[d + 1] = (int8_t)q1;
            out_row[d + 2] = (int8_t)q2;
            out_row[d + 3] = (int8_t)q3;
        }

        for (; d < head_dim; d++) {
            const int8_t *v_col = a->v_t_l1 + d * seq_len;
            int32_t acc = 0;

            for (int js = 0; js < simd_iters; js++) {
                v4u attn_vec = *((v4u *)(attn_row_uint8 + (js << 2)));
                v4s v_vec = *((v4s *)(v_col + (js << 2)));
                acc = SumDotpUS(attn_vec, v_vec, acc);
            }

            for (int jj = tail_start; jj < seq_len; jj++) {
                acc += (int32_t)attn_row_uint8[jj] * (int32_t)v_col[jj];
            }

            int32_t q_val = (acc + 128) >> 8;
            if (q_val > 127) q_val = 127;
            if (q_val < -128) q_val = -128;
            out_row[d] = (int8_t)q_val;
        }
    }
}

/**
 * Worker kernel: Fused integer softmax + AV using row-major V (v_l1)
 */
static void mhsa_tile_softmax_av_int_worker(void *arg) {
    mhsa_tile_args_t *a = (mhsa_tile_args_t *)arg;
    const int core_id = pi_core_id();
    const int seq_len = a->seq_len;
    const int head_dim = a->head_dim;
    const int simd_iters = seq_len >> 2;
    const int tail_start = simd_iters << 2;
    const int32_t requant_mul = a->requant_mul;
    const int32_t requant_shift = a->requant_shift;
    const int64_t round_val = ((requant_shift > 0) && (requant_shift < 63)) ? (1LL << (requant_shift - 1)) : 0;

    for (int i = core_id; i < a->tile_q; i += NUM_CORES) {
        const int32_t *score_row_int32 = a->scores_int32_l1 + i * seq_len;
        uint8_t *attn_row_uint8 = a->attn_uint8_l1 + i * seq_len;
        int8_t *out_row = a->m_tile_l1 + i * head_dim;

        int32_t max0 = score_row_int32[0];
        int32_t max1 = max0, max2 = max0, max3 = max0;
        int j;
        const int unroll_limit = seq_len - 3;
        for (j = 1; j < unroll_limit; j += 4) {
            int32_t s0 = score_row_int32[j];
            int32_t s1 = score_row_int32[j + 1];
            int32_t s2 = score_row_int32[j + 2];
            int32_t s3 = score_row_int32[j + 3];
            if (s0 > max0) max0 = s0;
            if (s1 > max1) max1 = s1;
            if (s2 > max2) max2 = s2;
            if (s3 > max3) max3 = s3;
        }
        for (; j < seq_len; j++) {
            if (score_row_int32[j] > max0) max0 = score_row_int32[j];
        }
        if (max1 > max0) max0 = max1;
        if (max2 > max0) max0 = max2;
        if (max3 > max0) max0 = max3;
        const int32_t max_score = max0;

        uint64_t y_sum = 0;
        for (j = 0; j < seq_len; j++) {
            int32_t diff = score_row_int32[j] - max_score;
            int32_t x_int = ((int64_t)diff * (int64_t)requant_mul + round_val) >> requant_shift;
            if (x_int < -128) x_int = -128;
            if (x_int > 0) x_int = 0;
            uint8_t idx = (uint8_t)(x_int + 128);
            attn_row_uint8[j] = idx;
            y_sum += i_softmax_lut_int8[idx];
        }

        if (y_sum > 0) {
            const uint64_t inv_sum = (255ULL << 24) / y_sum;
            const uint64_t round_norm = (1ULL << 23);
            for (j = 0; j + 4 <= seq_len; j += 4) {
                uint8_t idx0 = attn_row_uint8[j];
                uint8_t idx1 = attn_row_uint8[j + 1];
                uint8_t idx2 = attn_row_uint8[j + 2];
                uint8_t idx3 = attn_row_uint8[j + 3];
                uint32_t y0 = i_softmax_lut_int8[idx0];
                uint32_t y1 = i_softmax_lut_int8[idx1];
                uint32_t y2 = i_softmax_lut_int8[idx2];
                uint32_t y3 = i_softmax_lut_int8[idx3];
                attn_row_uint8[j]     = (uint8_t)(((uint64_t)y0 * inv_sum + round_norm) >> 24);
                attn_row_uint8[j + 1] = (uint8_t)(((uint64_t)y1 * inv_sum + round_norm) >> 24);
                attn_row_uint8[j + 2] = (uint8_t)(((uint64_t)y2 * inv_sum + round_norm) >> 24);
                attn_row_uint8[j + 3] = (uint8_t)(((uint64_t)y3 * inv_sum + round_norm) >> 24);
            }
            for (; j < seq_len; j++) {
                uint8_t idx = attn_row_uint8[j];
                uint32_t y_i = i_softmax_lut_int8[idx];
                attn_row_uint8[j] = (uint8_t)(((uint64_t)y_i * inv_sum + round_norm) >> 24);
            }
        } else {
            uint8_t uniform = 255 / seq_len;
            for (j = 0; j < seq_len; j++) {
                attn_row_uint8[j] = uniform;
            }
        }

        for (int d = 0; d < head_dim; d++) {
            int32_t acc = 0;

            for (int js = 0; js < simd_iters; js++) {
                v4u attn_vec = *((v4u *)(attn_row_uint8 + (js << 2)));
                v4s v_vec;
                int base_j = js << 2;
                ((int8_t*)&v_vec)[0] = a->v_l1[(base_j + 0) * head_dim + d];
                ((int8_t*)&v_vec)[1] = a->v_l1[(base_j + 1) * head_dim + d];
                ((int8_t*)&v_vec)[2] = a->v_l1[(base_j + 2) * head_dim + d];
                ((int8_t*)&v_vec)[3] = a->v_l1[(base_j + 3) * head_dim + d];

                acc = SumDotpUS(attn_vec, v_vec, acc);
            }

            for (int jj = tail_start; jj < seq_len; jj++) {
                acc += (int32_t)attn_row_uint8[jj] * (int32_t)a->v_l1[jj * head_dim + d];
            }

            int32_t q_val = (acc + 128) >> 8;
            if (q_val > 127) q_val = 127;
            if (q_val < -128) q_val = -128;
            out_row[d] = (int8_t)q_val;
        }
    }
}

/**
 * Worker kernel: Compute attn_weights x V (FP32 x INT8 → INT8) - FP32 path
 * Uses FP32 attention weights with INT8 values for context computation.
 */
static void mhsa_tile_av_fp32_worker(void *arg) {
    mhsa_tile_args_t *a = (mhsa_tile_args_t *)arg;
    const int core_id = pi_core_id();
    const int seq_len = a->seq_len;
    const int head_dim = a->head_dim;

    for (int i = core_id; i < a->tile_q; i += NUM_CORES) {
        const float *attn_row_fp32 = a->scores_l1 + i * seq_len;
        int8_t *out_row = a->m_tile_l1 + i * head_dim;

        for (int d = 0; d < head_dim; d++) {
            float acc = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                const int8_t v_val = a->v_l1[j * head_dim + d];
                acc += attn_row_fp32[j] * (float)v_val;
            }
            int32_t q_val = (int32_t)lrintf(acc);
            if (q_val > 127) q_val = 127;
            if (q_val < -128) q_val = -128;
            out_row[d] = (int8_t)q_val;
        }
    }
}

static void mhsa_tile_av_fp32_vt_worker(void *arg) {
    mhsa_tile_args_t *a = (mhsa_tile_args_t *)arg;
    const int core_id = pi_core_id();
    const int seq_len = a->seq_len;
    const int head_dim = a->head_dim;

    for (int i = core_id; i < a->tile_q; i += NUM_CORES) {
        const float *attn_row_fp32 = a->scores_l1 + i * seq_len;
        int8_t *out_row = a->m_tile_l1 + i * head_dim;

        for (int d = 0; d < head_dim; d++) {
            const int8_t *v_col = a->v_t_l1 + d * seq_len;
            float acc = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                acc += attn_row_fp32[j] * (float)v_col[j];
            }
            int32_t q_val = (int32_t)lrintf(acc);
            if (q_val > 127) q_val = 127;
            if (q_val < -128) q_val = -128;
            out_row[d] = (int8_t)q_val;
        }
    }
}

/**
 * Worker kernel: Compute attn_weights x V (UINT8 x INT8 → INT8) - Integer path
 * Uses UINT8 attention weights (from polynomial iSoftmax) with INT8 values.
 * SIMD-optimized using SumDotpUS for 4 UINT8xINT8 MACs per cycle.
 *
 * Requantization: The UINT8 attention weights sum to ~255 (n_levels-1) per row.
 * To get proper INT8 output: acc_int32 / 255 ≈ acc_int32 >> 8
 */
static void mhsa_tile_av_int_worker(void *arg) {
    mhsa_tile_args_t *a = (mhsa_tile_args_t *)arg;
    const int core_id = pi_core_id();
    const int seq_len = a->seq_len;
    const int head_dim = a->head_dim;
    const int simd_iters = seq_len >> 2;
    const int tail_start = simd_iters << 2;

    for (int i = core_id; i < a->tile_q; i += NUM_CORES) {
        const uint8_t *attn_row_uint8 = a->attn_uint8_l1 + i * seq_len;
        int8_t *out_row = a->m_tile_l1 + i * head_dim;

        // For each output dimension (head_dim outputs per row)
        for (int d = 0; d < head_dim; d++) {
            int32_t acc = 0;

            // SIMD path: 4 UINT8xINT8 MACs per cycle using sdotusp4
            // We iterate over the sequence (attention weights x V column)
            for (int js = 0; js < simd_iters; js++) {
                // Load 4 consecutive attention weights
                v4u attn_vec = *((v4u *)(attn_row_uint8 + (js << 2)));

                // Load 4 V values from column d (strided access)
                // V is stored as [seq_len x head_dim], so V[j, d] = V[j * head_dim + d]
                v4s v_vec;
                int base_j = js << 2;
                ((int8_t*)&v_vec)[0] = a->v_l1[(base_j + 0) * head_dim + d];
                ((int8_t*)&v_vec)[1] = a->v_l1[(base_j + 1) * head_dim + d];
                ((int8_t*)&v_vec)[2] = a->v_l1[(base_j + 2) * head_dim + d];
                ((int8_t*)&v_vec)[3] = a->v_l1[(base_j + 3) * head_dim + d];

                acc = SumDotpUS(attn_vec, v_vec, acc);
            }

            // Tail elements
            for (int j = tail_start; j < seq_len; j++) {
                acc += (int32_t)attn_row_uint8[j] * (int32_t)a->v_l1[j * head_dim + d];
            }

            // Requantize: attention weights sum to ~255, so divide by 255
            // Use >> 8 as approximation (divides by 256, close enough)
            // Adding 128 for rounding: (acc + 128) >> 8
            int32_t q_val = (acc + 128) >> 8;
            if (q_val > 127) q_val = 127;
            if (q_val < -128) q_val = -128;
            out_row[d] = (int8_t)q_val;
        }
    }
}

static void mhsa_tile_av_int_vt_worker(void *arg) {
    mhsa_tile_args_t *a = (mhsa_tile_args_t *)arg;
    const int core_id = pi_core_id();
    const int seq_len = a->seq_len;
    const int head_dim = a->head_dim;
    const int simd_iters = seq_len >> 2;
    const int tail_start = simd_iters << 2;

    for (int i = core_id; i < a->tile_q; i += NUM_CORES) {
        const uint8_t *attn_row_uint8 = a->attn_uint8_l1 + i * seq_len;
        int8_t *out_row = a->m_tile_l1 + i * head_dim;

        // Unroll over output dimension to reuse attn loads across multiple V columns.
        int d = 0;
        for (; d + 4 <= head_dim; d += 4) {
            const int8_t *v_col0 = a->v_t_l1 + (d + 0) * seq_len;
            const int8_t *v_col1 = a->v_t_l1 + (d + 1) * seq_len;
            const int8_t *v_col2 = a->v_t_l1 + (d + 2) * seq_len;
            const int8_t *v_col3 = a->v_t_l1 + (d + 3) * seq_len;

            int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

            for (int js = 0; js < simd_iters; js++) {
                v4u attn_vec = *((v4u *)(attn_row_uint8 + (js << 2)));
                v4s v_vec0 = *((v4s *)(v_col0 + (js << 2)));
                v4s v_vec1 = *((v4s *)(v_col1 + (js << 2)));
                v4s v_vec2 = *((v4s *)(v_col2 + (js << 2)));
                v4s v_vec3 = *((v4s *)(v_col3 + (js << 2)));
                acc0 = SumDotpUS(attn_vec, v_vec0, acc0);
                acc1 = SumDotpUS(attn_vec, v_vec1, acc1);
                acc2 = SumDotpUS(attn_vec, v_vec2, acc2);
                acc3 = SumDotpUS(attn_vec, v_vec3, acc3);
            }

            for (int j = tail_start; j < seq_len; j++) {
                const int32_t w = (int32_t)attn_row_uint8[j];
                acc0 += w * (int32_t)v_col0[j];
                acc1 += w * (int32_t)v_col1[j];
                acc2 += w * (int32_t)v_col2[j];
                acc3 += w * (int32_t)v_col3[j];
            }

            int32_t q0 = (acc0 + 128) >> 8;
            int32_t q1 = (acc1 + 128) >> 8;
            int32_t q2 = (acc2 + 128) >> 8;
            int32_t q3 = (acc3 + 128) >> 8;

            if (q0 > 127) q0 = 127; if (q0 < -128) q0 = -128;
            if (q1 > 127) q1 = 127; if (q1 < -128) q1 = -128;
            if (q2 > 127) q2 = 127; if (q2 < -128) q2 = -128;
            if (q3 > 127) q3 = 127; if (q3 < -128) q3 = -128;

            out_row[d + 0] = (int8_t)q0;
            out_row[d + 1] = (int8_t)q1;
            out_row[d + 2] = (int8_t)q2;
            out_row[d + 3] = (int8_t)q3;
        }

        // Tail output dimensions
        for (; d < head_dim; d++) {
            const int8_t *v_col = a->v_t_l1 + d * seq_len;
            int32_t acc = 0;

            for (int js = 0; js < simd_iters; js++) {
                v4u attn_vec = *((v4u *)(attn_row_uint8 + (js << 2)));
                v4s v_vec = *((v4s *)(v_col + (js << 2)));
                acc = SumDotpUS(attn_vec, v_vec, acc);
            }

            for (int j = tail_start; j < seq_len; j++) {
                acc += (int32_t)attn_row_uint8[j] * (int32_t)v_col[j];
            }

            int32_t q_val = (acc + 128) >> 8;
            if (q_val > 127) q_val = 127;
            if (q_val < -128) q_val = -128;
            out_row[d] = (int8_t)q_val;
        }
    }
}

/**
 * Worker kernel: Dispatcher for AV (selects FP32 or integer path)
 */
static void mhsa_tile_av_worker(void *arg) {
    mhsa_tile_args_t *a = (mhsa_tile_args_t *)arg;
    if (a->v_t_l1) {
        if (a->use_integer_softmax) {
            mhsa_tile_av_int_vt_worker(arg);
        } else {
            mhsa_tile_av_fp32_vt_worker(arg);
        }
    } else {
        if (a->use_integer_softmax) {
            mhsa_tile_av_int_worker(arg);
        } else {
            mhsa_tile_av_fp32_worker(arg);
        }
    }
}

static void mhsa_tile_qk_softmax_av_worker(void *arg) {
    mhsa_tile_args_t *a = (mhsa_tile_args_t *)arg;

    mhsa_tile_qk_worker(arg);
    if (a->use_integer_softmax) {
#if MHSA_FUSE_SOFTMAX_AV
        if (a->v_t_l1) {
            mhsa_tile_softmax_av_int_vt_worker(arg);
        } else {
            mhsa_tile_softmax_av_int_worker(arg);
        }
#else
        mhsa_tile_softmax_worker(arg);
        mhsa_tile_av_worker(arg);
#endif
    } else {
        mhsa_tile_softmax_worker(arg);
        mhsa_tile_av_worker(arg);
    }
}

static void mhsa_tile_qk_softmax_av_per_core_worker(void *arg) {
    mhsa_tile_args_t *a = (mhsa_tile_args_t *)arg;
    const int core_id = pi_core_id();
    const int head_dim = a->head_dim;
    const int seq_len = a->seq_len;
    const int qk_simd_iters = head_dim >> 2;
    const int qk_tail_start = qk_simd_iters << 2;
    const int attn_simd_iters = seq_len >> 2;
    const int attn_tail_start = attn_simd_iters << 2;
    const int32_t requant_mul = a->requant_mul;
    const int32_t requant_shift = a->requant_shift;
    const int64_t round_val = ((requant_shift > 0) && (requant_shift < 63)) ? (1LL << (requant_shift - 1)) : 0;

    int32_t *scores = a->scores_int32_l1 + core_id * seq_len;
    uint8_t *attn = a->attn_uint8_l1 + core_id * seq_len;

    for (int i = core_id; i < a->tile_q; i += NUM_CORES) {
        const int8_t *q_row = a->q_tile_l1 + i * head_dim;
        int8_t *out_row = a->m_tile_l1 + i * head_dim;

        // STEP 1: QK (per-row)
        int j = 0;
        for (; j + 4 <= seq_len; j += 4) {
            const int8_t *k_row0 = a->k_l1 + (j + 0) * head_dim;
            const int8_t *k_row1 = a->k_l1 + (j + 1) * head_dim;
            const int8_t *k_row2 = a->k_l1 + (j + 2) * head_dim;
            const int8_t *k_row3 = a->k_l1 + (j + 3) * head_dim;

            int32_t dot0 = 0, dot1 = 0, dot2 = 0, dot3 = 0;

            for (int d = 0; d < qk_simd_iters; d++) {
                v4s q_vec = *((v4s *)(q_row + (d << 2)));
                v4s k0 = *((v4s *)(k_row0 + (d << 2)));
                v4s k1 = *((v4s *)(k_row1 + (d << 2)));
                v4s k2 = *((v4s *)(k_row2 + (d << 2)));
                v4s k3 = *((v4s *)(k_row3 + (d << 2)));
                dot0 = SumDotpSS(q_vec, k0, dot0);
                dot1 = SumDotpSS(q_vec, k1, dot1);
                dot2 = SumDotpSS(q_vec, k2, dot2);
                dot3 = SumDotpSS(q_vec, k3, dot3);
            }
            for (int d = qk_tail_start; d < head_dim; d++) {
                const int32_t q = (int32_t)q_row[d];
                dot0 += q * (int32_t)k_row0[d];
                dot1 += q * (int32_t)k_row1[d];
                dot2 += q * (int32_t)k_row2[d];
                dot3 += q * (int32_t)k_row3[d];
            }

            scores[j + 0] = dot0;
            scores[j + 1] = dot1;
            scores[j + 2] = dot2;
            scores[j + 3] = dot3;
        }
        for (; j < seq_len; j++) {
            const int8_t *k_row = a->k_l1 + j * head_dim;
            int32_t dot = 0;

            for (int d = 0; d < qk_simd_iters; d++) {
                v4s q_vec = *((v4s *)(q_row + (d << 2)));
                v4s k_vec = *((v4s *)(k_row + (d << 2)));
                dot = SumDotpSS(q_vec, k_vec, dot);
            }
            for (int d = qk_tail_start; d < head_dim; d++) {
                dot += (int32_t)q_row[d] * (int32_t)k_row[d];
            }

            scores[j] = dot;
        }

        // STEP 2: Integer softmax (per-row)
        int32_t max0 = scores[0];
        int32_t max1 = max0, max2 = max0, max3 = max0;
        const int unroll_limit = seq_len - 3;
        for (j = 1; j < unroll_limit; j += 4) {
            int32_t s0 = scores[j];
            int32_t s1 = scores[j + 1];
            int32_t s2 = scores[j + 2];
            int32_t s3 = scores[j + 3];
            if (s0 > max0) max0 = s0;
            if (s1 > max1) max1 = s1;
            if (s2 > max2) max2 = s2;
            if (s3 > max3) max3 = s3;
        }
        for (; j < seq_len; j++) {
            if (scores[j] > max0) max0 = scores[j];
        }
        if (max1 > max0) max0 = max1;
        if (max2 > max0) max0 = max2;
        if (max3 > max0) max0 = max3;
        const int32_t max_score = max0;

        uint64_t y_sum = 0;
        for (j = 0; j < seq_len; j++) {
            int32_t diff = scores[j] - max_score;
            int32_t x_int = ((int64_t)diff * (int64_t)requant_mul + round_val) >> requant_shift;
            if (x_int < -128) x_int = -128;
            if (x_int > 0) x_int = 0;
            uint8_t idx = (uint8_t)(x_int + 128);
            attn[j] = idx;
            y_sum += i_softmax_lut_int8[idx];
        }

        if (y_sum > 0) {
            const uint64_t inv_sum = (255ULL << 24) / y_sum;
            const uint64_t round_norm = (1ULL << 23);
            for (j = 0; j + 4 <= seq_len; j += 4) {
                uint8_t idx0 = attn[j];
                uint8_t idx1 = attn[j + 1];
                uint8_t idx2 = attn[j + 2];
                uint8_t idx3 = attn[j + 3];
                uint32_t y0 = i_softmax_lut_int8[idx0];
                uint32_t y1 = i_softmax_lut_int8[idx1];
                uint32_t y2 = i_softmax_lut_int8[idx2];
                uint32_t y3 = i_softmax_lut_int8[idx3];
                attn[j]     = (uint8_t)(((uint64_t)y0 * inv_sum + round_norm) >> 24);
                attn[j + 1] = (uint8_t)(((uint64_t)y1 * inv_sum + round_norm) >> 24);
                attn[j + 2] = (uint8_t)(((uint64_t)y2 * inv_sum + round_norm) >> 24);
                attn[j + 3] = (uint8_t)(((uint64_t)y3 * inv_sum + round_norm) >> 24);
            }
            for (; j < seq_len; j++) {
                uint8_t idx = attn[j];
                uint32_t y_i = i_softmax_lut_int8[idx];
                attn[j] = (uint8_t)(((uint64_t)y_i * inv_sum + round_norm) >> 24);
            }
        } else {
            uint8_t uniform = 255 / seq_len;
            for (j = 0; j < seq_len; j++) {
                attn[j] = uniform;
            }
        }

        // STEP 3: AV (per-row)
        if (a->v_t_l1) {
            int d = 0;
            for (; d + 4 <= head_dim; d += 4) {
                const int8_t *v_col0 = a->v_t_l1 + (d + 0) * seq_len;
                const int8_t *v_col1 = a->v_t_l1 + (d + 1) * seq_len;
                const int8_t *v_col2 = a->v_t_l1 + (d + 2) * seq_len;
                const int8_t *v_col3 = a->v_t_l1 + (d + 3) * seq_len;

                int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

                for (int js = 0; js < attn_simd_iters; js++) {
                    v4u attn_vec = *((v4u *)(attn + (js << 2)));
                    v4s v_vec0 = *((v4s *)(v_col0 + (js << 2)));
                    v4s v_vec1 = *((v4s *)(v_col1 + (js << 2)));
                    v4s v_vec2 = *((v4s *)(v_col2 + (js << 2)));
                    v4s v_vec3 = *((v4s *)(v_col3 + (js << 2)));
                    acc0 = SumDotpUS(attn_vec, v_vec0, acc0);
                    acc1 = SumDotpUS(attn_vec, v_vec1, acc1);
                    acc2 = SumDotpUS(attn_vec, v_vec2, acc2);
                    acc3 = SumDotpUS(attn_vec, v_vec3, acc3);
                }

                for (int jj = attn_tail_start; jj < seq_len; jj++) {
                    const int32_t w = (int32_t)attn[jj];
                    acc0 += w * (int32_t)v_col0[jj];
                    acc1 += w * (int32_t)v_col1[jj];
                    acc2 += w * (int32_t)v_col2[jj];
                    acc3 += w * (int32_t)v_col3[jj];
                }

                int32_t q0 = (acc0 + 128) >> 8;
                int32_t q1 = (acc1 + 128) >> 8;
                int32_t q2 = (acc2 + 128) >> 8;
                int32_t q3 = (acc3 + 128) >> 8;

                if (q0 > 127) q0 = 127; if (q0 < -128) q0 = -128;
                if (q1 > 127) q1 = 127; if (q1 < -128) q1 = -128;
                if (q2 > 127) q2 = 127; if (q2 < -128) q2 = -128;
                if (q3 > 127) q3 = 127; if (q3 < -128) q3 = -128;

                out_row[d + 0] = (int8_t)q0;
                out_row[d + 1] = (int8_t)q1;
                out_row[d + 2] = (int8_t)q2;
                out_row[d + 3] = (int8_t)q3;
            }

            for (; d < head_dim; d++) {
                const int8_t *v_col = a->v_t_l1 + d * seq_len;
                int32_t acc = 0;

                for (int js = 0; js < attn_simd_iters; js++) {
                    v4u attn_vec = *((v4u *)(attn + (js << 2)));
                    v4s v_vec = *((v4s *)(v_col + (js << 2)));
                    acc = SumDotpUS(attn_vec, v_vec, acc);
                }

                for (int jj = attn_tail_start; jj < seq_len; jj++) {
                    acc += (int32_t)attn[jj] * (int32_t)v_col[jj];
                }

                int32_t q_val = (acc + 128) >> 8;
                if (q_val > 127) q_val = 127;
                if (q_val < -128) q_val = -128;
                out_row[d] = (int8_t)q_val;
            }
        } else {
            for (int d = 0; d < head_dim; d++) {
                int32_t acc = 0;

                for (int js = 0; js < attn_simd_iters; js++) {
                    v4u attn_vec = *((v4u *)(attn + (js << 2)));
                    v4s v_vec;
                    int base_j = js << 2;
                    ((int8_t *)&v_vec)[0] = a->v_l1[(base_j + 0) * head_dim + d];
                    ((int8_t *)&v_vec)[1] = a->v_l1[(base_j + 1) * head_dim + d];
                    ((int8_t *)&v_vec)[2] = a->v_l1[(base_j + 2) * head_dim + d];
                    ((int8_t *)&v_vec)[3] = a->v_l1[(base_j + 3) * head_dim + d];

                    acc = SumDotpUS(attn_vec, v_vec, acc);
                }

                for (int jj = attn_tail_start; jj < seq_len; jj++) {
                    acc += (int32_t)attn[jj] * (int32_t)a->v_l1[jj * head_dim + d];
                }

                int32_t q_val = (acc + 128) >> 8;
                if (q_val > 127) q_val = 127;
                if (q_val < -128) q_val = -128;
                out_row[d] = (int8_t)q_val;
            }
        }
    }
}

// ---
// OPTIMIZED MHSA: Fused Workers for L2 Fallback and L1-Cached Modes
// ---

// FUSED worker arguments: score + softmax + context in one kernel (pure L2 mode)
typedef struct {
    const int8_t *q_buffer_l2;
    const int8_t *k_buffer_l2;
    const int8_t *v_buffer_l2;
    int8_t *output_buffer_l2;
    int32_t *scores_int32;        // Per-core buffer in L1 (1 row per core)
    int8_t *scores_int8;          // Per-core buffer in L1 (1 row per core)
    uint8_t *attn_uint8;          // Per-core buffer in L1 (1 row per core)
    int tile_start;
    int tile_size;
    int seq_len;
    int embed_dim;
    int head_dim;
    int head_offset;
    int16_t requant_mul;
    int16_t requant_shift;
} mhsa_l2_fused_args_t;

// L1-CACHED FUSED worker arguments: K in L1, V may be in L2
// num_cores parameter allows 6 or 8 cores depending on L1 availability
typedef struct {
    const int8_t *q_buffer_l2;    // Q in L2 [seq_len, embed_dim] or [seq_len, head_dim] if permuted
    const int8_t *k_l1;           // K in L1 [seq_len, head_dim] - contiguous!
    const int8_t *v_l1;           // V in L1 or L2 [seq_len, head_dim]
    int8_t *output_buffer_l2;     // Output in L2
    int32_t *scores_int32;        // Per-core INT32 score buffer (1 row per core)
    int8_t *scores_int8;          // Per-core INT8 score buffer
    uint8_t *attn_uint8;          // Per-core UINT8 attention buffer
    int tile_start;               // Starting row in Q
    int tile_size;                // Number of rows to process
    int seq_len;                  // Full sequence length (for K/V access)
    int embed_dim;                // Q stride (head_dim if permuted, embed_dim otherwise)
    int head_dim;
    int head_offset;              // Offset in Q for this head (0 if permuted)
    int16_t requant_mul;
    int16_t requant_shift;
    int num_cores;                // 6 or 8 cores
} mhsa_l1_fused_args_t;

/**
 * FUSED Parallel worker: Score + Softmax + Context in one kernel (L2 mode)
 * Each core processes its assigned rows completely (score -> softmax -> context)
 * before moving to the next row.
 */
static void mhsa_l2_fused_worker(void *arg) {
    mhsa_l2_fused_args_t *a = (mhsa_l2_fused_args_t *)arg;
    const int core_id = pi_core_id();
    const int head_dim = a->head_dim;
    const int seq_len = a->seq_len;
    const int embed_dim = a->embed_dim;
    const int head_offset = a->head_offset;
    const int simd_iters = head_dim >> 2;
    const int tail_start = simd_iters << 2;
    const int16_t requant_mul = a->requant_mul;
    const int16_t requant_shift = a->requant_shift;

    // Each core gets its own slice of the score buffers
    int32_t *my_scores_int32 = a->scores_int32 + core_id * seq_len;
    int8_t *my_scores_int8 = a->scores_int8 + core_id * seq_len;
    uint8_t *my_attn_uint8 = a->attn_uint8 + core_id * seq_len;

    for (int i = core_id; i < a->tile_size; i += NUM_CORES) {
        const int global_i = a->tile_start + i;
        const int8_t *q_row = a->q_buffer_l2 + global_i * embed_dim + head_offset;
        int8_t *out_row = a->output_buffer_l2 + global_i * embed_dim + head_offset;

        // STEP 1: Compute scores for this row (QK^T)
        for (int j = 0; j < seq_len; j++) {
            const int8_t *k_row = a->k_buffer_l2 + j * embed_dim + head_offset;
            int32_t dot = 0;
            for (int d = 0; d < simd_iters; d++) {
                v4s q_vec = *((v4s *)(q_row + (d << 2)));
                v4s k_vec = *((v4s *)(k_row + (d << 2)));
                dot = SumDotpSS(q_vec, k_vec, dot);
            }
            for (int d = tail_start; d < head_dim; d++) {
                dot += (int32_t)q_row[d] * (int32_t)k_row[d];
            }
            my_scores_int32[j] = dot;
            my_scores_int8[j] = clip8((dot * requant_mul) >> requant_shift);
        }

        // STEP 2: Integer softmax on this row
        int8_t x_max = -128;
        for (int j = 0; j < seq_len; j++) {
            if (my_scores_int8[j] > x_max) x_max = my_scores_int8[j];
        }
        uint64_t y_sum = 0;
        for (int j = 0; j < seq_len; j++) {
            int16_t diff = (int16_t)my_scores_int8[j] - (int16_t)x_max;
            int idx = diff + 128;
            if (idx < 0) idx = 0;
            if (idx > 128) idx = 128;
            y_sum += i_softmax_lut_int8[idx];
        }
        if (y_sum > 0) {
            for (int j = 0; j < seq_len; j++) {
                int16_t diff = (int16_t)my_scores_int8[j] - (int16_t)x_max;
                int idx = diff + 128;
                if (idx < 0) idx = 0;
                if (idx > 128) idx = 128;
                uint32_t y_i = i_softmax_lut_int8[idx];
                my_attn_uint8[j] = (uint8_t)((y_i * 255) / y_sum);
            }
        } else {
            uint8_t uniform = 255 / seq_len;
            for (int j = 0; j < seq_len; j++) my_attn_uint8[j] = uniform;
        }

        // STEP 3: Compute context for this row (attn x V)
        int32_t acc[64];
        for (int d = 0; d < head_dim; d++) acc[d] = 0;

        for (int j = 0; j < seq_len; j++) {
            const int32_t attn_val = (int32_t)my_attn_uint8[j];
            const int8_t *v_row = a->v_buffer_l2 + j * embed_dim + head_offset;
            for (int d4 = 0; d4 < simd_iters; d4++) {
                int base = d4 << 2;
                acc[base + 0] += attn_val * (int32_t)v_row[base + 0];
                acc[base + 1] += attn_val * (int32_t)v_row[base + 1];
                acc[base + 2] += attn_val * (int32_t)v_row[base + 2];
                acc[base + 3] += attn_val * (int32_t)v_row[base + 3];
            }
            for (int d = tail_start; d < head_dim; d++) {
                acc[d] += attn_val * (int32_t)v_row[d];
            }
        }

        // Scale and write output
        for (int d4 = 0; d4 < simd_iters; d4++) {
            int base = d4 << 2;
            int32_t r0 = (acc[base + 0] + 127) >> 8;
            int32_t r1 = (acc[base + 1] + 127) >> 8;
            int32_t r2 = (acc[base + 2] + 127) >> 8;
            int32_t r3 = (acc[base + 3] + 127) >> 8;
            out_row[base + 0] = (int8_t)(r0 > 127 ? 127 : (r0 < -128 ? -128 : r0));
            out_row[base + 1] = (int8_t)(r1 > 127 ? 127 : (r1 < -128 ? -128 : r1));
            out_row[base + 2] = (int8_t)(r2 > 127 ? 127 : (r2 < -128 ? -128 : r2));
            out_row[base + 3] = (int8_t)(r3 > 127 ? 127 : (r3 < -128 ? -128 : r3));
        }
        for (int d = tail_start; d < head_dim; d++) {
            int32_t result = (acc[d] + 127) >> 8;
            if (result > 127) result = 127;
            if (result < -128) result = -128;
            out_row[d] = (int8_t)result;
        }
    }
}

/**
 * L1-CACHED FUSED worker: K is pre-loaded to L1
 * Processes score→softmax→context for each row before moving to next row.
 * K/V accesses hit L1 instead of L2 = ~10x faster!
 *
 * Optimizations:
 * - Multiplicative inverse softmax (avoid division in hot loop)
 * - J-unrolling (4x) for context computation
 * - num_cores parameter allows 6 or 8 cores depending on L1 availability
 */
static void mhsa_l1_fused_worker(void *arg) {
    mhsa_l1_fused_args_t *a = (mhsa_l1_fused_args_t *)arg;
    const int core_id = pi_core_id();
    const int num_cores = a->num_cores;
    const int head_dim = a->head_dim;
    const int seq_len = a->seq_len;
    const int embed_dim = a->embed_dim;
    const int head_offset = a->head_offset;
    const int simd_iters = head_dim >> 2;
    const int tail_start = simd_iters << 2;
    const int16_t requant_mul = a->requant_mul;
    const int16_t requant_shift = a->requant_shift;

    // Each core gets its own slice of the score buffers (1 row per core)
    int32_t *my_scores_int32 = a->scores_int32 + core_id * seq_len;
    int8_t *my_scores_int8 = a->scores_int8 + core_id * seq_len;
    uint8_t *my_attn_uint8 = a->attn_uint8 + core_id * seq_len;

    // Process rows assigned to this core
    for (int i = core_id; i < a->tile_size; i += num_cores) {
        const int global_i = a->tile_start + i;
        const int8_t *q_row = a->q_buffer_l2 + global_i * embed_dim + head_offset;
        int8_t *out_row = a->output_buffer_l2 + global_i * embed_dim + head_offset;

        // STEP 1: Compute scores (K is in L1 - FAST!)
        for (int j = 0; j < seq_len; j++) {
            const int8_t *k_row = a->k_l1 + j * head_dim;  // K contiguous in L1
            int32_t dot = 0;
            for (int d = 0; d < simd_iters; d++) {
                v4s q_vec = *((v4s *)(q_row + (d << 2)));
                v4s k_vec = *((v4s *)(k_row + (d << 2)));
                dot = SumDotpSS(q_vec, k_vec, dot);
            }
            for (int d = tail_start; d < head_dim; d++) {
                dot += (int32_t)q_row[d] * (int32_t)k_row[d];
            }
            my_scores_int32[j] = dot;
            my_scores_int8[j] = clip8((dot * requant_mul) >> requant_shift);
        }

        // STEP 2: Integer softmax (optimized with multiplicative inverse)
        int8_t x_max = -128;
        for (int j = 0; j < seq_len; j++) {
            if (my_scores_int8[j] > x_max) x_max = my_scores_int8[j];
        }

        // Compute exp values and sum
        uint64_t y_sum = 0;
        for (int j = 0; j < seq_len; j++) {
            int idx = (int)my_scores_int8[j] - (int)x_max + 128;
            if (idx < 0) idx = 0;
            uint32_t y_i = i_softmax_lut_int8[idx];
            my_scores_int32[j] = (int32_t)y_i;  // Reuse score buffer to store exp values
            y_sum += y_i;
        }

        // Normalize using multiplicative inverse with 24-bit precision (matches network_kernels.c)
        if (y_sum > 0) {
            const uint64_t inv_sum = (255ULL << 24) / y_sum;
            const uint64_t round_norm = (1ULL << 23);  // Rounding term for >>24
            for (int j = 0; j < seq_len; j++) {
                uint32_t y_i = (uint32_t)my_scores_int32[j];
                my_attn_uint8[j] = (uint8_t)(((uint64_t)y_i * inv_sum + round_norm) >> 24);
            }
        } else {
            uint8_t uniform = 255 / seq_len;
            for (int j = 0; j < seq_len; j++) my_attn_uint8[j] = uniform;
        }

        // STEP 3: Context (V access, optimized with j-unrolling)
        int32_t acc[64];
        for (int d = 0; d < head_dim; d++) acc[d] = 0;

        // Unroll j loop by 4 to reduce loop overhead
        const int8_t *v_base = a->v_l1;
        const int j_unroll = (seq_len >> 2) << 2;

        for (int j = 0; j < j_unroll; j += 4) {
            const int32_t a0 = (int32_t)my_attn_uint8[j];
            const int32_t a1 = (int32_t)my_attn_uint8[j + 1];
            const int32_t a2 = (int32_t)my_attn_uint8[j + 2];
            const int32_t a3 = (int32_t)my_attn_uint8[j + 3];
            const int8_t *v0 = v_base + j * head_dim;
            const int8_t *v1 = v0 + head_dim;
            const int8_t *v2 = v1 + head_dim;
            const int8_t *v3 = v2 + head_dim;

            for (int d4 = 0; d4 < simd_iters; d4++) {
                int base = d4 << 2;
                acc[base + 0] += a0 * (int32_t)v0[base + 0] + a1 * (int32_t)v1[base + 0]
                               + a2 * (int32_t)v2[base + 0] + a3 * (int32_t)v3[base + 0];
                acc[base + 1] += a0 * (int32_t)v0[base + 1] + a1 * (int32_t)v1[base + 1]
                               + a2 * (int32_t)v2[base + 1] + a3 * (int32_t)v3[base + 1];
                acc[base + 2] += a0 * (int32_t)v0[base + 2] + a1 * (int32_t)v1[base + 2]
                               + a2 * (int32_t)v2[base + 2] + a3 * (int32_t)v3[base + 2];
                acc[base + 3] += a0 * (int32_t)v0[base + 3] + a1 * (int32_t)v1[base + 3]
                               + a2 * (int32_t)v2[base + 3] + a3 * (int32_t)v3[base + 3];
            }
            for (int d = tail_start; d < head_dim; d++) {
                acc[d] += a0 * (int32_t)v0[d] + a1 * (int32_t)v1[d]
                        + a2 * (int32_t)v2[d] + a3 * (int32_t)v3[d];
            }
        }
        // Handle remaining elements
        for (int j = j_unroll; j < seq_len; j++) {
            const int32_t attn_val = (int32_t)my_attn_uint8[j];
            const int8_t *v_row = v_base + j * head_dim;
            for (int d = 0; d < head_dim; d++) {
                acc[d] += attn_val * (int32_t)v_row[d];
            }
        }

        // Scale and write output
        for (int d4 = 0; d4 < simd_iters; d4++) {
            int base = d4 << 2;
            int32_t r0 = (acc[base + 0] + 127) >> 8;
            int32_t r1 = (acc[base + 1] + 127) >> 8;
            int32_t r2 = (acc[base + 2] + 127) >> 8;
            int32_t r3 = (acc[base + 3] + 127) >> 8;
            out_row[base + 0] = (int8_t)(r0 > 127 ? 127 : (r0 < -128 ? -128 : r0));
            out_row[base + 1] = (int8_t)(r1 > 127 ? 127 : (r1 < -128 ? -128 : r1));
            out_row[base + 2] = (int8_t)(r2 > 127 ? 127 : (r2 < -128 ? -128 : r2));
            out_row[base + 3] = (int8_t)(r3 > 127 ? 127 : (r3 < -128 ? -128 : r3));
        }
        for (int d = tail_start; d < head_dim; d++) {
            int32_t result = (acc[d] + 127) >> 8;
            if (result > 127) result = 127;
            if (result < -128) result = -128;
            out_row[d] = (int8_t)result;
        }
    }
}

/**
 * MHSA L2 Fallback: Compute attention directly in L2 memory.
 *
 * Used when L1 memory is insufficient for tiling.
 * Computes multi-head self-attention with all data residing in L2:
 * - For each head:
 * 1. Compute scores = Q x K^T (INT8 x INT8 → FP32)
 * 2. Apply softmax to scores
 * 3. Compute context = scores x V (FP32 x INT8 → INT8)
 *
 * This is slower than L1 tiling but guaranteed to work regardless of L1 size.
 */
// ---
// MHSA L2 Fallback (Inner Loop - operates on data already in L2)
// ---
static void mhsa_l2_fallback_inner(mhsa_pipeline_config_t *cfg) {
    // Path B: FP32 projections (no q/k/v output scales provided)
    if (cfg->use_fp32_projections) {
        const int seq_len = cfg->seq_len;
        const int embed_dim = cfg->embed_dim;
        const int num_heads = cfg->num_heads;
        const int head_dim = cfg->head_dim;
        const int pool_mode = cfg->pool_mode;
        const size_t token_count = (size_t)seq_len * embed_dim;

        const float mult_q = cfg->scale_input * cfg->scale_q_weight;
        const float mult_k = cfg->scale_input * cfg->scale_k_weight;
        const float mult_v = cfg->scale_input * cfg->scale_v_weight;
        const float bias_scale_out = cfg->scale_input * cfg->scale_out_weight;

        // Use L1 for scores tiling: process tile_q rows at a time
        // Avoids allocating full seq_len * seq_len scores matrix in L2

        float *q_fp32 = (float *)pi_l2_malloc(token_count * sizeof(float));
        float *k_fp32 = (float *)pi_l2_malloc(token_count * sizeof(float));
        float *v_fp32 = (float *)pi_l2_malloc(token_count * sizeof(float));
        float *context_fp32 = (float *)pi_l2_malloc(token_count * sizeof(float));
        // NO L2 scores allocation - will use L1 buffer instead
        float *out_proj = NULL;
        float *pooled = NULL;

        if (!q_fp32 || !k_fp32 || !v_fp32 || !context_fp32) {
#ifndef MINIMAL_OUTPUT
            printf("MHSA FP32 Projections: L2 malloc failed (q=%p k=%p v=%p ctx=%p)\n",
                   (void *)q_fp32, (void *)k_fp32, (void *)v_fp32, (void *)context_fp32);
#endif
            if (q_fp32) pi_l2_free(q_fp32, token_count * sizeof(float));
            if (k_fp32) pi_l2_free(k_fp32, token_count * sizeof(float));
            if (v_fp32) pi_l2_free(v_fp32, token_count * sizeof(float));
            if (context_fp32) pi_l2_free(context_fp32, token_count * sizeof(float));
            return;
        }

        // Get L1 buffer for scores tiling
        // scores_l1 will hold tile_q rows of scores (tile_q * seq_len * 4 bytes)
        const int tile_q = cfg->tile_q;  // From planner (e.g., 44 for test_15)
        float *scores_l1 = (float *)cfg->l1_buffer;
        const size_t scores_tile_bytes = tile_q * seq_len * sizeof(float);

#ifndef MINIMAL_OUTPUT
        printf("MHSA FP32 Projections: Using L1 tiling (tile_q=%d, scores_tile=%zu bytes)\n", tile_q, scores_tile_bytes);
#endif

        if (pool_mode == MHSA_POOL_MEAN) {
            pooled = (float *)pi_l2_malloc(embed_dim * sizeof(float));
            if (!pooled) {
#ifndef MINIMAL_OUTPUT
                printf("MHSA FP32 Projections: malloc failed for pooled\n");
#endif
                // No scores to free (using L1)
                pi_l2_free(context_fp32, token_count * sizeof(float));
                pi_l2_free(v_fp32, token_count * sizeof(float));
                pi_l2_free(k_fp32, token_count * sizeof(float));
                pi_l2_free(q_fp32, token_count * sizeof(float));
                return;
            }
            memset(pooled, 0, embed_dim * sizeof(float));
        } else {
            out_proj = (float *)pi_l2_malloc(token_count * sizeof(float));
            if (!out_proj) {
#ifndef MINIMAL_OUTPUT
                printf("MHSA FP32 Projections: malloc failed for out_proj\n");
#endif
                // No scores to free (using L1)
                pi_l2_free(context_fp32, token_count * sizeof(float));
                pi_l2_free(v_fp32, token_count * sizeof(float));
                pi_l2_free(k_fp32, token_count * sizeof(float));
                pi_l2_free(q_fp32, token_count * sizeof(float));
                if (pooled) pi_l2_free(pooled, embed_dim * sizeof(float));
                return;
            }
        }

#ifndef MINIMAL_OUTPUT
        printf("MHSA L2 Fallback: Running FP32 projections. scale_in=%.10f, scale_out=%.10f, pool_mode=%d\n", cfg->scale_input, cfg->scale_output, pool_mode);
#endif

        // Compute Q/K/V in FP32 directly from INT8 input/weights
        for (int t = 0; t < seq_len; t++) {
            const int8_t *x_row = cfg->input_buffer_l2 + t * embed_dim;
            for (int d = 0; d < embed_dim; d++) {
                float acc_q = cfg->q_bias_l2 ? (float)cfg->q_bias_l2[d] * mult_q : 0.0f;
                float acc_k = cfg->k_bias_l2 ? (float)cfg->k_bias_l2[d] * mult_k : 0.0f;
                float acc_v = cfg->v_bias_l2 ? (float)cfg->v_bias_l2[d] * mult_v : 0.0f;
                const int8_t *wq = cfg->q_weight_l2 + d * embed_dim;
                const int8_t *wk = cfg->k_weight_l2 + d * embed_dim;
                const int8_t *wv = cfg->v_weight_l2 + d * embed_dim;
                for (int j = 0; j < embed_dim; j++) {
                    const float x_val = (float)x_row[j];
                    acc_q += x_val * (float)wq[j] * mult_q;
                    acc_k += x_val * (float)wk[j] * mult_k;
                    acc_v += x_val * (float)wv[j] * mult_v;
                }
                q_fp32[t * embed_dim + d] = acc_q;
                k_fp32[t * embed_dim + d] = acc_k;
                v_fp32[t * embed_dim + d] = acc_v;
            }
        }

        // Attention per head (TILED: process tile_q rows at a time using L1 scores buffer)
        for (int head = 0; head < num_heads; head++) {
            const int head_offset = head * head_dim;

            // Process Q in tiles of size tile_q
            for (int tile_start = 0; tile_start < seq_len; tile_start += tile_q) {
                const int current_tile_size = (tile_start + tile_q > seq_len) ? (seq_len - tile_start) : tile_q;

                // Compute scores for this tile: tile_size rows x seq_len cols
                for (int i = 0; i < current_tile_size; i++) {
                    const int global_i = tile_start + i;
                    const float *q_row = q_fp32 + global_i * embed_dim + head_offset;
                    for (int j = 0; j < seq_len; j++) {
                        const float *k_row = k_fp32 + j * embed_dim + head_offset;
                        float score = 0.0f;
                        for (int d = 0; d < head_dim; d++) {
                            score += q_row[d] * k_row[d];
                        }
                        scores_l1[i * seq_len + j] = score * cfg->softmax_scale;
                    }
                }

                // Softmax rows (only for this tile)
                for (int i = 0; i < current_tile_size; i++) {
                    float *score_row = scores_l1 + i * seq_len;
                    if (cfg->softmax_lut != NULL) {
                        // i-Softmax path: use LUT for bit-exact matching with Python
                        i_softmax_row(score_row, score_row, seq_len, cfg->softmax_lut);
                    } else {
                        // Legacy FP32 path
                        float max_val = score_row[0];
                        for (int j = 1; j < seq_len; j++) if (score_row[j] > max_val) max_val = score_row[j];
                        float sum = 0.0f;
                        for (int j = 0; j < seq_len; j++) { score_row[j] = fast_exp(score_row[j] - max_val); sum += score_row[j]; }
                        float inv_sum = (sum > 1e-8f) ? (1.0f / sum) : 0.0f;
                        for (int j = 0; j < seq_len; j++) score_row[j] *= inv_sum;
                    }
                }

                // Context = softmax(scores) x V (only for this tile)
                for (int i = 0; i < current_tile_size; i++) {
                    const int global_i = tile_start + i;
                    float *ctx_row = context_fp32 + global_i * embed_dim + head_offset;
                    for (int d = 0; d < head_dim; d++) {
                        float acc = 0.0f;
                        for (int j = 0; j < seq_len; j++) {
                            const float v_val = v_fp32[j * embed_dim + head_offset + d];
                            acc += scores_l1[i * seq_len + j] * v_val;
                        }
                        ctx_row[d] = acc;
                    }
                }
            }
        }

        const float mult_out = cfg->scale_out_weight;
        for (int t = 0; t < seq_len; t++) {
            const float *ctx_row = context_fp32 + t * embed_dim;
            for (int d = 0; d < embed_dim; d++) {
                float acc = cfg->out_bias_l2 ? (float)cfg->out_bias_l2[d] * bias_scale_out : 0.0f;
                const int8_t *w_row = cfg->out_weight_l2 + d * embed_dim;
                for (int j = 0; j < embed_dim; j++) {
                    acc += ctx_row[j] * (float)w_row[j] * mult_out;
                }
                if (pool_mode == MHSA_POOL_MEAN) {
                    pooled[d] += acc;
                } else {
                    out_proj[t * embed_dim + d] = acc;
                }
            }
        }

        int8_t *out_int8 = cfg->output_buffer_l2;
        if (pool_mode == MHSA_POOL_MEAN) {
            for (int d = 0; d < embed_dim; d++) {
                float mean = pooled[d] / seq_len;
                int32_t q_val = (int32_t)lrintf(mean / cfg->scale_output);
                if (q_val > 127) q_val = 127;
                if (q_val < -128) q_val = -128;
                out_int8[d] = (int8_t)q_val;
            }
        } else {
            const size_t total = token_count;
            for (size_t idx = 0; idx < total; idx++) {
                int32_t q_val = (int32_t)lrintf(out_proj[idx] / cfg->scale_output);
                if (q_val > 127) q_val = 127;
                if (q_val < -128) q_val = -128;
                out_int8[idx] = (int8_t)q_val;
            }
        }

        if (pooled) pi_l2_free(pooled, embed_dim * sizeof(float));
        if (out_proj) pi_l2_free(out_proj, token_count * sizeof(float));
        // No scores to free (using L1 buffer)
        pi_l2_free(context_fp32, token_count * sizeof(float));
        pi_l2_free(v_fp32, token_count * sizeof(float));
        pi_l2_free(k_fp32, token_count * sizeof(float));
        pi_l2_free(q_fp32, token_count * sizeof(float));
        return;
    }

    // INT8 projections path (Path A) with 8-core hybrid mode
    // Uses L1 caching for K and optimized fused workers
    const int seq_len = cfg->seq_len;
    const int num_heads = cfg->num_heads;
    const int head_dim = cfg->head_dim;
    const int embed_dim = cfg->embed_dim;

    int8_t *l1_workspace = cfg->l1_buffer;

    // Requantization parameters for INT32 → INT8 scores
    // Use log2(head_dim) for shift - divide by head_dim to normalize dot product
    const int16_t requant_mul = 1;
    int16_t requant_shift = 0;
    int temp_hd = head_dim;
    while (temp_hd > 1) { temp_hd >>= 1; requant_shift++; }  // log2(head_dim)

    // Check if head-contiguous layout is available (required for efficient L1 caching)
    if (!cfg->use_head_contiguous || !cfg->k_permuted_l2 || !cfg->v_permuted_l2) {
        goto pure_l2_fallback;
    }

    // Buffer sizes
    const size_t kv_buffer_size = seq_len * head_dim;

    // Dynamic 8-core tile sizing
    // Formula: 8 cores x seq_len x 6 bytes + K (kv_buffer_size) ≤ L1 available
    const size_t score_buffer_8c = 8 * seq_len * 6;  // 6 = sizeof(int32) + sizeof(int8) + sizeof(uint8)
    const size_t total_l1_8c = score_buffer_8c + kv_buffer_size;

    if (cfg->l1_buffer_size >= total_l1_8c) {
#ifndef MINIMAL_OUTPUT
        printf("CL: [MHSA L2 FB] 8-core path: q_src=%p k_src=%p v_src=%p out=%p\n",
               (void *)cfg->q_permuted_l2, (void *)cfg->k_permuted_l2, (void *)cfg->v_permuted_l2, (void *)cfg->output_buffer_l2);
        printf("CL: [MHSA L2 FB] seq_len=%d head_dim=%d num_heads=%d shift=%d\n",
               seq_len, head_dim, num_heads, requant_shift);
#endif
        // L1 Layout: [score buffers for 8 cores][K cache]
        int32_t *scores_int32_base = (int32_t *)l1_workspace;
        int8_t *scores_int8_base = (int8_t *)(scores_int32_base + 8 * seq_len);
        uint8_t *attn_uint8_base = (uint8_t *)(scores_int8_base + 8 * seq_len);
        int8_t *k_l1 = (int8_t *)(attn_uint8_base + 8 * seq_len);

        // Source buffers (head-contiguous layout)
        const int8_t *q_src = cfg->q_permuted_l2;
        const int8_t *k_src = cfg->k_permuted_l2;
        const int8_t *v_src = cfg->v_permuted_l2;
        const size_t head_data_size = seq_len * head_dim;

        for (int head = 0; head < num_heads; head++) {
            const size_t head_offset = head * head_data_size;

#ifndef MINIMAL_OUTPUT
            if (head == 0) {
                printf("CL: [MHSA L2 FB] head0 Q[0..4]: %d %d %d %d %d\n",
                       q_src[0], q_src[1], q_src[2], q_src[3], q_src[4]);
            }
#endif

            // DMA K to L1 (once per head)
            pi_cl_dma_copy_t k_dma;
            k_dma.dir = PI_CL_DMA_DIR_EXT2LOC;
            k_dma.size = kv_buffer_size;
            k_dma.ext = (uint32_t)(k_src + head_offset);
            k_dma.loc = (uint32_t)k_l1;
            k_dma.merge = 0;
            pi_cl_dma_memcpy(&k_dma);
            pi_cl_dma_wait(&k_dma);

            // Process all Q rows with 8 cores
            mhsa_l1_fused_args_t args = {
                .q_buffer_l2 = q_src + head_offset,
                .k_l1 = k_l1,                             // K in L1!
                .v_l1 = (int8_t *)(v_src + head_offset),  // V in L2
                .output_buffer_l2 = cfg->output_buffer_l2 + head_offset,  // Each head writes to its own location
                .scores_int32 = scores_int32_base,
                .scores_int8 = scores_int8_base,
                .attn_uint8 = attn_uint8_base,
                .tile_start = 0,
                .tile_size = seq_len,
                .seq_len = seq_len,
                .embed_dim = head_dim,
                .head_dim = head_dim,
                .head_offset = 0,
                .requant_mul = requant_mul,
                .requant_shift = requant_shift,
                .num_cores = 8
            };
#ifndef MINIMAL_OUTPUT
            printf("CL: [MHSA L2 FB] Fork head %d from core %d\n", head, pi_core_id());
#endif
            pi_cl_team_fork(8, mhsa_l1_fused_worker, &args);
#ifndef MINIMAL_OUTPUT
            printf("CL: [MHSA L2 FB] Fork done, out[0]=%d q[0]=%d k[0]=%d v[0]=%d\n",
                   args.output_buffer_l2[0], args.q_buffer_l2[0],
                   args.k_l1[0], args.v_l1[0]);
#endif
        }
#ifndef MINIMAL_OUTPUT
        printf("CL: [MHSA L2 FB] Done. out[0..4]: %d %d %d %d %d\n",
               cfg->output_buffer_l2[0], cfg->output_buffer_l2[1], cfg->output_buffer_l2[2],
               cfg->output_buffer_l2[3], cfg->output_buffer_l2[4]);
#endif
        return;
    }

    // Fallback: 6-core mode with K in L1
    {
        const int num_cores_to_use = 6;
        const size_t score_buffer_6c = num_cores_to_use * seq_len * 6;
        const size_t total_l1_6c = score_buffer_6c + kv_buffer_size;

        if (cfg->l1_buffer_size >= total_l1_6c) {
            int32_t *scores_int32_base = (int32_t *)l1_workspace;
            int8_t *scores_int8_base = (int8_t *)(scores_int32_base + num_cores_to_use * seq_len);
            uint8_t *attn_uint8_base = (uint8_t *)(scores_int8_base + num_cores_to_use * seq_len);
            int8_t *k_l1 = (int8_t *)(attn_uint8_base + num_cores_to_use * seq_len);

            const int8_t *q_src = cfg->q_permuted_l2;
            const int8_t *k_src = cfg->k_permuted_l2;
            const int8_t *v_src = cfg->v_permuted_l2;
            const size_t head_data_size = seq_len * head_dim;

            for (int head = 0; head < num_heads; head++) {
                const size_t head_offset = head * head_data_size;

                pi_cl_dma_copy_t k_dma;
                k_dma.dir = PI_CL_DMA_DIR_EXT2LOC;
                k_dma.size = kv_buffer_size;
                k_dma.ext = (uint32_t)(k_src + head_offset);
                k_dma.loc = (uint32_t)k_l1;
                k_dma.merge = 0;
                pi_cl_dma_memcpy(&k_dma);
                pi_cl_dma_wait(&k_dma);

                mhsa_l1_fused_args_t args = {
                    .q_buffer_l2 = q_src + head_offset,
                    .k_l1 = k_l1,
                    .v_l1 = (int8_t *)(v_src + head_offset),
                    .output_buffer_l2 = cfg->output_buffer_l2 + head_offset,  // Each head writes to its own location
                    .scores_int32 = scores_int32_base,
                    .scores_int8 = scores_int8_base,
                    .attn_uint8 = attn_uint8_base,
                    .tile_start = 0,
                    .tile_size = seq_len,
                    .seq_len = seq_len,
                    .embed_dim = head_dim,
                    .head_dim = head_dim,
                    .head_offset = 0,
                    .requant_mul = requant_mul,
                    .requant_shift = requant_shift,
                    .num_cores = num_cores_to_use
                };
                pi_cl_team_fork(num_cores_to_use, mhsa_l1_fused_worker, &args);
            }
            return;
        }
    }

pure_l2_fallback:
    {
        // Pure L2 mode: No K caching, 8 cores with score buffers in L1
        const size_t fb_per_core = seq_len * 6;
        const size_t fb_total_buffer = NUM_CORES * fb_per_core;

        if (cfg->l1_buffer_size < fb_total_buffer) {
            return;  // Cannot run - not enough L1 even for score buffers
        }

        const int fb_head_contig = cfg->use_head_contiguous &&
                                   cfg->q_permuted_l2 && cfg->k_permuted_l2 && cfg->v_permuted_l2;
        const int fb_embed_dim = fb_head_contig ? head_dim : embed_dim;
        const int fb_head_data_offset = fb_head_contig ? (seq_len * head_dim) : 0;

        int32_t *fb_scores_int32 = (int32_t *)l1_workspace;
        int8_t *fb_scores_int8 = (int8_t *)(fb_scores_int32 + NUM_CORES * seq_len);
        uint8_t *fb_attn_uint8 = (uint8_t *)(fb_scores_int8 + NUM_CORES * seq_len);

        const int8_t *fb_q_src = fb_head_contig ? cfg->q_permuted_l2 : cfg->q_buffer_l2;
        const int8_t *fb_k_src = fb_head_contig ? cfg->k_permuted_l2 : cfg->k_buffer_l2;
        const int8_t *fb_v_src = fb_head_contig ? cfg->v_permuted_l2 : cfg->v_buffer_l2;

        for (int head = 0; head < num_heads; head++) {
            const int fb_head_offset = fb_head_contig ? (head * fb_head_data_offset) : (head * head_dim);

            mhsa_l2_fused_args_t fused_args = {
                .q_buffer_l2 = fb_q_src + (fb_head_contig ? fb_head_offset : 0),
                .k_buffer_l2 = fb_k_src + (fb_head_contig ? fb_head_offset : 0),
                .v_buffer_l2 = fb_v_src + (fb_head_contig ? fb_head_offset : 0),
                .output_buffer_l2 = cfg->output_buffer_l2 + (fb_head_contig ? fb_head_offset : 0),  // Each head writes to its own location
                .scores_int32 = fb_scores_int32,
                .scores_int8 = fb_scores_int8,
                .attn_uint8 = fb_attn_uint8,
                .tile_start = 0,
                .tile_size = seq_len,
                .seq_len = seq_len,
                .embed_dim = fb_embed_dim,
                .head_dim = head_dim,
                .head_offset = fb_head_contig ? 0 : (head * head_dim),
                .requant_mul = requant_mul,
                .requant_shift = requant_shift
            };
            pi_cl_team_fork(NUM_CORES, mhsa_l2_fused_worker, &fused_args);
        }
    }
}

static void mhsa_tiled_l1_inner_loop(mhsa_pipeline_config_t *cfg) {
    // Calculate actual required L1 size
    // Two paths:
    // A) FP32 path: QK→INT32→FP32 softmax→FP32 AV→INT8 (shared INT32/FP32 buffer)
    // B) Integer path: QK→INT32+INT8→UINT8 iSoftmax→INT8 AV (separate INT8 and UINT8 buffers)
    int8_t *l1_workspace = cfg->l1_buffer;
    size_t k_size = cfg->seq_len * cfg->head_dim;
    size_t v_size = cfg->seq_len * cfg->head_dim;
    int tile_q = cfg->tile_q;
    int num_tiles = cfg->num_tiles;
    size_t q_tile_size = tile_q * cfg->head_dim;
    size_t m_tile_size = tile_q * cfg->head_dim;
    const int use_int_softmax = cfg->use_integer_softmax;
    const int use_fused_qk_softmax_av = use_int_softmax && MHSA_FUSE_QK_SOFTMAX_AV;
    const int use_per_core_fused = use_fused_qk_softmax_av && MHSA_FUSE_QK_SOFTMAX_AV_PER_CORE;
    const int use_head_contiguous = cfg->use_head_contiguous;

    size_t v_t_size = 0;
    int use_v_transpose = 0;
#if MHSA_V_TRANSPOSE_L1
    if (use_int_softmax) {
        use_v_transpose = 1;
#if !MHSA_V_TRANSPOSE_REUSE_V_BUFFER
        v_t_size = v_size;
#endif
    }
#endif

    // Scores buffer size depends on path
    // FP32 path: tile_q * seq_len * 4 bytes (shared INT32/FP32)
    // Integer path: tile_q * seq_len * 4 (INT32) + tile_q * seq_len (INT8) + tile_q * seq_len (UINT8)
    size_t scores_int32_size = use_per_core_fused
        ? (NUM_CORES * cfg->seq_len * sizeof(int32_t))
        : (tile_q * cfg->seq_len * sizeof(int32_t));
    size_t scores_int8_size = (use_int_softmax && !use_per_core_fused && MHSA_TILED_QK_STORE_INT8_SCORES)
        ? (tile_q * cfg->seq_len) : 0;
    size_t attn_uint8_size = use_int_softmax
        ? (use_per_core_fused ? (NUM_CORES * cfg->seq_len) : (tile_q * cfg->seq_len))
        : 0;
    size_t scores_fp32_size = use_int_softmax ? 0 : (tile_q * cfg->seq_len * sizeof(float));

    size_t total_required = k_size + v_size + v_t_size + 2*q_tile_size + scores_int32_size +
                            scores_int8_size + attn_uint8_size + scores_fp32_size + 2*m_tile_size;

    // Check if we have enough L1 space
    if (cfg->l1_buffer && cfg->l1_buffer_size < total_required && use_v_transpose) {
        while (tile_q > NUM_CORES && cfg->l1_buffer_size < total_required) {
            tile_q--;
            num_tiles = (cfg->seq_len + tile_q - 1) / tile_q;
            q_tile_size = tile_q * cfg->head_dim;
            m_tile_size = tile_q * cfg->head_dim;
            scores_int32_size = use_per_core_fused
                ? (NUM_CORES * cfg->seq_len * sizeof(int32_t))
                : (tile_q * cfg->seq_len * sizeof(int32_t));
            scores_int8_size = (use_int_softmax && !use_per_core_fused && MHSA_TILED_QK_STORE_INT8_SCORES)
                ? (tile_q * cfg->seq_len) : 0;
            attn_uint8_size = use_int_softmax
                ? (use_per_core_fused ? (NUM_CORES * cfg->seq_len) : (tile_q * cfg->seq_len))
                : 0;
            scores_fp32_size = use_int_softmax ? 0 : (tile_q * cfg->seq_len * sizeof(float));
            total_required = k_size + v_size + v_t_size + 2*q_tile_size + scores_int32_size +
                             scores_int8_size + attn_uint8_size + scores_fp32_size + 2*m_tile_size;
        }
    }
    if ((!cfg->l1_buffer || cfg->l1_buffer_size < total_required) && use_v_transpose) {
        v_t_size = 0;
        use_v_transpose = 0;
        tile_q = cfg->tile_q;
        num_tiles = cfg->num_tiles;
        q_tile_size = tile_q * cfg->head_dim;
        m_tile_size = tile_q * cfg->head_dim;
        scores_int32_size = use_per_core_fused
            ? (NUM_CORES * cfg->seq_len * sizeof(int32_t))
            : (tile_q * cfg->seq_len * sizeof(int32_t));
        scores_int8_size = (use_int_softmax && !use_per_core_fused && MHSA_TILED_QK_STORE_INT8_SCORES)
            ? (tile_q * cfg->seq_len) : 0;
        attn_uint8_size = use_int_softmax
            ? (use_per_core_fused ? (NUM_CORES * cfg->seq_len) : (tile_q * cfg->seq_len))
            : 0;
        scores_fp32_size = use_int_softmax ? 0 : (tile_q * cfg->seq_len * sizeof(float));
        total_required = k_size + v_size + v_t_size + 2*q_tile_size + scores_int32_size +
                         scores_int8_size + attn_uint8_size + scores_fp32_size + 2*m_tile_size;
    }
    if (use_per_core_fused && cfg->l1_buffer && cfg->l1_buffer_size >= total_required) {
        int max_tile_q = tile_q;

        while (max_tile_q < cfg->seq_len) {
            int next_tile_q = max_tile_q + 1;
            size_t next_q_tile_size = next_tile_q * cfg->head_dim;
            size_t next_total = k_size + v_size + v_t_size + 2*next_q_tile_size +
                                scores_int32_size + scores_int8_size + attn_uint8_size +
                                scores_fp32_size + 2*next_q_tile_size;
            if (next_total > cfg->l1_buffer_size) {
                break;
            }
            max_tile_q = next_tile_q;
        }

        int aligned_tile_q = (max_tile_q / NUM_CORES) * NUM_CORES;
        if (aligned_tile_q < NUM_CORES) aligned_tile_q = NUM_CORES;
        if (aligned_tile_q < tile_q) aligned_tile_q = tile_q;

        if (aligned_tile_q != tile_q) {
            tile_q = aligned_tile_q;
            num_tiles = (cfg->seq_len + tile_q - 1) / tile_q;
            q_tile_size = tile_q * cfg->head_dim;
            m_tile_size = tile_q * cfg->head_dim;
            total_required = k_size + v_size + v_t_size + 2*q_tile_size +
                             scores_int32_size + scores_int8_size + attn_uint8_size +
                             scores_fp32_size + 2*m_tile_size;
        }
    }
    if (!cfg->l1_buffer || cfg->l1_buffer_size < total_required) {
#ifndef MINIMAL_OUTPUT
        printf("CL: [MHSA L1] FALLBACK - need %u bytes, have %u (l1_buffer=%p)\n",
               (unsigned)total_required, (unsigned)cfg->l1_buffer_size, (void *)cfg->l1_buffer);
#endif
        mhsa_l2_fallback_inner(cfg);
        return;
    }
#ifndef MINIMAL_OUTPUT
    printf("CL: [MHSA L1] Using L1 tiled path (need %u, have %u)\n",
           (unsigned)total_required, (unsigned)cfg->l1_buffer_size);
#endif

    int8_t *k_l1 = l1_workspace;
    int8_t *v_l1 = k_l1 + k_size;
    int8_t *v_t_l1 = NULL;
    if (use_v_transpose) {
#if MHSA_V_TRANSPOSE_REUSE_V_BUFFER
        v_t_l1 = v_l1;          // Reuse V buffer for transposed layout
#else
        v_t_l1 = v_l1 + v_size; // Separate Vᵀ buffer
#endif
    }
    int8_t *q_tile_l1_a = v_l1 + v_size + v_t_size;
    int8_t *q_tile_l1_b = q_tile_l1_a + q_tile_size;

    // Buffer layout depends on path
    int32_t *scores_int32_l1 = (int32_t *)(q_tile_l1_b + q_tile_size);
    int8_t *scores_int8_l1 = NULL;
    uint8_t *attn_uint8_l1 = NULL;
    float *scores_l1 = NULL;

    if (use_int_softmax) {
        // Integer path: INT32 scores + (optional INT8 intermediate) + UINT8 attention
        if (scores_int8_size > 0) {
            scores_int8_l1 = (int8_t *)scores_int32_l1 + scores_int32_size;
            attn_uint8_l1 = (uint8_t *)scores_int8_l1 + scores_int8_size;
        } else {
            scores_int8_l1 = NULL;
            attn_uint8_l1 = (uint8_t *)((int8_t *)scores_int32_l1 + scores_int32_size);
        }
        // m_tile starts after attn_uint8
    } else {
        // FP32 path: Shared INT32/FP32 buffer
        scores_l1 = (float *)scores_int32_l1;  // Same memory, different view
    }

    int8_t *m_tile_l1_a = use_int_softmax
        ? ((int8_t *)attn_uint8_l1 + attn_uint8_size)
        : ((int8_t *)scores_l1 + scores_int32_size);
    int8_t *m_tile_l1_b = m_tile_l1_a + m_tile_size;
    const float softmax_scale = cfg->softmax_scale;

    // Batch DMA descriptor limit (GAP9 hardware has 16 counters; see `ares_config.h`)
    pi_cl_dma_copy_t dma_batch[MHSA_DMA_BATCH_SIZE];

    // Head-contiguous layout sizes for bulk DMA (when enabled)
    const size_t head_data_size = cfg->seq_len * cfg->head_dim;  // e.g., 196 * 32 = 6272 bytes

    // Debug: Print DMA mode once at start
#ifndef MINIMAL_OUTPUT
    if (cfg->num_heads > 0) {
        if (use_head_contiguous && cfg->k_permuted_l2 && cfg->v_permuted_l2 && cfg->q_permuted_l2) {
            printf("CL: [MHSA DMA] Using BULK DMA (head-contiguous layout, %d bytes/head)\n", (int)head_data_size);
        } else {
            printf("CL: [MHSA DMA] Using STRIDED DMA (original layout, %d batches of %d transfers)\n",
                   (cfg->seq_len + MHSA_DMA_BATCH_SIZE - 1) / MHSA_DMA_BATCH_SIZE, MHSA_DMA_BATCH_SIZE);
        }
    }
#endif

    for (int head = 0; head < cfg->num_heads; head++) {
        const int head_offset = head * cfg->head_dim;

        // ---
        // K LOADING: Bulk DMA (head-contiguous) vs Strided batch DMA (original)
        // ---
        if (use_head_contiguous && cfg->k_permuted_l2) {
            // HEAD-CONTIGUOUS PATH: Single bulk DMA per head
            // Data layout: [num_heads, seq_len, head_dim] - each head's K is contiguous
            pi_cl_dma_copy_t k_bulk;
            k_bulk.dir = PI_CL_DMA_DIR_EXT2LOC;
            k_bulk.size = head_data_size;
            k_bulk.ext = (uint32_t)(cfg->k_permuted_l2 + head * head_data_size);
            k_bulk.loc = (uint32_t)k_l1;
            k_bulk.merge = 0;
            pi_cl_dma_memcpy(&k_bulk);
            pi_cl_dma_wait(&k_bulk);
	        } else {
	            // STRIDED PATH: Batch DMA with 8 transfers at a time
	            for (int batch_start = 0; batch_start < cfg->seq_len; batch_start += MHSA_DMA_BATCH_SIZE) {
                int batch_end = batch_start + MHSA_DMA_BATCH_SIZE;
                if (batch_end > cfg->seq_len) batch_end = cfg->seq_len;
                int batch_count = batch_end - batch_start;

                // Queue batch of K row DMAs
                for (int i = 0; i < batch_count; i++) {
                    int t = batch_start + i;
                    dma_batch[i].dir = PI_CL_DMA_DIR_EXT2LOC;
                    dma_batch[i].size = cfg->head_dim;
                    dma_batch[i].ext = (uint32_t)(cfg->k_buffer_l2 + t * cfg->embed_dim + head_offset);
                    dma_batch[i].loc = (uint32_t)(k_l1 + t * cfg->head_dim);
                    dma_batch[i].merge = 0;
                    pi_cl_dma_memcpy(&dma_batch[i]);
                }
                // Wait for batch
                for (int i = 0; i < batch_count; i++) {
                    pi_cl_dma_wait(&dma_batch[i]);
                }
            }
        }

        // ---
        // V LOADING: Optionally load directly into transposed layout in L1
        // ---
        if (use_v_transpose) {
#if MHSA_V_TRANSPOSE_REUSE_V_BUFFER
            // Load V directly as Vᵀ[d, j] (row-major by d) using 2D DMA column gathers.
            // This avoids allocating both V and Vᵀ in L1, allowing a larger tile_q.
            const uint32_t v_src_base = (uint32_t)(
                (use_head_contiguous && cfg->v_permuted_l2)
                    ? (cfg->v_permuted_l2 + head * head_data_size)
                    : (cfg->v_buffer_l2 + head_offset)
            );
            const uint32_t v_src_stride = (uint32_t)(
                (use_head_contiguous && cfg->v_permuted_l2) ? cfg->head_dim : cfg->embed_dim
            );

            pi_cl_dma_copy_2d_t v_col_dma;
            v_col_dma.dir = PI_CL_DMA_DIR_EXT2LOC;
            v_col_dma.merge = 0;
            v_col_dma.size = (uint32_t)cfg->seq_len;  // total bytes
            v_col_dma.length = 1;                     // bytes per line
            v_col_dma.stride = v_src_stride;          // bytes between lines in L2

            for (int d = 0; d < cfg->head_dim; d++) {
                v_col_dma.ext = v_src_base + (uint32_t)d;
                v_col_dma.loc = (uint32_t)(v_t_l1 + d * cfg->seq_len);
                pi_cl_dma_memcpy_2d(&v_col_dma);
                pi_cl_dma_wait(&v_col_dma);
            }
#else
            // Baseline: load V in original layout, then transpose into Vᵀ buffer.
            if (use_head_contiguous && cfg->v_permuted_l2) {
                pi_cl_dma_copy_t v_bulk;
                v_bulk.dir = PI_CL_DMA_DIR_EXT2LOC;
                v_bulk.size = head_data_size;
                v_bulk.ext = (uint32_t)(cfg->v_permuted_l2 + head * head_data_size);
                v_bulk.loc = (uint32_t)v_l1;
                v_bulk.merge = 0;
                pi_cl_dma_memcpy(&v_bulk);
                pi_cl_dma_wait(&v_bulk);
            } else {
                for (int batch_start = 0; batch_start < cfg->seq_len; batch_start += MHSA_DMA_BATCH_SIZE) {
                    int batch_end = batch_start + MHSA_DMA_BATCH_SIZE;
                    if (batch_end > cfg->seq_len) batch_end = cfg->seq_len;
                    int batch_count = batch_end - batch_start;

                    for (int i = 0; i < batch_count; i++) {
                        int t = batch_start + i;
                        dma_batch[i].dir = PI_CL_DMA_DIR_EXT2LOC;
                        dma_batch[i].size = cfg->head_dim;
                        dma_batch[i].ext = (uint32_t)(cfg->v_buffer_l2 + t * cfg->embed_dim + head_offset);
                        dma_batch[i].loc = (uint32_t)(v_l1 + t * cfg->head_dim);
                        dma_batch[i].merge = 0;
                        pi_cl_dma_memcpy(&dma_batch[i]);
                    }
                    for (int i = 0; i < batch_count; i++) {
                        pi_cl_dma_wait(&dma_batch[i]);
                    }
                }
            }

            for (int j = 0; j < cfg->seq_len; j++) {
                for (int d = 0; d < cfg->head_dim; d++) {
                    v_t_l1[d * cfg->seq_len + j] = v_l1[j * cfg->head_dim + d];
                }
            }
#endif  // MHSA_V_TRANSPOSE_REUSE_V_BUFFER
        } else {
            // No transpose: load V in original layout.
            if (use_head_contiguous && cfg->v_permuted_l2) {
                pi_cl_dma_copy_t v_bulk;
                v_bulk.dir = PI_CL_DMA_DIR_EXT2LOC;
                v_bulk.size = head_data_size;
                v_bulk.ext = (uint32_t)(cfg->v_permuted_l2 + head * head_data_size);
                v_bulk.loc = (uint32_t)v_l1;
                v_bulk.merge = 0;
                pi_cl_dma_memcpy(&v_bulk);
                pi_cl_dma_wait(&v_bulk);
            } else {
                for (int batch_start = 0; batch_start < cfg->seq_len; batch_start += MHSA_DMA_BATCH_SIZE) {
                    int batch_end = batch_start + MHSA_DMA_BATCH_SIZE;
                    if (batch_end > cfg->seq_len) batch_end = cfg->seq_len;
                    int batch_count = batch_end - batch_start;

                    for (int i = 0; i < batch_count; i++) {
                        int t = batch_start + i;
                        dma_batch[i].dir = PI_CL_DMA_DIR_EXT2LOC;
                        dma_batch[i].size = cfg->head_dim;
                        dma_batch[i].ext = (uint32_t)(cfg->v_buffer_l2 + t * cfg->embed_dim + head_offset);
                        dma_batch[i].loc = (uint32_t)(v_l1 + t * cfg->head_dim);
                        dma_batch[i].merge = 0;
                        pi_cl_dma_memcpy(&dma_batch[i]);
                    }
                    for (int i = 0; i < batch_count; i++) {
                        pi_cl_dma_wait(&dma_batch[i]);
                    }
                }
            }
        }

	        int current_q_buffer = 0;
        int current_m_buffer = 0;
        for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
            const int q_start = tile_idx * tile_q;
            int q_end = q_start + tile_q;
            if (q_end > cfg->seq_len) q_end = cfg->seq_len;
            const int actual_tile_q = q_end - q_start;
            int8_t *q_tile_l1 = (current_q_buffer == 0) ? q_tile_l1_a : q_tile_l1_b;
            int8_t *m_tile_l1 = (current_m_buffer == 0) ? m_tile_l1_a : m_tile_l1_b;

            // ---
            // Q TILE LOADING: Bulk DMA (head-contiguous) vs Strided batch DMA
            // ---
            if (use_head_contiguous && cfg->q_permuted_l2) {
                // HEAD-CONTIGUOUS PATH: Single bulk DMA for Q tile
                // Data layout: [num_heads, seq_len, head_dim]
                pi_cl_dma_copy_t q_tile_bulk;
                q_tile_bulk.dir = PI_CL_DMA_DIR_EXT2LOC;
                q_tile_bulk.size = actual_tile_q * cfg->head_dim;
                q_tile_bulk.ext = (uint32_t)(cfg->q_permuted_l2 + head * head_data_size + q_start * cfg->head_dim);
                q_tile_bulk.loc = (uint32_t)q_tile_l1;
                q_tile_bulk.merge = 0;
                pi_cl_dma_memcpy(&q_tile_bulk);
                pi_cl_dma_wait(&q_tile_bulk);
            } else {
                // STRIDED PATH: Batch DMA with 8 transfers at a time
                for (int batch_start = 0; batch_start < actual_tile_q; batch_start += MHSA_DMA_BATCH_SIZE) {
                    int batch_end = batch_start + MHSA_DMA_BATCH_SIZE;
                    if (batch_end > actual_tile_q) batch_end = actual_tile_q;
                    int batch_count = batch_end - batch_start;

                    for (int i = 0; i < batch_count; i++) {
                        int t = q_start + batch_start + i;
                        dma_batch[i].dir = PI_CL_DMA_DIR_EXT2LOC;
                        dma_batch[i].size = cfg->head_dim;
                        dma_batch[i].ext = (uint32_t)(cfg->q_buffer_l2 + t * cfg->embed_dim + head_offset);
                        dma_batch[i].loc = (uint32_t)(q_tile_l1 + (batch_start + i) * cfg->head_dim);
                        dma_batch[i].merge = 0;
                        pi_cl_dma_memcpy(&dma_batch[i]);
                    }
                    for (int i = 0; i < batch_count; i++) {
                        pi_cl_dma_wait(&dma_batch[i]);
                    }
                }
            }

            if (use_fused_qk_softmax_av) {
                unsigned int inner_fused_start = pi_perf_read(PI_PERF_CYCLES);
                mhsa_tile_args_t fused_args = {
                    .q_tile_l1 = q_tile_l1,
                    .k_l1 = k_l1,
                    .v_l1 = v_l1,
                    .v_t_l1 = v_t_l1,
                    .scores_int32_l1 = scores_int32_l1,
                    .scores_l1 = scores_l1,
                    .scores_int8_l1 = scores_int8_l1,
                    .attn_uint8_l1 = attn_uint8_l1,
                    .m_tile_l1 = m_tile_l1,
                    .tile_q = actual_tile_q,
                    .seq_len = cfg->seq_len,
                    .head_dim = cfg->head_dim,
                    .scale_q = cfg->scale_q,
                    .scale_k = cfg->scale_k,
                    .softmax_scale = softmax_scale,
                    .softmax_lut = cfg->softmax_lut,
                    .use_integer_softmax = use_int_softmax,
                    .requant_mul = cfg->requant_mul,
                    .requant_shift = cfg->requant_shift,
                    .isoftmax_coeffA = cfg->isoftmax_coeffA,
                    .isoftmax_coeffB = cfg->isoftmax_coeffB,
                    .isoftmax_coeffC = cfg->isoftmax_coeffC,
                    .isoftmax_log2 = cfg->isoftmax_log2,
                    .isoftmax_n_levels = cfg->isoftmax_n_levels
                };
                if (use_per_core_fused) {
                    pi_cl_team_fork(NUM_CORES, mhsa_tile_qk_softmax_av_per_core_worker, &fused_args);
                } else {
                    pi_cl_team_fork(NUM_CORES, mhsa_tile_qk_softmax_av_worker, &fused_args);
                }
                unsigned int inner_fused_end = pi_perf_read(PI_PERF_CYCLES);
                cfg->inner_qk_cycles += (inner_fused_end - inner_fused_start);
            } else {
                // QK kernel: Q_tile x K^T → INT32 scores (+ INT8 if integer path)
                unsigned int inner_qk_start = pi_perf_read(PI_PERF_CYCLES);
                mhsa_tile_args_t qk_args = {
                    .q_tile_l1 = q_tile_l1,
                    .k_l1 = k_l1,
                    .scores_int32_l1 = scores_int32_l1,
                    .scores_int8_l1 = scores_int8_l1,
                    .tile_q = actual_tile_q,
                    .seq_len = cfg->seq_len,
                    .head_dim = cfg->head_dim,
                    .use_integer_softmax = use_int_softmax,
                    .requant_mul = cfg->requant_mul,
                    .requant_shift = cfg->requant_shift
                };
                pi_cl_team_fork(NUM_CORES, mhsa_tile_qk_worker, &qk_args);
                unsigned int inner_qk_end = pi_perf_read(PI_PERF_CYCLES);
                cfg->inner_qk_cycles += (inner_qk_end - inner_qk_start);

                // Softmax + AV (fused for integer path when enabled)
                if (use_int_softmax) {
#if MHSA_FUSE_SOFTMAX_AV
                    unsigned int inner_softmax_start = pi_perf_read(PI_PERF_CYCLES);
                    mhsa_tile_args_t fused_args = {
                        .scores_int32_l1 = scores_int32_l1,
                        .scores_int8_l1 = scores_int8_l1,
                        .attn_uint8_l1 = attn_uint8_l1,
                        .m_tile_l1 = m_tile_l1,
                        .v_l1 = v_l1,
                        .v_t_l1 = v_t_l1,
                        .tile_q = actual_tile_q,
                        .seq_len = cfg->seq_len,
                        .head_dim = cfg->head_dim,
                        .scale_q = cfg->scale_q,
                        .scale_k = cfg->scale_k,
                        .softmax_scale = softmax_scale,
                        .softmax_lut = cfg->softmax_lut,
                        .use_integer_softmax = use_int_softmax,
                        .requant_mul = cfg->requant_mul,
                        .requant_shift = cfg->requant_shift,
                        .isoftmax_coeffA = cfg->isoftmax_coeffA,
                        .isoftmax_coeffB = cfg->isoftmax_coeffB,
                        .isoftmax_coeffC = cfg->isoftmax_coeffC,
                        .isoftmax_log2 = cfg->isoftmax_log2,
                        .isoftmax_n_levels = cfg->isoftmax_n_levels
                    };
                    if (v_t_l1) {
                        pi_cl_team_fork(NUM_CORES, mhsa_tile_softmax_av_int_vt_worker, &fused_args);
                    } else {
                        pi_cl_team_fork(NUM_CORES, mhsa_tile_softmax_av_int_worker, &fused_args);
                    }
                    unsigned int inner_softmax_end = pi_perf_read(PI_PERF_CYCLES);
                    cfg->inner_softmax_cycles += (inner_softmax_end - inner_softmax_start);
#else
                    // Softmax kernel: INT32→UINT8 (integer path)
                    unsigned int inner_softmax_start = pi_perf_read(PI_PERF_CYCLES);
                    mhsa_tile_args_t softmax_args = {
                        .scores_int32_l1 = scores_int32_l1,
                        .scores_l1 = scores_l1,
                        .scores_int8_l1 = scores_int8_l1,
                        .attn_uint8_l1 = attn_uint8_l1,
                        .tile_q = actual_tile_q,
                        .seq_len = cfg->seq_len,
                        .scale_q = cfg->scale_q,
                        .scale_k = cfg->scale_k,
                        .softmax_scale = softmax_scale,
                        .softmax_lut = cfg->softmax_lut,
                        .use_integer_softmax = use_int_softmax,
                        .requant_mul = cfg->requant_mul,      // For fully-integer softmax
                        .requant_shift = cfg->requant_shift,  // For fully-integer softmax
                        .isoftmax_coeffA = cfg->isoftmax_coeffA,
                        .isoftmax_coeffB = cfg->isoftmax_coeffB,
                        .isoftmax_coeffC = cfg->isoftmax_coeffC,
                        .isoftmax_log2 = cfg->isoftmax_log2,
                        .isoftmax_n_levels = cfg->isoftmax_n_levels
                    };
                    pi_cl_team_fork(NUM_CORES, mhsa_tile_softmax_worker, &softmax_args);
                    unsigned int inner_softmax_end = pi_perf_read(PI_PERF_CYCLES);
                    cfg->inner_softmax_cycles += (inner_softmax_end - inner_softmax_start);

                    // AV kernel: UINT8xINT8 (integer path) → INT8
                    unsigned int inner_av_start = pi_perf_read(PI_PERF_CYCLES);
                    mhsa_tile_args_t av_args = {
                        .v_l1 = v_l1,
                        .v_t_l1 = v_t_l1,
                        .scores_l1 = scores_l1,
                        .attn_uint8_l1 = attn_uint8_l1,
                        .m_tile_l1 = m_tile_l1,
                        .tile_q = actual_tile_q,
                        .seq_len = cfg->seq_len,
                        .head_dim = cfg->head_dim,
                        .use_integer_softmax = use_int_softmax
                    };
                    pi_cl_team_fork(NUM_CORES, mhsa_tile_av_worker, &av_args);
                    unsigned int inner_av_end = pi_perf_read(PI_PERF_CYCLES);
                    cfg->inner_av_cycles += (inner_av_end - inner_av_start);
#endif
                } else {
                    // Softmax kernel: INT32→FP32 (FP32 path)
                    unsigned int inner_softmax_start = pi_perf_read(PI_PERF_CYCLES);
                    mhsa_tile_args_t softmax_args = {
                        .scores_int32_l1 = scores_int32_l1,
                        .scores_l1 = scores_l1,
                        .scores_int8_l1 = scores_int8_l1,
                        .attn_uint8_l1 = attn_uint8_l1,
                        .tile_q = actual_tile_q,
                        .seq_len = cfg->seq_len,
                        .scale_q = cfg->scale_q,
                        .scale_k = cfg->scale_k,
                        .softmax_scale = softmax_scale,
                        .softmax_lut = cfg->softmax_lut,
                        .use_integer_softmax = use_int_softmax,
                        .requant_mul = cfg->requant_mul,      // For fully-integer softmax
                        .requant_shift = cfg->requant_shift,  // For fully-integer softmax
                        .isoftmax_coeffA = cfg->isoftmax_coeffA,
                        .isoftmax_coeffB = cfg->isoftmax_coeffB,
                        .isoftmax_coeffC = cfg->isoftmax_coeffC,
                        .isoftmax_log2 = cfg->isoftmax_log2,
                        .isoftmax_n_levels = cfg->isoftmax_n_levels
                    };
                    pi_cl_team_fork(NUM_CORES, mhsa_tile_softmax_worker, &softmax_args);
                    unsigned int inner_softmax_end = pi_perf_read(PI_PERF_CYCLES);
                    cfg->inner_softmax_cycles += (inner_softmax_end - inner_softmax_start);

                    // AV kernel: FP32xINT8 (FP32 path) → INT8
                    unsigned int inner_av_start = pi_perf_read(PI_PERF_CYCLES);
                    mhsa_tile_args_t av_args = {
                        .v_l1 = v_l1,
                        .v_t_l1 = v_t_l1,
                        .scores_l1 = scores_l1,
                        .attn_uint8_l1 = attn_uint8_l1,
                        .m_tile_l1 = m_tile_l1,
                        .tile_q = actual_tile_q,
                        .seq_len = cfg->seq_len,
                        .head_dim = cfg->head_dim,
                        .use_integer_softmax = use_int_softmax
                    };
                    pi_cl_team_fork(NUM_CORES, mhsa_tile_av_worker, &av_args);
                    unsigned int inner_av_end = pi_perf_read(PI_PERF_CYCLES);
                    cfg->inner_av_cycles += (inner_av_end - inner_av_start);
                }
            }

            // ---
            // M (context) STORING: Bulk DMA (head-contiguous) vs Strided batch DMA
            // ---
            if (use_head_contiguous && cfg->m_permuted_l2) {
                // HEAD-CONTIGUOUS PATH: Single bulk DMA for M tile
                // Data layout: [num_heads, seq_len, head_dim]
                pi_cl_dma_copy_t m_tile_bulk;
                m_tile_bulk.dir = PI_CL_DMA_DIR_LOC2EXT;
                m_tile_bulk.size = actual_tile_q * cfg->head_dim;
                m_tile_bulk.ext = (uint32_t)(cfg->m_permuted_l2 + head * head_data_size + q_start * cfg->head_dim);
                m_tile_bulk.loc = (uint32_t)m_tile_l1;
                m_tile_bulk.merge = 0;
                pi_cl_dma_memcpy(&m_tile_bulk);
                pi_cl_dma_wait(&m_tile_bulk);
            } else {
                // STRIDED PATH: Batch DMA with 8 transfers at a time (original code)
                for (int batch_start = 0; batch_start < actual_tile_q; batch_start += MHSA_DMA_BATCH_SIZE) {
                    int batch_end = batch_start + MHSA_DMA_BATCH_SIZE;
                    if (batch_end > actual_tile_q) batch_end = actual_tile_q;
                    int batch_count = batch_end - batch_start;

                    for (int i = 0; i < batch_count; i++) {
                        int t = q_start + batch_start + i;
                        dma_batch[i].dir = PI_CL_DMA_DIR_LOC2EXT;
                        dma_batch[i].size = cfg->head_dim;
                        dma_batch[i].loc = (uint32_t)(m_tile_l1 + (batch_start + i) * cfg->head_dim);
                        dma_batch[i].ext = (uint32_t)(cfg->output_buffer_l2 + t * cfg->embed_dim + head_offset);
                        dma_batch[i].merge = 0;
                        pi_cl_dma_memcpy(&dma_batch[i]);
                    }
                    for (int i = 0; i < batch_count; i++) {
                        pi_cl_dma_wait(&dma_batch[i]);
                    }
                }
            }

            current_q_buffer = 1 - current_q_buffer;
            current_m_buffer = 1 - current_m_buffer;
        }
    }
    #undef MHSA_DMA_BATCH_SIZE
}

// ---
// KV Cache Operations for Autoregressive Generation (GQA/Llama-style)
// ---

/**
 * Store current K/V projections into the KV cache at the given position.
 * Used in autoregressive generation mode to accumulate K/V across tokens.
 *
 * @param cfg MHSA pipeline configuration with kv_cache_* fields populated
 * @param k_projected Current K projection [n_kv_heads, 1, head_dim] (for single token)
 * @param v_projected Current V projection [n_kv_heads, 1, head_dim]
 * @param cache_pos Position in the cache to store (0-indexed)
 */
void mhsa_kv_cache_store(
    mhsa_pipeline_config_t *cfg,
    const int8_t *k_projected,
    const int8_t *v_projected,
    int cache_pos
) {
    if (!cfg->kv_cache_enabled || !cfg->kv_cache_k || !cfg->kv_cache_v) return;
    if (cache_pos < 0 || cache_pos >= cfg->kv_cache_max_seq_len) return;

    const int n_kv_heads = (cfg->n_kv_heads > 0) ? cfg->n_kv_heads : cfg->num_heads;
    const int head_dim = cfg->head_dim;
    const int kv_stride = cfg->kv_cache_max_seq_len * head_dim;  // Stride per KV head

    // Store K at cache position: cache[h, pos, :] = k[h, 0, :]
    for (int h = 0; h < n_kv_heads; h++) {
        int8_t *cache_k_dst = cfg->kv_cache_k + h * kv_stride + cache_pos * head_dim;
        const int8_t *k_src = k_projected + h * head_dim;
        for (int d = 0; d < head_dim; d++) {
            cache_k_dst[d] = k_src[d];
        }
    }

    // Store V at cache position
    for (int h = 0; h < n_kv_heads; h++) {
        int8_t *cache_v_dst = cfg->kv_cache_v + h * kv_stride + cache_pos * head_dim;
        const int8_t *v_src = v_projected + h * head_dim;
        for (int d = 0; d < head_dim; d++) {
            cache_v_dst[d] = v_src[d];
        }
    }
}

/**
 * Retrieve K/V from cache for attention computation.
 * Copies cached K/V up to (cache_pos + 1) positions to output buffers.
 *
 * @param cfg MHSA pipeline configuration
 * @param k_out Output K buffer [n_kv_heads, cache_pos+1, head_dim]
 * @param v_out Output V buffer [n_kv_heads, cache_pos+1, head_dim]
 * @param cache_pos Current cache position (will retrieve 0..cache_pos inclusive)
 * @return Number of tokens retrieved (cache_pos + 1)
 */
int mhsa_kv_cache_retrieve(
    mhsa_pipeline_config_t *cfg,
    int8_t *k_out,
    int8_t *v_out,
    int cache_pos
) {
    if (!cfg->kv_cache_enabled || !cfg->kv_cache_k || !cfg->kv_cache_v) return 0;
    if (cache_pos < 0) return 0;

    const int n_kv_heads = (cfg->n_kv_heads > 0) ? cfg->n_kv_heads : cfg->num_heads;
    const int head_dim = cfg->head_dim;
    const int kv_len = cache_pos + 1;  // Include current position
    const int cache_stride = cfg->kv_cache_max_seq_len * head_dim;
    const int out_stride = kv_len * head_dim;

    // Copy K from cache to output
    for (int h = 0; h < n_kv_heads; h++) {
        const int8_t *cache_k_src = cfg->kv_cache_k + h * cache_stride;
        int8_t *k_dst = k_out + h * out_stride;
        for (int t = 0; t < kv_len; t++) {
            for (int d = 0; d < head_dim; d++) {
                k_dst[t * head_dim + d] = cache_k_src[t * head_dim + d];
            }
        }
    }

    // Copy V from cache to output
    for (int h = 0; h < n_kv_heads; h++) {
        const int8_t *cache_v_src = cfg->kv_cache_v + h * cache_stride;
        int8_t *v_dst = v_out + h * out_stride;
        for (int t = 0; t < kv_len; t++) {
            for (int d = 0; d < head_dim; d++) {
                v_dst[t * head_dim + d] = cache_v_src[t * head_dim + d];
            }
        }
    }

    return kv_len;
}

// ---
// MHSA Main Pipeline (Outer Loop: L3 -> L2)
// ---

void mhsa_tiled_l1_pipeline(mhsa_pipeline_config_t *cfg) {
    // INT8 projections only (FP32 path disabled for memory efficiency)
    if (cfg->use_fp32_projections) {
        printf("ERROR: FP32 projections path disabled. Use INT8 projections.\n");
        return;
    }

    // Helper macro: run a square (embed_dim x embed_dim) INT8 projection using
    // the existing 3D linear weight-tiling path (keeps weights in L1).
    // This is used as a fast fallback when full projection weights cannot be
    // cached in L1 alongside the MHSA inner-loop buffers.
#define MHSA_TILED_PROJ(_name, _input, _weights, _bias, _output, _seq_len, _scale_in, _scale_w, _scale_out) \
    do { \
        linear_int8_pipeline_config_t lin_cfg = { \
            .layer_name = (_name), \
            .input_buffer_l2 = (int8_t *)(_input), \
            .output_buffer_l2 = (int8_t *)(_output), \
            .weight_l2 = (int8_t *)(_weights), \
            .bias_l2 = (int32_t *)(_bias), \
            .l1_buffer = cfg->l1_buffer, \
            .l1_buffer_size = cfg->l1_buffer_size, \
            .in_features = cfg->embed_dim, \
            .out_features = cfg->embed_dim, \
            .batch_tokens = (_seq_len), \
            .tile_out_features = 0, \
            .scale_input = (_scale_in), \
            .scale_weight = (_scale_w), \
            .scale_output = (_scale_out), \
            .fusion_relu = 0, \
            .fusion_quant = 0, \
            .ram_dev = cfg->ram_dev, \
        }; \
        linear_int8_tiled_l1_inner_loop(&lin_cfg); \
    } while (0)

#define MHSA_TILED_PROJ_SLICE(_name, _input, _weights, _bias, _output, _seq_len, _out_features, _scale_in, _scale_w, _scale_out) \
    do { \
        linear_int8_pipeline_config_t lin_cfg = { \
            .layer_name = (_name), \
            .input_buffer_l2 = (int8_t *)(_input), \
            .output_buffer_l2 = (int8_t *)(_output), \
            .weight_l2 = (int8_t *)(_weights), \
            .bias_l2 = (int32_t *)(_bias), \
            .l1_buffer = cfg->l1_buffer, \
            .l1_buffer_size = cfg->l1_buffer_size, \
            .in_features = cfg->embed_dim, \
            .out_features = (_out_features), \
            .batch_tokens = (_seq_len), \
            .tile_out_features = 0, \
            .scale_input = (_scale_in), \
            .scale_weight = (_scale_w), \
            .scale_output = (_scale_out), \
            .fusion_relu = 0, \
            .fusion_quant = 0, \
            .ram_dev = cfg->ram_dev, \
        }; \
        linear_int8_tiled_l1_inner_loop(&lin_cfg); \
    } while (0)

    // ---
    // L1 WEIGHT CACHING: Load projection weights to L1 for faster access
    // ---
    // L1 layout for weight caching (if enabled):
    //   [Q_weight_l1][K_weight_l1][V_weight_l1][Out_weight_l1]
    // Each is embed_dim * embed_dim bytes
    //
    // The inner loop uses separate L1 space for K, V, Q tiles, scores, M tiles.
    // Weight L1 buffers are allocated from the END of L1 buffer (after inner loop space).

    int8_t *q_weight_l1 = NULL;
    int8_t *k_weight_l1 = NULL;
    int8_t *v_weight_l1 = NULL;
    int8_t *out_weight_l1 = NULL;

    if (cfg->l1_weight_caching_enabled && cfg->l1_proj_weight_size > 0) {
        // Calculate inner loop L1 usage to determine weight buffer start
        size_t k_size = cfg->seq_len * cfg->head_dim;
        size_t v_size = cfg->seq_len * cfg->head_dim;
        size_t q_tile_size = cfg->tile_q * cfg->head_dim;
        size_t m_tile_size = cfg->tile_q * cfg->head_dim;

        // Scores buffer size depends on softmax path:
        // - FP32 path: INT32/FP32 shared buffer (tile_q * seq_len * 4)
        // - Integer path: INT32 + (optional INT8) + UINT8 buffers
        size_t scores_size;
        const int use_per_core_fused =
            cfg->use_integer_softmax && MHSA_FUSE_QK_SOFTMAX_AV && MHSA_FUSE_QK_SOFTMAX_AV_PER_CORE;
        if (cfg->use_integer_softmax) {
            if (use_per_core_fused) {
                // Per-core scratch: INT32 scores + UINT8 attention per core
                scores_size = NUM_CORES * cfg->seq_len * (sizeof(int32_t) + 1);
            } else {
                // Integer path: INT32 scores + (optional INT8 intermediate) + UINT8 attention weights
                const size_t scores_int8_size = MHSA_TILED_QK_STORE_INT8_SCORES ? (cfg->tile_q * cfg->seq_len) : 0;
                scores_size = cfg->tile_q * cfg->seq_len * sizeof(int32_t)  // INT32 scores
                            + scores_int8_size                                  // Optional INT8 intermediate
                            + cfg->tile_q * cfg->seq_len;                     // UINT8 attention
            }
        } else {
            // FP32 path: shared INT32/FP32 buffer
            scores_size = cfg->tile_q * cfg->seq_len * sizeof(float);
        }
        size_t inner_loop_l1 = k_size + v_size + 2*q_tile_size + scores_size + 2*m_tile_size;

        // Weight buffers start after inner loop space
        size_t weight_start_offset = inner_loop_l1;
        size_t total_weight_space = 4 * cfg->l1_proj_weight_size;

        // Check if we have space for all 4 projection weights
        if (cfg->l1_buffer && (weight_start_offset + total_weight_space <= cfg->l1_buffer_size)) {
            q_weight_l1 = cfg->l1_buffer + weight_start_offset;
            k_weight_l1 = q_weight_l1 + cfg->l1_proj_weight_size;
            v_weight_l1 = k_weight_l1 + cfg->l1_proj_weight_size;
            out_weight_l1 = v_weight_l1 + cfg->l1_proj_weight_size;

            // DMA: Load all 4 projection weights from L2 to L1
            pi_cl_dma_copy_t q_w_copy, k_w_copy, v_w_copy, o_w_copy;

            q_w_copy.dir = PI_CL_DMA_DIR_EXT2LOC;
            q_w_copy.size = cfg->l1_proj_weight_size;
            q_w_copy.ext = (uint32_t)cfg->q_weight_l2;
            q_w_copy.loc = (uint32_t)q_weight_l1;
            q_w_copy.merge = 0;

            k_w_copy.dir = PI_CL_DMA_DIR_EXT2LOC;
            k_w_copy.size = cfg->l1_proj_weight_size;
            k_w_copy.ext = (uint32_t)cfg->k_weight_l2;
            k_w_copy.loc = (uint32_t)k_weight_l1;
            k_w_copy.merge = 0;

            v_w_copy.dir = PI_CL_DMA_DIR_EXT2LOC;
            v_w_copy.size = cfg->l1_proj_weight_size;
            v_w_copy.ext = (uint32_t)cfg->v_weight_l2;
            v_w_copy.loc = (uint32_t)v_weight_l1;
            v_w_copy.merge = 0;

            o_w_copy.dir = PI_CL_DMA_DIR_EXT2LOC;
            o_w_copy.size = cfg->l1_proj_weight_size;
            o_w_copy.ext = (uint32_t)cfg->out_weight_l2;
            o_w_copy.loc = (uint32_t)out_weight_l1;
            o_w_copy.merge = 0;

            // Issue all 4 DMAs
            pi_cl_dma_memcpy(&q_w_copy);
            pi_cl_dma_memcpy(&k_w_copy);
            pi_cl_dma_memcpy(&v_w_copy);
            pi_cl_dma_memcpy(&o_w_copy);

            // Wait for all to complete
            pi_cl_dma_wait(&q_w_copy);
            pi_cl_dma_wait(&k_w_copy);
            pi_cl_dma_wait(&v_w_copy);
            pi_cl_dma_wait(&o_w_copy);

#ifndef MINIMAL_OUTPUT
            printf("CL:   [L1_CACHE] Loaded MHSA projection weights to L1 (4 x %u bytes)\n",
                   (unsigned)cfg->l1_proj_weight_size);
#endif
        } else {
            // Not enough L1 space - fall back to L2 weights
#ifndef MINIMAL_OUTPUT
            size_t available = (cfg->l1_buffer_size > weight_start_offset)
                             ? (cfg->l1_buffer_size - weight_start_offset) : 0;
            printf("CL:   [L1_CACHE] Not enough L1 for MHSA weights (need %u, have %u after inner loop)\n",
                   (unsigned)total_weight_space, (unsigned)available);
#endif
        }
    }

    // Use L1 weights if loaded, otherwise fall back to L2
    const int8_t *q_weights_ptr = q_weight_l1 ? q_weight_l1 : cfg->q_weight_l2;
    const int8_t *k_weights_ptr = k_weight_l1 ? k_weight_l1 : cfg->k_weight_l2;
    const int8_t *v_weights_ptr = v_weight_l1 ? v_weight_l1 : cfg->v_weight_l2;
    const int8_t *out_weights_ptr = out_weight_l1 ? out_weight_l1 : cfg->out_weight_l2;

    const int enable_head_contig_proj =
        cfg->use_head_contiguous &&
        cfg->q_permuted_l2 && cfg->k_permuted_l2 && cfg->v_permuted_l2 &&
        (cfg->head_dim > 0) && ((cfg->embed_dim % cfg->head_dim) == 0);

    // ALWAYS use parallel projections - the fused kernel is much faster than per-token sequential
    // L1 weight caching provides additional speedup but isn't required for parallelism
    int use_parallel_projections = 1;  // Always enable parallel projections

    if (cfg->l3_tiling_enabled) {
        for (int slab_idx = 0; slab_idx < cfg->num_l3_tiles; slab_idx++) {
            int seq_start = slab_idx * cfg->l3_seq_len;
            int seq_end   = seq_start + cfg->l3_seq_len;
            if (seq_end > cfg->seq_len) seq_end = cfg->seq_len;
            int actual_slab_seq = seq_end - seq_start;
            size_t slab_bytes = actual_slab_seq * cfg->embed_dim;

            // Load Input Slab (L3 -> L2)
            if (cfg->l3_input_addr) {
                dma_layout_t l3_in = { .base_addr = (int8_t*)cfg->l3_input_addr + (seq_start * cfg->embed_dim), .width = slab_bytes, .height = 1, .channels = 1, .loc = MEM_LOC_L3, .ram_dev = cfg->ram_dev };
                dma_layout_t l2_in = { .base_addr = cfg->input_buffer_l2, .width = slab_bytes, .height = 1, .channels = 1, .loc = MEM_LOC_L2, .ram_dev = cfg->ram_dev };
                execute_dma_transfer(&l3_in, &l2_in);
            }

	            int qkv_head_contiguous = 0;
	            // Perform Projections
	            if (use_parallel_projections) {
	                if (enable_head_contig_proj && (q_weight_l1 && k_weight_l1 && v_weight_l1)) {
	                    // Project directly into head-contiguous layout to avoid the explicit permutation.
	                    for (int head = 0; head < cfg->num_heads; head++) {
	                        const int out_off = head * cfg->head_dim;
	                        const int out_bytes = actual_slab_seq * cfg->head_dim;

                        int8_t *q_dst = cfg->q_permuted_l2 + head * out_bytes;
                        int8_t *k_dst = cfg->k_permuted_l2 + head * out_bytes;
                        int8_t *v_dst = cfg->v_permuted_l2 + head * out_bytes;

	                        const int32_t *q_bias = cfg->q_bias_l2 ? (cfg->q_bias_l2 + out_off) : NULL;
	                        const int32_t *k_bias = cfg->k_bias_l2 ? (cfg->k_bias_l2 + out_off) : NULL;
	                        const int32_t *v_bias = cfg->v_bias_l2 ? (cfg->v_bias_l2 + out_off) : NULL;

	                        network_linear_int8_fused_qkv_parallel_tokens_rect(
	                            cfg->input_buffer_l2,
	                            q_weight_l1 + out_off * cfg->embed_dim,
	                            k_weight_l1 + out_off * cfg->embed_dim,
	                            v_weight_l1 + out_off * cfg->embed_dim,
	                            q_bias, k_bias, v_bias,
	                            q_dst, k_dst, v_dst,
	                            actual_slab_seq, cfg->embed_dim, cfg->head_dim,
	                            cfg->scale_input,
	                            cfg->scale_q_weight, cfg->scale_k_weight, cfg->scale_v_weight,
	                            cfg->scale_q, cfg->scale_k, cfg->scale_v
	                        );
	                    }
	                    qkv_head_contiguous = 1;
	                }
                if (!qkv_head_contiguous) {
                    // Prefer L1-cached fused QKV if all three weights are available in L1.
                    // Otherwise fall back to the faster 3D weight-tiling path (3 launches).
	                    if (q_weight_l1 && k_weight_l1 && v_weight_l1) {
	                        network_linear_int8_fused_qkv(
	                            cfg->input_buffer_l2,
	                            q_weights_ptr, k_weights_ptr, v_weights_ptr,
	                            cfg->q_bias_l2, cfg->k_bias_l2, cfg->v_bias_l2,
	                            cfg->q_buffer_l2, cfg->k_buffer_l2, cfg->v_buffer_l2,
	                            actual_slab_seq, cfg->embed_dim,
	                            cfg->scale_input,
	                            cfg->scale_q_weight, cfg->scale_k_weight, cfg->scale_v_weight,
	                            cfg->scale_q, cfg->scale_k, cfg->scale_v
	                        );
	                    } else {
	                        const size_t proj_w_bytes = (size_t)cfg->embed_dim * (size_t)cfg->embed_dim;
	                        const size_t scratch_bytes = 3u * proj_w_bytes;
	                        if (cfg->l1_buffer && (scratch_bytes <= cfg->l1_buffer_size)) {
	                            int8_t *q_w_l1 = cfg->l1_buffer;
	                            int8_t *k_w_l1 = q_w_l1 + proj_w_bytes;
	                            int8_t *v_w_l1 = k_w_l1 + proj_w_bytes;

	                            pi_cl_dma_copy_t q_w_copy;
	                            pi_cl_dma_copy_t k_w_copy;
	                            pi_cl_dma_copy_t v_w_copy;

	                            q_w_copy.dir = PI_CL_DMA_DIR_EXT2LOC;
	                            q_w_copy.size = proj_w_bytes;
	                            q_w_copy.ext = (uint32_t)cfg->q_weight_l2;
	                            q_w_copy.loc = (uint32_t)q_w_l1;
	                            q_w_copy.merge = 0;

	                            k_w_copy.dir = PI_CL_DMA_DIR_EXT2LOC;
	                            k_w_copy.size = proj_w_bytes;
	                            k_w_copy.ext = (uint32_t)cfg->k_weight_l2;
	                            k_w_copy.loc = (uint32_t)k_w_l1;
	                            k_w_copy.merge = 0;

	                            v_w_copy.dir = PI_CL_DMA_DIR_EXT2LOC;
	                            v_w_copy.size = proj_w_bytes;
	                            v_w_copy.ext = (uint32_t)cfg->v_weight_l2;
	                            v_w_copy.loc = (uint32_t)v_w_l1;
	                            v_w_copy.merge = 0;

	                            pi_cl_dma_memcpy(&q_w_copy);
	                            pi_cl_dma_memcpy(&k_w_copy);
	                            pi_cl_dma_memcpy(&v_w_copy);
	                            pi_cl_dma_wait(&q_w_copy);
	                            pi_cl_dma_wait(&k_w_copy);
	                            pi_cl_dma_wait(&v_w_copy);

	                            network_linear_int8_fused_qkv(
	                                cfg->input_buffer_l2,
	                                q_w_l1, k_w_l1, v_w_l1,
	                                cfg->q_bias_l2, cfg->k_bias_l2, cfg->v_bias_l2,
	                                cfg->q_buffer_l2, cfg->k_buffer_l2, cfg->v_buffer_l2,
	                                actual_slab_seq, cfg->embed_dim,
	                                cfg->scale_input,
	                                cfg->scale_q_weight, cfg->scale_k_weight, cfg->scale_v_weight,
	                                cfg->scale_q, cfg->scale_k, cfg->scale_v
	                            );
	                        } else {
	                            MHSA_TILED_PROJ("mhsa.q_proj", cfg->input_buffer_l2, cfg->q_weight_l2, cfg->q_bias_l2, cfg->q_buffer_l2,
	                                            actual_slab_seq, cfg->scale_input, cfg->scale_q_weight, cfg->scale_q);
	                            MHSA_TILED_PROJ("mhsa.k_proj", cfg->input_buffer_l2, cfg->k_weight_l2, cfg->k_bias_l2, cfg->k_buffer_l2,
	                                            actual_slab_seq, cfg->scale_input, cfg->scale_k_weight, cfg->scale_k);
	                            MHSA_TILED_PROJ("mhsa.v_proj", cfg->input_buffer_l2, cfg->v_weight_l2, cfg->v_bias_l2, cfg->v_buffer_l2,
	                                            actual_slab_seq, cfg->scale_input, cfg->scale_v_weight, cfg->scale_v);
	                        }
	                    }
	                }
            } else {
                // Sequential fallback: loop over all tokens in this slab
                for (int t = 0; t < actual_slab_seq; t++) {
                    network_linear_int8_sequential(cfg->input_buffer_l2 + t * cfg->embed_dim, cfg->q_weight_l2, cfg->q_bias_l2, cfg->q_buffer_l2 + t * cfg->embed_dim, cfg->embed_dim, cfg->embed_dim, cfg->scale_input, cfg->scale_q_weight, cfg->scale_q);
                    network_linear_int8_sequential(cfg->input_buffer_l2 + t * cfg->embed_dim, cfg->k_weight_l2, cfg->k_bias_l2, cfg->k_buffer_l2 + t * cfg->embed_dim, cfg->embed_dim, cfg->embed_dim, cfg->scale_input, cfg->scale_k_weight, cfg->scale_k);
                    network_linear_int8_sequential(cfg->input_buffer_l2 + t * cfg->embed_dim, cfg->v_weight_l2, cfg->v_bias_l2, cfg->v_buffer_l2 + t * cfg->embed_dim, cfg->embed_dim, cfg->embed_dim, cfg->scale_input, cfg->scale_v_weight, cfg->scale_v);
                }
            }

            // Run Inner Loop
            // ---
            // HEAD-CONTIGUOUS PERMUTATION (L3 SLAB): [seq_len, embed_dim] -> [num_heads, seq_len, head_dim]
            // ---
	        if (!qkv_head_contiguous && cfg->use_head_contiguous && cfg->q_permuted_l2 && cfg->k_permuted_l2 && cfg->v_permuted_l2) {
                mhsa_permute_to_heads(cfg->q_buffer_l2, cfg->q_permuted_l2, actual_slab_seq, cfg->embed_dim, cfg->num_heads);
                mhsa_permute_to_heads(cfg->k_buffer_l2, cfg->k_permuted_l2, actual_slab_seq, cfg->embed_dim, cfg->num_heads);
                mhsa_permute_to_heads(cfg->v_buffer_l2, cfg->v_permuted_l2, actual_slab_seq, cfg->embed_dim, cfg->num_heads);
            }

            // Optional RoPE rotation on Q/K (head-contiguous layout) before attention.
            if (cfg->use_rope && cfg->rope_cos_q15 && cfg->rope_sin_q15 &&
                cfg->use_head_contiguous && cfg->q_permuted_l2 && cfg->k_permuted_l2) {
                network_rope_int8_inplace_parallel(
                    cfg->q_permuted_l2,
                    cfg->rope_cos_q15,
                    cfg->rope_sin_q15,
                    (uint32_t)cfg->num_heads,
                    (uint32_t)actual_slab_seq,
                    (uint32_t)cfg->head_dim,
                    (uint32_t)seq_start
                );
                network_rope_int8_inplace_parallel(
                    cfg->k_permuted_l2,
                    cfg->rope_cos_q15,
                    cfg->rope_sin_q15,
                    (uint32_t)cfg->num_heads,
                    (uint32_t)actual_slab_seq,
                    (uint32_t)cfg->head_dim,
                    (uint32_t)seq_start
                );
            }

            //  Reuse Q buffer for Attention Context (L2) to save memory
            // The inner loop writes Context to 'output_buffer_l2', so we point it to Q buffer temporarily
            int8_t *original_output_buffer = cfg->output_buffer_l2;
            if (cfg->use_head_contiguous && cfg->m_permuted_l2) {
                cfg->output_buffer_l2 = cfg->m_permuted_l2;
            } else {
                cfg->output_buffer_l2 = cfg->q_buffer_l2;
            }

            int original_seq = cfg->seq_len;
            cfg->seq_len = actual_slab_seq;
            mhsa_tiled_l1_inner_loop(cfg);
            cfg->seq_len = original_seq;

            // Restore output buffer pointer
            cfg->output_buffer_l2 = original_output_buffer;

            // ---
            // INVERSE PERMUTATION (L3 SLAB): [num_heads, seq_len, head_dim] -> [seq_len, embed_dim]
            // ---
            if (cfg->use_head_contiguous && cfg->m_permuted_l2) {
                mhsa_permute_from_heads(cfg->m_permuted_l2, cfg->q_buffer_l2, actual_slab_seq, cfg->embed_dim, cfg->num_heads);
            }

            // Output Projection: Input is now in Q buffer (Context), Output goes to real Output Buffer
            float out_proj_scale = cfg->scale_output;
            if (use_parallel_projections) {
                if (out_weight_l1) {
                    network_linear_int8_parallel_tokens(cfg->q_buffer_l2, out_weights_ptr, cfg->out_bias_l2, cfg->output_buffer_l2,
                                                        actual_slab_seq, cfg->embed_dim, cfg->embed_dim,
                                                        cfg->scale_v, cfg->scale_out_weight, out_proj_scale);
                } else {
                    MHSA_TILED_PROJ("mhsa.out_proj", cfg->q_buffer_l2, cfg->out_weight_l2, cfg->out_bias_l2, cfg->output_buffer_l2,
                                    actual_slab_seq, cfg->scale_v, cfg->scale_out_weight, out_proj_scale);
                }
            } else {
                for (int t = 0; t < actual_slab_seq; t++) {
                    network_linear_int8_sequential(cfg->q_buffer_l2 + t * cfg->embed_dim,
                                                   cfg->out_weight_l2, cfg->out_bias_l2,
                                                   cfg->output_buffer_l2 + t * cfg->embed_dim,
                                                   cfg->embed_dim, cfg->embed_dim,
                                                   cfg->scale_v, cfg->scale_out_weight, out_proj_scale);
                }
            }

            // Store L2 -> L3
            dma_layout_t l2_o = { .base_addr = cfg->output_buffer_l2, .width = slab_bytes, .height = 1, .channels = 1, .loc = MEM_LOC_L2, .ram_dev = cfg->ram_dev };
            dma_layout_t l3_o = { .base_addr = (int8_t*)cfg->l3_output_addr + (seq_start * cfg->embed_dim), .width = slab_bytes, .height = 1, .channels = 1, .loc = MEM_LOC_L3, .ram_dev = cfg->ram_dev };
            execute_dma_transfer(&l2_o, &l3_o);
        }
    } else {
        // Standard L2: Project and run
        // STAGE PROFILING
        unsigned int phase_start, phase_end;

	        // STAGE 1: Q/K/V projections
	        phase_start = pi_perf_read(PI_PERF_CYCLES);
	        int qkv_head_contiguous = 0;
	        if (use_parallel_projections) {
	            if (enable_head_contig_proj && (q_weight_l1 && k_weight_l1 && v_weight_l1)) {
	                for (int head = 0; head < cfg->num_heads; head++) {
	                    const int out_off = head * cfg->head_dim;
	                    const int out_bytes = cfg->seq_len * cfg->head_dim;

                    int8_t *q_dst = cfg->q_permuted_l2 + head * out_bytes;
                    int8_t *k_dst = cfg->k_permuted_l2 + head * out_bytes;
                    int8_t *v_dst = cfg->v_permuted_l2 + head * out_bytes;

	                    const int32_t *q_bias = cfg->q_bias_l2 ? (cfg->q_bias_l2 + out_off) : NULL;
	                    const int32_t *k_bias = cfg->k_bias_l2 ? (cfg->k_bias_l2 + out_off) : NULL;
	                    const int32_t *v_bias = cfg->v_bias_l2 ? (cfg->v_bias_l2 + out_off) : NULL;

	                    network_linear_int8_fused_qkv_parallel_tokens_rect(
	                        cfg->input_buffer_l2,
	                        q_weight_l1 + out_off * cfg->embed_dim,
	                        k_weight_l1 + out_off * cfg->embed_dim,
	                        v_weight_l1 + out_off * cfg->embed_dim,
	                        q_bias, k_bias, v_bias,
	                        q_dst, k_dst, v_dst,
	                        cfg->seq_len, cfg->embed_dim, cfg->head_dim,
	                        cfg->scale_input,
	                        cfg->scale_q_weight, cfg->scale_k_weight, cfg->scale_v_weight,
	                        cfg->scale_q, cfg->scale_k, cfg->scale_v
	                    );
	                }
	                qkv_head_contiguous = 1;
	            }
            if (!qkv_head_contiguous) {
                // Prefer L1-cached fused QKV if all three weights are available in L1.
                // Otherwise use the 3D weight-tiling path (3 launches, but much higher throughput).
	                if (q_weight_l1 && k_weight_l1 && v_weight_l1) {
	                    network_linear_int8_fused_qkv(
	                        cfg->input_buffer_l2,
	                        q_weights_ptr, k_weights_ptr, v_weights_ptr,
	                        cfg->q_bias_l2, cfg->k_bias_l2, cfg->v_bias_l2,
	                        cfg->q_buffer_l2, cfg->k_buffer_l2, cfg->v_buffer_l2,
	                        cfg->seq_len, cfg->embed_dim,
	                        cfg->scale_input,
	                        cfg->scale_q_weight, cfg->scale_k_weight, cfg->scale_v_weight,
	                        cfg->scale_q, cfg->scale_k, cfg->scale_v
	                    );
	                } else {
	                    const size_t proj_w_bytes = (size_t)cfg->embed_dim * (size_t)cfg->embed_dim;
	                    const size_t scratch_bytes = 3u * proj_w_bytes;
	                    if (cfg->l1_buffer && (scratch_bytes <= cfg->l1_buffer_size)) {
	                        int8_t *q_w_l1 = cfg->l1_buffer;
	                        int8_t *k_w_l1 = q_w_l1 + proj_w_bytes;
	                        int8_t *v_w_l1 = k_w_l1 + proj_w_bytes;

	                        pi_cl_dma_copy_t q_w_copy;
	                        pi_cl_dma_copy_t k_w_copy;
	                        pi_cl_dma_copy_t v_w_copy;

	                        q_w_copy.dir = PI_CL_DMA_DIR_EXT2LOC;
	                        q_w_copy.size = proj_w_bytes;
	                        q_w_copy.ext = (uint32_t)cfg->q_weight_l2;
	                        q_w_copy.loc = (uint32_t)q_w_l1;
	                        q_w_copy.merge = 0;

	                        k_w_copy.dir = PI_CL_DMA_DIR_EXT2LOC;
	                        k_w_copy.size = proj_w_bytes;
	                        k_w_copy.ext = (uint32_t)cfg->k_weight_l2;
	                        k_w_copy.loc = (uint32_t)k_w_l1;
	                        k_w_copy.merge = 0;

	                        v_w_copy.dir = PI_CL_DMA_DIR_EXT2LOC;
	                        v_w_copy.size = proj_w_bytes;
	                        v_w_copy.ext = (uint32_t)cfg->v_weight_l2;
	                        v_w_copy.loc = (uint32_t)v_w_l1;
	                        v_w_copy.merge = 0;

	                        pi_cl_dma_memcpy(&q_w_copy);
	                        pi_cl_dma_memcpy(&k_w_copy);
	                        pi_cl_dma_memcpy(&v_w_copy);
	                        pi_cl_dma_wait(&q_w_copy);
	                        pi_cl_dma_wait(&k_w_copy);
	                        pi_cl_dma_wait(&v_w_copy);

	                        network_linear_int8_fused_qkv(
	                            cfg->input_buffer_l2,
	                            q_w_l1, k_w_l1, v_w_l1,
	                            cfg->q_bias_l2, cfg->k_bias_l2, cfg->v_bias_l2,
	                            cfg->q_buffer_l2, cfg->k_buffer_l2, cfg->v_buffer_l2,
	                            cfg->seq_len, cfg->embed_dim,
	                            cfg->scale_input,
	                            cfg->scale_q_weight, cfg->scale_k_weight, cfg->scale_v_weight,
	                            cfg->scale_q, cfg->scale_k, cfg->scale_v
	                        );
	                    } else {
	                        MHSA_TILED_PROJ("mhsa.q_proj", cfg->input_buffer_l2, cfg->q_weight_l2, cfg->q_bias_l2, cfg->q_buffer_l2,
	                                        cfg->seq_len, cfg->scale_input, cfg->scale_q_weight, cfg->scale_q);
	                        MHSA_TILED_PROJ("mhsa.k_proj", cfg->input_buffer_l2, cfg->k_weight_l2, cfg->k_bias_l2, cfg->k_buffer_l2,
	                                        cfg->seq_len, cfg->scale_input, cfg->scale_k_weight, cfg->scale_k);
	                        MHSA_TILED_PROJ("mhsa.v_proj", cfg->input_buffer_l2, cfg->v_weight_l2, cfg->v_bias_l2, cfg->v_buffer_l2,
	                                        cfg->seq_len, cfg->scale_input, cfg->scale_v_weight, cfg->scale_v);
	                    }
	                }
	            }
        } else {
            // Sequential fallback: loop over all tokens
            for (int t = 0; t < cfg->seq_len; t++) {
                network_linear_int8_sequential(cfg->input_buffer_l2 + t * cfg->embed_dim, cfg->q_weight_l2, cfg->q_bias_l2, cfg->q_buffer_l2 + t * cfg->embed_dim, cfg->embed_dim, cfg->embed_dim, cfg->scale_input, cfg->scale_q_weight, cfg->scale_q);
                network_linear_int8_sequential(cfg->input_buffer_l2 + t * cfg->embed_dim, cfg->k_weight_l2, cfg->k_bias_l2, cfg->k_buffer_l2 + t * cfg->embed_dim, cfg->embed_dim, cfg->embed_dim, cfg->scale_input, cfg->scale_k_weight, cfg->scale_k);
                network_linear_int8_sequential(cfg->input_buffer_l2 + t * cfg->embed_dim, cfg->v_weight_l2, cfg->v_bias_l2, cfg->v_buffer_l2 + t * cfg->embed_dim, cfg->embed_dim, cfg->embed_dim, cfg->scale_input, cfg->scale_v_weight, cfg->scale_v);
            }
        }
        phase_end = pi_perf_read(PI_PERF_CYCLES);
        unsigned int qkv_proj_cycles = phase_end - phase_start;

        // STAGE 2: Permutation to head-contiguous
        phase_start = pi_perf_read(PI_PERF_CYCLES);
        // ---
        // HEAD-CONTIGUOUS PERMUTATION: [seq_len, embed_dim] -> [num_heads, seq_len, head_dim]
        // ---
        // This transforms Q/K/V data so each head's data is contiguous in memory,
        // enabling bulk DMA in the inner loop instead of strided transfers.
        if (!enable_head_contig_proj && cfg->use_head_contiguous && cfg->q_permuted_l2 && cfg->k_permuted_l2 && cfg->v_permuted_l2) {
            if (cfg->use_inplace_permute && cfg->permute_scratch_l2) {
                // In-place permutation: use output buffer as scratch
                // Algorithm: permute src→scratch, memcpy scratch→src (now src is permuted)
                // This saves 3 * seq_len * embed_dim bytes (450KB for test_20)!
                const int buffer_size = cfg->seq_len * cfg->embed_dim;

                // Permute Q in-place
                mhsa_permute_to_heads(cfg->q_buffer_l2, cfg->permute_scratch_l2, cfg->seq_len, cfg->embed_dim, cfg->num_heads);
                memcpy(cfg->q_buffer_l2, cfg->permute_scratch_l2, buffer_size);

                // Permute K in-place
                mhsa_permute_to_heads(cfg->k_buffer_l2, cfg->permute_scratch_l2, cfg->seq_len, cfg->embed_dim, cfg->num_heads);
                memcpy(cfg->k_buffer_l2, cfg->permute_scratch_l2, buffer_size);

                // Permute V in-place
                mhsa_permute_to_heads(cfg->v_buffer_l2, cfg->permute_scratch_l2, cfg->seq_len, cfg->embed_dim, cfg->num_heads);
                memcpy(cfg->v_buffer_l2, cfg->permute_scratch_l2, buffer_size);

                // q_permuted_l2, k_permuted_l2, v_permuted_l2 already point to q/k/v_buffer_l2
                // which are now in permuted layout
            } else {
                // Original approach: separate destination buffers
                mhsa_permute_to_heads(cfg->q_buffer_l2, cfg->q_permuted_l2, cfg->seq_len, cfg->embed_dim, cfg->num_heads);
                mhsa_permute_to_heads(cfg->k_buffer_l2, cfg->k_permuted_l2, cfg->seq_len, cfg->embed_dim, cfg->num_heads);
                mhsa_permute_to_heads(cfg->v_buffer_l2, cfg->v_permuted_l2, cfg->seq_len, cfg->embed_dim, cfg->num_heads);
            }
        }
        phase_end = pi_perf_read(PI_PERF_CYCLES);
        unsigned int permute_cycles = phase_end - phase_start;

        // Optional RoPE rotation on Q/K (head-contiguous layout) before attention.
        if (cfg->use_rope && cfg->rope_cos_q15 && cfg->rope_sin_q15 &&
            cfg->use_head_contiguous && cfg->q_permuted_l2 && cfg->k_permuted_l2) {
            network_rope_int8_inplace_parallel(
                cfg->q_permuted_l2,
                cfg->rope_cos_q15,
                cfg->rope_sin_q15,
                (uint32_t)cfg->num_heads,
                (uint32_t)cfg->seq_len,
                (uint32_t)cfg->head_dim,
                0
            );
            network_rope_int8_inplace_parallel(
                cfg->k_permuted_l2,
                cfg->rope_cos_q15,
                cfg->rope_sin_q15,
                (uint32_t)cfg->num_heads,
                (uint32_t)cfg->seq_len,
                (uint32_t)cfg->head_dim,
                0
            );
        }

        // Reuse Q buffer for Attention Context (or M permuted buffer if head-contiguous)
        int8_t *original_output_buffer = cfg->output_buffer_l2;
        if (cfg->use_head_contiguous && cfg->m_permuted_l2) {
            cfg->output_buffer_l2 = cfg->m_permuted_l2;  // Inner loop writes to permuted M buffer
        } else {
            cfg->output_buffer_l2 = cfg->q_buffer_l2;
        }

        // STAGE 3: Inner loop (QK^T + softmax + AV)
        // Reset inner loop cycle counters before execution
        cfg->inner_qk_cycles = 0;
        cfg->inner_softmax_cycles = 0;
        cfg->inner_av_cycles = 0;
        phase_start = pi_perf_read(PI_PERF_CYCLES);
        mhsa_tiled_l1_inner_loop(cfg);
        phase_end = pi_perf_read(PI_PERF_CYCLES);
        unsigned int inner_loop_cycles = phase_end - phase_start;

        // Restore output buffer pointer
        cfg->output_buffer_l2 = original_output_buffer;

        // STAGE 4: Inverse permutation
        phase_start = pi_perf_read(PI_PERF_CYCLES);
        // ---
        // INVERSE PERMUTATION: [num_heads, seq_len, head_dim] -> [seq_len, embed_dim]
        // ---
        // Transform M back to standard layout for output projection
        if (cfg->use_head_contiguous && cfg->m_permuted_l2) {
            mhsa_permute_from_heads(cfg->m_permuted_l2, cfg->q_buffer_l2, cfg->seq_len, cfg->embed_dim, cfg->num_heads);
        }
        phase_end = pi_perf_read(PI_PERF_CYCLES);
        unsigned int inv_permute_cycles = phase_end - phase_start;

        // STAGE 5: Output projection
        phase_start = pi_perf_read(PI_PERF_CYCLES);
        // Output Projection: Input is now in Q buffer (Context)
        // Reuse K buffer as scratch storage for the unpooled output (seq_len * embed_dim).
        if (use_parallel_projections) {
            if (out_weight_l1) {
                network_linear_int8_parallel_tokens(cfg->q_buffer_l2, out_weights_ptr, cfg->out_bias_l2, cfg->k_buffer_l2,
                                                    cfg->seq_len, cfg->embed_dim, cfg->embed_dim,
                                                    cfg->scale_v, cfg->scale_out_weight, cfg->scale_output);
            } else {
                MHSA_TILED_PROJ("mhsa.out_proj", cfg->q_buffer_l2, cfg->out_weight_l2, cfg->out_bias_l2, cfg->k_buffer_l2,
                                cfg->seq_len, cfg->scale_v, cfg->scale_out_weight, cfg->scale_output);
            }
        } else {
            for (int t = 0; t < cfg->seq_len; t++) {
                network_linear_int8_sequential(cfg->q_buffer_l2 + t * cfg->embed_dim, cfg->out_weight_l2, cfg->out_bias_l2, cfg->k_buffer_l2 + t * cfg->embed_dim, cfg->embed_dim, cfg->embed_dim, cfg->scale_v, cfg->scale_out_weight, cfg->scale_output);
            }
        }
        phase_end = pi_perf_read(PI_PERF_CYCLES);
        unsigned int out_proj_cycles = phase_end - phase_start;

#ifdef ENABLE_PERF_COUNTERS
        // Populate perf counter so network.c can report MHSA MACs/cycle and the
        // global perf summary includes MHSA as a compute-heavy layer.
        if (cfg->perf_counter) {
            cfg->perf_counter->compute_cycles +=
                qkv_proj_cycles +
                permute_cycles +
                inner_loop_cycles +
                inv_permute_cycles +
                out_proj_cycles;
        }
#endif

#if defined(ENABLE_PERF_COUNTERS) && !defined(MINIMAL_OUTPUT)
        printf("CL: [MHSA STAGE] QKV_proj=%u, Permute=%u, InnerLoop=%u, InvPerm=%u, OutProj=%u\n",
               qkv_proj_cycles, permute_cycles, inner_loop_cycles, inv_permute_cycles, out_proj_cycles);
        printf("CL: [MHSA INNER] QK=%u, Softmax=%u, AV=%u, DMA=%u\n",
               cfg->inner_qk_cycles, cfg->inner_softmax_cycles, cfg->inner_av_cycles,
               inner_loop_cycles - cfg->inner_qk_cycles - cfg->inner_softmax_cycles - cfg->inner_av_cycles);
#endif

        // Conditional Pooling based on pool_mode
        const int pool_mode = cfg->pool_mode;
        if (pool_mode == MHSA_POOL_MEAN) {
            // Mean Pooling: K buffer (SeqLen * EmbedDim) -> Output Buffer (1 * EmbedDim)
            // Pool over sequence dimension (dim 0 of the buffer)
            // Matches Python reference: dequantize, mean in FP32, requantize
            for (int d = 0; d < cfg->embed_dim; d++) {
                float sum_fp32 = 0.0f;
                for (int s = 0; s < cfg->seq_len; s++) {
                    // Dequantize each INT8 value to FP32
                    sum_fp32 += (float)cfg->k_buffer_l2[s * cfg->embed_dim + d] * cfg->scale_output;
                }
                // Compute mean in FP32
                float mean_fp32 = sum_fp32 / (float)cfg->seq_len;
                // Requantize to INT8
                int32_t quantized = qround(mean_fp32 / cfg->scale_output);
                if (quantized > 127) quantized = 127;
                if (quantized < -128) quantized = -128;
                cfg->output_buffer_l2[d] = (int8_t)quantized;
            }
        } else {
            // No pooling (pool_mode == MHSA_POOL_NONE): Copy unpooled output directly
            // K buffer (SeqLen * EmbedDim) -> Output Buffer (SeqLen * EmbedDim)
            size_t output_size = cfg->seq_len * cfg->embed_dim;
            for (size_t i = 0; i < output_size; i++) {
                cfg->output_buffer_l2[i] = cfg->k_buffer_l2[i];
            }
        }
    }

#undef MHSA_TILED_PROJ
#undef MHSA_TILED_PROJ_SLICE
}
