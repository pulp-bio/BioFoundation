/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

// Centralized compile-time tuning configuration
// Override any value via Makefile `EXTRA_CFLAGS += -D<MACRO>=<VALUE>`.
#pragma once

// -----------------------------------------------------------------------------
// Core configuration
// -----------------------------------------------------------------------------

// Number of cluster cores for parallel execution.
// The Makefile defines -DNUM_CORES=$(CORE) so this provides a fallback default.
#ifndef NUM_CORES
#define NUM_CORES 8
#endif

// Alias used by code paths that reference CL_NUM_CORES.
#ifndef CL_NUM_CORES
#define CL_NUM_CORES NUM_CORES
#endif

// -----------------------------------------------------------------------------
// DMA / pipeline sizing
// -----------------------------------------------------------------------------

// Global DMA descriptor arrays shared across layers (generated project allocates them).
#ifndef MAX_DMA_DESCRIPTORS
#define MAX_DMA_DESCRIPTORS 1000
#endif

// Row-batched 2D DMA (generated network.c paths).
#ifndef MAX_DMA_ROWS
#define MAX_DMA_ROWS 8
#endif

// Row-batched DMA staging inside conv2d tiled pipeline (runtime).
// GAP9 MCHAN supports 16 concurrent descriptors - use full capacity.
#ifndef MAX_DMA_BATCH
#define MAX_DMA_BATCH 16
#endif

// -----------------------------------------------------------------------------
// Linear INT8 tuning
// -----------------------------------------------------------------------------

// Optional per-token input caching into L1 (TCDM) to avoid repeated L2 reads
// during INT8 linear matrix-vector loops.
#ifndef LINEAR_INT8_INPUT_L1_CACHE
#define LINEAR_INT8_INPUT_L1_CACHE 1
#endif

#ifndef LINEAR_INT8_INPUT_L1_CACHE_MAX_BYTES
//  Lowered from 2048 to 512 to prevent stack overflow on worker cores.
// With PI_CL_SLAVE_STACK_SIZE=0x400 (1KB), caching 1024+ byte inputs
// consumes the entire stack, causing corruption. 512 bytes is safe.
#define LINEAR_INT8_INPUT_L1_CACHE_MAX_BYTES 512
#endif

// -----------------------------------------------------------------------------
// MHSA tuning
// -----------------------------------------------------------------------------

// The LUT-based integer softmax uses INT32 scores + UINT8 attention weights and
// does not consume the intermediate INT8 score buffer produced by the QK kernel.
// Disable it by default to save L1 and avoid extra stores in QK.
#ifndef MHSA_TILED_QK_STORE_INT8_SCORES
#define MHSA_TILED_QK_STORE_INT8_SCORES 0
#endif

#ifndef MHSA_FUSE_SOFTMAX_AV
#define MHSA_FUSE_SOFTMAX_AV 0
#endif

#ifndef MHSA_FUSE_QK_SOFTMAX_AV
#define MHSA_FUSE_QK_SOFTMAX_AV 1
#endif

#ifndef MHSA_FUSE_QK_SOFTMAX_AV_PER_CORE
#define MHSA_FUSE_QK_SOFTMAX_AV_PER_CORE 0
#endif

#ifndef MHSA_QK_UNROLL_HEAD_DIM_64
#define MHSA_QK_UNROLL_HEAD_DIM_64 0
#endif

// Transpose V in L1 for integer softmax AV (improves AV locality).
#ifndef MHSA_V_TRANSPOSE_L1
#define MHSA_V_TRANSPOSE_L1 1
#endif

// When enabled, store V only in transposed layout in L1 (no separate V + Vᵀ buffers).
// This reduces L1 footprint and can keep tile_q higher, at the cost of using 2D DMA
// (strided external reads) to load V into the transposed buffer.
#ifndef MHSA_V_TRANSPOSE_REUSE_V_BUFFER
#define MHSA_V_TRANSPOSE_REUSE_V_BUFFER 1
#endif

// Batch DMA descriptor limit for MHSA strided K/V loads (kept <= 16 HW counters).
// GAP9 MCHAN supports 16 concurrent descriptors - use full capacity.
#ifndef MHSA_DMA_BATCH_SIZE
#define MHSA_DMA_BATCH_SIZE 16
#endif

// -----------------------------------------------------------------------------
// Memory thresholds and alignment
// -----------------------------------------------------------------------------

// Weight tensor size threshold for L3 staging vs L1 resident.
// Weights larger than this are streamed from L3 to L2 in chunks.
#ifndef L1_WEIGHT_SIZE_THRESHOLD
#define L1_WEIGHT_SIZE_THRESHOLD 65536
#endif

// Arena and buffer alignment (bytes).
#ifndef L2_ARENA_ALIGNMENT
#define L2_ARENA_ALIGNMENT 4
#endif

// Golden output chunk size for streamed L3 comparison.
#ifndef GOLDEN_CHUNK_SIZE
#define GOLDEN_CHUNK_SIZE 65536
#endif

// Default L3 staging buffer size when no L3-tiled layers present.
#ifndef DEFAULT_L3_STAGING_SIZE
#define DEFAULT_L3_STAGING_SIZE 65536
#endif
