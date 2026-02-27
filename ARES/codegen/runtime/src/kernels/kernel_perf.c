/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * kernel_perf.c - Performance Counter Utilities
 *
 * Provides cycle-accurate performance measurement for layer-by-layer
 * profiling on GAP9. Tracks compute, DMA load, DMA store, and idle cycles.
 *
 * Enable with: make ENABLE_PERF=1
 *
 * Part of the ARES modular kernel system.
 */

// ---
// Performance Counter Implementations
// ---

#ifdef ENABLE_PERF_COUNTERS

// Static variables for performance tracking
static uint32_t perf_layer_start_cycles;
static uint32_t perf_dma_load_start_cycles;
static uint32_t perf_dma_store_start_cycles;
static uint32_t perf_compute_start_cycles;

// Running totals for summary
static uint32_t perf_total_compute_cycles = 0;
static uint32_t perf_total_dma_load_cycles = 0;
static uint32_t perf_total_dma_store_cycles = 0;
static uint32_t perf_total_cycles = 0;
static uint32_t perf_num_layers = 0;

void perf_counter_init(void) {
    // Configure hardware performance counter for cycle counting
    pi_perf_conf((1 << PI_PERF_CYCLES));
    pi_perf_reset();
    pi_perf_start();

    // Reset totals
    perf_total_compute_cycles = 0;
    perf_total_dma_load_cycles = 0;
    perf_total_dma_store_cycles = 0;
    perf_total_cycles = 0;
    perf_num_layers = 0;
}

void perf_counter_reset(void) {
    pi_perf_reset();
    pi_perf_start();
}

uint32_t perf_counter_get_cycles(void) {
    return pi_perf_read(PI_PERF_CYCLES);
}

void perf_layer_start(const char *layer_name) {
    perf_layer_start_cycles = pi_perf_read(PI_PERF_CYCLES);
}

void perf_layer_end(const char *layer_name, layer_perf_t *perf) {
    uint32_t end_cycles = pi_perf_read(PI_PERF_CYCLES);
    perf->total_cycles = end_cycles - perf_layer_start_cycles;

    // Update running totals
    perf_total_cycles += perf->total_cycles;
    perf_total_compute_cycles += perf->compute_cycles;
    perf_total_dma_load_cycles += perf->dma_load_cycles;
    perf_total_dma_store_cycles += perf->dma_store_cycles;
    perf_num_layers++;

    // Calculate idle cycles (time not spent in compute or DMA)
    uint32_t active_cycles = perf->compute_cycles + perf->dma_load_cycles + perf->dma_store_cycles;
    if (active_cycles < perf->total_cycles) {
        perf->idle_cycles = perf->total_cycles - active_cycles;
    } else {
        // Overlapped execution - compute and DMA running in parallel
        perf->idle_cycles = 0;
    }

    // Calculate overlap percentage
    if (perf->total_cycles > 0) {
        perf->overlap_percent = 100.0f * (1.0f - (float)perf->idle_cycles / (float)perf->total_cycles);
    } else {
        perf->overlap_percent = 0.0f;
    }
}

void perf_layer_record(const char *layer_name, const layer_perf_t *perf) {
#ifndef MINIMAL_OUTPUT
    printf("  PERF %-24s: total=%8u compute=%8u dma_load=%8u dma_store=%8u idle=%8u overlap=%.1f%%\n",
           layer_name,
           perf->total_cycles,
           perf->compute_cycles,
           perf->dma_load_cycles,
           perf->dma_store_cycles,
           perf->idle_cycles,
           perf->overlap_percent);
#endif
}

void perf_summary_print(void) {
#ifndef MINIMAL_OUTPUT
    printf("\n");
    printf("PERFORMANCE SUMMARY\n");
    printf("Total layers:        %u\n", perf_num_layers);
    printf("Total cycles:        %u\n", perf_total_cycles);
    printf("  Compute cycles:    %u (%.1f%%)\n",
           perf_total_compute_cycles,
           perf_total_cycles > 0 ? 100.0f * perf_total_compute_cycles / perf_total_cycles : 0.0f);
    printf("  DMA load cycles:   %u (%.1f%%)\n",
           perf_total_dma_load_cycles,
           perf_total_cycles > 0 ? 100.0f * perf_total_dma_load_cycles / perf_total_cycles : 0.0f);
    printf("  DMA store cycles:  %u (%.1f%%)\n",
           perf_total_dma_store_cycles,
           perf_total_cycles > 0 ? 100.0f * perf_total_dma_store_cycles / perf_total_cycles : 0.0f);

    uint32_t accounted = perf_total_compute_cycles + perf_total_dma_load_cycles + perf_total_dma_store_cycles;
    if (accounted < perf_total_cycles) {
        uint32_t overhead = perf_total_cycles - accounted;
        printf("  Overhead/idle:     %u (%.1f%%)\n",
               overhead,
               100.0f * overhead / perf_total_cycles);
    }
#endif
}

// DMA timing helpers
void perf_dma_load_start(void) {
    perf_dma_load_start_cycles = pi_perf_read(PI_PERF_CYCLES);
}

uint32_t perf_dma_load_end(void) {
    return pi_perf_read(PI_PERF_CYCLES) - perf_dma_load_start_cycles;
}

void perf_dma_store_start(void) {
    perf_dma_store_start_cycles = pi_perf_read(PI_PERF_CYCLES);
}

uint32_t perf_dma_store_end(void) {
    return pi_perf_read(PI_PERF_CYCLES) - perf_dma_store_start_cycles;
}

// Compute timing helpers
void perf_compute_start(void) {
    perf_compute_start_cycles = pi_perf_read(PI_PERF_CYCLES);
}

uint32_t perf_compute_end(void) {
    return pi_perf_read(PI_PERF_CYCLES) - perf_compute_start_cycles;
}

#endif // ENABLE_PERF_COUNTERS
