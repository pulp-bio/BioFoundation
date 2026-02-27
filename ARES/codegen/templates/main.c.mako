/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <stdint.h>
#include "pmsis.h"
#include "mem.h"
#include "network_data.h"
#include "network.h"

#ifndef CL_NUM_CORES
#define CL_NUM_CORES 8
#endif

static inline uint32_t checksum_bytes(const void *d, size_t n)
{
    const uint8_t *p = (const uint8_t *)d;
    uint32_t s = 0;
    for (size_t i = 0; i < n; ++i) s += p[i];
    return s;
}

static void *load_one_to_l3(const char *file, size_t sz, uint32_t exp_ck, const char *what)
{
    void *l3 = ram_malloc(sz);
    if (!l3) {
        printf("FC ERR: L3 alloc failed for %s (%zu bytes)\n", what, sz);
        return NULL;
    }

    size_t got = load_file_to_ram(l3, file);
    if (got != sz) {
        printf("FC ERR: load %s (\"%s\") got=%zu exp=%zu\n", what, file, got, sz);
        ram_free(l3, sz);
        return NULL;
    }

    // Skip checksum verification for files > 1MB (they may not fit in L2 scratch allocation)
    #define CHECKSUM_SIZE_LIMIT (1024 * 1024)
    if (sz > CHECKSUM_SIZE_LIMIT) {
#ifndef MINIMAL_OUTPUT
        printf("FC: Loaded %-40s from \"%s\" (size=%6zu, ck=SKIPPED)\n", what, file, sz);
#endif
        return l3;
    }

    void *l2 = pi_l2_malloc(sz);
    if (!l2) {
        printf("FC ERR: L2 temp alloc for checksum (%s)\n", what);
        ram_free(l3, sz);
        return NULL;
    }
    ram_read(l2, l3, sz);
    uint32_t ck = checksum_bytes(l2, sz);
    pi_l2_free(l2, sz);

    if (exp_ck && ck != exp_ck) {
        printf("FC ERR: checksum mismatch for %s: exp=0x%08x got=0x%08x\n", what, exp_ck, ck);
        ram_free(l3, sz);
        return NULL;
    }
#ifndef MINIMAL_OUTPUT
    printf("FC: Loaded %-40s from \"%s\" (size=%6zu, ck=0x%08x)\n", what, file, sz, ck);
#endif
    return l3;
}

// Binary file descriptors
% for file in binary_files:
const network_file_t g_${file['c_symbol']} = {
    .path = "${file['filename']}",
    .size_bytes = ${file['size']},
    .checksum = 0x${'%08X' % file['checksum']}u,
    .label = "${file['label']}",
};
% endfor

void network_dump_manifest(void)
{
#ifndef MINIMAL_OUTPUT
    printf("Network Binary Files:\n");
% for file in binary_files:
    printf("  - %-40s %6zu bytes  ck=0x%08X\n",
           g_${file['c_symbol']}.label,
           g_${file['c_symbol']}.size_bytes,
           g_${file['c_symbol']}.checksum);
% endfor
#endif
}

int main(void)
{
#ifndef MINIMAL_OUTPUT
    printf("--- Starting ${network_name} INT8 Test ---\n");
#endif

    mem_init();
#ifndef MINIMAL_OUTPUT
    printf("FC: L3 Memory and Filesystem initialized.\n");
#endif

    network_dump_manifest();

    // Define file list (static to avoid stack overflow for large networks)
    static const network_file_t *files[] = {
% for file in binary_files:
        &g_${file['c_symbol']},
% endfor
    };
    static void *handles[${len(binary_files)}] = {0};

    int status = 0;

    // Load all files
    for (size_t i = 0; i < sizeof(files) / sizeof(files[0]); ++i) {
        const network_file_t *desc = files[i];
        handles[i] = load_one_to_l3(desc->path, desc->size_bytes, desc->checksum, desc->label);
        if (!handles[i]) {
            printf("FC ERR: failed to load \"%s\".\n", desc->path);
            for (size_t j = 0; j < i; ++j) {
                if (handles[j]) {
                    ram_free(handles[j], files[j]->size_bytes);
                }
            }
            pmsis_exit(-1);
            return -1;
        }
    }

#ifndef MINIMAL_OUTPUT
    printf("FC: All network assets loaded successfully.\n");
#endif

<%
    # Identify L3 fallback activation buffers
    l3_fallback_buffers = [b for b in activation_buffers if b.get('use_l3_fallback', False)]
    l3_fallback_pools = [p for p in shared_activation_pool if p.get('use_l3_fallback', False)]
    all_l3_fallback = l3_fallback_buffers + l3_fallback_pools
%>
% if all_l3_fallback:
    // Allocate L3 buffers for oversized activation buffers
<%
    def _sizeof(ctype):
        if ctype == 'int8_t':
            return 1
        elif ctype == 'int32_t':
            return 4
        elif ctype == 'float':
            return 4
        else:
            raise ValueError(f"Unknown C type: {ctype}")
%>
    static void *l3_activation_handles[${len(all_l3_fallback)}] = {0};
%   for i, buf in enumerate(all_l3_fallback):
<%
    size_bytes = buf['numel'] * _sizeof(buf['ctype'])
    size_kb_str = f"{size_bytes / 1024.0:.1f}"
%>
    // ${buf['name']}: ${buf['numel']} x ${buf['ctype']} = ${size_kb_str} KB
    l3_activation_handles[${i}] = ram_malloc(${size_bytes});
    if (!l3_activation_handles[${i}]) {
        printf("FC ERR: L3 allocation failed for ${buf['name']} (${size_bytes} bytes)\n");
%     for j in range(i):
        if (l3_activation_handles[${j}]) {
            ram_free(l3_activation_handles[${j}], ${all_l3_fallback[j]['numel'] * _sizeof(all_l3_fallback[j]['ctype'])});
        }
%     endfor
        pmsis_exit(-1);
        return -1;
    }
#ifndef MINIMAL_OUTPUT
    printf("FC: Allocated L3 buffer for ${buf['name']}: ${size_kb_str} KB\n");
#endif
%   endfor
#ifndef MINIMAL_OUTPUT
    printf("FC: L3 activation buffers allocated successfully (${len(all_l3_fallback)} buffer(s))\n");
#endif
% endif

    // Initialize cluster
    struct pi_device cluster_dev = {0};
    struct pi_cluster_conf cl_conf;
    pi_cluster_conf_init(&cl_conf);
    cl_conf.id = 0;
% if target_name == "gap9":
    cl_conf.cc_stack_size = 8192;  // Stack per core (bytes) - Increased for deep networks (ResNet-18+)
    cl_conf.icache_conf   = PI_CLUSTER_MASTER_CORE_ICACHE_ENABLE |
                            PI_CLUSTER_ICACHE_PREFETCH_ENABLE   |
                            PI_CLUSTER_ICACHE_ENABLE;
% endif
    pi_open_from_conf(&cluster_dev, &cl_conf);
    if (pi_cluster_open(&cluster_dev)) {
        printf("FC ERR: cluster open failed\n");
        status = -2;
        goto cleanup;
    }

    // Organize handles into proper arrays (weights, biases)
    void *weights_l3[${len(param_layers)}] = {0};
    void *biases_l3[${len(param_layers)}] = {0};
    size_t widx = 0;
% for layer in param_layers:
% if layer.get('weight_index') is not None:
    weights_l3[widx++] = handles[${layer['weight_index']}];
% else:
    weights_l3[widx++] = NULL;  // No weight for ${layer.get('name', 'layer')}
% endif
% endfor
    size_t bidx = 0;
% for layer in param_layers:
% if layer.get('bias_index') is not None:
    biases_l3[bidx++] = handles[${layer['bias_index']}];
% else:
    biases_l3[bidx++] = NULL;  // No bias for ${layer.get('name', 'layer')}
% endif
% endfor

<%
    ne16_param_layers = [l for l in param_layers if l.get('ne16_eligible', False)]
%>
% if ne16_param_layers:
    // NE16 packed weights and bias corrections (pre-packed for NE16 accelerator)
    void *ne16_packed_l3[${len(ne16_param_layers)}] = {0};
    void *ne16_bias_corr_l3[${len(ne16_param_layers)}] = {0};
    void *ne16_hw_scale_l3[${len(ne16_param_layers)}] = {0};
    void *ne16_hw_scale_shift_l3[${len(ne16_param_layers)}] = {0};
    size_t ne16_idx = 0;
% for layer in ne16_param_layers:
    ne16_packed_l3[ne16_idx] = handles[${layer['ne16_packed_weight_index']}];
    ne16_bias_corr_l3[ne16_idx] = handles[${layer['ne16_bias_corr_index']}];
% if layer.get('ne16_use_hw_requant', False):
    ne16_hw_scale_l3[ne16_idx] = handles[${layer['ne16_hw_scale_index']}];
    ne16_hw_scale_shift_l3[ne16_idx] = handles[${layer['ne16_hw_scale_shift_index']}];
% endif
    ne16_idx++;
% endfor
% endif

% if alt_attn_ne16_entries:
    // Alternating Attention NE16 packed weights (for Cerebro transformer)
    // Each entry has: [0]=qkv_packed, [1]=qkv_bias, [2]=out_packed, [3]=out_bias
% for entry in alt_attn_ne16_entries:
    void *${entry['c_name']}_alt_attn_ne16_l3[4] = {
        handles[${entry['qkv_ne16_packed_index']}],  // QKV packed weights
        handles[${entry['qkv_ne16_bias_index']}],    // QKV bias corrected
        handles[${entry['out_ne16_packed_index']}],  // Output packed weights
        handles[${entry['out_ne16_bias_index']}],    // Output bias corrected
    };
% endfor
% endif

% if ssm_entries:
    // SSM parameters (integer-only, no FP32)
    // Each SSM layer has: x_proj_weight, dt_proj_weight, dt_proj_bias_q16_16, A_q15, D_q15, softplus_lut, exp_lut
% for entry in ssm_entries:
    void *${entry['c_name']}_ssm_params_l3[7] = {
        handles[${entry['x_proj_weight_index']}],         // [0] x_proj_weight
        handles[${entry['dt_proj_weight_index']}],        // [1] dt_proj_weight
        handles[${entry['dt_proj_bias_q16_16_index']}],   // [2] dt_proj_bias_q16_16 (Q16.16)
        handles[${entry['A_q15_index']}],                 // [3] A_q15 (Q15)
        handles[${entry['D_q15_index']}],                 // [4] D_q15 (Q15)
        handles[${entry['softplus_lut_index']}],          // [5] softplus_lut (Q8.8)
        handles[${entry['exp_lut_index']}],               // [6] exp_lut (Q15)
    };
    // Scale factors: dt_scale_q=${entry['dt_scale_q']}, dt_scale_shift=${entry['dt_scale_shift']}
% endfor
% endif

% if mamba_block_entries:
    // MambaBlock parameters (integer-only, no FP32)
    // Each MambaBlock has: in_proj, conv1d, conv1d_bias, silu_lut, silu_gate_lut_q13, softplus_lut, exp_lut, x_proj, dt_proj, dt_proj_bias_q16_16, A_q15, D_q15, out_proj
% for entry in mamba_block_entries:
    void *${entry['c_name']}_mamba_params_l3[13] = {
        handles[${entry['in_proj_weight_index']}],        // [0] in_proj_weight
        handles[${entry['conv1d_weight_index']}],         // [1] conv1d_weight
        handles[${entry['conv1d_bias_index']}],           // [2] conv1d_bias (INT32)
        handles[${entry['silu_lut_index']}],              // [3] silu_lut
        handles[${entry['silu_gate_lut_q13_index']}],     // [4] silu_gate_lut_q13 (Q13 gating)
        handles[${entry['softplus_lut_index']}],          // [5] softplus_lut (Q8.8)
        handles[${entry['exp_lut_index']}],               // [6] exp_lut (Q15)
        handles[${entry['x_proj_weight_index']}],         // [7] x_proj_weight
        handles[${entry['dt_proj_weight_index']}],        // [8] dt_proj_weight (INT8)
        handles[${entry['dt_proj_bias_q16_16_index']}],   // [9] dt_proj_bias_q16_16 (Q16.16)
        handles[${entry['A_q15_index']}],                 // [10] A_q15 (Q15)
        handles[${entry['D_q15_index']}],                 // [11] D_q15 (Q15)
        handles[${entry['out_proj_weight_index']}],       // [12] out_proj_weight
    };
    // Scale factors: dt_scale_q=${entry['dt_scale_q']}, dt_scale_shift=${entry['dt_scale_shift']}
% endfor
% endif


% if intermediate_entries:
#ifndef DISABLE_INTERMEDIATE_GOLDEN
    void *intermediate_golden_l3[${len(intermediate_entries)}] = {0};
    size_t iidx = 0;
% for entry in intermediate_entries:
    intermediate_golden_l3[iidx++] = handles[${entry['index']}];
% endfor
#ifndef MINIMAL_OUTPUT
    printf("FC: Intermediate golden outputs loaded for layer-wise comparison\n");
#endif
#else
    void **intermediate_golden_l3 = NULL;
#endif
% else:
    void **intermediate_golden_l3 = NULL;
% endif

% if additional_input_entries:
    // Build additional inputs array for multi-input model
    void *additional_inputs_l3[${len(additional_input_entries)}] = {
%   for entry in additional_input_entries:
        handles[${entry['index']}],  // ${entry['quant_layer']}
%   endfor
    };
% endif

    // Run network test
    status = network_run_test_from_l3(
        handles[${input_entry['index']}],
% if additional_input_entries:
        additional_inputs_l3,
        ${len(additional_input_entries)},
% endif
        weights_l3,
        ${len(param_layers)},
        biases_l3,
        ${len(param_layers)},
        handles[${golden_entry['index']}],
% if intermediate_entries:
#ifndef DISABLE_INTERMEDIATE_GOLDEN
        intermediate_golden_l3,
        ${len(intermediate_entries)},
#else
        NULL,
        0,
#endif
% else:
        NULL,
        0,
% endif
% if all_l3_fallback:
        l3_activation_handles,
        ${len(all_l3_fallback)},
% else:
        NULL,
        0,
% endif
% if ssm_entries:
        // SSM parameter arrays
% for entry in ssm_entries:
        ${entry['c_name']}_ssm_params_l3,
% endfor
% endif
% if mamba_block_entries:
        // MambaBlock parameter arrays
% for entry in mamba_block_entries:
        ${entry['c_name']}_mamba_params_l3,
% endfor
% endif
% if ne16_param_layers:
        // NE16 packed weights arrays and HW outquant scales
        ne16_packed_l3,
        ne16_bias_corr_l3,
        ne16_hw_scale_l3,
        ne16_hw_scale_shift_l3,
        ${len(ne16_param_layers)},
% endif
% if alt_attn_ne16_entries:
        // Alternating Attention NE16 parameter arrays
% for entry in alt_attn_ne16_entries:
        ${entry['c_name']}_alt_attn_ne16_l3,
% endfor
% endif
        &cluster_dev);

    if (status != 0) {
        printf("FC ERR: Network test failed (status=%d)\n", status);
    } else {
        printf("FC: Network test completed successfully.\n");
    }

    pi_cluster_close(&cluster_dev);

cleanup:
% if all_l3_fallback:
    // Free L3 activation buffers
%   for i, buf in enumerate(all_l3_fallback):
    if (l3_activation_handles[${i}]) {
        ram_free(l3_activation_handles[${i}], ${buf['numel'] * _sizeof(buf['ctype'])});
    }
%   endfor
% endif

    for (size_t i = 0; i < sizeof(files) / sizeof(files[0]); ++i) {
        if (handles[i]) {
            ram_free(handles[i], files[i]->size_bytes);
        }
    }

    pmsis_exit(status);
    return status;
}
