/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

/* ---
 * Network Arguments Structure
 * Holds all pointers and state needed during network execution
 * --- */

<%page args="l3_fallback_buffers, l3_fallback_pools"/>

typedef struct {
    void *input_l3;
% if additional_input_entries:
    void *additional_inputs_l3[${len(additional_input_entries)}];
    size_t num_additional_inputs;
% endif
    void *golden_l3;
% for layer in param_layers:
    void *${layer['c_name']}_weight_l3;
% if layer.get('bias_elements', 0) > 0:
    void *${layer['c_name']}_bias_l3;
% endif
% endfor
    struct pi_device *cluster_dev;
    struct pi_device *ram_dev; // HyperRAM device handle
    uint32_t cluster_cycles;
    int status;

    // L2 Memory Arena (packed activation buffers)
    void *l2_arena;

    int8_t *l1_buffer;
    size_t l1_buffer_size;

    // NE16 Persistent L1 Buffers (allocated once, reused across NE16 layers)
    // This avoids per-layer L1 alloc/free overhead for NE16 weight transfers
    uint8_t *ne16_weight_l1;
    size_t ne16_weight_l1_size;
    int32_t *ne16_bias_l1;
    size_t ne16_bias_l1_size;

    // L3 fallback pointers (for activations that don't fit in L2)
% for buf in l3_fallback_buffers:
    void *${buf['c_name']}_l3;
% endfor
% for pool_buf in l3_fallback_pools:
    void *${pool_buf['c_name']}_l3;
% endfor
% for buf in l3_fallback_buffers:
    ${buf['ctype']} *${buf['c_name']}_staging;
% endfor
% for pool_buf in l3_fallback_pools:
    ${pool_buf['ctype']} *${pool_buf['c_name']}_staging;
% endfor
% if l3_fallback_buffers or l3_fallback_pools:
    int8_t *l3_staging_buffer;  // Shared 64KB staging buffer for all L3 fallback operations
% endif

    // Shared weight slab for transformer block streaming
    // Uses a single shared slab and streams weights operation-by-operation from L3
    int8_t *block_weight_slab;   // Shared weight staging slab (max weight size)
    int8_t *block_bias_slab;     // Shared bias staging slab (max bias size)
    size_t block_weight_slab_size;
    size_t block_bias_slab_size;

    // Activation Pointers
% for buf in activation_buffers:
    ${buf['ctype']} *${buf['c_name']};
% endfor
% for pool_buf in shared_activation_pool:
    ${pool_buf['ctype']} *${pool_buf['c_name']};
% endfor
% for block_id, block_bufs in sorted(block_activation_buffers.items()):
% for buf in block_bufs:
    ${buf['ctype']} *${buf['c_name']};
% endfor
% endfor

    // Parameters
% for layer in param_layers:
% if layer.get('weight_type') == 'fp32':
    float *${layer['c_name']}_weight;
% elif layer.get('weight_type') == 'int16':
    int16_t *${layer['c_name']}_weight;
% else:
    int8_t *${layer['c_name']}_weight;
% endif
% if layer.get('bias_elements', 0) > 0:
% if layer.get('bias_type') == 'int32':
    int32_t *${layer['c_name']}_bias;
% else:
    float *${layer['c_name']}_bias;
% endif
% endif
% endfor

    // NE16 Accelerator Packed Weights (L3 addresses + L2 buffers)
% for layer in param_layers:
% if layer.get('ne16_eligible', False):
    void *${layer['c_name']}_ne16_packed_l3;
    void *${layer['c_name']}_ne16_bias_corr_l3;
    int8_t *${layer['c_name']}_ne16_packed;
    int32_t *${layer['c_name']}_ne16_bias_corr;
% if layer.get('ne16_use_hw_requant', False):
    // NE16 HW Outquant scale arrays
    void *${layer['c_name']}_ne16_hw_scale_l3;
    void *${layer['c_name']}_ne16_hw_scale_shift_l3;
    uint8_t *${layer['c_name']}_ne16_hw_scale;
    uint8_t *${layer['c_name']}_ne16_hw_scale_shift;
% endif
% endif
% endfor

    // Alternating Attention NE16 Packed Weights (L3 addresses + L2 buffers)
% for spec in layer_specs:
% if spec.get('op') == 'alternating_attention' and (spec.get('use_ne16_qkv', False) or spec.get('use_ne16_out', False)):
% if spec.get('use_ne16_qkv', False):
    // NE16 QKV projection for ${spec['c_name']}
    void *${spec['c_name']}_qkv_ne16_packed_l3;
    void *${spec['c_name']}_qkv_ne16_bias_l3;
    int8_t *${spec['c_name']}_qkv_ne16_packed;
    int32_t *${spec['c_name']}_qkv_ne16_bias;
% endif
% if spec.get('use_ne16_out', False):
    // NE16 output projection for ${spec['c_name']}
    void *${spec['c_name']}_out_ne16_packed_l3;
    void *${spec['c_name']}_out_ne16_bias_l3;
    int8_t *${spec['c_name']}_out_ne16_packed;
    int32_t *${spec['c_name']}_out_ne16_bias;
% endif
% endif
% endfor

    // SSM Layer Parameters (integer-only, no FP32)
% for spec in layer_specs:
% if spec.get('op') == 'ssm':
    void *${spec['c_name']}_x_proj_weight_l3;
    void *${spec['c_name']}_dt_proj_weight_l3;
    void *${spec['c_name']}_dt_proj_bias_q16_16_l3;  // Q16.16 bias
    void *${spec['c_name']}_A_q15_l3;  // Q15 A
    void *${spec['c_name']}_D_q15_l3;  // Q15 D
    void *${spec['c_name']}_softplus_lut_l3;  // Q8.8 softplus
    void *${spec['c_name']}_exp_lut_l3;  // Q15 exp
    int8_t *${spec['c_name']}_x_proj_weight;
    int8_t *${spec['c_name']}_dt_proj_weight;
    int32_t *${spec['c_name']}_dt_proj_bias_q16_16;  // Q16.16 bias
    int16_t *${spec['c_name']}_A_q15;  // Q15 A
    int16_t *${spec['c_name']}_D_q15;  // Q15 D
    int16_t *${spec['c_name']}_softplus_lut;  // Q8.8 softplus LUT
    int16_t *${spec['c_name']}_exp_lut;  // Q15 exp LUT
% endif
% endfor


    // MambaBlock Layer Parameters (L3 Streaming)
    // Uses shared weight slab - loads one direction at a time from L3
% if mamba_block_entries:
% for entry in mamba_block_entries:
    // MambaBlock ${entry['c_name']} L3 addresses
    void *${entry['c_name']}_in_proj_weight_l3;
    void *${entry['c_name']}_conv1d_weight_l3;
    void *${entry['c_name']}_conv1d_bias_l3;
    void *${entry['c_name']}_silu_lut_l3;
    void *${entry['c_name']}_silu_gate_lut_q13_l3;
    void *${entry['c_name']}_softplus_lut_l3;
    void *${entry['c_name']}_exp_lut_l3;
    void *${entry['c_name']}_x_proj_weight_l3;
    void *${entry['c_name']}_dt_proj_weight_l3;
    void *${entry['c_name']}_A_q15_l3;
    void *${entry['c_name']}_D_q15_l3;
    void *${entry['c_name']}_out_proj_weight_l3;
    void *${entry['c_name']}_dt_proj_bias_q16_16_l3;
% endfor
    // Shared weight slab for L3 streaming (one direction at a time)
    int8_t *mamba_weight_slab;
    size_t mamba_weight_slab_size;
    // Shared scratch buffer for all Mamba directions (run sequentially)
    int8_t *mamba_shared_scratch;
    size_t mamba_shared_scratch_size;
    // Current direction's weights (pointers into slab)
    int8_t *mamba_cur_in_proj_weight;      // Ping buffer for chunked in_proj
    int8_t *mamba_cur_in_proj_weight_alt;  // Pong buffer for double-buffered async prefetch
    int8_t *mamba_cur_conv1d_weight;
    int32_t *mamba_cur_conv1d_bias;
    int8_t *mamba_cur_silu_lut;
    int16_t *mamba_cur_silu_gate_lut_q13;
    int16_t *mamba_cur_softplus_lut;
    int16_t *mamba_cur_exp_lut;
    int8_t *mamba_cur_x_proj_weight;
    int8_t *mamba_cur_dt_proj_weight;
    int32_t *mamba_cur_dt_proj_bias_q16_16;
    int16_t *mamba_cur_A_q15;
    int16_t *mamba_cur_D_q15;
    int8_t *mamba_cur_out_proj_weight;
% endif

% for spec in layer_specs:
% if spec.get('golden_slot') is not None:
<%
    golden_var_name = spec.get('golden_c_name', spec['c_name'])
%>\
    void *${golden_var_name}_golden_l3;
% if not use_streamed_golden:
    int8_t *${golden_var_name}_golden;
% endif
% endif
% endfor
% if use_streamed_golden:
    // L3 Streamed Golden Validation: Single shared staging buffer
    // Streams each golden from L3 on-demand before comparison
    int8_t *golden_staging;  // Size: ${golden_chunk_size} bytes (chunked comparison)
% endif
} network_cl_args_t;

static network_cl_args_t g_network_cl_args;
