# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Mamba/SSM layer handlers extracted from generate_c_code."""

from __future__ import annotations

from typing import Any, Dict


def handle_ssm(
    generator: Any,
    ctx: Any,
    layer_name: str,
    layer_data: Dict[str, Any],
    spec: Dict[str, Any],
) -> bool:
    """Handle SSM (Selective State Space Model) layer - MAMBA core.

    Input shape: [batch, seq_len, d_inner] or [batch, d_inner, seq_len]
    Output shape: same as input
    """
    if len(ctx.current_shape) != 3:
        raise ValueError(f"SSM {layer_name} expects 3D input, got {ctx.current_shape}")

    batch = ctx.current_shape[0]
    seq_len = layer_data.get('seq_len', ctx.current_shape[1])
    d_inner = layer_data.get('d_inner', ctx.current_shape[2])
    d_state = layer_data.get('d_state', 16)
    dt_rank = layer_data.get('dt_rank', d_inner // 16)

    out_shape = [batch, seq_len, d_inner]
    output_numel = generator._numel(out_shape)

    buffer_name = f"{layer_name}_out"
    output_entry = generator._ctx_register_buffer(
        ctx, buffer_name, 'int8_t', output_numel,
        f"{layer_name} output", spec.get('block_id')
    )

    # Query KB BEFORE override check to allow auto-application of learned configs
    generator._prepare_kb_config(
        layer_name=layer_name,
        op_type='ssm_int8',
        shape={
            'seq_len': seq_len,
            'd_inner': d_inner,
            'd_state': d_state,
            'dt_rank': dt_rank,
        }
    )

    # Check for config override (auto-tuning) - extract SSM kernel hints
    ssm_kernel_hints = {}
    override = generator._get_layer_override(layer_name)
    if override:
        if 'kernel_config' in override:
            kc = override['kernel_config']
            ssm_kernel_hints = {
                'ph1_xproj_l1_tile': kc.get('ph1_xproj_l1_tile', False),
                'ph2_dtproj_l1_tile': kc.get('ph2_dtproj_l1_tile', False),
                'ph3_bc_l1_staging': kc.get('ph3_bc_l1_staging', False),
                'ph1_xt_staging': kc.get('ph1_xt_staging', False),
                'ph2_delta_l1_staging': kc.get('ph2_delta_l1_staging', False),
            }
            if any(ssm_kernel_hints.values()):
                print(f"  [TUNE] Using SSM kernel hints for {layer_name}: {ssm_kernel_hints}")
        if 'tile_config' in override:
            tc = override['tile_config']
            if tc.get('ph3_channel_tile'):
                ssm_kernel_hints['ph3_channel_tile'] = tc['ph3_channel_tile']
                print(f"  [TUNE] Using SSM tile hint for {layer_name}: ph3_channel_tile={tc['ph3_channel_tile']}")

    spec.update({
        'op': 'ssm',
        'input_buffer': generator._ctx_buffer_c_name(ctx, ctx.current_buffer),
        'output_buffer': output_entry['c_name'],
        'batch': batch,
        'seq_len': seq_len,
        'd_inner': d_inner,
        'd_state': d_state,
        'dt_rank': dt_rank,
        'scale_x': layer_data.get('scale_input', ctx.current_scale),
        'scale_x_proj': layer_data.get('x_proj_scale_weight', 0.001),
        'scale_dt_proj': layer_data.get('dt_proj_scale_weight', 0.01),
        'scale_output': layer_data.get('scale_output', ctx.current_scale),
        # I-Mamba step 7: Fixed-point scale for dt_acc to Q16.16 conversion
        'dt_scale_q': generator.ssm_dt_scale_q.get(layer_name, 0),
        'dt_scale_shift': generator.ssm_dt_scale_shift.get(layer_name, 24),
        # I-Mamba step 8: Fixed-point scale for B/C to Q15 conversion
        'bc_scale_factor': generator.ssm_bc_scale_factor.get(layer_name, 0),
        # I-Mamba step 9: Fixed-point scale for output conversion
        'output_scale_q': generator.ssm_output_scale_q.get(layer_name, 0),
        # Auto-tuning kernel hints
        'kernel_hints': ssm_kernel_hints,
    })

    # Register SSM weights
    generator._register_mamba_weight(layer_name, 'dt_proj_weight', dt_rank * d_inner)
    generator._register_mamba_param(layer_name, 'A', d_inner * d_state, 'float')  # Pre-computed A = -exp(A_log)
    generator._register_mamba_param(layer_name, 'D', d_inner, 'float')

    generator._ctx_attach_golden(ctx, spec, layer_name, output_numel, buffer_name)
    ctx.specs.append(spec)
    ctx.current_buffer = buffer_name
    ctx.current_shape = out_shape
    ctx.current_scale = layer_data.get('scale_output', ctx.current_scale)
    ctx.layer_output_buffer[layer_name] = ctx.current_buffer
    ctx.layer_output_scale[layer_name] = ctx.current_scale
    ctx.buffer_scale[ctx.current_buffer] = ctx.current_scale
    return True


def handle_mambablock(
    generator: Any,
    ctx: Any,
    layer_name: str,
    layer_data: Dict[str, Any],
    spec: Dict[str, Any],
) -> bool:
    """Handle MambaBlock (composite: in_proj -> conv1d -> silu -> ssm -> out_proj).

    Input shape: [batch, seq_len, d_model]
    Output shape: same as input (residual connection)
    """
    if len(ctx.current_shape) != 3:
        raise ValueError(f"MambaBlock {layer_name} expects 3D input [B, L, D], got {ctx.current_shape}")

    batch = ctx.current_shape[0]
    seq_len = ctx.current_shape[1]
    d_model = ctx.current_shape[2]
    d_inner = layer_data.get('d_inner', d_model * 2)
    d_state = layer_data.get('d_state', 16)
    dt_rank = layer_data.get('dt_rank', d_inner // 16)
    kernel_size = layer_data.get('kernel_size', 4)

    out_shape = [batch, seq_len, d_model]
    output_numel = generator._numel(out_shape)

    buffer_name = f"{layer_name}_out"
    output_entry = generator._ctx_register_buffer(
        ctx, buffer_name, 'int8_t', output_numel,
        f"{layer_name} output", spec.get('block_id')
    )

    # Calculate L2 scratch requirement for MambaBlock intermediate buffers
    # INT8 buffers (with F1 fusion: conv_out eliminated, x_transposed reused for ssm_out):
    xz_proj_size = batch * seq_len * 2 * d_inner      # [B, L, 2*d_inner] in_proj output
    x_part_size = batch * seq_len * d_inner           # x-part of xz_proj (freed after transpose)
    x_transposed_size = batch * d_inner * seq_len     # [B, d_inner, L] for conv1d (reused for ssm_out)
    ssm_in_size = batch * seq_len * d_inner           # [B, L, d_inner] from fused conv

    # Fixed-point SSM scratch (INT16/INT32, not FP32):
    proj_size = dt_rank + 2 * d_state
    h_state_size = batch * d_inner * d_state * 2      # int16
    proj_all_size = seq_len * proj_size * 4           # int32
    dt_all_size = d_inner * seq_len * 2               # int16
    B_all_size = seq_len * d_state * 2                # int16
    C_all_size = seq_len * d_state * 2                # int16

    # interleaved layout [x0, z0, x1, z1, ...] - reusing "x-part" destroys z values
    # needed for gating. All small SSM scratch placed AFTER dt_all.
    small_ssm_scratch = h_state_size + proj_all_size + B_all_size + C_all_size
    int8_scratch = xz_proj_size + x_transposed_size + ssm_in_size
    ssm_scratch_size = dt_all_size + small_ssm_scratch

    scratch_needed = int8_scratch + ssm_scratch_size

    # Extract all intermediate scales for MambaBlock
    scale_input = layer_data.get('scale_input', ctx.current_scale)
    scale_output = layer_data.get('scale_output', ctx.current_scale)
    in_proj_scale_w = layer_data.get('in_proj_scale_weight', 0.001)
    in_proj_scale_out = layer_data.get('in_proj_scale_output', scale_input)
    conv1d_scale_w = layer_data.get('conv1d_scale_weight', 0.001)
    conv1d_scale_out = layer_data.get('conv1d_scale_output', scale_input)
    silu_scale_out = layer_data.get('silu_scale_output', scale_input)
    ssm_x_proj_scale_w = layer_data.get('ssm_x_proj_scale_weight', 0.001)
    ssm_scale_out = layer_data.get('ssm_scale_output', scale_input)
    out_proj_scale_w = layer_data.get('out_proj_scale_weight', 0.001)

    # Get dt_proj scale for INT8 kernel
    ssm_dt_proj_scale_w = layer_data.get('ssm_dt_proj_scale_weight', 0.01)

    spec.update({
        'op': 'mamba_block',
        'input_buffer': generator._ctx_buffer_c_name(ctx, ctx.current_buffer),
        'output_buffer': output_entry['c_name'],
        'batch': batch,
        'seq_len': seq_len,
        'd_model': d_model,
        'd_inner': d_inner,
        'd_state': d_state,
        'dt_rank': dt_rank,
        'kernel_size': kernel_size,
        'scale_in': scale_input,
        'scale_out': scale_output,
        # Sub-operation scales
        'scale_in_proj': in_proj_scale_w,          # in_proj weight scale
        'scale_xz': in_proj_scale_out,             # in_proj output scale (x and z)
        'scale_conv': conv1d_scale_w,              # conv1d weight scale
        'scale_conv_out': conv1d_scale_out,        # conv1d output scale
        'scale_silu_out': silu_scale_out,          # silu output scale
        'scale_ssm_in': silu_scale_out,            # SSM input (after silu)
        'scale_x_proj': ssm_x_proj_scale_w,        # SSM x_proj weight scale
        'scale_dt_proj': ssm_dt_proj_scale_w,      # SSM dt_proj weight scale (INT8)
        'scale_ssm_out': ssm_scale_out,            # SSM output scale
        'scale_gate': silu_scale_out,              # z-branch silu output scale
        'scale_gated': ssm_scale_out,              # After multiply (same as ssm_out)
        'scale_out_proj': out_proj_scale_w,        # out_proj weight scale
        'scratch_size': scratch_needed,
        # I-Mamba step 7: Fixed-point scale for dt_acc to Q16.16 conversion
        'dt_scale_q': generator.ssm_dt_scale_q.get(layer_name, 0),
        'dt_scale_shift': generator.ssm_dt_scale_shift.get(layer_name, 24),
        # I-Mamba step 8: Fixed-point scale for B/C to Q15 conversion
        'bc_scale_factor': generator.ssm_bc_scale_factor.get(layer_name, 0),
        # I-Mamba step 9: Fixed-point scale for output conversion
        'output_scale_q': generator.ssm_output_scale_q.get(layer_name, 0),
    })

    # Register all MambaBlock weights
    generator._register_mamba_weight(layer_name, 'in_proj_weight', d_model * 2 * d_inner)
    generator._register_mamba_weight(layer_name, 'conv1d_weight', d_inner * kernel_size)
    generator._register_mamba_param(layer_name, 'conv1d_bias', d_inner, 'int32')  # INT32 bias
    generator._register_mamba_weight(layer_name, 'x_proj_weight', (dt_rank + 2 * d_state) * d_inner)
    generator._register_mamba_weight(layer_name, 'dt_proj_weight', dt_rank * d_inner)  # Now INT8
    generator._register_mamba_param(layer_name, 'dt_proj_bias', d_inner, 'float')
    generator._register_mamba_param(layer_name, 'A', d_inner * d_state, 'float')  # Pre-computed A = -exp(A_log)
    generator._register_mamba_param(layer_name, 'D', d_inner, 'float')
    generator._register_mamba_weight(layer_name, 'out_proj_weight', d_inner * d_model)
    generator._register_mamba_lut(layer_name, 'silu_lut', 256)
    generator._register_mamba_lut(layer_name, 'softplus_lut', 256)  # I-Mamba: Integer softplus
    generator._register_mamba_lut(layer_name, 'exp_lut', 256)  # I-Mamba: Integer exp for discretization

    # Create entry for mamba_block_entries (like ssm_entries)
    mamba_entry = {
        'c_name': generator.sanitize_c_name(layer_name),
        'layer_name': layer_name,
        'in_proj_weight_index': generator.weight_entries.get(f"{layer_name}::in_proj_weight", {}).get('index'),
        'in_proj_weight_elements': d_model * 2 * d_inner,
        'conv1d_weight_index': generator.weight_entries.get(f"{layer_name}::conv1d_weight", {}).get('index'),
        'conv1d_weight_elements': d_inner * kernel_size,
        'conv1d_bias_index': generator.weight_entries.get(f"{layer_name}::conv1d_bias", {}).get('index'),
        'conv1d_bias_elements': d_inner,
        'x_proj_weight_index': generator.weight_entries.get(f"{layer_name}::x_proj_weight", {}).get('index'),
        'x_proj_weight_elements': (dt_rank + 2 * d_state) * d_inner,
        'dt_proj_weight_index': generator.weight_entries.get(f"{layer_name}::dt_proj_weight", {}).get('index'),
        'dt_proj_weight_elements': dt_rank * d_inner,
        'dt_proj_bias_index': generator.weight_entries.get(f"{layer_name}::dt_proj_bias", {}).get('index'),
        'dt_proj_bias_elements': d_inner,
        'A_index': generator.weight_entries.get(f"{layer_name}::A", {}).get('index'),
        'A_elements': d_inner * d_state,
        # I-Mamba step 4: Q15 A for full dyadic SSM
        'A_q15_index': generator.weight_entries.get(f"{layer_name}::A_q15", {}).get('index'),
        'D_index': generator.weight_entries.get(f"{layer_name}::D", {}).get('index'),
        'D_elements': d_inner,
        # I-Mamba step 2c: Q15 D for dyadic arithmetic
        'D_q15_index': generator.weight_entries.get(f"{layer_name}::D_q15", {}).get('index'),
        'out_proj_weight_index': generator.weight_entries.get(f"{layer_name}::out_proj_weight", {}).get('index'),
        'out_proj_weight_elements': d_inner * d_model,
        'silu_lut_index': generator.weight_entries.get(f"{layer_name}::silu_lut", {}).get('index'),
        'silu_gate_lut_q13_index': generator.weight_entries.get(f"{layer_name}::silu_gate_lut_q13", {}).get('index'),
        'softplus_lut_index': generator.weight_entries.get(f"{layer_name}::softplus_lut", {}).get('index'),
        'exp_lut_index': generator.weight_entries.get(f"{layer_name}::exp_lut", {}).get('index'),
        # I-Mamba step 6: Q16.16 bias and scale factors for full integer dt_proj
        'dt_proj_bias_q16_16_index': generator.weight_entries.get(f"{layer_name}::dt_proj_bias_q16_16", {}).get('index'),
        'dt_scale_q': generator.ssm_dt_scale_q.get(layer_name, 0),
        'dt_scale_shift': generator.ssm_dt_scale_shift.get(layer_name, 24),
        'silu_lut_elements': 256,  # 256-entry LUT for SiLU, Q13 gate, softplus, exp
        'l2_scratch_size': scratch_needed,  # For intermediate MambaBlock buffers
        'd_model': d_model,
        'd_inner': d_inner,
        'd_state': d_state,
        'dt_rank': dt_rank,
        'kernel_size': kernel_size,
    }
    generator.mamba_block_entries.append(mamba_entry)

    generator._ctx_attach_golden(ctx, spec, layer_name, output_numel, buffer_name)
    ctx.specs.append(spec)

    ctx.current_buffer = buffer_name
    ctx.current_shape = out_shape
    ctx.current_scale = layer_data.get('scale_output', ctx.current_scale)
    ctx.layer_output_buffer[layer_name] = ctx.current_buffer
    ctx.layer_output_scale[layer_name] = ctx.current_scale
    ctx.buffer_scale[ctx.current_buffer] = ctx.current_scale
    return True


def handle_mambawrapper(
    generator: Any,
    ctx: Any,
    layer_name: str,
    layer_data: Dict[str, Any],
    spec: Dict[str, Any],
) -> bool:
    """Handle bidirectional MambaWrapper (flip -> mamba_fwd + mamba_rev -> flip -> add).

    Input shape: [batch, seq_len, d_model]
    Output shape: same as input (or double features if concat strategy)
    """
    if len(ctx.current_shape) != 3:
        raise ValueError(f"MambaWrapper {layer_name} expects 3D input [B, L, D], got {ctx.current_shape}")

    batch = ctx.current_shape[0]
    seq_len = ctx.current_shape[1]
    d_model = ctx.current_shape[2]
    d_inner = layer_data.get('d_inner', d_model * 2)
    d_state = layer_data.get('d_state', 16)
    dt_rank = layer_data.get('fwd_dt_rank', d_inner // 16)
    kernel_size = layer_data.get('fwd_kernel_size', 4)
    bidirectional_strategy = layer_data.get('bidirectional_strategy', 'add')

    # Output shape same as input (or double features if concat strategy)
    if bidirectional_strategy == 'concat':
        out_shape = [batch, seq_len, d_model * 2]
    else:
        out_shape = [batch, seq_len, d_model]
    output_numel = generator._numel(out_shape)

    buffer_name = f"{layer_name}_out"
    output_entry = generator._ctx_register_buffer(
        ctx, buffer_name, 'int8_t', output_numel,
        f"{layer_name} output", spec.get('block_id')
    )

    # Calculate L2 scratch for MambaWrapper (2x MambaBlock + flip buffers)
    xz_proj_size = batch * seq_len * 2 * d_inner
    x_part_size = batch * seq_len * d_inner
    x_transposed_size = batch * d_inner * seq_len
    ssm_in_size = batch * seq_len * d_inner
    int8_per_dir = xz_proj_size + x_transposed_size + ssm_in_size

    # Fixed-point SSM scratch per direction
    proj_size = dt_rank + 2 * d_state
    h_state_size = batch * d_inner * d_state * 2
    proj_all_size = seq_len * proj_size * 4
    dt_all_size = d_inner * seq_len * 2
    B_all_size = seq_len * d_state * 2
    C_all_size = seq_len * d_state * 2

    # OPTIMIZATION: Reuse xz_proj's x-part for small SSM buffers
    small_ssm_scratch = h_state_size + proj_all_size + B_all_size + C_all_size

    if small_ssm_scratch <= x_part_size:
        ssm_scratch_separate = dt_all_size
    else:
        ssm_scratch_separate = dt_all_size + small_ssm_scratch

    scratch_per_dir = int8_per_dir + ssm_scratch_separate

    # OPTIMIZATION: flip_buffer overlaps with SSM scratch
    flip_buffer_size = batch * seq_len * d_model
    ssm_scratch_with_flip = max(ssm_scratch_separate, flip_buffer_size)
    scratch_needed = int8_per_dir + ssm_scratch_with_flip

    # Extract scales
    scale_input = layer_data.get('scale_input', ctx.current_scale)
    scale_output = layer_data.get('scale_output', ctx.current_scale)
    scale_equalizer = layer_data.get('scale_equalizer', scale_output)

    spec.update({
        'op': 'mamba_wrapper',
        'input_buffer': generator._ctx_buffer_c_name(ctx, ctx.current_buffer),
        'output_buffer': output_entry['c_name'],
        'batch': batch,
        'seq_len': seq_len,
        'd_model': d_model,
        'd_inner': d_inner,
        'd_state': d_state,
        'dt_rank': dt_rank,
        'kernel_size': kernel_size,
        'bidirectional_strategy': bidirectional_strategy,
        'scale_in': scale_input,
        'scale_out': scale_output,
        'scale_equalizer': scale_equalizer,
        'scratch_size': scratch_needed,
        # Forward direction scales
        'fwd_scale_in_proj': layer_data.get('fwd_in_proj_scale_weight', 0.001),
        'fwd_scale_xz': layer_data.get('fwd_in_proj_scale_output', scale_input),
        'fwd_scale_conv': layer_data.get('fwd_conv1d_scale_weight', 0.001),
        'fwd_scale_conv_out': layer_data.get('fwd_conv1d_scale_output', scale_input),
        'fwd_scale_silu_out': layer_data.get('fwd_silu_scale_output', scale_input),
        'fwd_scale_x_proj': layer_data.get('fwd_ssm_x_proj_scale_weight', 0.001),
        'fwd_scale_dt_proj': layer_data.get('fwd_ssm_dt_proj_scale_weight', 0.01),
        'fwd_scale_ssm_out': layer_data.get('fwd_ssm_scale_output', scale_input),
        'fwd_scale_out_proj': layer_data.get('fwd_out_proj_scale_weight', 0.001),
        'fwd_dt_scale_q': generator.ssm_dt_scale_q.get(f"{layer_name}_fwd", 0),
        'fwd_dt_scale_shift': generator.ssm_dt_scale_shift.get(f"{layer_name}_fwd", 24),
        'fwd_bc_scale_factor': generator.ssm_bc_scale_factor.get(f"{layer_name}_fwd", 0),
        'fwd_output_scale_q': generator.ssm_output_scale_q.get(f"{layer_name}_fwd", 0),
        # Reverse direction scales
        'rev_scale_in_proj': layer_data.get('rev_in_proj_scale_weight', 0.001),
        'rev_scale_xz': layer_data.get('rev_in_proj_scale_output', scale_input),
        'rev_scale_conv': layer_data.get('rev_conv1d_scale_weight', 0.001),
        'rev_scale_conv_out': layer_data.get('rev_conv1d_scale_output', scale_input),
        'rev_scale_silu_out': layer_data.get('rev_silu_scale_output', scale_input),
        'rev_scale_x_proj': layer_data.get('rev_ssm_x_proj_scale_weight', 0.001),
        'rev_scale_dt_proj': layer_data.get('rev_ssm_dt_proj_scale_weight', 0.01),
        'rev_scale_ssm_out': layer_data.get('rev_ssm_scale_output', scale_input),
        'rev_scale_out_proj': layer_data.get('rev_out_proj_scale_weight', 0.001),
        'rev_dt_scale_q': generator.ssm_dt_scale_q.get(f"{layer_name}_rev", 0),
        'rev_dt_scale_shift': generator.ssm_dt_scale_shift.get(f"{layer_name}_rev", 24),
        'rev_bc_scale_factor': generator.ssm_bc_scale_factor.get(f"{layer_name}_rev", 0),
        'rev_output_scale_q': generator.ssm_output_scale_q.get(f"{layer_name}_rev", 0),
    })

    # Add bit width info for 2-bit weight support
    fwd_prefix = f"{layer_name}_fwd"
    rev_prefix = f"{layer_name}_rev"

    fwd_in_proj_bw = generator.weight_entries.get(f"{fwd_prefix}::in_proj_weight_bit_width", 8)
    fwd_x_proj_bw = generator.weight_entries.get(f"{fwd_prefix}::x_proj_weight_bit_width", 8)
    fwd_dt_proj_bw = generator.weight_entries.get(f"{fwd_prefix}::dt_proj_weight_bit_width", 8)
    fwd_out_proj_bw = generator.weight_entries.get(f"{fwd_prefix}::out_proj_weight_bit_width", 8)
    rev_in_proj_bw = generator.weight_entries.get(f"{rev_prefix}::in_proj_weight_bit_width", 8)
    rev_x_proj_bw = generator.weight_entries.get(f"{rev_prefix}::x_proj_weight_bit_width", 8)
    rev_dt_proj_bw = generator.weight_entries.get(f"{rev_prefix}::dt_proj_weight_bit_width", 8)
    rev_out_proj_bw = generator.weight_entries.get(f"{rev_prefix}::out_proj_weight_bit_width", 8)

    # Calculate packed byte sizes for 2-bit weights
    in_proj_elems = d_model * 2 * d_inner
    x_proj_elems = (dt_rank + 2 * d_state) * d_inner
    dt_proj_elems = dt_rank * d_inner
    out_proj_elems = d_inner * d_model

    spec.update({
        # Forward direction bit widths and packed sizes
        'fwd_in_proj_bit_width': fwd_in_proj_bw,
        'fwd_in_proj_bytes': (in_proj_elems + 3) // 4 if fwd_in_proj_bw == 2 else in_proj_elems,
        'fwd_x_proj_bit_width': fwd_x_proj_bw,
        'fwd_x_proj_bytes': (x_proj_elems + 3) // 4 if fwd_x_proj_bw == 2 else x_proj_elems,
        'fwd_dt_proj_bit_width': fwd_dt_proj_bw,
        'fwd_dt_proj_bytes': (dt_proj_elems + 3) // 4 if fwd_dt_proj_bw == 2 else dt_proj_elems,
        'fwd_out_proj_bit_width': fwd_out_proj_bw,
        'fwd_out_proj_bytes': (out_proj_elems + 3) // 4 if fwd_out_proj_bw == 2 else out_proj_elems,
        # Reverse direction bit widths and packed sizes
        'rev_in_proj_bit_width': rev_in_proj_bw,
        'rev_in_proj_bytes': (in_proj_elems + 3) // 4 if rev_in_proj_bw == 2 else in_proj_elems,
        'rev_x_proj_bit_width': rev_x_proj_bw,
        'rev_x_proj_bytes': (x_proj_elems + 3) // 4 if rev_x_proj_bw == 2 else x_proj_elems,
        'rev_dt_proj_bit_width': rev_dt_proj_bw,
        'rev_dt_proj_bytes': (dt_proj_elems + 3) // 4 if rev_dt_proj_bw == 2 else dt_proj_elems,
        'rev_out_proj_bit_width': rev_out_proj_bw,
        'rev_out_proj_bytes': (out_proj_elems + 3) // 4 if rev_out_proj_bw == 2 else out_proj_elems,
    })

    # Register weights for both directions
    for direction in ['fwd', 'rev']:
        prefix = f"{layer_name}_{direction}"
        generator._register_mamba_weight(prefix, 'in_proj_weight', d_model * 2 * d_inner)
        generator._register_mamba_weight(prefix, 'conv1d_weight', d_inner * kernel_size)
        generator._register_mamba_param(prefix, 'conv1d_bias', d_inner, 'int32')
        generator._register_mamba_weight(prefix, 'x_proj_weight', (dt_rank + 2 * d_state) * d_inner)
        generator._register_mamba_weight(prefix, 'dt_proj_weight', dt_rank * d_inner)
        generator._register_mamba_param(prefix, 'dt_proj_bias', d_inner, 'float')
        generator._register_mamba_param(prefix, 'A', d_inner * d_state, 'float')
        generator._register_mamba_param(prefix, 'D', d_inner, 'float')
        generator._register_mamba_weight(prefix, 'out_proj_weight', d_inner * d_model)
        generator._register_mamba_lut(prefix, 'silu_lut', 256)
        generator._register_mamba_lut(prefix, 'softplus_lut', 256)
        generator._register_mamba_lut(prefix, 'exp_lut', 256)

    # Create entry for mamba_block_entries (one for fwd, one for rev)
    for direction in ['fwd', 'rev']:
        prefix = f"{layer_name}_{direction}"
        in_proj_bit_width = generator.weight_entries.get(f"{prefix}::in_proj_weight_bit_width", 8)
        x_proj_bit_width = generator.weight_entries.get(f"{prefix}::x_proj_weight_bit_width", 8)
        dt_proj_bit_width = generator.weight_entries.get(f"{prefix}::dt_proj_weight_bit_width", 8)
        out_proj_bit_width = generator.weight_entries.get(f"{prefix}::out_proj_weight_bit_width", 8)

        in_proj_elements = d_model * 2 * d_inner
        in_proj_packed = (in_proj_elements + 3) // 4 if in_proj_bit_width == 2 else in_proj_elements
        x_proj_elements = (dt_rank + 2 * d_state) * d_inner
        x_proj_packed = (x_proj_elements + 3) // 4 if x_proj_bit_width == 2 else x_proj_elements
        dt_proj_elements = dt_rank * d_inner
        dt_proj_packed = (dt_proj_elements + 3) // 4 if dt_proj_bit_width == 2 else dt_proj_elements
        out_proj_elements = d_inner * d_model
        out_proj_packed = (out_proj_elements + 3) // 4 if out_proj_bit_width == 2 else out_proj_elements

        mamba_entry = {
            'c_name': generator.sanitize_c_name(prefix),
            'layer_name': prefix,
            'direction': direction,
            'parent_wrapper': layer_name,
            'in_proj_weight_index': generator.weight_entries.get(f"{prefix}::in_proj_weight", {}).get('index'),
            'in_proj_weight_elements': in_proj_elements,
            'in_proj_weight_packed_bytes': in_proj_packed,
            'in_proj_bit_width': in_proj_bit_width,
            'conv1d_weight_index': generator.weight_entries.get(f"{prefix}::conv1d_weight", {}).get('index'),
            'conv1d_weight_elements': d_inner * kernel_size,
            'conv1d_bias_index': generator.weight_entries.get(f"{prefix}::conv1d_bias", {}).get('index'),
            'conv1d_bias_elements': d_inner,
            'x_proj_weight_index': generator.weight_entries.get(f"{prefix}::x_proj_weight", {}).get('index'),
            'x_proj_weight_elements': x_proj_elements,
            'x_proj_weight_packed_bytes': x_proj_packed,
            'x_proj_bit_width': x_proj_bit_width,
            'dt_proj_weight_index': generator.weight_entries.get(f"{prefix}::dt_proj_weight", {}).get('index'),
            'dt_proj_weight_elements': dt_proj_elements,
            'dt_proj_weight_packed_bytes': dt_proj_packed,
            'dt_proj_bit_width': dt_proj_bit_width,
            'dt_proj_bias_index': generator.weight_entries.get(f"{prefix}::dt_proj_bias", {}).get('index'),
            'dt_proj_bias_elements': d_inner,
            'A_index': generator.weight_entries.get(f"{prefix}::A", {}).get('index'),
            'A_elements': d_inner * d_state,
            'A_q15_index': generator.weight_entries.get(f"{prefix}::A_q15", {}).get('index'),
            'D_index': generator.weight_entries.get(f"{prefix}::D", {}).get('index'),
            'D_elements': d_inner,
            'D_q15_index': generator.weight_entries.get(f"{prefix}::D_q15", {}).get('index'),
            'out_proj_weight_index': generator.weight_entries.get(f"{prefix}::out_proj_weight", {}).get('index'),
            'out_proj_weight_elements': out_proj_elements,
            'out_proj_weight_packed_bytes': out_proj_packed,
            'out_proj_bit_width': out_proj_bit_width,
            'silu_lut_index': generator.weight_entries.get(f"{prefix}::silu_lut", {}).get('index'),
            'silu_gate_lut_q13_index': generator.weight_entries.get(f"{prefix}::silu_gate_lut_q13", {}).get('index'),
            'softplus_lut_index': generator.weight_entries.get(f"{prefix}::softplus_lut", {}).get('index'),
            'exp_lut_index': generator.weight_entries.get(f"{prefix}::exp_lut", {}).get('index'),
            'dt_proj_bias_q16_16_index': generator.weight_entries.get(f"{prefix}::dt_proj_bias_q16_16", {}).get('index'),
            'dt_scale_q': generator.ssm_dt_scale_q.get(prefix, 0),
            'dt_scale_shift': generator.ssm_dt_scale_shift.get(prefix, 24),
            'silu_lut_elements': 256,
            'l2_scratch_size': scratch_needed,
            'd_model': d_model,
            'd_inner': d_inner,
            'd_state': d_state,
            'dt_rank': dt_rank,
            'kernel_size': kernel_size,
        }
        generator.mamba_block_entries.append(mamba_entry)

    generator._ctx_attach_golden(ctx, spec, layer_name, output_numel, buffer_name)
    ctx.specs.append(spec)

    ctx.current_buffer = buffer_name
    ctx.current_shape = out_shape
    ctx.current_scale = layer_data.get('scale_output', ctx.current_scale)
    ctx.layer_output_buffer[layer_name] = ctx.current_buffer
    ctx.layer_output_scale[layer_name] = ctx.current_scale
    ctx.buffer_scale[ctx.current_buffer] = ctx.current_scale
    return True


