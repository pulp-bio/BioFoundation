# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Embedding-family layer handlers extracted from generate_c_code."""

from __future__ import annotations

from typing import Any, Callable, Dict

from ..gap9_model import WEIGHT_RESIDENCY_L2, WEIGHT_RESIDENCY_MAMBA_SCRATCH


def handle_embedding(
    generator: Any,
    ctx: Any,
    layer_name: str,
    layer_data: Dict[str, Any],
    spec: Dict[str, Any],
    determine_weight_residency: Callable[..., str],
) -> bool:
    """Handle Embedding lookup using a captured (constant) indices tensor."""
    output_shape = layer_data.get("output_shape")
    if output_shape is None:
        indices_shape = layer_data.get("indices_shape")
        if indices_shape is None:
            raise ValueError(
                f"Embedding {layer_name}: missing output_shape/indices_shape metadata. "
                f"Ensure extractor forward hooks capture embedding indices."
            )
        output_shape = list(indices_shape) + [int(layer_data.get("embedding_dim", 0))]

    if output_shape[0] != 1:
        raise ValueError(f"Embedding {layer_name}: only batch size 1 supported, got {output_shape[0]}")

    vocab_size = int(layer_data.get("num_embeddings", 0))
    embed_dim = int(layer_data.get("embedding_dim", 0))
    if vocab_size <= 0 or embed_dim <= 0:
        raise ValueError(f"Embedding {layer_name}: invalid vocab_size={vocab_size}, embed_dim={embed_dim}")

    output_numel = generator._numel(output_shape)
    if output_numel % embed_dim != 0:
        raise ValueError(
            f"Embedding {layer_name}: output numel {output_numel} not divisible by embed_dim={embed_dim}"
        )
    num_indices = int(output_numel // embed_dim)

    buffer_name = f"{layer_name}_out"
    output_entry = generator._ctx_register_buffer(
        ctx, buffer_name, "int8_t", output_numel, f"{layer_name} output", spec.get("block_id")
    )

    weight_entry = generator.weight_entries.get(layer_name)
    indices_entry = generator.bias_entries.get(layer_name)
    if weight_entry is None or indices_entry is None:
        raise ValueError(
            f"Embedding {layer_name}: missing binaries. "
            f"Expected {layer_name}_weight_int8.npy and {layer_name}_indices_int32.npy in weights dir."
        )

    scale_out = float(layer_data.get("scale", 1.0))

    spec.update(
        {
            "op": "embedding",
            "output_buffer": output_entry["c_name"],
            "num_indices": num_indices,
            "embed_dim": embed_dim,
            "vocab_size": vocab_size,
            "scale_out": scale_out,
            "weight_index": weight_entry["index"],
            "bias_index": indices_entry["index"],
            "weight_elements": vocab_size * embed_dim,
            "bias_elements": num_indices,
            "bias_type": "int32",
            "weight_residency": determine_weight_residency(
                weight_size_bytes=vocab_size * embed_dim,
                layer_type="embedding",
            ),
        }
    )

    generator._ctx_attach_golden(ctx, spec, layer_name, output_numel, buffer_name)
    ctx.specs.append(spec)
    ctx.param_layers.append(spec)

    ctx.current_buffer = buffer_name
    ctx.current_shape = output_shape
    ctx.current_scale = scale_out
    ctx.layer_output_buffer[layer_name] = ctx.current_buffer
    ctx.layer_output_scale[layer_name] = ctx.current_scale
    ctx.buffer_scale[ctx.current_buffer] = ctx.current_scale
    return True


def handle_patchembed(
    generator: Any,
    ctx: Any,
    layer_name: str,
    layer_data: Dict[str, Any],
    spec: Dict[str, Any],
    determine_weight_residency: Callable[..., str],
) -> bool:
    """Handle PatchEmbed: Conv2D + reshape + permute."""
    if len(ctx.current_shape) != 4:
        raise ValueError(f"PatchEmbed {layer_name} expects 4D input [B, C, H, W], got {ctx.current_shape}")

    batch = ctx.current_shape[0]
    in_chans = ctx.current_shape[1]
    inp_h = ctx.current_shape[2]
    inp_w = ctx.current_shape[3]

    patch_size = layer_data.get("patch_size", 2)
    stride = layer_data.get("stride", patch_size)
    embed_dim = layer_data.get("embed_dim", 35)

    if isinstance(patch_size, (list, tuple)):
        patch_h, patch_w = patch_size[0], patch_size[1]
    else:
        patch_h = patch_w = patch_size
    if isinstance(stride, (list, tuple)):
        stride_h, stride_w = stride[0], stride[1]
    else:
        stride_h = stride_w = stride

    grid_h = layer_data.get("grid_h", (inp_h - patch_h) // stride_h + 1)
    grid_w = layer_data.get("grid_w", (inp_w - patch_w) // stride_w + 1)
    seq_len = layer_data.get("seq_len", grid_w)
    d_model = layer_data.get("d_model", grid_h * embed_dim)

    out_shape = [batch, seq_len, d_model]
    output_numel = generator._numel(out_shape)

    buffer_name = f"{layer_name}_out"
    output_entry = generator._ctx_register_buffer(
        ctx, buffer_name, "int8_t", output_numel, f"{layer_name} output", spec.get("block_id")
    )

    scale_input = layer_data.get("scale_input", ctx.current_scale)
    scale_output = layer_data.get("scale_output", ctx.current_scale)
    scale_weight = layer_data.get("proj_weight_scale", 0.001)

    weight_elements = embed_dim * in_chans * patch_h * patch_w
    bias_elements = embed_dim

    spec.update(
        {
            "op": "patch_embed",
            "input_buffer": generator._ctx_buffer_c_name(ctx, ctx.current_buffer),
            "output_buffer": output_entry["c_name"],
            "batch": batch,
            "in_chans": in_chans,
            "inp_h": inp_h,
            "inp_w": inp_w,
            "patch_h": patch_h,
            "patch_w": patch_w,
            "stride_h": stride_h,
            "stride_w": stride_w,
            "embed_dim": embed_dim,
            "grid_h": grid_h,
            "grid_w": grid_w,
            "seq_len": seq_len,
            "d_model": d_model,
            "scale_in": scale_input,
            "scale_out": scale_output,
            "scale_weight": scale_weight,
            "weight_index": generator.weight_entries.get(layer_name, {}).get("index"),
            "bias_index": generator.bias_entries.get(layer_name, {}).get("index"),
            "weight_elements": weight_elements,
            "bias_elements": bias_elements,
            "bias_type": "int32",
            "weight_residency": determine_weight_residency(
                weight_size_bytes=weight_elements,
                layer_type="patch_embed",
            ),
        }
    )

    generator._ctx_attach_golden(ctx, spec, layer_name, output_numel, buffer_name)
    ctx.specs.append(spec)
    ctx.param_layers.append(spec)

    ctx.current_buffer = buffer_name
    ctx.current_shape = out_shape
    ctx.current_scale = scale_output
    ctx.layer_output_buffer[layer_name] = ctx.current_buffer
    ctx.layer_output_scale[layer_name] = ctx.current_scale
    ctx.buffer_scale[ctx.current_buffer] = ctx.current_scale
    return True


def handle_positionalembedding(
    generator: Any,
    ctx: Any,
    layer_name: str,
    layer_data: Dict[str, Any],
    spec: Dict[str, Any],
    has_mamba_layers: bool,
) -> bool:
    """Handle PositionalEmbedding: in-place add of learned position tensor."""
    if len(ctx.current_shape) != 3:
        raise ValueError(
            f"PositionalEmbedding {layer_name} expects 3D input [B, seq_len, d_model], got {ctx.current_shape}"
        )

    batch = ctx.current_shape[0]
    seq_len = ctx.current_shape[1]
    d_model = ctx.current_shape[2]
    output_numel = generator._numel(ctx.current_shape)

    scale_pos = layer_data.get("scale", 1.0)
    scale_equalizer = layer_data.get("scale_equalizer", ctx.current_scale)
    scale_input = ctx.current_scale
    scale_out = scale_equalizer if scale_equalizer else ctx.current_scale

    weight_elements = seq_len * d_model
    uses_mamba_scratch = has_mamba_layers

    spec.update(
        {
            "op": "pos_embed",
            "buffer": generator._ctx_buffer_c_name(ctx, ctx.current_buffer),
            "numel": output_numel,
            "batch": batch,
            "seq_len": seq_len,
            "d_model": d_model,
            "scale_pos": scale_pos,
            "scale_input": scale_input,
            "scale_out": scale_out,
            "weight_index": generator.weight_entries.get(layer_name, {}).get("index"),
            "weight_elements": weight_elements,
            "bias_elements": 0,
            "bias_type": None,
            "weight_residency": WEIGHT_RESIDENCY_MAMBA_SCRATCH if uses_mamba_scratch else WEIGHT_RESIDENCY_L2,
            "uses_mamba_scratch": uses_mamba_scratch,
        }
    )

    generator._ctx_attach_golden(ctx, spec, layer_name, output_numel, ctx.current_buffer)
    ctx.specs.append(spec)
    ctx.param_layers.append(spec)

    ctx.current_scale = scale_out
    ctx.layer_output_buffer[layer_name] = ctx.current_buffer
    ctx.layer_output_scale[layer_name] = ctx.current_scale
    ctx.buffer_scale[ctx.current_buffer] = ctx.current_scale
    return True
