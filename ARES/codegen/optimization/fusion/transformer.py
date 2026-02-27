# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Fusion spec transformer scaffolding."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

LayerSpec = Dict[str, Any]
FusionSpec = Dict[str, Any]


def _transfer_golden_output(dst: LayerSpec, src: LayerSpec) -> None:
    """Copy golden-output metadata from a fused-away layer."""
    if "golden_slot" not in src:
        return

    dst["golden_slot"] = src["golden_slot"]
    dst["golden_buffer"] = src["golden_buffer"]
    if "c_name" in src:
        dst["golden_c_name"] = src["c_name"]


def _transfer_compare_buffer(dst: LayerSpec, src: LayerSpec) -> None:
    if "compare_buffer" in src:
        dst["compare_buffer"] = src["compare_buffer"]


class FusionTransformer:
    """Apply fusion descriptors onto layer specs while tracking skipped indices."""

    def transform(
        self,
        specs: List[LayerSpec],
        fusions: Sequence[FusionSpec],
        fused_layers: Optional[List[str]] = None,
    ) -> Tuple[List[LayerSpec], Set[int], List[str]]:
        tracked_fusions = fused_layers if fused_layers is not None else []
        skip_indices: Set[int] = set()

        for fusion in fusions:
            fusion_type = fusion["type"]
            layer_indices = fusion["layers"]

            if fusion_type == "conv_relu":
                conv_idx, relu_idx = layer_indices
                conv_name = specs[conv_idx]["name"]
                relu_name = specs[relu_idx]["name"]

                specs[conv_idx]["fusion_relu"] = True
                specs[conv_idx]["op"] = "conv2d"

                _transfer_golden_output(specs[conv_idx], specs[relu_idx])
                _transfer_compare_buffer(specs[conv_idx], specs[relu_idx])

                skip_indices.add(relu_idx)
                tracked_fusions.append(f"{conv_name}+{relu_name} (conv_relu)")

            elif fusion_type == "linear_relu":
                linear_idx, relu_idx = layer_indices
                linear_name = specs[linear_idx]["name"]
                relu_name = specs[relu_idx]["name"]

                specs[linear_idx]["fusion_relu"] = True
                relu_scale = specs[relu_idx].get("scale", specs[linear_idx].get("scale_output", 0.0))
                specs[linear_idx]["relu_output_scale"] = relu_scale

                _transfer_golden_output(specs[linear_idx], specs[relu_idx])
                _transfer_compare_buffer(specs[linear_idx], specs[relu_idx])

                skip_indices.add(relu_idx)
                tracked_fusions.append(f"{linear_name}+{relu_name} (linear_relu)")

            elif fusion_type == "pool_quant":
                pool_idx, quant_idx = layer_indices
                pool_spec = specs[pool_idx]

                pool_spec["fusion_quant"] = True
                pool_spec["quant_scale_in"] = specs[quant_idx]["scale_in"]
                pool_spec["quant_scale_out"] = specs[quant_idx]["scale_out"]

                _transfer_golden_output(pool_spec, specs[quant_idx])
                _transfer_compare_buffer(pool_spec, specs[quant_idx])

                skip_indices.add(quant_idx)

                pool_name = pool_spec["name"]
                quant_name = specs[quant_idx]["name"]
                tracked_fusions.append(f"{pool_name}+{quant_name} (pool_quant)")

            elif fusion_type == "conv_relu_quant":
                conv_idx, relu_idx, quant_idx = layer_indices
                conv_name = specs[conv_idx]["name"]
                relu_name = specs[relu_idx]["name"]
                quant_name = specs[quant_idx]["name"]

                specs[conv_idx]["fusion_relu"] = True
                specs[conv_idx]["fusion_quant"] = True
                specs[conv_idx]["quant_scale_in"] = specs[quant_idx]["scale_in"]
                specs[conv_idx]["quant_scale_out"] = specs[quant_idx]["scale_out"]
                specs[conv_idx]["op"] = "conv2d"

                _transfer_golden_output(specs[conv_idx], specs[quant_idx])
                _transfer_compare_buffer(specs[conv_idx], specs[quant_idx])

                skip_indices.add(relu_idx)
                skip_indices.add(quant_idx)
                tracked_fusions.append(f"{conv_name}+{relu_name}+{quant_name} (conv_relu_quant)")

            elif fusion_type == "linear_relu_quant":
                linear_idx, relu_idx, quant_idx = layer_indices
                linear_name = specs[linear_idx]["name"]
                relu_name = specs[relu_idx]["name"]
                quant_name = specs[quant_idx]["name"]

                specs[linear_idx]["fusion_relu"] = True
                specs[linear_idx]["fusion_quant"] = True
                specs[linear_idx]["relu_output_scale"] = specs[relu_idx]["scale"]
                specs[linear_idx]["quant_scale_in"] = specs[quant_idx]["scale_in"]
                specs[linear_idx]["quant_scale_out"] = specs[quant_idx]["scale_out"]

                _transfer_golden_output(specs[linear_idx], specs[quant_idx])
                _transfer_compare_buffer(specs[linear_idx], specs[quant_idx])

                skip_indices.add(relu_idx)
                skip_indices.add(quant_idx)
                tracked_fusions.append(f"{linear_name}+{relu_name}+{quant_name} (linear_relu_quant)")

            elif fusion_type == "conv_relu_maxpool":
                conv_idx, relu_idx, pool_idx = layer_indices
                conv_spec = specs[conv_idx]
                pool_spec = specs[pool_idx]
                conv_name = conv_spec["name"]
                relu_name = specs[relu_idx]["name"]
                pool_name = pool_spec["name"]

                conv_spec["fusion_relu"] = True
                conv_spec["fusion_maxpool"] = True

                conv_spec["pool_kernel_h"] = pool_spec.get("kernel_h", pool_spec.get("kernel_size", 1))
                conv_spec["pool_kernel_w"] = pool_spec.get("kernel_w", pool_spec.get("kernel_size", 1))
                conv_spec["pool_stride_h"] = pool_spec.get("stride_h", pool_spec.get("stride", 1))
                conv_spec["pool_stride_w"] = pool_spec.get("stride_w", pool_spec.get("stride", 1))
                conv_spec["pool_out_h"] = pool_spec.get("out_h", 1)
                conv_spec["pool_out_w"] = pool_spec.get("out_w", 1)
                conv_spec["fused_output_buffer"] = pool_spec.get("output_buffer")
                conv_spec["fused_out_h"] = pool_spec.get("out_h", 1)
                conv_spec["fused_out_w"] = pool_spec.get("out_w", 1)

                if "golden_slot" in pool_spec:
                    conv_spec["golden_slot"] = pool_spec["golden_slot"]
                    conv_spec["golden_buffer"] = pool_spec["golden_buffer"]
                    conv_spec["golden_c_name"] = pool_spec["c_name"]
                    conv_spec["golden_size"] = pool_spec.get("golden_size", pool_spec.get("numel", 0))
                _transfer_compare_buffer(conv_spec, pool_spec)

                skip_indices.add(relu_idx)
                skip_indices.add(pool_idx)
                tracked_fusions.append(f"{conv_name}+{relu_name}+{pool_name} (conv_relu_maxpool)")

        fused_specs = [spec for index, spec in enumerate(specs) if index not in skip_indices]
        return fused_specs, skip_indices, tracked_fusions


def transform_specs_for_fusion(
    specs: List[LayerSpec],
    fusions: Sequence[FusionSpec],
    fused_layers: Optional[List[str]] = None,
) -> Tuple[List[LayerSpec], List[str]]:
    """Convenience wrapper returning transformed specs and tracked fusion labels."""
    transformer = FusionTransformer()
    fused_specs, _skip_indices, tracked = transformer.transform(specs, fusions, fused_layers)
    return fused_specs, tracked

