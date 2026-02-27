# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Shared compatibility metadata for ARES usability tooling."""

from __future__ import annotations

from typing import Dict, List, Sequence, Set


# Layer type strings produced by tools/pytorch_extractor.py::BrevitasExtractor.
SUPPORTED_LAYER_TYPES: Sequence[str] = (
    "QuantIdentity",
    "QuantConv2d",
    "QuantReLU",
    "MaxPool2d",
    "AvgPool2d",
    "GlobalAvgPool",
    "AdaptiveAvgPool1d",
    "ZeroPad2d",
    "Squeeze",
    "LayerNorm",
    "RMSNorm",
    "GroupNorm",
    "Embedding",
    "GELU",
    "Flatten",
    "Permute",
    "Reshape",
    "RFFT",
    "QuantLinear",
    "Add",
    "Concatenate",
    "Mean",
    "MultiheadSelfAttention",
    "CrossAttention",
    "AlternatingAttention",
    "Conv1dDepthwise",
    "SiLU",
    "SSM",
    "MambaWrapper",
    "PatchEmbed",
    "MambaBlock",
)


# Extractor intentionally skips submodules for these composite/custom blocks.
COMPOSITE_LAYER_TYPES: Set[str] = {
    "Mean",
    "MultiheadSelfAttention",
    "CrossAttention",
    "AlternatingAttention",
    "SSM",
    "MambaWrapper",
    "PatchEmbed",
    "MambaBlock",
}


# Common migration guidance for unsupported modules.
REPLACEMENT_SUGGESTIONS: Dict[str, str] = {
    "BatchNorm1d": "Use LayerNorm or GroupNorm, or fold BatchNorm into the preceding linear layer.",
    "BatchNorm2d": "Use LayerNorm/GroupNorm or fold BatchNorm into the preceding convolution.",
    "Dropout": "Dropout is a no-op in eval mode. Remove it for cleaner extraction.",
    "MultiheadAttention": "Use QuantMultiHeadAttention from tests.test_networks.brevitas_custom_layers or ares.nn.",
    "Sigmoid": "Prefer ReLU, GELU, or SiLU for ARES-validated INT8 paths.",
    "Hardtanh": "Prefer ReLU, GELU, or SiLU for ARES-validated INT8 paths.",
    "ConvTranspose1d": "ConvTranspose is not currently extractable. Restructure with supported convolutions/upsampling.",
    "ConvTranspose2d": "ConvTranspose is not currently extractable. Restructure with supported convolutions/upsampling.",
    "Conv1d": "Use QuantConv1dDepthwise (custom) or reformulate to supported ops.",
    "Conv3d": "Conv3d is not currently extractable.",
    "LSTM": "Reformulate with supported blocks (Mamba/attention/linear) for ARES extraction.",
    "GRU": "Reformulate with supported blocks (Mamba/attention/linear) for ARES extraction.",
}


# Mapping from extractor layer type to C runtime OpType enum entry.
OPTYPE_MAPPING: Dict[str, str] = {
    "QuantIdentity": "OP_REQUANTIZE",
    "QuantConv2d": "OP_CONV2D",
    "QuantReLU": "OP_RELU",
    "MaxPool2d": "OP_MAXPOOL",
    "AvgPool2d": "OP_AVGPOOL",
    "GlobalAvgPool": "OP_GLOBAL_AVGPOOL",
    "AdaptiveAvgPool1d": "OP_AVGPOOL",  # AdaptiveAvgPool1d currently follows avgpool runtime handling.
    "ZeroPad2d": "OP_ZEROPAD2D",
    "Squeeze": "OP_SQUEEZE",
    "LayerNorm": "OP_LAYERNORM",
    "RMSNorm": "OP_RMSNORM (when ARES_LLAMA_SUPPORT enabled)",
    "GroupNorm": "OP_GROUPNORM",
    "Embedding": "OP_EMBEDDING",
    "GELU": "OP_GELU",
    "Flatten": "OP_FLATTEN",
    "Permute": "OP_TRANSPOSE_2D",
    "Reshape": "OP_RESHAPE",
    "RFFT": "OP_RFFT",
    "QuantLinear": "OP_LINEAR_INT8 / OP_LINEAR_FP32 (depending on output path)",
    "Add": "OP_ADD",
    "Concatenate": "OP_CONCAT",
    "Mean": "OP_MEAN",
    "MultiheadSelfAttention": "OP_MHSA",
    "CrossAttention": "OP_CROSS_ATTENTION",
    "AlternatingAttention": "OP_ALTERNATING_ATTENTION",
    "Conv1dDepthwise": "OP_CONV1D_DEPTHWISE",
    "SiLU": "OP_SILU",
    "SSM": "OP_SSM",
    "MambaWrapper": "OP_MAMBA_WRAPPER",
    "PatchEmbed": "OP_PATCH_EMBED",
    "MambaBlock": "OP_MAMBA_BLOCK",
}


WARNING_RULES: List[Dict[str, str]] = [
    {
        "class_name": "Dropout",
        "reason": "Dropout is a no-op in eval mode and extractor will effectively ignore it.",
        "suggestion": REPLACEMENT_SUGGESTIONS["Dropout"],
    },
    {
        "class_name": "QuantMultiheadAttention",
        "reason": "Brevitas built-in attention may not match ARES custom quantization points.",
        "suggestion": REPLACEMENT_SUGGESTIONS["MultiheadAttention"],
    },
]


def get_supported_layer_types() -> Sequence[str]:
    """Return extractor-supported layer type names."""
    return SUPPORTED_LAYER_TYPES


def get_composite_layer_types() -> Set[str]:
    """Return supported layer types whose children should be skipped."""
    return COMPOSITE_LAYER_TYPES


def get_warning_rules() -> List[Dict[str, str]]:
    """Return warning rules used by the compatibility checker."""
    return WARNING_RULES


def get_replacement_suggestions() -> Dict[str, str]:
    """Return mapping from unsupported class name to migration advice."""
    return REPLACEMENT_SUGGESTIONS


def get_optype_mapping() -> Dict[str, str]:
    """Return mapping from extractor layer types to runtime OpType names."""
    return OPTYPE_MAPPING
