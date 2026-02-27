# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch Extractor for Brevitas Models

Extracts network structure, quantization scales, and weights directly from
Brevitas quantized models. This bypasses ONNX entirely and works directly
with the PyTorch model.

Purpose:
- Extract quantization scales for each layer
- Extract and quantize weights to INT8
- Build a clean layer-by-layer structure for INT8 inference
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

from brevitas import nn as qnn
import re

try:
    from tests.test_networks.brevitas_custom_layers import (
        QuantAdd,
        QuantConcatenate,
        QuantMean,
        QuantSelfAttention,
        QuantMultiHeadAttention,
        QuantRoPESelfAttention,
        QuantCrossAttention,
        QuantCrossAttentionWithSelfRefine,
        QuantClassificationHeadWithMLP,
        QuantAlternatingAttention,
        QuantConv1dDepthwise,
        QuantSiLU,
        QuantSSM,
        QuantMambaBlock,
        QuantMambaWrapper,
        QuantPatchEmbed,
    )
except (ImportError, ModuleNotFoundError):
    try:
        from test_networks.brevitas_custom_layers import (
            QuantAdd,
            QuantConcatenate,
            QuantMean,
            QuantSelfAttention,
            QuantMultiHeadAttention,
            QuantRoPESelfAttention,
            QuantCrossAttention,
            QuantCrossAttentionWithSelfRefine,
            QuantClassificationHeadWithMLP,
            QuantAlternatingAttention,
            QuantConv1dDepthwise,
            QuantSiLU,
            QuantSSM,
            QuantMambaBlock,
            QuantMambaWrapper,
            QuantPatchEmbed,
        )
    except (ImportError, ModuleNotFoundError):
        QuantAdd = None  # type: ignore
        QuantConcatenate = None  # type: ignore
        QuantMean = None  # type: ignore
        QuantSelfAttention = None  # type: ignore
        QuantMultiHeadAttention = None  # type: ignore
        QuantRoPESelfAttention = None  # type: ignore
        QuantCrossAttention = None  # type: ignore
        QuantCrossAttentionWithSelfRefine = None  # type: ignore
        QuantClassificationHeadWithMLP = None  # type: ignore
        QuantAlternatingAttention = None  # type: ignore
        QuantConv1dDepthwise = None  # type: ignore
        QuantSiLU = None  # type: ignore
        QuantSSM = None  # type: ignore
        QuantMambaBlock = None  # type: ignore
        QuantMambaWrapper = None  # type: ignore
        QuantPatchEmbed = None  # type: ignore

# Import QuantMHSA for TinyMyo support
try:
    from tests.test_networks.test_20_tinymyo import QuantMHSA
except (ImportError, ModuleNotFoundError):
    try:
        from test_networks.test_20_tinymyo import QuantMHSA
    except (ImportError, ModuleNotFoundError):
        QuantMHSA = None  # type: ignore


def _extract_block_id(layer_name: str) -> Optional[int]:
    """
    Extract transformer block ID from layer name.

    Patterns recognized:
    - blocks.0.norm1 → block_id=0
    - blocks.3.mlp_fc2 → block_id=3
    - encoder.layers.2.attn → block_id=2
    - transformer.blocks.1.mhsa → block_id=1

    Args:
        layer_name: Full layer name from named_modules()

    Returns:
        Block ID (int) if pattern matches, None otherwise
    """
    # Pattern: blocks.N.* or layers.N.* or transformer.blocks.N.*
    # Use token boundaries so names like "conv_layers.0.0" do not get
    # misclassified as transformer block IDs.
    patterns = [
        r'(?:^|\.)blocks\.(\d+)\.',      # blocks.0.norm1 / classifier.blocks.0.norm1
        r'(?:^|\.)layers\.(\d+)\.',      # layers.2.attn / encoder.layers.2.attn
        r'encoder\.layers\.(\d+)\.', # encoder.layers.3.mlp
    ]

    for pattern in patterns:
        match = re.search(pattern, layer_name)
        if match:
            return int(match.group(1))

    return None


def _is_quant_add(module: nn.Module) -> bool:
    """Check whether a module is our custom QuantAdd."""
    if isinstance(QuantAdd, type) and isinstance(module, QuantAdd):
        return True
    return module.__class__.__name__ == "QuantAdd"


def _is_quant_concatenate(module: nn.Module) -> bool:
    """Check whether a module is our custom QuantConcatenate."""
    if isinstance(QuantConcatenate, type) and isinstance(module, QuantConcatenate):
        return True
    return module.__class__.__name__ == "QuantConcatenate"


def _is_quant_mean(module: nn.Module) -> bool:
    """Check whether a module is our custom QuantMean."""
    if isinstance(QuantMean, type) and isinstance(module, QuantMean):
        return True
    return module.__class__.__name__ == "QuantMean"


def _is_quant_self_attention(module: nn.Module) -> bool:
    if isinstance(QuantSelfAttention, type) and isinstance(module, QuantSelfAttention):
        return True
    return module.__class__.__name__ == "QuantSelfAttention"


def _is_quant_multihead_attention(module: nn.Module) -> bool:
    """Detect QuantMultiHeadAttention or QuantMHSA modules."""
    if isinstance(QuantMultiHeadAttention, type) and isinstance(module, QuantMultiHeadAttention):
        return True
    if isinstance(QuantMHSA, type) and isinstance(module, QuantMHSA):
        return True
    return module.__class__.__name__ in ("QuantMultiHeadAttention", "QuantMHSA")

def _is_quant_rope_self_attention(module: nn.Module) -> bool:
    """Detect RoPE-enabled MHSA modules."""
    if isinstance(QuantRoPESelfAttention, type) and isinstance(module, QuantRoPESelfAttention):
        return True
    return module.__class__.__name__ == "QuantRoPESelfAttention"

def _is_quant_cross_attention(module: nn.Module) -> bool:
    """Detect QuantCrossAttention modules."""
    if isinstance(QuantCrossAttention, type) and isinstance(module, QuantCrossAttention):
        return True
    return module.__class__.__name__ == "QuantCrossAttention"

def _is_quant_cross_attention_with_self_refine(module: nn.Module) -> bool:
    """Detect QuantCrossAttentionWithSelfRefine modules."""
    if isinstance(QuantCrossAttentionWithSelfRefine, type) and isinstance(module, QuantCrossAttentionWithSelfRefine):
        return True
    return module.__class__.__name__ == "QuantCrossAttentionWithSelfRefine"

def _is_quant_classification_head_with_mlp(module: nn.Module) -> bool:
    """Detect QuantClassificationHeadWithMLP modules."""
    if isinstance(QuantClassificationHeadWithMLP, type) and isinstance(module, QuantClassificationHeadWithMLP):
        return True
    return module.__class__.__name__ == "QuantClassificationHeadWithMLP"


def _is_quant_alternating_attention(module: nn.Module) -> bool:
    """Detect QuantAlternatingAttention modules (Cerebro transformer)."""
    if isinstance(QuantAlternatingAttention, type) and isinstance(module, QuantAlternatingAttention):
        return True
    return module.__class__.__name__ == "QuantAlternatingAttention"


def _is_quant_conv1d_depthwise(module: nn.Module) -> bool:
    """Check whether a module is our custom QuantConv1dDepthwise."""
    if isinstance(QuantConv1dDepthwise, type) and isinstance(module, QuantConv1dDepthwise):
        return True
    return module.__class__.__name__ == "QuantConv1dDepthwise"


def _is_quant_silu(module: nn.Module) -> bool:
    """Check whether a module is our custom QuantSiLU."""
    if isinstance(QuantSiLU, type) and isinstance(module, QuantSiLU):
        return True
    return module.__class__.__name__ == "QuantSiLU"


def _is_quant_ssm(module: nn.Module) -> bool:
    """Check whether a module is our custom QuantSSM."""
    if isinstance(QuantSSM, type) and isinstance(module, QuantSSM):
        return True
    return module.__class__.__name__ == "QuantSSM"


def _is_quant_mamba_block(module: nn.Module) -> bool:
    """Check whether a module is our custom QuantMambaBlock."""
    if isinstance(QuantMambaBlock, type) and isinstance(module, QuantMambaBlock):
        return True
    return module.__class__.__name__ == "QuantMambaBlock"


def _is_quant_mamba_wrapper(module: nn.Module) -> bool:
    """Check whether a module is our custom QuantMambaWrapper (bidirectional)."""
    if isinstance(QuantMambaWrapper, type) and isinstance(module, QuantMambaWrapper):
        return True
    return module.__class__.__name__ == "QuantMambaWrapper"


def _is_quant_patch_embed(module: nn.Module) -> bool:
    """Check whether a module is our custom QuantPatchEmbed."""
    if isinstance(QuantPatchEmbed, type) and isinstance(module, QuantPatchEmbed):
        return True
    return module.__class__.__name__ == "QuantPatchEmbed"




class BrevitasExtractor:
    """Extract network structure and quantization info from Brevitas model."""

    def __init__(self, model: nn.Module):
        """
        Initialize extractor with a Brevitas model.

        Args:
            model: Brevitas quantized model (nn.Module)
        """
        self.model = model
        self.model.eval()
        self.layer_info = {}
        self._last_activation_scale: Optional[float] = None
        self._skip_module_prefixes = set()

    def extract_all(self, sample_input=None) -> Dict[str, Any]:
        """
        Extract complete network information.

        Args:
            sample_input: Either a single tensor or a tuple/list of tensors for multi-input models

        Returns:
            Dictionary with layer-by-layer information including:
            - Layer types
            - Quantization scales
            - Weights (INT8)
            - Biases (INT32)
            - Layer parameters (kernel_size, stride, etc.)
        """
        print("="*80)
        print("Extracting Network Information from Brevitas Model")
        print("="*80)

        # Iterate through all named modules
        for name, module in self.model.named_modules():
            if name == '':  # Skip root module
                continue
            skip = False
            for prefix in self._skip_module_prefixes:
                if name.startswith(prefix):
                    skip = True
                    break
            if skip:
                continue

            layer_data = {}
            layer_data['name'] = name

            # Extract transformer block ID for block-level weight staging
            block_id = _extract_block_id(name)
            if block_id is not None:
                layer_data['block_id'] = block_id

            if isinstance(module, qnn.QuantIdentity):
                layer_data['type'] = 'QuantIdentity'
                layer_data['scale'] = self._extract_activation_scale(module)
                print(f"  {name}: QuantIdentity, scale={layer_data['scale']:.6f}")
                self._last_activation_scale = layer_data['scale']

            elif isinstance(module, qnn.QuantConv2d):
                layer_data['type'] = 'QuantConv2d'
                layer_data['scale_input'] = self._last_activation_scale or 1.0
                layer_data.update(self._extract_conv2d_info(module))
                groups = layer_data.get('groups', 1)
                groups_str = f", groups={groups}" if groups > 1 else ""
                dw_str = " [DEPTHWISE]" if groups == layer_data['in_channels'] and groups > 1 else ""
                print(f"  {name}: QuantConv2d({layer_data['in_channels']}→{layer_data['out_channels']}, "
                      f"k={layer_data['kernel_size']}{groups_str}, scale_w={layer_data['scale_weight']:.6f}){dw_str}")

            elif isinstance(module, qnn.QuantReLU):
                layer_data['type'] = 'QuantReLU'
                layer_data['scale'] = self._extract_activation_scale(module)
                print(f"  {name}: QuantReLU, scale={layer_data['scale']:.6f}")
                self._last_activation_scale = layer_data['scale']

            elif isinstance(module, nn.MaxPool2d):
                layer_data['type'] = 'MaxPool2d'
                layer_data['kernel_size'] = module.kernel_size
                layer_data['stride'] = module.stride if module.stride is not None else module.kernel_size
                layer_data['padding'] = module.padding
                if isinstance(layer_data['kernel_size'], int):
                    layer_data['kernel_size'] = (layer_data['kernel_size'], layer_data['kernel_size'])
                if isinstance(layer_data['stride'], int):
                    layer_data['stride'] = (layer_data['stride'], layer_data['stride'])
                if isinstance(layer_data['padding'], int):
                    layer_data['padding'] = (layer_data['padding'], layer_data['padding'])
                print(f"  {name}: MaxPool2d, kernel={layer_data['kernel_size']}, stride={layer_data['stride']}")

            elif isinstance(module, nn.AvgPool2d):
                layer_data['type'] = 'AvgPool2d'
                layer_data['kernel_size'] = module.kernel_size
                layer_data['stride'] = module.stride if module.stride is not None else module.kernel_size
                layer_data['padding'] = module.padding
                if isinstance(layer_data['kernel_size'], int):
                    layer_data['kernel_size'] = (layer_data['kernel_size'], layer_data['kernel_size'])
                if isinstance(layer_data['stride'], int):
                    layer_data['stride'] = (layer_data['stride'], layer_data['stride'])
                if isinstance(layer_data['padding'], int):
                    layer_data['padding'] = (layer_data['padding'], layer_data['padding'])
                print(f"  {name}: AvgPool2d, kernel={layer_data['kernel_size']}, stride={layer_data['stride']}")

            elif isinstance(module, (nn.AdaptiveAvgPool2d,)):
                layer_data['type'] = 'GlobalAvgPool'
                layer_data['output_size'] = module.output_size if hasattr(module, 'output_size') else (1, 1)
                print(f"  {name}: GlobalAvgPool (AdaptiveAvgPool2d)")

            elif isinstance(module, nn.AdaptiveAvgPool1d):
                layer_data['type'] = 'AdaptiveAvgPool1d'
                layer_data['output_size'] = module.output_size
                print(f"  {name}: AdaptiveAvgPool1d(output_size={module.output_size})")

            elif isinstance(module, nn.ZeroPad2d):
                layer_data['type'] = 'ZeroPad2d'
                # ZeroPad2d padding is (left, right, top, bottom)
                padding = module.padding
                if isinstance(padding, int):
                    padding = (padding, padding, padding, padding)
                elif len(padding) == 2:
                    padding = (padding[0], padding[0], padding[1], padding[1])
                layer_data['padding'] = padding  # (left, right, top, bottom)
                print(f"  {name}: ZeroPad2d(padding={padding})")

            elif module.__class__.__name__ == 'Squeeze':
                layer_data['type'] = 'Squeeze'
                layer_data['dim'] = module.dim
                print(f"  {name}: Squeeze(dim={module.dim})")

            elif isinstance(module, nn.LayerNorm):
                layer_data['type'] = 'LayerNorm'
                layer_data['normalized_shape'] = module.normalized_shape
                layer_data['eps'] = module.eps
                layer_data['scale_input'] = self._last_activation_scale or 1.0
                # Extract weight and bias
                if module.weight is not None:
                    layer_data['weight'] = module.weight.detach().cpu().numpy().astype(np.float32)
                else:
                    layer_data['weight'] = np.ones(module.normalized_shape, dtype=np.float32)
                if module.bias is not None:
                    layer_data['bias'] = module.bias.detach().cpu().numpy().astype(np.float32)
                else:
                    layer_data['bias'] = np.zeros(module.normalized_shape, dtype=np.float32)
                print(f"  {name}: LayerNorm(normalized_shape={layer_data['normalized_shape']}, eps={layer_data['eps']})")

            elif hasattr(nn, 'RMSNorm') and isinstance(module, nn.RMSNorm) or module.__class__.__name__ == 'RMSNorm':
                # RMSNorm (used in Llama and other LLMs)
                layer_data['type'] = 'RMSNorm'
                # Get normalized_shape - handle different attribute names
                if hasattr(module, 'normalized_shape'):
                    layer_data['normalized_shape'] = module.normalized_shape
                elif hasattr(module, 'weight'):
                    layer_data['normalized_shape'] = module.weight.shape[0]
                else:
                    raise ValueError(f"Cannot determine normalized_shape for RMSNorm module {name}")
                layer_data['eps'] = getattr(module, 'eps', 1e-5)
                layer_data['scale_input'] = self._last_activation_scale or 1.0
                # Extract weight (gamma) - RMSNorm has no bias
                if hasattr(module, 'weight') and module.weight is not None:
                    layer_data['weight'] = module.weight.detach().cpu().numpy().astype(np.float32)
                else:
                    ns = layer_data['normalized_shape']
                    if isinstance(ns, (tuple, list)):
                        ns = ns[-1]
                    layer_data['weight'] = np.ones((ns,), dtype=np.float32)
                print(f"  {name}: RMSNorm(normalized_shape={layer_data['normalized_shape']}, eps={layer_data['eps']})")

            elif isinstance(module, nn.GroupNorm):
                layer_data['type'] = 'GroupNorm'
                layer_data['num_groups'] = int(module.num_groups)
                layer_data['num_channels'] = int(module.num_channels)
                layer_data['eps'] = float(module.eps)
                layer_data['scale_input'] = self._last_activation_scale or 1.0

                # Extract weight and bias (gamma/beta)
                if module.weight is not None:
                    layer_data['weight'] = module.weight.detach().cpu().numpy().astype(np.float32)
                else:
                    layer_data['weight'] = np.ones((module.num_channels,), dtype=np.float32)
                if module.bias is not None:
                    layer_data['bias'] = module.bias.detach().cpu().numpy().astype(np.float32)
                else:
                    layer_data['bias'] = np.zeros((module.num_channels,), dtype=np.float32)

                print(f"  {name}: GroupNorm(groups={layer_data['num_groups']}, channels={layer_data['num_channels']}, eps={layer_data['eps']})")

            elif isinstance(module, nn.Embedding):
                layer_data['type'] = 'Embedding'
                layer_data['num_embeddings'] = int(module.num_embeddings)
                layer_data['embedding_dim'] = int(module.embedding_dim)

                weight_fp32 = module.weight.detach().cpu().numpy().astype(np.float32)
                max_abs = float(np.max(np.abs(weight_fp32))) if weight_fp32.size else 0.0
                scale = max_abs / 127.0 if max_abs > 0.0 else 1.0

                layer_data['scale'] = float(scale)
                layer_data['weight_int8'] = self._quantize_to_int8(weight_fp32, scale)

                # Indices are captured during the sample forward pass via forward hooks
                print(f"  {name}: Embedding(num_embeddings={layer_data['num_embeddings']}, "
                      f"embedding_dim={layer_data['embedding_dim']}, scale={layer_data['scale']:.6f})")

            elif isinstance(module, nn.GELU):
                layer_data['type'] = 'GELU'
                layer_data['scale_input'] = self._last_activation_scale or 1.0
                # GELU doesn't change scale, preserve input scale
                # (but QuantIdentity after GELU will set the actual output scale)
                print(f"  {name}: GELU")

            elif isinstance(module, nn.Flatten):
                layer_data['type'] = 'Flatten'
                layer_data['start_dim'] = module.start_dim
                print(f"  {name}: Flatten, start_dim={layer_data['start_dim']}")

            elif module.__class__.__name__ == 'Permute':
                layer_data['type'] = 'Permute'
                layer_data['dims'] = module.dims
                print(f"  {name}: Permute{module.dims}")

            elif module.__class__.__name__ == 'Reshape':
                layer_data['type'] = 'Reshape'
                layer_data['shape'] = module.shape
                print(f"  {name}: Reshape{module.shape}")

            elif module.__class__.__name__ in ('RFFT', 'RFFTFeatures', 'RFFTFeature'):
                layer_data['type'] = 'RFFT'
                patch_size = getattr(module, 'patch_size', None)
                if patch_size is None:
                    patch_size = getattr(module, 'n_fft', None)
                layer_data['patch_size'] = int(patch_size) if patch_size is not None else 40
                layer_data['scale_input'] = self._last_activation_scale or 1.0
                print(f"  {name}: RFFT(patch_size={layer_data['patch_size']})")

            elif isinstance(module, qnn.QuantLinear):
                layer_data['type'] = 'QuantLinear'
                layer_data['scale_input'] = self._last_activation_scale or 1.0
                layer_data.update(self._extract_linear_info(module))
                print(f"  {name}: QuantLinear({layer_data['in_features']}→{layer_data['out_features']}, "
                      f"scale_w={layer_data['scale_weight']:.6f})")

            elif _is_quant_add(module):
                layer_data['type'] = 'Add'
                layer_data['inputs'] = []
                layer_data['scale_output'] = self._extract_activation_scale(module.quant)
                print(f"  {name}: QuantAdd")
                self._last_activation_scale = layer_data['scale_output']

            elif _is_quant_concatenate(module):
                layer_data['type'] = 'Concatenate'
                layer_data['inputs'] = []
                layer_data['dim'] = getattr(module, 'dim', 1)
                layer_data['scale_output'] = self._extract_activation_scale(module.quant)
                print(f"  {name}: QuantConcatenate, dim={layer_data['dim']}")
                self._last_activation_scale = layer_data['scale_output']

            elif _is_quant_mean(module):
                layer_data['type'] = 'Mean'
                layer_data['scale_input'] = self._last_activation_scale or 1.0
                layer_data['dim'] = getattr(module, 'dim', 1)
                layer_data['keepdim'] = getattr(module, 'keepdim', False)
                layer_data['scale_output'] = self._extract_activation_scale(module.quant)
                print(f"  {name}: QuantMean, dim={layer_data['dim']}, keepdim={layer_data['keepdim']}")
                self._last_activation_scale = layer_data['scale_output']
                # Skip the inner .quant module
                self._skip_module_prefixes.add(f"{name}.")

            elif _is_quant_self_attention(module) or _is_quant_multihead_attention(module) or _is_quant_rope_self_attention(module):
                layer_data['type'] = 'MultiheadSelfAttention'
                layer_data['scale_input'] = self._last_activation_scale or 1.0
                # Handle both 'dim' and 'embed_dim' attributes
                layer_data['embed_dim'] = getattr(module, 'embed_dim', None) or getattr(module, 'dim', None)
                layer_data['num_heads'] = getattr(module, 'num_heads', 1)
                layer_data['head_dim'] = getattr(module, 'head_dim', None)
                layer_data['sequence_length'] = getattr(module, 'seq_len', None)
                layer_data['pool_sequence'] = getattr(module, 'pool_sequence', 'none')
                layer_data['use_rope'] = _is_quant_rope_self_attention(module) or bool(getattr(module, 'use_rope', False))
                if layer_data['use_rope']:
                    layer_data['rope_base'] = float(getattr(module, 'rope_base', 10000.0))
                # Calculate softmax_scale if not provided (1/sqrt(head_dim))
                softmax_scale = getattr(module, 'softmax_scale', None)
                if softmax_scale is None and layer_data['head_dim'] is not None:
                    softmax_scale = float(layer_data['head_dim'] ** -0.5)
                layer_data['softmax_scale'] = softmax_scale
                attn_info = self._extract_attention_info(module)
                layer_data.update(attn_info)
                # Handle both 'output_quant' and 'out_quant' attributes
                output_quant = getattr(module, 'output_quant', None) or getattr(module, 'out_quant', None)
                layer_data['scale_output'] = self._extract_activation_scale(output_quant)
                rope_str = ", rope" if layer_data.get('use_rope') else ""
                print(f"  {name}: MultiHeadAttention(embed_dim={layer_data['embed_dim']}, "
                      f"heads={layer_data['num_heads']}{rope_str})")
                self._last_activation_scale = layer_data['scale_output']
                self._skip_module_prefixes.add(f"{name}.")

            elif _is_quant_cross_attention_with_self_refine(module):
                layer_data['type'] = 'CrossAttentionWithSelfRefine'
                layer_data['scale_input'] = self._last_activation_scale or 1.0
                layer_data['embed_dim'] = module.embed_dim
                layer_data['num_heads'] = module.num_heads
                layer_data['head_dim'] = module.head_dim
                layer_data['num_queries'] = module.num_queries
                layer_data['ff_dim'] = module.ff_dim
                layer_data['kv_len'] = getattr(module, 'kv_len', None)
                layer_data['softmax_scale'] = module.softmax_scale

                # Learned query embedding
                query_quant = getattr(module, 'query_quant', None)
                layer_data['query_scale'] = self._extract_activation_scale(query_quant) if query_quant else 1.0
                query_embed_fp32 = module.query_embed.detach().cpu().numpy().astype(np.float32)
                if query_embed_fp32.ndim == 3 and query_embed_fp32.shape[0] == 1:
                    query_embed_fp32 = query_embed_fp32[0]
                layer_data['query_embed_int8'] = self._quantize_to_int8(query_embed_fp32, layer_data['query_scale'])

                # Layer norms (queries, keys, values)
                ln_quant_map = {
                    'queries_norm': 'post_qnorm_quant',
                    'keys_norm': 'post_knorm_quant',
                    'values_norm': 'post_vnorm_quant',
                }
                for ln_name, quant_name in ln_quant_map.items():
                    ln = getattr(module, ln_name)
                    layer_data[f'{ln_name}_weight'] = ln.weight.detach().cpu().numpy().astype(np.float32)
                    layer_data[f'{ln_name}_bias'] = ln.bias.detach().cpu().numpy().astype(np.float32)
                    post_quant = getattr(module, quant_name)
                    layer_data[f'{ln_name}_scale_output'] = self._extract_activation_scale(post_quant)

                # Cross-attention projections (q, k, v, cross_out)
                for prefix, proj_name, quant_name in [
                    ('q', 'q_proj', 'q_quant'),
                    ('k', 'k_proj', 'k_quant'),
                    ('v', 'v_proj', 'v_quant'),
                    ('out', 'cross_out_proj', 'cross_out_quant'),
                ]:
                    proj = getattr(module, proj_name)
                    proj_info = self._extract_linear_info(proj)
                    layer_data[f'{prefix}_weight_int8'] = proj_info['weight_int8']
                    layer_data[f'{prefix}_bias_fp32'] = proj_info['bias_fp32']
                    layer_data[f'{prefix}_scale_weight'] = proj_info['scale_weight']
                    layer_data[f'{prefix}_in_features'] = proj_info['in_features']
                    layer_data[f'{prefix}_out_features'] = proj_info['out_features']
                    quant = getattr(module, quant_name)
                    layer_data[f'{prefix}_scale_output'] = self._extract_activation_scale(quant)

                # FFN after cross-attention
                ffn_fc1_info = self._extract_linear_info(module.ffn_fc1)
                for k, v in ffn_fc1_info.items():
                    layer_data[f'ffn_fc1_{k}'] = v
                layer_data['ffn_gelu_scale'] = self._extract_activation_scale(module.post_ffn_gelu_quant)
                ffn_fc2_info = self._extract_linear_info(module.ffn_fc2)
                for k, v in ffn_fc2_info.items():
                    layer_data[f'ffn_fc2_{k}'] = v
                layer_data['ffn_add_scale'] = self._extract_activation_scale(module.ffn_add.quant)

                # 3x self-attention refinement blocks
                layer_data['num_self_attn_blocks'] = len(module.self_attn_blocks)
                for i, block in enumerate(module.self_attn_blocks):
                    pfx = f'sa{i}'

                    # Norm1
                    layer_data[f'{pfx}_norm1_weight'] = block['norm1'].weight.detach().cpu().numpy().astype(np.float32)
                    layer_data[f'{pfx}_norm1_bias'] = block['norm1'].bias.detach().cpu().numpy().astype(np.float32)
                    layer_data[f'{pfx}_norm1_scale_output'] = self._extract_activation_scale(block['post_norm1_quant'])

                    # Self-attention projections
                    for proj_prefix, proj_key, quant_key in [
                        ('q', 'q_proj', 'q_quant'),
                        ('k', 'k_proj', 'k_quant'),
                        ('v', 'v_proj', 'v_quant'),
                        ('out', 'out_proj', 'out_quant'),
                    ]:
                        proj = block[proj_key]
                        proj_info = self._extract_linear_info(proj)
                        layer_data[f'{pfx}_{proj_prefix}_weight_int8'] = proj_info['weight_int8']
                        layer_data[f'{pfx}_{proj_prefix}_bias_fp32'] = proj_info['bias_fp32']
                        layer_data[f'{pfx}_{proj_prefix}_scale_weight'] = proj_info['scale_weight']
                        layer_data[f'{pfx}_{proj_prefix}_in_features'] = proj_info['in_features']
                        layer_data[f'{pfx}_{proj_prefix}_out_features'] = proj_info['out_features']
                        quant = block[quant_key]
                        layer_data[f'{pfx}_{proj_prefix}_scale_output'] = self._extract_activation_scale(quant)

                    layer_data[f'{pfx}_add1_scale'] = self._extract_activation_scale(block['add1'].quant)

                    # Norm2
                    layer_data[f'{pfx}_norm2_weight'] = block['norm2'].weight.detach().cpu().numpy().astype(np.float32)
                    layer_data[f'{pfx}_norm2_bias'] = block['norm2'].bias.detach().cpu().numpy().astype(np.float32)
                    layer_data[f'{pfx}_norm2_scale_output'] = self._extract_activation_scale(block['post_norm2_quant'])

                    # MLP
                    mlp_fc1_info = self._extract_linear_info(block['mlp_fc1'])
                    for k, v in mlp_fc1_info.items():
                        layer_data[f'{pfx}_mlp_fc1_{k}'] = v
                    layer_data[f'{pfx}_mlp_gelu_scale'] = self._extract_activation_scale(block['post_mlp_gelu_quant'])
                    mlp_fc2_info = self._extract_linear_info(block['mlp_fc2'])
                    for k, v in mlp_fc2_info.items():
                        layer_data[f'{pfx}_mlp_fc2_{k}'] = v
                    layer_data[f'{pfx}_add2_scale'] = self._extract_activation_scale(block['add2'].quant)

                layer_data['scale_output'] = self._extract_activation_scale(module.output_quant)
                print(f"  {name}: CrossAttentionWithSelfRefine(embed_dim={layer_data['embed_dim']}, "
                      f"heads={layer_data['num_heads']}, queries={layer_data['num_queries']}, "
                      f"sa_blocks={layer_data['num_self_attn_blocks']})")
                self._last_activation_scale = layer_data['scale_output']
                self._skip_module_prefixes.add(f"{name}.")

            elif _is_quant_classification_head_with_mlp(module):
                layer_data['type'] = 'ClassificationHeadWithMLP'
                layer_data['scale_input'] = self._last_activation_scale or 1.0
                layer_data['embed_dim'] = module.embed_dim
                layer_data['num_queries'] = module.num_queries
                layer_data['hidden_dim'] = module.hidden_dim
                layer_data['num_heads'] = module.num_heads
                layer_data['head_dim'] = module.head_dim
                layer_data['num_classes'] = module.num_classes
                layer_data['softmax_scale'] = module.softmax_scale

                # Learned aggregation query
                agg_quant = getattr(module, 'agg_quant', None)
                layer_data['agg_scale'] = self._extract_activation_scale(agg_quant) if agg_quant else 1.0
                agg_fp32 = module.learned_agg.detach().cpu().numpy().astype(np.float32)
                if agg_fp32.ndim == 3 and agg_fp32.shape[0] == 1:
                    agg_fp32 = agg_fp32[0]
                layer_data['learned_agg_int8'] = self._quantize_to_int8(agg_fp32, layer_data['agg_scale'])

                # Cross-attention projections for pooling
                for prefix, proj_name, quant_name in [
                    ('q', 'q_proj', 'q_quant'),
                    ('k', 'k_proj', 'k_quant'),
                    ('v', 'v_proj', 'v_quant'),
                    ('out', 'attn_out_proj', 'attn_out_quant'),
                ]:
                    proj = getattr(module, proj_name)
                    proj_info = self._extract_linear_info(proj)
                    layer_data[f'{prefix}_weight_int8'] = proj_info['weight_int8']
                    layer_data[f'{prefix}_bias_fp32'] = proj_info['bias_fp32']
                    layer_data[f'{prefix}_scale_weight'] = proj_info['scale_weight']
                    layer_data[f'{prefix}_in_features'] = proj_info['in_features']
                    layer_data[f'{prefix}_out_features'] = proj_info['out_features']
                    quant = getattr(module, quant_name)
                    layer_data[f'{prefix}_scale_output'] = self._extract_activation_scale(quant)

                # MLP classifier
                mlp_fc1_info = self._extract_linear_info(module.mlp_fc1)
                for k, v in mlp_fc1_info.items():
                    layer_data[f'mlp_fc1_{k}'] = v
                layer_data['mlp_gelu_scale'] = self._extract_activation_scale(module.post_mlp_gelu_quant)
                mlp_fc2_info = self._extract_linear_info(module.mlp_fc2)
                for k, v in mlp_fc2_info.items():
                    layer_data[f'mlp_fc2_{k}'] = v

                layer_data['scale_output'] = layer_data.get('mlp_fc2_scale_weight', 1.0)
                print(f"  {name}: ClassificationHeadWithMLP(hidden_dim={layer_data['hidden_dim']}, "
                      f"heads={layer_data['num_heads']}, classes={layer_data['num_classes']})")
                self._last_activation_scale = layer_data['scale_output']
                self._skip_module_prefixes.add(f"{name}.")

            elif _is_quant_cross_attention(module):
                layer_data['type'] = 'CrossAttention'
                layer_data['scale_input'] = self._last_activation_scale or 1.0
                layer_data['embed_dim'] = getattr(module, 'embed_dim', None) or getattr(module, 'dim', None)
                layer_data['num_heads'] = getattr(module, 'num_heads', 1)
                layer_data['head_dim'] = getattr(module, 'head_dim', None)
                layer_data['num_queries'] = getattr(module, 'num_queries', None)
                layer_data['kv_len'] = getattr(module, 'kv_len', None)

                # Learned query embedding table: export quantized INT8 + its input scale.
                query_quant = getattr(module, 'query_quant', None)
                layer_data['query_scale'] = self._extract_activation_scale(query_quant) if query_quant is not None else 1.0
                query_embed_fp32 = module.query_embed.detach().cpu().numpy().astype(np.float32)
                if query_embed_fp32.ndim == 3 and query_embed_fp32.shape[0] == 1:
                    query_embed_fp32 = query_embed_fp32[0]
                layer_data['query_embed_int8'] = self._quantize_to_int8(query_embed_fp32, layer_data['query_scale'])

                softmax_scale = getattr(module, 'softmax_scale', None)
                if softmax_scale is None and layer_data['head_dim'] is not None:
                    softmax_scale = float(layer_data['head_dim'] ** -0.5)
                layer_data['softmax_scale'] = softmax_scale

                layer_data.update(self._extract_attention_info(module))
                layer_data['scale_output'] = self._extract_activation_scale(module.output_quant)
                print(f"  {name}: CrossAttention(embed_dim={layer_data['embed_dim']}, "
                      f"heads={layer_data['num_heads']}, queries={layer_data.get('num_queries')})")
                self._last_activation_scale = layer_data['scale_output']
                self._skip_module_prefixes.add(f"{name}.")

            elif _is_quant_alternating_attention(module):
                layer_data['type'] = 'AlternatingAttention'
                layer_data['scale_input'] = self._last_activation_scale or 1.0
                layer_data['embed_dim'] = getattr(module, 'embed_dim', None)
                layer_data['num_heads'] = getattr(module, 'num_heads', 1)
                layer_data['head_dim'] = getattr(module, 'head_dim', None)
                layer_data['num_channels'] = getattr(module, 'num_channels', None)
                layer_data['temporal_len'] = getattr(module, 'temporal_len', None)
                layer_data['block_idx'] = getattr(module, 'block_idx', 0)
                layer_data['scaling_factor'] = getattr(module, 'scaling_factor', None)

                # Calculate softmax_scale if not provided (1/sqrt(head_dim))
                softmax_scale = getattr(module, 'softmax_scale', None)
                if softmax_scale is None and layer_data['head_dim'] is not None:
                    softmax_scale = float(layer_data['head_dim'] ** -0.5)
                layer_data['softmax_scale'] = softmax_scale

                # Extract QKV and output projection info
                layer_data.update(self._extract_alternating_attention_info(module))
                layer_data['scale_output'] = self._extract_activation_scale(module.output_quant)

                attn_type = "channel" if layer_data['block_idx'] % 2 == 0 else "temporal"
                print(f"  {name}: AlternatingAttention(embed_dim={layer_data['embed_dim']}, "
                      f"heads={layer_data['num_heads']}, block={layer_data['block_idx']} ({attn_type}))")
                self._last_activation_scale = layer_data['scale_output']
                self._skip_module_prefixes.add(f"{name}.")

            elif _is_quant_conv1d_depthwise(module):
                layer_data['type'] = 'Conv1dDepthwise'
                layer_data['scale_input'] = self._last_activation_scale or 1.0
                layer_data.update(self._extract_conv1d_depthwise_info(module))
                print(f"  {name}: Conv1dDepthwise(channels={layer_data['channels']}, "
                      f"k={layer_data['kernel_size']}, causal={layer_data['causal']})")

            elif _is_quant_silu(module):
                layer_data['type'] = 'SiLU'
                layer_data['scale_input'] = self._last_activation_scale or 1.0
                layer_data['scale_output'] = self._extract_activation_scale(module.output_quant)
                print(f"  {name}: SiLU(scale_in={layer_data['scale_input']:.6f}, "
                      f"scale_out={layer_data['scale_output']:.6f})")
                self._last_activation_scale = layer_data['scale_output']

            elif _is_quant_ssm(module):
                layer_data['type'] = 'SSM'
                layer_data['scale_input'] = self._last_activation_scale or 1.0
                layer_data.update(self._extract_ssm_info(module))
                print(f"  {name}: SSM(d_inner={layer_data['d_inner']}, "
                      f"d_state={layer_data['d_state']})")
                self._last_activation_scale = layer_data['scale_output']
                # Skip submodules of SSM (they're handled internally)
                self._skip_module_prefixes.add(f"{name}.")

            elif _is_quant_mamba_wrapper(module):
                layer_data['type'] = 'MambaWrapper'
                layer_data['scale_input'] = self._last_activation_scale or 1.0
                layer_data.update(self._extract_mamba_wrapper_info(module))
                print(f"  {name}: MambaWrapper(d_model={layer_data['d_model']}, "
                      f"d_inner={layer_data['d_inner']}, d_state={layer_data['d_state']}, "
                      f"strategy={layer_data['bidirectional_strategy']})")
                self._last_activation_scale = layer_data['scale_output']
                # Skip all submodules of MambaWrapper (handled as composite)
                self._skip_module_prefixes.add(f"{name}.")

            elif _is_quant_patch_embed(module):
                layer_data['type'] = 'PatchEmbed'
                layer_data['scale_input'] = self._last_activation_scale or 1.0
                layer_data.update(self._extract_patch_embed_info(module))
                print(f"  {name}: PatchEmbed(inp={layer_data['inp_size']}, "
                      f"patch={layer_data['patch_size']}, embed_dim={layer_data['embed_dim']}, "
                      f"out=[{layer_data['seq_len']}, {layer_data['d_model']}])")
                self._last_activation_scale = layer_data['scale_output']
                # Skip all submodules of PatchEmbed (handled as composite)
                self._skip_module_prefixes.add(f"{name}.")


            elif _is_quant_mamba_block(module):
                layer_data['type'] = 'MambaBlock'
                layer_data['scale_input'] = self._last_activation_scale or 1.0
                layer_data.update(self._extract_mamba_block_info(module))
                print(f"  {name}: MambaBlock(d_model={layer_data['d_model']}, "
                      f"d_inner={layer_data['d_inner']}, d_state={layer_data['d_state']})")
                self._last_activation_scale = layer_data['scale_output']
                # Skip all submodules of MambaBlock (handled as composite)
                self._skip_module_prefixes.add(f"{name}.")

            else:
                # Skip other module types (Sequential, etc.)
                continue

            self.layer_info[name] = layer_data

        # Extract positional embeddings (nn.Parameter instances)
        self._extract_positional_embeddings()

        print("="*80)
        print(f"[PASS] Extracted {len(self.layer_info)} layers")
        print("="*80)

        self.layer_order = list(self.layer_info.keys())
        self._capture_output_shapes(sample_input)
        self._finalize_rope_tables()
        return self.layer_info

    def _default_sample_input(self) -> torch.Tensor:
        # Assume single-channel 28x28 input if not provided
        return torch.randn(1, 1, 28, 28)

    def _finalize_rope_tables(self) -> None:
        """
        Populate Q15 RoPE sin/cos tables for RoPE-enabled MHSA layers.

        We finalize after the sample forward pass so that dynamic sequence lengths
        (captured from input shapes) are available.
        """
        for layer_name, layer_data in self.layer_info.items():
            if layer_data.get('type') != 'MultiheadSelfAttention' or not layer_data.get('use_rope', False):
                continue
            if 'rope_cos_q15' in layer_data and 'rope_sin_q15' in layer_data:
                continue

            seq_len = layer_data.get('sequence_length')
            embed_dim = layer_data.get('embed_dim')
            num_heads = layer_data.get('num_heads', 1)
            head_dim = layer_data.get('head_dim') or (embed_dim // num_heads if embed_dim and num_heads else None)
            if seq_len is None or head_dim is None:
                raise ValueError(f"RoPE MHSA {layer_name}: missing seq_len/head_dim metadata")
            if head_dim % 2 != 0:
                raise ValueError(f"RoPE MHSA {layer_name}: head_dim must be even, got {head_dim}")

            base = float(layer_data.get('rope_base', 10000.0))
            inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float64) / np.float64(head_dim)))
            positions = np.arange(int(seq_len), dtype=np.float64)
            angles = np.outer(positions, inv_freq)

            cos = np.cos(angles)
            sin = np.sin(angles)

            q15 = 32768.0
            cos_q15 = np.round(cos * q15).astype(np.int64)
            sin_q15 = np.round(sin * q15).astype(np.int64)
            cos_q15 = np.clip(cos_q15, -int(q15), int(q15) - 1).astype(np.int16)
            sin_q15 = np.clip(sin_q15, -int(q15), int(q15) - 1).astype(np.int16)

            layer_data['rope_cos_q15'] = cos_q15
            layer_data['rope_sin_q15'] = sin_q15

    def _extract_positional_embeddings(self):
        """
        Extract positional embedding parameters (nn.Parameter instances).

        Looks for parameters named 'pos_embed' and their associated quantizers
        (e.g., 'pos_quant'). Creates layer entries for positional embeddings.

        IMPORTANT: Inserts pos_embed right after patch_embed in layer_info order.
        """
        # Find all positional embedding parameters
        pos_embed_params = {}
        for name, param in self.model.named_parameters():
            if 'pos_embed' in name:
                # Store the parameter with its full path
                pos_embed_params[name] = param

        if not pos_embed_params:
            return

        print("\n  Extracting positional embeddings...")

        for param_name, param in pos_embed_params.items():
            # Find the associated quantizer (e.g., 'pos_quant' for 'pos_embed')
            # The parameter name is like 'pos_embed' or 'encoder.pos_embed'
            # The quantizer should be at the same level as 'pos_quant'
            parts = param_name.rsplit('.', 1)
            if len(parts) == 2:
                prefix, local_name = parts
                quant_name = f"{prefix}.pos_quant"
            else:
                local_name = parts[0]
                quant_name = "pos_quant"

            # Try to find the quantizer module
            scale = 1.0
            for name, module in self.model.named_modules():
                if name == quant_name or name.endswith('.pos_quant'):
                    if isinstance(module, qnn.QuantIdentity):
                        scale = self._extract_activation_scale(module)
                        break

            # Also check for scale_equalizer which may be used for the addition
            scale_equalizer = None
            for name, module in self.model.named_modules():
                if 'scale_equalizer' in name:
                    if isinstance(module, qnn.QuantIdentity):
                        scale_equalizer = self._extract_activation_scale(module)
                        break

            # Extract the parameter values
            pos_embed_fp32 = param.detach().cpu().numpy()

            # Quantize to INT8
            pos_embed_int8 = self._quantize_to_int8(pos_embed_fp32, scale)

            # Create layer entry for positional embedding
            layer_data = {
                'name': param_name,
                'type': 'PositionalEmbedding',
                'shape': list(pos_embed_fp32.shape),
                'scale': scale,
                'scale_equalizer': scale_equalizer,
                'pos_embed_fp32': pos_embed_fp32,
                'pos_embed_int8': pos_embed_int8,
            }

            # Find patch_embed in layer_info and insert pos_embed right after it
            # This ensures correct execution order in int8_inference.py
            patch_embed_key = None
            for key in self.layer_info.keys():
                if self.layer_info[key].get('type') == 'PatchEmbed':
                    patch_embed_key = key
                    break

            # Fallback anchor for models without PatchEmbed (e.g., CCT tokenizer stack):
            # insert before the first transformer block layer.
            transformer_anchor_key = None
            if patch_embed_key is None:
                for key in self.layer_info.keys():
                    if '.blocks.0.' in key or '.layers.0.' in key:
                        transformer_anchor_key = key
                        break

            if patch_embed_key:
                # Rebuild layer_info with pos_embed inserted after patch_embed
                new_layer_info = {}
                for key, value in self.layer_info.items():
                    new_layer_info[key] = value
                    if key == patch_embed_key:
                        # Insert pos_embed right after patch_embed
                        new_layer_info[param_name] = layer_data
                self.layer_info = new_layer_info
            elif transformer_anchor_key:
                # Rebuild layer_info with pos_embed inserted before the first block layer
                new_layer_info = {}
                for key, value in self.layer_info.items():
                    if key == transformer_anchor_key:
                        new_layer_info[param_name] = layer_data
                    new_layer_info[key] = value
                self.layer_info = new_layer_info
            else:
                # No patch_embed found, append at end
                self.layer_info[param_name] = layer_data

            print(f"  {param_name}: PositionalEmbedding(shape={layer_data['shape']}, "
                  f"scale={scale:.6f})")

    def _capture_output_shapes(self, sample_input):
        """Capture output shapes via forward pass. Supports single or multiple inputs."""
        device = next(self.model.parameters()).device

        # Handle multi-input case (tuple/list of tensors)
        if isinstance(sample_input, (tuple, list)):
            sample_inputs = []
            for inp in sample_input:
                if not isinstance(inp, torch.Tensor):
                    inp = torch.tensor(inp)
                sample_inputs.append(inp.to(device))
            sample_inputs = tuple(sample_inputs)
            is_multi_input = True
        else:
            # Single input case
            if sample_input is None:
                sample_input = self._default_sample_input()
            if not isinstance(sample_input, torch.Tensor):
                sample_input = torch.tensor(sample_input)
            sample_inputs = (sample_input.to(device),)
            is_multi_input = False

        shapes = {}
        tensor_to_layer = {}
        hooks = []

        def register_hook(name):
            def hook(module, inputs, output):
                if hasattr(output, 'value'):
                    tensor_to_layer[id(output)] = name
                    tensor = output.value
                else:
                    tensor = output
                if isinstance(tensor, torch.Tensor):
                    shapes[name] = list(tensor.shape)
                    tensor_to_layer[id(tensor)] = name

                layer_entry = self.layer_info.get(name)
                if layer_entry is None:
                    return

                # Extract sequence_length for attention layers from input shape
                if layer_entry.get('type') == 'MultiheadSelfAttention':
                    # Input should be [B, N, D] where N is sequence length
                    if inputs and len(inputs) > 0:
                        input_tensor = inputs[0]
                        if hasattr(input_tensor, 'value'):
                            input_tensor = input_tensor.value
                        if isinstance(input_tensor, torch.Tensor) and input_tensor.dim() == 3:
                            seq_len = input_tensor.size(1)  # [B, N, D] -> N
                            if layer_entry.get('sequence_length') is None:
                                layer_entry['sequence_length'] = seq_len

                if layer_entry.get('type') == 'Embedding':
                    # Capture the concrete indices used during the sample forward pass so codegen
                    # can embed them as a constant INT32 array.
                    if inputs and len(inputs) > 0:
                        idx_tensor = inputs[0]
                        if hasattr(idx_tensor, 'value'):
                            idx_tensor = idx_tensor.value
                        if isinstance(idx_tensor, torch.Tensor):
                            idx_np = idx_tensor.detach().cpu().numpy().astype(np.int32, copy=False)
                            layer_entry['indices'] = idx_np
                            layer_entry['indices_shape'] = list(idx_np.shape)
                            layer_entry['num_indices'] = int(idx_np.size)

                if layer_entry.get('type') in ('Add', 'Concatenate'):
                    input_names = self._resolve_input_names(inputs, tensor_to_layer)
                    if layer_entry['type'] == 'Add' and input_names:
                        deduped = self._dedupe_add_inputs(name, input_names, shapes)
                        if len(deduped) >= 2:
                            layer_entry['inputs'] = deduped[:2]
                    elif layer_entry['type'] == 'Concatenate' and input_names:
                        channels = []
                        for inp_name in input_names:
                            shape = shapes.get(inp_name) or self.layer_info.get(inp_name, {}).get('output_shape')
                            if shape and len(shape) > 1:
                                channels.append(shape[1])

                        pairs = list(zip(input_names, channels))
                        preferred_pairs = [
                            (n, ch) for n, ch in pairs
                            if self.layer_info.get(n, {}).get('type') not in ('QuantConv2d',)
                        ]
                        if preferred_pairs:
                            pairs = preferred_pairs
                        input_names = [n for n, _ in pairs]
                        channels = [ch for _, ch in pairs]

                        # Some Brevitas hooks report duplicate tensors (e.g., both QuantIdentity
                        # output and its upstream source). Trim any extra entries so the summed
                        # channel count matches the recorded concat output shape.
                        target_channels = None
                        current_shape = shapes.get(name)
                        if current_shape and len(current_shape) > 1:
                            target_channels = current_shape[1]

                        if (target_channels is not None and channels and
                                len(channels) == len(input_names) and
                                sum(channels) >= target_channels):
                            trimmed_inputs = []
                            trimmed_channels = []
                            running = 0
                            for inp_name, ch in zip(input_names, channels):
                                if running + ch > target_channels:
                                    break
                                trimmed_inputs.append(inp_name)
                                trimmed_channels.append(ch)
                                running += ch
                            if running == target_channels and trimmed_inputs:
                                input_names = trimmed_inputs
                                channels = trimmed_channels

                        layer_entry['inputs'] = input_names
                        if channels:
                            layer_entry['channels_per_input'] = channels
            return hook

        for name, module in self.model.named_modules():
            if name in self.layer_info:
                hooks.append(module.register_forward_hook(register_hook(name)))

        try:
            with torch.no_grad():
                # Use unpacking for multi-input models
                _ = self.model(*sample_inputs)
        finally:
            for h in hooks:
                h.remove()

        for name, shape in shapes.items():
            self.layer_info[name]['output_shape'] = shape

    def _dedupe_add_inputs(self, layer_name, candidates, shapes):
        """
        Ensure Add layers reference two distinct upstream tensors.
        Prefer the most recent layers in layer_order to avoid stale references.
        """
        unique = []
        seen = set()

        # Sort candidates by their position in layer_order (most recent first)
        # This ensures we prefer quant2 over conv2, quant1 over conv1, etc.
        def get_layer_index(name):
            try:
                return self.layer_order.index(name)
            except ValueError:
                return -1

        sorted_candidates = sorted(candidates, key=get_layer_index, reverse=True)

        for name in sorted_candidates:
            if name in seen:
                continue
            seen.add(name)
            unique.append(name)
            if len(unique) == 2:
                return unique

        target_shape = shapes.get(layer_name) or self.layer_info.get(layer_name, {}).get('output_shape')
        idx = None
        if layer_name in self.layer_order:
            idx = self.layer_order.index(layer_name)

        if idx is not None:
            for prev in reversed(self.layer_order[:idx]):
                if prev in seen:
                    continue
                prev_shape = shapes.get(prev) or self.layer_info.get(prev, {}).get('output_shape')
                if target_shape and prev_shape and len(prev_shape) == len(target_shape):
                    if prev_shape[1:] != target_shape[1:]:
                        continue
                unique.append(prev)
                seen.add(prev)
                if len(unique) == 2:
                    break

        return unique

    def _extract_activation_scale(self, module) -> float:
        """
        Extract activation quantization scale from QuantIdentity or QuantReLU.

        Args:
            module: Brevitas quantization module

        Returns:
            Activation scale (float)
        """
        # The most reliable way is to run a forward pass and extract from QuantTensor
        return self._extract_scale_via_forward(module)

    def _resolve_input_names(self, inputs, tensor_map) -> List[str]:
        """
        Resolve the layer names that produced the tensors feeding into a module.
        """
        names = []

        def collect(obj):
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    collect(item)
                return
            tensor = obj
            name = tensor_map.get(id(tensor))
            if not name and hasattr(tensor, 'value'):
                tensor = tensor.value
                name = tensor_map.get(id(tensor))
            if isinstance(tensor, torch.Tensor) and name:
                names.append(name)

        if isinstance(inputs, tuple):
            for item in inputs:
                collect(item)
        else:
            collect(inputs)

        return names

    def _extract_scale_via_forward(self, module, input_shape=None) -> float:
        """
        Extract scale by running a forward pass and checking the QuantTensor.

        Args:
            module: Brevitas module
            input_shape: Optional input shape tuple. If None, defaults to (1, 1, 28, 28)

        Returns:
            Scale value
        """
        try:
            # Create a dummy input with appropriate shape
            if input_shape is None:
                dummy_input = torch.randn(1, 1, 28, 28)
            else:
                dummy_input = torch.randn(*input_shape)

            # Move to same device as module
            if hasattr(module, 'parameters'):
                try:
                    device = next(module.parameters()).device
                    dummy_input = dummy_input.to(device)
                except StopIteration:
                    pass  # No parameters, use CPU

            # Forward pass
            with torch.no_grad():
                output = module(dummy_input)

            # Check if output is a QuantTensor
            if hasattr(output, 'scale'):
                return float(output.scale.detach().cpu().item())

        except Exception as e:
            print(f"    Warning: Could not extract scale via forward: {e}")

        return 1.0

    def _extract_conv2d_info(self, module: qnn.QuantConv2d) -> Dict[str, Any]:
        """
        Extract Conv2D layer information.

        Args:
            module: QuantConv2d module

        Returns:
            Dictionary with conv2d parameters, weights, bias, and scales
        """
        info = {}

        # Layer parameters
        info['in_channels'] = module.in_channels
        info['out_channels'] = module.out_channels
        info['kernel_size'] = module.kernel_size
        info['stride'] = module.stride
        info['padding'] = module.padding
        info['groups'] = module.groups  # 1=normal conv, in_channels=depthwise conv

        # Extract weight quantization scale and bit width
        info['scale_weight'] = self._extract_weight_scale(module)
        info['weight_bit_width'] = self._extract_weight_bit_width(module)

        # Extract and quantize weights (respects bit_width)
        weight_fp32 = module.weight.detach().cpu().numpy()
        weight_int8 = self._quantize_weights(weight_fp32, info['scale_weight'], info['weight_bit_width'])
        info['weight_int8'] = weight_int8

        # Extract bias (if exists)
        if module.bias is not None:
            bias_fp32 = module.bias.detach().cpu().numpy()
            # Bias is quantized with scale_x * scale_w
            # We'll handle this during inference when we know scale_x
            info['bias_fp32'] = bias_fp32
        else:
            info['bias_fp32'] = None

        return info

    def _extract_linear_info(self, module: qnn.QuantLinear) -> Dict[str, Any]:
        """
        Extract Linear (Fully Connected) layer information.

        Args:
            module: QuantLinear module

        Returns:
            Dictionary with linear parameters, weights, bias, and scales
        """
        info = {}

        # Layer parameters
        info['in_features'] = module.in_features
        info['out_features'] = module.out_features

        # Extract weight quantization scale and bit width
        info['scale_weight'] = self._extract_weight_scale(module)
        info['weight_bit_width'] = self._extract_weight_bit_width(module)

        # Extract and quantize weights (respects bit_width)
        weight_fp32 = module.weight.detach().cpu().numpy()
        weight_int8 = self._quantize_weights(weight_fp32, info['scale_weight'], info['weight_bit_width'])
        info['weight_int8'] = weight_int8

        # For 2-bit weights, also create packed version (4 weights per byte)
        if info['weight_bit_width'] == 2:
            info['weight_packed_2bit'] = self._pack_2bit_weights_fast(weight_int8)

        # Extract bias (if exists)
        if module.bias is not None:
            bias_fp32 = module.bias.detach().cpu().numpy()
            info['bias_fp32'] = bias_fp32
        else:
            info['bias_fp32'] = None

        return info

    def _extract_attention_info(self, module) -> Dict[str, Any]:
        """Extract Q/K/V/Out projection weights for self-attention."""
        info: Dict[str, Any] = {}

        # Support both naming conventions:
        # - QuantMultiHeadAttention: q_proj, k_proj, v_proj, out_proj
        # - QuantMHSA: q, k, v, proj
        projections = {}
        if hasattr(module, 'q_proj'):
            # QuantMultiHeadAttention naming
            projections = {
                'q': module.q_proj,
                'k': module.k_proj,
                'v': module.v_proj,
                'out': module.out_proj,
            }
        elif hasattr(module, 'q'):
            # QuantMHSA naming
            projections = {
                'q': module.q,
                'k': module.k,
                'v': module.v,
                'out': module.proj,
            }
        else:
            raise ValueError(f"Unknown attention module structure: {type(module)}")

        for prefix, proj in projections.items():
            proj_info = self._extract_linear_info(proj)
            info[f'{prefix}_weight_int8'] = proj_info['weight_int8']
            info[f'{prefix}_bias_fp32'] = proj_info['bias_fp32']
            info[f'{prefix}_scale_weight'] = proj_info['scale_weight']
            info[f'{prefix}_in_features'] = proj_info['in_features']
            info[f'{prefix}_out_features'] = proj_info['out_features']

            # Extract output scale for Q/K/V projections (critical for attention scores)
            # PRIORITY 1: Check for separate quantizer layer (e.g., q_quant, k_quant, v_quant)
            # These are explicit QuantIdentity layers added for Path A
            separate_quant = getattr(module, f'{prefix}_quant', None)
            if separate_quant is not None:
                info[f'{prefix}_scale_output'] = self._extract_activation_scale(separate_quant)
            else:
                # PRIORITY 2: Check if projection has output_quant attribute
                output_quant = getattr(proj, 'output_quant', None) or getattr(proj, 'out_quant', None)
                if output_quant is not None:
                    info[f'{prefix}_scale_output'] = self._extract_activation_scale(output_quant)
                else:
                    # PRIORITY 3: Try runtime forward pass extraction
                    try:
                        in_features = proj_info['in_features']
                        dummy_input = torch.randn(1, in_features).to(next(proj.parameters()).device)
                        with torch.no_grad():
                            output = proj(dummy_input)

                        # Try to extract scale from output
                        if hasattr(output, 'scale'):
                            scale = float(output.scale.detach().cpu().item())
                            info[f'{prefix}_scale_output'] = scale
                        else:
                            # Last resort: use 1.0
                            print(f"    WARNING: No output quantizer found for {prefix} projection, using 1.0")
                            info[f'{prefix}_scale_output'] = 1.0
                    except Exception as e:
                        print(f"    WARNING: Failed to extract {prefix}_scale_output: {e}")
                        info[f'{prefix}_scale_output'] = 1.0
        return info

    def _extract_alternating_attention_info(self, module) -> Dict[str, Any]:
        """
        Extract QKV projection and output projection weights for alternating attention.

        AlternatingAttention uses combined QKV projection (3*embed_dim output)
        instead of separate Q/K/V projections.
        """
        info: Dict[str, Any] = {}

        # Extract combined QKV projection
        qkv_proj = getattr(module, 'qkv_proj', None)
        if qkv_proj is not None:
            qkv_info = self._extract_linear_info(qkv_proj)
            info['qkv_weight_int8'] = qkv_info['weight_int8']
            info['qkv_bias_fp32'] = qkv_info['bias_fp32']
            info['qkv_scale_weight'] = qkv_info['scale_weight']
            info['qkv_in_features'] = qkv_info['in_features']
            info['qkv_out_features'] = qkv_info['out_features']

            # QKV projection output scale (before splitting Q/K/V)
            output_quant = getattr(qkv_proj, 'output_quant', None)
            if output_quant is not None:
                info['qkv_scale_output'] = self._extract_activation_scale(output_quant)
            else:
                try:
                    dummy_input = torch.randn(1, qkv_info['in_features']).to(next(qkv_proj.parameters()).device)
                    with torch.no_grad():
                        output = qkv_proj(dummy_input)
                    if hasattr(output, 'scale'):
                        info['qkv_scale_output'] = float(output.scale.detach().cpu().item())
                    else:
                        info['qkv_scale_output'] = 1.0
                except Exception:
                    info['qkv_scale_output'] = 1.0

        # Extract separate Q/K/V quantizer scales (after splitting and scaling Q)
        for prefix in ['q', 'k', 'v']:
            quant = getattr(module, f'{prefix}_quant', None)
            if quant is not None:
                info[f'{prefix}_scale_output'] = self._extract_activation_scale(quant)
            else:
                info[f'{prefix}_scale_output'] = 1.0

        # Extract output projection
        out_proj = getattr(module, 'out_proj', None)
        if out_proj is not None:
            out_info = self._extract_linear_info(out_proj)
            info['out_weight_int8'] = out_info['weight_int8']
            info['out_bias_fp32'] = out_info['bias_fp32']
            info['out_scale_weight'] = out_info['scale_weight']
            info['out_in_features'] = out_info['in_features']
            info['out_out_features'] = out_info['out_features']

        return info

    def _extract_conv1d_depthwise_info(self, module) -> Dict[str, Any]:
        """
        Extract Conv1D Depthwise layer information for MAMBA.

        Args:
            module: QuantConv1dDepthwise module

        Returns:
            Dictionary with conv1d parameters, weights, bias, and scales
        """
        info = {}

        # Layer parameters
        info['channels'] = getattr(module, 'in_channels', getattr(module, 'channels', None))
        info['kernel_size'] = module.kernel_size
        info['causal'] = getattr(module, 'causal', True)

        # Extract weight quantization scale and bit width from the conv layer
        conv = module.conv
        info['scale_weight'] = self._extract_weight_scale(conv)
        info['weight_bit_width'] = self._extract_weight_bit_width(conv)

        # Extract and quantize weights (respects bit_width)
        # Weight shape: [channels, 1, kernel_size] for depthwise
        weight_fp32 = conv.weight.detach().cpu().numpy()
        # Reshape to [channels, kernel_size] for our atomic op
        weight_fp32 = weight_fp32.squeeze(1)  # [C, K]
        weight_int8 = self._quantize_weights(weight_fp32, info['scale_weight'], info['weight_bit_width'])
        info['weight_int8'] = weight_int8

        # Extract bias (if exists)
        if conv.bias is not None:
            bias_fp32 = conv.bias.detach().cpu().numpy()
            info['bias_fp32'] = bias_fp32
        else:
            info['bias_fp32'] = None

        # Extract output scale
        info['scale_output'] = self._extract_activation_scale(module.output_quant)

        return info

    def _extract_ssm_info(self, module) -> Dict[str, Any]:
        """
        Extract SSM (State Space Model) layer information for MAMBA.

        Args:
            module: QuantSSM module

        Returns:
            Dictionary with SSM parameters including:
            - x_proj: [d_inner, d_state * 2] for B, C projection
            - dt_proj: [d_inner, dt_rank] for dt projection
            - A_log: [d_inner, d_state] log of A matrix
            - D: [d_inner] or scalar
        """
        info = {}

        # Layer parameters
        info['d_inner'] = module.d_inner
        info['d_state'] = module.d_state
        info['dt_rank'] = module.dt_rank

        # Extract x_proj (projects input to B, C)
        x_proj = module.x_proj
        x_proj_info = self._extract_linear_info(x_proj)
        info['x_proj_weight_int8'] = x_proj_info['weight_int8']
        info['x_proj_bias_fp32'] = x_proj_info['bias_fp32']
        info['x_proj_scale_weight'] = x_proj_info['scale_weight']
        info['x_proj_weight_bit_width'] = x_proj_info.get('weight_bit_width', 8)
        if 'weight_packed_2bit' in x_proj_info:
            info['x_proj_weight_packed_2bit'] = x_proj_info['weight_packed_2bit']

        # Extract dt_proj (projects to dt values)
        dt_proj = module.dt_proj
        dt_proj_info = self._extract_linear_info(dt_proj)
        info['dt_proj_weight_int8'] = dt_proj_info['weight_int8']
        info['dt_proj_bias_fp32'] = dt_proj_info['bias_fp32']
        info['dt_proj_scale_weight'] = dt_proj_info['scale_weight']
        info['dt_proj_weight_bit_width'] = dt_proj_info.get('weight_bit_width', 8)
        if 'weight_packed_2bit' in dt_proj_info:
            info['dt_proj_weight_packed_2bit'] = dt_proj_info['weight_packed_2bit']

        # Extract A_log parameter (FP32, will be converted to Q16 at runtime)
        A_log = module.A_log.detach().cpu().numpy()
        info['A_log_fp32'] = A_log

        # Extract D parameter (FP32)
        D = module.D.detach().cpu().numpy()
        info['D_fp32'] = D

        # Extract output scale from output_quant
        info['scale_output'] = self._extract_activation_scale(module.output_quant)

        return info

    def _extract_mamba_block_info(self, module) -> Dict[str, Any]:
        """
        Extract full MambaBlock layer information.

        A MambaBlock contains:
        - in_proj: Linear projection from d_model to 2*d_inner (x and z branches)
        - conv1d: Depthwise 1D convolution on x branch
        - silu: SiLU activation after conv1d
        - ssm: State Space Model core
        - out_proj: Linear projection from d_inner back to d_model

        Args:
            module: QuantMambaBlock module

        Returns:
            Dictionary with all MambaBlock parameters
        """
        info = {}

        # Layer parameters
        info['d_model'] = module.d_model
        info['d_inner'] = module.d_inner
        info['d_state'] = module.d_state
        info['dt_rank'] = module.ssm.dt_rank  # dt_rank is on SSM sub-module
        info['kernel_size'] = module.kernel_size

        # Extract in_proj (d_model -> 2*d_inner)
        in_proj = module.in_proj
        in_proj_info = self._extract_linear_info(in_proj)
        info['in_proj_weight_int8'] = in_proj_info['weight_int8']
        info['in_proj_bias_fp32'] = in_proj_info['bias_fp32']
        info['in_proj_scale_weight'] = in_proj_info['scale_weight']
        info['in_proj_weight_bit_width'] = in_proj_info.get('weight_bit_width', 8)
        if 'weight_packed_2bit' in in_proj_info:
            info['in_proj_weight_packed_2bit'] = in_proj_info['weight_packed_2bit']
        info['in_proj_scale_output'] = self._extract_activation_scale(module.in_proj_quant)

        # Extract conv1d (depthwise on x branch)
        conv1d_info = self._extract_conv1d_depthwise_info(module.conv1d)
        for key, value in conv1d_info.items():
            info[f'conv1d_{key}'] = value

        # Extract SiLU scales (QuantSiLU only has output_quant, not input_quant)
        info['silu_scale_output'] = self._extract_activation_scale(module.silu.output_quant)

        # Extract SSM parameters
        ssm_info = self._extract_ssm_info(module.ssm)
        for key, value in ssm_info.items():
            info[f'ssm_{key}'] = value

        # Extract out_proj (d_inner -> d_model)
        out_proj = module.out_proj
        out_proj_info = self._extract_linear_info(out_proj)
        info['out_proj_weight_int8'] = out_proj_info['weight_int8']
        info['out_proj_bias_fp32'] = out_proj_info['bias_fp32']
        info['out_proj_scale_weight'] = out_proj_info['scale_weight']
        info['out_proj_weight_bit_width'] = out_proj_info.get('weight_bit_width', 8)
        if 'weight_packed_2bit' in out_proj_info:
            info['out_proj_weight_packed_2bit'] = out_proj_info['weight_packed_2bit']

        # Extract output scale
        info['scale_output'] = self._extract_activation_scale(module.output_quant)

        return info

    def _extract_mamba_wrapper_info(self, module) -> Dict[str, Any]:
        """
        Extract full MambaWrapper (bidirectional) layer information.

        A MambaWrapper contains:
        - mamba_fwd: Forward direction MambaBlock
        - mamba_rev: Reverse direction MambaBlock
        - scale_equalizer: QuantIdentity for combining outputs
        - flip operations for bidirectional processing

        Args:
            module: QuantMambaWrapper module

        Returns:
            Dictionary with all MambaWrapper parameters including both mamba_fwd and mamba_rev
        """
        info = {}

        # Layer parameters
        info['d_model'] = module.d_model
        info['d_inner'] = module.d_inner
        info['d_state'] = module.d_state
        info['bidirectional_strategy'] = module.bidirectional_strategy

        # Extract forward MambaBlock parameters
        fwd_info = self._extract_mamba_block_info(module.mamba_fwd)
        for key, value in fwd_info.items():
            info[f'fwd_{key}'] = value

        # Extract reverse MambaBlock parameters
        rev_info = self._extract_mamba_block_info(module.mamba_rev)
        for key, value in rev_info.items():
            info[f'rev_{key}'] = value

        # Extract scale equalizer scale (used for combining fwd and rev outputs)
        info['scale_equalizer'] = self._extract_activation_scale(module.scale_equalizer)

        # Extract flip quantization scales
        info['scale_post_flip_input'] = self._extract_activation_scale(module.post_flip_quant)
        info['scale_post_flip_output'] = self._extract_activation_scale(module.rev_post_flip_quant)

        # Extract output scale
        info['scale_output'] = self._extract_activation_scale(module.output_quant)

        return info


    def _extract_patch_embed_info(self, module) -> Dict[str, Any]:
        """
        Extract PatchEmbed layer information.

        A PatchEmbed contains:
        - proj: QuantConv2d for patch projection
        - output_quant: QuantIdentity for output quantization

        Args:
            module: QuantPatchEmbed module

        Returns:
            Dictionary with PatchEmbed parameters including Conv2D weights
        """
        info = {}

        # Layer parameters from module - handle both FEMBA and TinyMyo naming conventions
        # FEMBA: inp_size, stride, grid_h, grid_w, seq_len, d_model
        # TinyMyo: img_size, in_chans, num_patches
        info['inp_size'] = getattr(module, 'inp_size', None) or getattr(module, 'img_size', (0, 0))
        info['patch_size'] = module.patch_size
        info['stride'] = getattr(module, 'stride', module.patch_size)  # Default stride = patch_size
        info['in_chans'] = module.in_chans
        info['embed_dim'] = module.embed_dim
        info['grid_h'] = getattr(module, 'grid_h', 1)  # TinyMyo doesn't have grid_h
        info['grid_w'] = getattr(module, 'grid_w', getattr(module, 'num_patches', 0))
        info['seq_len'] = getattr(module, 'seq_len', getattr(module, 'num_patches', 0))
        info['d_model'] = getattr(module, 'd_model', module.embed_dim)

        # Extract projection Conv2D weights
        proj = module.proj
        weight_fp32 = proj.weight.detach().cpu().numpy()
        weight_scale = self._extract_weight_scale(proj)
        weight_bit_width = self._extract_weight_bit_width(proj)

        # Quantize weights (respects bit_width)
        info['proj_weight_int8'] = self._quantize_weights(weight_fp32, weight_scale, weight_bit_width)
        info['proj_weight_scale'] = weight_scale
        info['proj_weight_bit_width'] = weight_bit_width

        if proj.bias is not None:
            info['proj_bias_fp32'] = proj.bias.detach().cpu().numpy()

        # Extract output quantization scale - handle both naming conventions
        # FEMBA: output_quant, TinyMyo: quant
        output_quant = getattr(module, 'output_quant', None) or getattr(module, 'quant', None)
        if output_quant is not None:
            info['scale_output'] = self._extract_activation_scale(output_quant)
        else:
            info['scale_output'] = 1.0  # Default scale if no quantizer found

        return info

    def _extract_weight_scale(self, module) -> float:
        """
        Extract weight quantization scale from QuantConv2d or QuantLinear.

        Args:
            module: Brevitas quantization module with weights

        Returns:
            Weight scale (float)
        """
        try:
            # Run a forward pass to get the quantized weight tensor
            dummy_input_shape = None

            if isinstance(module, qnn.QuantConv2d):
                # For Conv2d, create appropriate input shape
                dummy_input_shape = (1, module.in_channels, 8, 8)
            elif isinstance(module, qnn.QuantLinear):
                # For Linear, create appropriate input shape
                dummy_input_shape = (1, module.in_features)

            if dummy_input_shape is not None:
                dummy_input = torch.randn(*dummy_input_shape)

                # Forward pass to trigger quantization
                with torch.no_grad():
                    output = module(dummy_input)

                # Try to access the quantized weight scale
                if hasattr(module, 'quant_weight'):
                    quant_weight = module.quant_weight()
                    if hasattr(quant_weight, 'scale'):
                        scale_val = quant_weight.scale
                        if isinstance(scale_val, (torch.Tensor, nn.Parameter)):
                            return float(scale_val.detach().cpu().item())
                        else:
                            return float(scale_val)

            # Fallback: estimate from weight range
            weight = module.weight.detach().cpu()
            w_max = torch.max(torch.abs(weight)).item()
            scale = w_max / 127.0
            return scale

        except Exception as e:
            # Fallback: estimate from weight range
            weight = module.weight.detach().cpu()
            w_max = torch.max(torch.abs(weight)).item()
            scale = w_max / 127.0
            return scale

    def _extract_weight_bit_width(self, module) -> int:
        """
        Extract weight bit width from Brevitas quantized module.

        Args:
            module: Brevitas quantization module with weights

        Returns:
            Bit width (int), defaults to 8 if not detectable
        """
        try:
            if hasattr(module, 'quant_weight'):
                qw = module.quant_weight()
                if hasattr(qw, 'bit_width'):
                    bw = qw.bit_width
                    if isinstance(bw, (torch.Tensor,)):
                        return int(bw.item())
                    return int(bw)
            # Fallback: assume 8-bit
            return 8
        except Exception:
            return 8

    def _quantize_weights(self, x: np.ndarray, scale: float, bit_width: int = 8) -> np.ndarray:
        """
        Quantize FP32 array to integer with specified bit width.

        For symmetric quantization:
        - 2-bit: clip to [-1, 1]
        - 3-bit: clip to [-3, 3]
        - 4-bit: clip to [-7, 7]
        - 8-bit: clip to [-127, 127]

        Args:
            x: FP32 array
            scale: Quantization scale
            bit_width: Number of bits for quantization (2, 3, 4, or 8)

        Returns:
            INT8 array (values limited to bit_width range)
        """
        # Symmetric quantization: max value is 2^(bit_width-1) - 1
        max_val = (1 << (bit_width - 1)) - 1
        min_val = -max_val

        x_scaled = x / scale
        x_rounded = np.round(x_scaled)
        x_clipped = np.clip(x_rounded, min_val, max_val)
        return x_clipped.astype(np.int8)

    def _quantize_to_int8(self, x: np.ndarray, scale: float) -> np.ndarray:
        """
        Quantize FP32 array to INT8 (8-bit).

        Formula: q = clip(round(x / scale), -128, 127)

        Args:
            x: FP32 array
            scale: Quantization scale

        Returns:
            INT8 array
        """
        return self._quantize_weights(x, scale, bit_width=8)

    def _pack_2bit_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Pack 2-bit weights into uint8 (4 weights per byte).

        Input values are in {-1, 0, 1}, mapped to {0, 1, 2} for packing.
        Packing order: w0 in bits [1:0], w1 in bits [3:2], w2 in bits [5:4], w3 in bits [7:6]

        Args:
            weights: INT8 array with values in {-1, 0, 1}

        Returns:
            UINT8 array with 4x compression (4 weights per byte)
        """
        # Flatten weights for packing
        flat = weights.flatten().astype(np.int8)

        # Pad to multiple of 4 if necessary
        pad_len = (4 - len(flat) % 4) % 4
        if pad_len > 0:
            flat = np.concatenate([flat, np.zeros(pad_len, dtype=np.int8)])

        # Map {-1, 0, 1} -> {0, 1, 2} for unsigned packing
        mapped = (flat + 1).astype(np.uint8)

        # Pack 4 weights per byte
        packed_len = len(flat) // 4
        packed = np.zeros(packed_len, dtype=np.uint8)

        for i in range(packed_len):
            idx = i * 4
            packed[i] = (mapped[idx] |
                        (mapped[idx + 1] << 2) |
                        (mapped[idx + 2] << 4) |
                        (mapped[idx + 3] << 6))

        return packed

    def _pack_2bit_weights_fast(self, weights: np.ndarray) -> np.ndarray:
        """
        Fast vectorized 2-bit weight packing (4 weights per byte).

        Same as _pack_2bit_weights but uses numpy vectorization for speed.

        Args:
            weights: INT8 array with values in {-1, 0, 1}

        Returns:
            UINT8 array with 4x compression
        """
        flat = weights.flatten().astype(np.int8)

        # Pad to multiple of 4
        pad_len = (4 - len(flat) % 4) % 4
        if pad_len > 0:
            flat = np.concatenate([flat, np.zeros(pad_len, dtype=np.int8)])

        # Map {-1, 0, 1} -> {0, 1, 2}
        mapped = (flat + 1).astype(np.uint8)

        # Reshape to [N/4, 4] and pack
        reshaped = mapped.reshape(-1, 4)
        packed = (reshaped[:, 0] |
                 (reshaped[:, 1] << 2) |
                 (reshaped[:, 2] << 4) |
                 (reshaped[:, 3] << 6))

        return packed.astype(np.uint8)

    def _compute_residency_policy(self):
        """
        Compute L2 vs L3 residency policy for each layer based on weight size.

        Heuristic:
        - weight_size < 4KB: L2-resident (small, negligible footprint)
        - weight_size < 32KB AND Conv2D: L2-resident (high reuse across tiles)
        - weight_size >= 32KB OR large Linear: Stage from L3 just-in-time

        Adds 'weight_residency' and 'activation_residency' fields to layer_info.
        """
        SIZE_THRESHOLD_SMALL = 4 * 1024  # 4KB
        SIZE_THRESHOLD_CONV = 64 * 1024  # 64KB (increased from 32KB to avoid boundary issues)

        for layer_name, layer_data in self.layer_info.items():
            # Calculate weight size in bytes
            weight_size = 0
            if 'weight_int8' in layer_data:
                weight_shape = layer_data['weight_int8'].shape
                weight_size = np.prod(weight_shape)  # INT8: 1 byte per element
            elif layer_data.get('type') == 'MultiheadSelfAttention':
                # MHSA has 4 separate projection weights: Q, K, V, Out
                # Calculate total size of all 4 projections
                for proj in ['q', 'k', 'v', 'out']:
                    weight_key = f'{proj}_weight_int8'
                    if weight_key in layer_data:
                        weight_array = np.array(layer_data[weight_key])
                        weight_size += np.prod(weight_array.shape)
            elif layer_data.get('type') == 'CrossAttention':
                # CrossAttention: 4 projection weights + learned query embedding table
                for proj in ['q', 'k', 'v', 'out']:
                    weight_key = f'{proj}_weight_int8'
                    if weight_key in layer_data:
                        weight_array = np.array(layer_data[weight_key])
                        weight_size += np.prod(weight_array.shape)
                if 'query_embed_int8' in layer_data:
                    weight_size += int(np.prod(np.array(layer_data['query_embed_int8']).shape))
            elif layer_data.get('type') in ('CrossAttentionWithSelfRefine', 'ClassificationHeadWithMLP'):
                # Count all int8 weight arrays in the flat dict
                for key, val in layer_data.items():
                    if key.endswith('_weight_int8') and isinstance(val, np.ndarray):
                        weight_size += np.prod(val.shape)
                # Learned query/aggregation embedding
                for embed_key in ('query_embed_int8', 'learned_agg_int8'):
                    if embed_key in layer_data:
                        weight_size += int(np.prod(np.array(layer_data[embed_key]).shape))
            elif layer_data.get('type') == 'SSM':
                # SSM has x_proj and dt_proj weights
                for proj in ['x_proj', 'dt_proj']:
                    weight_key = f'{proj}_weight_int8'
                    if weight_key in layer_data:
                        weight_array = np.array(layer_data[weight_key])
                        weight_size += np.prod(weight_array.shape)
            elif layer_data.get('type') == 'MambaBlock':
                # MambaBlock has in_proj, conv1d, ssm (x_proj, dt_proj), out_proj
                for proj in ['in_proj', 'conv1d', 'ssm_x_proj', 'ssm_dt_proj', 'out_proj']:
                    weight_key = f'{proj}_weight_int8'
                    if weight_key in layer_data:
                        weight_array = np.array(layer_data[weight_key])
                        weight_size += np.prod(weight_array.shape)
            elif layer_data.get('type') == 'MambaWrapper':
                # MambaWrapper has fwd and rev MambaBlocks
                for direction in ['fwd', 'rev']:
                    for proj in ['in_proj', 'conv1d', 'ssm_x_proj', 'ssm_dt_proj', 'out_proj']:
                        weight_key = f'{direction}_{proj}_weight_int8'
                        if weight_key in layer_data:
                            weight_array = np.array(layer_data[weight_key])
                            weight_size += np.prod(weight_array.shape)
            elif layer_data.get('type') == 'PatchEmbed':
                # PatchEmbed has proj (Conv2D) weights
                if 'proj_weight_int8' in layer_data:
                    weight_array = np.array(layer_data['proj_weight_int8'])
                    weight_size = np.prod(weight_array.shape)

            # Apply residency heuristic
            layer_type = layer_data.get('type', '')

            if weight_size == 0:
                # No weights (ReLU, Pool, etc.) - keep activations in L2
                weight_residency = 'N/A'
                activation_residency = 'L2'
            elif weight_size < SIZE_THRESHOLD_SMALL:
                # Small weights - always L2 resident
                weight_residency = 'L2'
                activation_residency = 'L2'
            elif weight_size < SIZE_THRESHOLD_CONV and 'Conv' in layer_type:
                # Conv2D with moderate weights - keep in L2 (high reuse across tiles)
                weight_residency = 'L2'
                activation_residency = 'L2'
            else:
                # Large weights - stage from L3 just-in-time
                weight_residency = 'L3_STAGED'
                activation_residency = 'L2'  # Activations still L2 (smaller, frequently accessed)

            # Add residency info to layer data
            layer_data['weight_residency'] = weight_residency
            layer_data['activation_residency'] = activation_residency
            layer_data['weight_size_bytes'] = int(weight_size)

            # Log decision
            residency_str = f"{weight_residency:12}" if weight_residency != 'N/A' else "N/A (no wt)"
            print(f"  {layer_name:20} | size: {weight_size:8d} bytes | residency: {residency_str}")

    def save_to_json(self, output_path: str = "golden_outputs/network_info.json"):
        """
        Save extracted network information to JSON file.

        Note: INT8 arrays are converted to lists for JSON serialization.

        Args:
            output_path: Path to save JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy arrays to lists for JSON serialization
        layer_info_serializable = {}
        for layer_name, layer_data in self.layer_info.items():
            data_copy = layer_data.copy()

            # Convert numpy arrays to lists
            for key, value in data_copy.items():
                if isinstance(value, np.ndarray):
                    data_copy[key] = value.tolist()
                elif isinstance(value, torch.Tensor):
                    data_copy[key] = value.detach().cpu().numpy().tolist()

            layer_info_serializable[layer_name] = data_copy

        layer_info_serializable['__layer_order__'] = list(self.layer_info.keys())

        with open(output_path, 'w') as f:
            json.dump(layer_info_serializable, f, indent=2)

        print(f"[PASS] Saved network info to {output_path}")

    def save_weights(self, output_dir: str = "golden_outputs/weights/"):
        """
        Save INT8 weights and FP32 biases to separate numpy files.

        Args:
            output_dir: Directory to save weight files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for layer_name, layer_data in self.layer_info.items():
            # Save INT8 weights
            if 'weight_int8' in layer_data:
                weight_path = output_dir / f"{layer_name}_weight_int8.npy"
                np.save(weight_path, layer_data['weight_int8'])
                print(f"  Saved {layer_name} weights: {layer_data['weight_int8'].shape}")

            # Save FP32 bias
            if 'bias_fp32' in layer_data and layer_data['bias_fp32'] is not None:
                bias_path = output_dir / f"{layer_name}_bias_fp32.npy"
                np.save(bias_path, layer_data['bias_fp32'])
                print(f"  Saved {layer_name} bias: {layer_data['bias_fp32'].shape}")

            # Save LayerNorm/RMSNorm FP32 weight and bias (gamma and beta)
            if layer_data.get('type') in ('LayerNorm', 'GroupNorm', 'RMSNorm'):
                if 'weight' in layer_data and layer_data['weight'] is not None:
                    weight_path = output_dir / f"{layer_name}_weight_fp32.npy"
                    np.save(weight_path, layer_data['weight'])
                    print(f"  Saved {layer_name} weights: {layer_data['weight'].shape}")
                if 'bias' in layer_data and layer_data['bias'] is not None:
                    bias_path = output_dir / f"{layer_name}_bias_fp32.npy"
                    np.save(bias_path, layer_data['bias'])
                    print(f"  Saved {layer_name} bias: {layer_data['bias'].shape}")

            # Save PatchEmbed projection weights
            if layer_data.get('type') == 'PatchEmbed':
                if 'proj_weight_int8' in layer_data:
                    weight_path = output_dir / f"{layer_name}_proj_weight_int8.npy"
                    np.save(weight_path, layer_data['proj_weight_int8'])
                    print(f"  Saved {layer_name}.proj weights: {layer_data['proj_weight_int8'].shape}")
                if 'proj_bias_fp32' in layer_data and layer_data['proj_bias_fp32'] is not None:
                    bias_path = output_dir / f"{layer_name}_proj_bias_fp32.npy"
                    np.save(bias_path, layer_data['proj_bias_fp32'])
                    print(f"  Saved {layer_name}.proj bias: {layer_data['proj_bias_fp32'].shape}")

            # Save PositionalEmbedding (INT8)
            if layer_data.get('type') == 'PositionalEmbedding':
                if 'pos_embed_int8' in layer_data:
                    weight_path = output_dir / f"{layer_name}_int8.npy"
                    np.save(weight_path, layer_data['pos_embed_int8'])
                    print(f"  Saved {layer_name}: {layer_data['pos_embed_int8'].shape}")
                if 'pos_embed_fp32' in layer_data:
                    weight_path = output_dir / f"{layer_name}_fp32.npy"
                    np.save(weight_path, layer_data['pos_embed_fp32'])
                    print(f"  Saved {layer_name} (FP32): {layer_data['pos_embed_fp32'].shape}")

            # Save Embedding indices (INT32) alongside the embedding weight table
            if layer_data.get('type') == 'Embedding':
                indices = layer_data.get('indices')
                if indices is not None:
                    indices_path = output_dir / f"{layer_name}_indices_int32.npy"
                    np.save(indices_path, np.asarray(indices, dtype=np.int32))
                    print(f"  Saved {layer_name} indices: {np.asarray(indices).shape}")

            if layer_data.get('type') == 'MultiheadSelfAttention':
                for prefix in ('q', 'k', 'v', 'out'):
                    weight_key = f"{prefix}_weight_int8"
                    bias_key = f"{prefix}_bias_fp32"
                    if weight_key in layer_data:
                        weight_path = output_dir / f"{layer_name}_{prefix}_weight_int8.npy"
                        np.save(weight_path, layer_data[weight_key])
                        print(f"  Saved {layer_name}.{prefix} weights: {layer_data[weight_key].shape}")
                    if bias_key in layer_data and layer_data[bias_key] is not None:
                        bias_path = output_dir / f"{layer_name}_{prefix}_bias_fp32.npy"
                        np.save(bias_path, layer_data[bias_key])
                        print(f"  Saved {layer_name}.{prefix} bias: {layer_data[bias_key].shape}")
                if layer_data.get('use_rope', False):
                    if 'rope_cos_q15' in layer_data:
                        rope_cos_path = output_dir / f"{layer_name}_rope_cos_q15.npy"
                        np.save(rope_cos_path, np.asarray(layer_data['rope_cos_q15'], dtype=np.int16))
                        print(f"  Saved {layer_name}.rope_cos_q15: {np.asarray(layer_data['rope_cos_q15']).shape}")
                    if 'rope_sin_q15' in layer_data:
                        rope_sin_path = output_dir / f"{layer_name}_rope_sin_q15.npy"
                        np.save(rope_sin_path, np.asarray(layer_data['rope_sin_q15'], dtype=np.int16))
                        print(f"  Saved {layer_name}.rope_sin_q15: {np.asarray(layer_data['rope_sin_q15']).shape}")

            if layer_data.get('type') == 'CrossAttention':
                if 'query_embed_int8' in layer_data:
                    query_path = output_dir / f"{layer_name}_query_embed_int8.npy"
                    np.save(query_path, np.asarray(layer_data['query_embed_int8'], dtype=np.int8))
                    print(f"  Saved {layer_name}.query_embed: {np.asarray(layer_data['query_embed_int8']).shape}")
                for prefix in ('q', 'k', 'v', 'out'):
                    weight_key = f"{prefix}_weight_int8"
                    bias_key = f"{prefix}_bias_fp32"
                    if weight_key in layer_data:
                        weight_path = output_dir / f"{layer_name}_{prefix}_weight_int8.npy"
                        np.save(weight_path, layer_data[weight_key])
                        print(f"  Saved {layer_name}.{prefix} weights: {layer_data[weight_key].shape}")
                    if bias_key in layer_data and layer_data[bias_key] is not None:
                        bias_path = output_dir / f"{layer_name}_{prefix}_bias_fp32.npy"
                        np.save(bias_path, layer_data[bias_key])
                        print(f"  Saved {layer_name}.{prefix} bias: {layer_data[bias_key].shape}")

            # Save CrossAttentionWithSelfRefine / ClassificationHeadWithMLP parameters
            if layer_data.get('type') in ('CrossAttentionWithSelfRefine', 'ClassificationHeadWithMLP'):
                # Save learned embedding (query or aggregation)
                for embed_key in ('query_embed_int8', 'learned_agg_int8'):
                    if embed_key in layer_data:
                        path = output_dir / f"{layer_name}_{embed_key}.npy"
                        np.save(path, np.asarray(layer_data[embed_key], dtype=np.int8))
                        print(f"  Saved {layer_name}.{embed_key}: {np.asarray(layer_data[embed_key]).shape}")
                # Save all weight and bias arrays
                for key, val in layer_data.items():
                    if not isinstance(val, np.ndarray):
                        continue
                    if key.endswith('_weight_int8'):
                        path = output_dir / f"{layer_name}_{key}.npy"
                        np.save(path, val)
                    elif key.endswith('_bias_fp32') and val is not None:
                        path = output_dir / f"{layer_name}_{key}.npy"
                        np.save(path, val)
                    elif key.endswith('_norm_weight') or key.endswith('_norm1_weight') or key.endswith('_norm2_weight'):
                        path = output_dir / f"{layer_name}_{key}.npy"
                        np.save(path, val)
                    elif key.endswith('_norm_bias') or key.endswith('_norm1_bias') or key.endswith('_norm2_bias'):
                        path = output_dir / f"{layer_name}_{key}.npy"
                        np.save(path, val)

            # Save SSM parameters (x_proj, dt_proj, A_log, D)
            if layer_data.get('type') == 'SSM':
                # Save x_proj weights
                if 'x_proj_weight_int8' in layer_data:
                    weight_path = output_dir / f"{layer_name}_x_proj_weight_int8.npy"
                    np.save(weight_path, layer_data['x_proj_weight_int8'])
                    print(f"  Saved {layer_name}.x_proj weights: {layer_data['x_proj_weight_int8'].shape}")
                if 'x_proj_bias_fp32' in layer_data and layer_data['x_proj_bias_fp32'] is not None:
                    bias_path = output_dir / f"{layer_name}_x_proj_bias_fp32.npy"
                    np.save(bias_path, layer_data['x_proj_bias_fp32'])
                    print(f"  Saved {layer_name}.x_proj bias: {layer_data['x_proj_bias_fp32'].shape}")

                # Save dt_proj weights
                if 'dt_proj_weight_int8' in layer_data:
                    weight_path = output_dir / f"{layer_name}_dt_proj_weight_int8.npy"
                    np.save(weight_path, layer_data['dt_proj_weight_int8'])
                    print(f"  Saved {layer_name}.dt_proj weights: {layer_data['dt_proj_weight_int8'].shape}")
                if 'dt_proj_bias_fp32' in layer_data and layer_data['dt_proj_bias_fp32'] is not None:
                    bias_path = output_dir / f"{layer_name}_dt_proj_bias_fp32.npy"
                    np.save(bias_path, layer_data['dt_proj_bias_fp32'])
                    print(f"  Saved {layer_name}.dt_proj bias: {layer_data['dt_proj_bias_fp32'].shape}")

                # Save A_log (FP32)
                if 'A_log_fp32' in layer_data:
                    A_path = output_dir / f"{layer_name}_A_log_fp32.npy"
                    np.save(A_path, layer_data['A_log_fp32'])
                    print(f"  Saved {layer_name}.A_log: {layer_data['A_log_fp32'].shape}")

                # Save D (FP32)
                if 'D_fp32' in layer_data:
                    D_path = output_dir / f"{layer_name}_D_fp32.npy"
                    np.save(D_path, layer_data['D_fp32'])
                    print(f"  Saved {layer_name}.D: {layer_data['D_fp32'].shape}")

            # Save MambaBlock parameters (comprehensive)
            if layer_data.get('type') == 'MambaBlock':
                # in_proj weights
                if 'in_proj_weight_int8' in layer_data:
                    weight_path = output_dir / f"{layer_name}_in_proj_weight_int8.npy"
                    np.save(weight_path, layer_data['in_proj_weight_int8'])
                    print(f"  Saved {layer_name}.in_proj weights: {layer_data['in_proj_weight_int8'].shape}")
                if 'in_proj_bias_fp32' in layer_data and layer_data['in_proj_bias_fp32'] is not None:
                    bias_path = output_dir / f"{layer_name}_in_proj_bias_fp32.npy"
                    np.save(bias_path, layer_data['in_proj_bias_fp32'])
                    print(f"  Saved {layer_name}.in_proj bias: {layer_data['in_proj_bias_fp32'].shape}")

                # conv1d weights
                if 'conv1d_weight_int8' in layer_data:
                    weight_path = output_dir / f"{layer_name}_conv1d_weight_int8.npy"
                    np.save(weight_path, layer_data['conv1d_weight_int8'])
                    print(f"  Saved {layer_name}.conv1d weights: {layer_data['conv1d_weight_int8'].shape}")
                if 'conv1d_bias_fp32' in layer_data and layer_data['conv1d_bias_fp32'] is not None:
                    bias_path = output_dir / f"{layer_name}_conv1d_bias_fp32.npy"
                    np.save(bias_path, layer_data['conv1d_bias_fp32'])
                    print(f"  Saved {layer_name}.conv1d bias: {layer_data['conv1d_bias_fp32'].shape}")

                # SSM parameters within MambaBlock
                ssm_params = [
                    ('ssm_x_proj_weight_int8', 'ssm.x_proj weights'),
                    ('ssm_x_proj_bias_fp32', 'ssm.x_proj bias'),
                    ('ssm_dt_proj_weight_int8', 'ssm.dt_proj weights'),
                    ('ssm_dt_proj_bias_fp32', 'ssm.dt_proj bias'),
                    ('ssm_A_log_fp32', 'ssm.A_log'),
                    ('ssm_D_fp32', 'ssm.D'),
                ]
                for key, desc in ssm_params:
                    if key in layer_data and layer_data[key] is not None:
                        param_path = output_dir / f"{layer_name}_{key}.npy"
                        np.save(param_path, layer_data[key])
                        print(f"  Saved {layer_name}.{desc}: {np.array(layer_data[key]).shape}")

                # out_proj weights
                if 'out_proj_weight_int8' in layer_data:
                    weight_path = output_dir / f"{layer_name}_out_proj_weight_int8.npy"
                    np.save(weight_path, layer_data['out_proj_weight_int8'])
                    print(f"  Saved {layer_name}.out_proj weights: {layer_data['out_proj_weight_int8'].shape}")
                if 'out_proj_bias_fp32' in layer_data and layer_data['out_proj_bias_fp32'] is not None:
                    bias_path = output_dir / f"{layer_name}_out_proj_bias_fp32.npy"
                    np.save(bias_path, layer_data['out_proj_bias_fp32'])
                    print(f"  Saved {layer_name}.out_proj bias: {layer_data['out_proj_bias_fp32'].shape}")

            # Save MambaWrapper parameters (bidirectional - fwd and rev blocks)
            if layer_data.get('type') == 'MambaWrapper':
                # Save forward and reverse MambaBlock parameters
                for direction in ['fwd', 'rev']:
                    # in_proj weights
                    in_proj_key = f'{direction}_in_proj_weight_int8'
                    if in_proj_key in layer_data:
                        weight_path = output_dir / f"{layer_name}_{in_proj_key}.npy"
                        np.save(weight_path, layer_data[in_proj_key])
                        print(f"  Saved {layer_name}.{direction}.in_proj weights: {np.array(layer_data[in_proj_key]).shape}")

                    in_proj_bias_key = f'{direction}_in_proj_bias_fp32'
                    if in_proj_bias_key in layer_data and layer_data[in_proj_bias_key] is not None:
                        bias_path = output_dir / f"{layer_name}_{in_proj_bias_key}.npy"
                        np.save(bias_path, layer_data[in_proj_bias_key])
                        print(f"  Saved {layer_name}.{direction}.in_proj bias: {np.array(layer_data[in_proj_bias_key]).shape}")

                    # conv1d weights
                    conv_key = f'{direction}_conv1d_weight_int8'
                    if conv_key in layer_data:
                        weight_path = output_dir / f"{layer_name}_{conv_key}.npy"
                        np.save(weight_path, layer_data[conv_key])
                        print(f"  Saved {layer_name}.{direction}.conv1d weights: {np.array(layer_data[conv_key]).shape}")

                    conv_bias_key = f'{direction}_conv1d_bias_fp32'
                    if conv_bias_key in layer_data and layer_data[conv_bias_key] is not None:
                        bias_path = output_dir / f"{layer_name}_{conv_bias_key}.npy"
                        np.save(bias_path, layer_data[conv_bias_key])
                        print(f"  Saved {layer_name}.{direction}.conv1d bias: {np.array(layer_data[conv_bias_key]).shape}")

                    # SSM parameters
                    ssm_params = [
                        (f'{direction}_ssm_x_proj_weight_int8', f'{direction}.ssm.x_proj weights'),
                        (f'{direction}_ssm_x_proj_bias_fp32', f'{direction}.ssm.x_proj bias'),
                        (f'{direction}_ssm_dt_proj_weight_int8', f'{direction}.ssm.dt_proj weights'),
                        (f'{direction}_ssm_dt_proj_bias_fp32', f'{direction}.ssm.dt_proj bias'),
                        (f'{direction}_ssm_A_log_fp32', f'{direction}.ssm.A_log'),
                        (f'{direction}_ssm_D_fp32', f'{direction}.ssm.D'),
                    ]
                    for key, desc in ssm_params:
                        if key in layer_data and layer_data[key] is not None:
                            param_path = output_dir / f"{layer_name}_{key}.npy"
                            np.save(param_path, layer_data[key])
                            print(f"  Saved {layer_name}.{desc}: {np.array(layer_data[key]).shape}")

                    # out_proj weights
                    out_proj_key = f'{direction}_out_proj_weight_int8'
                    if out_proj_key in layer_data:
                        weight_path = output_dir / f"{layer_name}_{out_proj_key}.npy"
                        np.save(weight_path, layer_data[out_proj_key])
                        print(f"  Saved {layer_name}.{direction}.out_proj weights: {np.array(layer_data[out_proj_key]).shape}")

                    out_proj_bias_key = f'{direction}_out_proj_bias_fp32'
                    if out_proj_bias_key in layer_data and layer_data[out_proj_bias_key] is not None:
                        bias_path = output_dir / f"{layer_name}_{out_proj_bias_key}.npy"
                        np.save(bias_path, layer_data[out_proj_bias_key])
                        print(f"  Saved {layer_name}.{direction}.out_proj bias: {np.array(layer_data[out_proj_bias_key]).shape}")

        print(f"[PASS] Saved weights to {output_dir}")


def test_extractor():
    """Test BrevitasExtractor with SimpleCNN model."""
    print("="*80)
    print("Testing BrevitasExtractor")
    print("="*80)

    # Import the model
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from brevitas_clean import SimpleCNNQuantizedClean

    # Try to load trained model, fallback to untrained
    trained_model_path = Path(__file__).parent.parent / "artifacts" / "trained_brevitas_model.pth"

    if trained_model_path.exists():
        print(f"[PASS] Loading TRAINED model from: {trained_model_path}")
        print()
        checkpoint = torch.load(trained_model_path, map_location='cpu')
        model = SimpleCNNQuantizedClean(in_chans=1, num_classes=10, bit_width=8)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    else:
        print("[WARN]  Trained model not found, using untrained model with random weights")
        print(f"   Expected at: {trained_model_path}")
        print(f"   Run 'python train_brevitas_model.py' first for meaningful weights")
        print()

        # Create and initialize model with random weights
        model = SimpleCNNQuantizedClean(in_chans=1, num_classes=10, bit_width=8)
        model.eval()

        # Run a forward pass to initialize quantization parameters
        dummy_input = torch.randn(1, 1, 28, 28)
        with torch.no_grad():
            _ = model(dummy_input)

    # Extract network info
    extractor = BrevitasExtractor(model)
    network_info = extractor.extract_all()

    # Print summary
    print("\n" + "="*80)
    print("Network Summary")
    print("="*80)

    for layer_name, layer_data in network_info.items():
        layer_type = layer_data['type']
        print(f"\n{layer_name} ({layer_type}):")

        if layer_type == 'QuantIdentity':
            print(f"  scale: {layer_data['scale']:.6f}")

        elif layer_type == 'QuantConv2d':
            print(f"  in_channels: {layer_data['in_channels']}")
            print(f"  out_channels: {layer_data['out_channels']}")
            print(f"  kernel_size: {layer_data['kernel_size']}")
            print(f"  scale_weight: {layer_data['scale_weight']:.6f}")
            print(f"  weight_int8 shape: {layer_data['weight_int8'].shape}")

        elif layer_type == 'QuantLinear':
            print(f"  in_features: {layer_data['in_features']}")
            print(f"  out_features: {layer_data['out_features']}")
            print(f"  scale_weight: {layer_data['scale_weight']:.6f}")
            print(f"  weight_int8 shape: {layer_data['weight_int8'].shape}")

        elif layer_type == 'MaxPool2d':
            print(f"  kernel_size: {layer_data['kernel_size']}")
            print(f"  stride: {layer_data['stride']}")

        elif layer_type == 'Flatten':
            print(f"  start_dim: {layer_data['start_dim']}")

    # Compute residency policy (L2 vs L3)
    print("\n" + "="*80)
    print("Computing Residency Policy (L2 vs L3)")
    print("="*80)
    extractor._compute_residency_policy()

    # Save to files
    print("\n" + "="*80)
    print("Saving Network Information")
    print("="*80)

    extractor.save_to_json("golden_outputs/network_info.json")
    extractor.save_weights("golden_outputs/weights/")

    print("\n[PASS] BrevitasExtractor test complete!")


if __name__ == "__main__":
    test_extractor()
