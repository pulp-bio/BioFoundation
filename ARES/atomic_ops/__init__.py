# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Atomic Operations Library for True INT8 Neural Networks

This library provides verified INT8 implementations of neural network operations.
Each operation is tested independently and can be composed to build complex networks.

Available operations:
- Quantization: quantize_linear, dequantize_linear
- Conv2D: conv2d_int8 (with INT32 accumulation)
- ReLU: relu_int8 (preserves quantization)
- MaxPool: maxpool2d_int8 (order-preserving)
- AvgPool: avgpool2d_int8 (with rescaling)
- GlobalAvgPool: global_avgpool_int8 (reduces spatial dims)
- Linear: linear_int8 (fully connected with INT32 accumulation)
- Flatten: flatten_int8 (reshape only, zero computation)
- Add: add_int8 (element-wise addition with rescaling)
- Concat: concat_int8 (channel concatenation with rescaling)
- LayerNorm: layernorm_int8 (normalization with affine transform)
- RMSNorm: rmsnorm_int8 (root mean square normalization for Llama/LLMs)
- SwiGLU: swiglu_ffn_int8 (Swish-Gated Linear Unit FFN for Llama/LLMs)
- LlamaBlock: llama_block_int8 (Complete Llama transformer decoder block)
- KVCache: KVCache (Key-Value cache for autoregressive generation)
- GELU: gelu_int8 (Gaussian Error Linear Unit activation)
- SSM: ssm_forward_int8 (Mamba v1 State Space Model)
"""

from .quantize import quantize_linear, dequantize_linear, compute_quantization_params

__all__ = [
    'quantize_linear',
    'dequantize_linear',
    'compute_quantization_params',
]

# Import all atomic operations
try:
    from .conv2d import conv2d_int8, conv2d_fp32_reference
    __all__.extend(['conv2d_int8', 'conv2d_fp32_reference'])
except ImportError:
    pass

try:
    from .relu import relu_int8, relu_fp32_reference
    __all__.extend(['relu_int8', 'relu_fp32_reference'])
except ImportError:
    pass

try:
    from .maxpool import maxpool2d_int8, maxpool2d_fp32_reference
    __all__.extend(['maxpool2d_int8', 'maxpool2d_fp32_reference'])
except ImportError:
    pass

try:
    from .linear import linear_int8, linear_fp32_reference
    __all__.extend(['linear_int8', 'linear_fp32_reference'])
except ImportError:
    pass

try:
    from .flatten import flatten_int8
    __all__.extend(['flatten_int8'])
except ImportError:
    pass

try:
    from .avgpool import avgpool2d_int8
    __all__.extend(['avgpool2d_int8'])
except ImportError:
    pass

try:
    from .globalavgpool import global_avgpool_int8, global_avgpool_int8_fast
    __all__.extend(['global_avgpool_int8', 'global_avgpool_int8_fast'])
except ImportError:
    pass

try:
    from .add import add_int8, add_int8_optimized
    __all__.extend(['add_int8', 'add_int8_optimized'])
except ImportError:
    pass

try:
    from .concat import concat_int8, concat_int8_channel
    __all__.extend(['concat_int8', 'concat_int8_channel'])
except ImportError:
    pass

try:
    from .requantize import requantize_int8
    __all__.extend(['requantize_int8'])
except ImportError:
    pass

try:
    from .mhsa import mhsa_int8_hybrid, mhsa_autoregressive_step, repeat_kv
    __all__.extend(['mhsa_int8_hybrid', 'mhsa_autoregressive_step', 'repeat_kv'])
except ImportError:
    pass

try:
    from .kv_cache import KVCache
    __all__.extend(['KVCache'])
except ImportError:
    pass

try:
    from .layernorm import (
        layernorm_int8,
        layernorm_int8_fixed_point,
        layernorm_int8_lut,
        get_builtin_layernorm_isqrt_lut,
        i_sqrt_newton,
        sqrt_q64
    )
    __all__.extend([
        'layernorm_int8',
        'layernorm_int8_fixed_point',
        'layernorm_int8_lut',
        'get_builtin_layernorm_isqrt_lut',
        'i_sqrt_newton',
        'sqrt_q64'
    ])
except ImportError:
    pass

try:
    from .rmsnorm import (
        rmsnorm_int8,
        rmsnorm_int8_fixed_point,
        rmsnorm_int8_lut,
        rmsnorm_fp32_reference,
        get_builtin_rmsnorm_isqrt_lut
    )
    __all__.extend([
        'rmsnorm_int8',
        'rmsnorm_int8_fixed_point',
        'rmsnorm_int8_lut',
        'rmsnorm_fp32_reference',
        'get_builtin_rmsnorm_isqrt_lut'
    ])
except ImportError:
    pass

try:
    from .embedding import embedding_int8
    __all__.extend(['embedding_int8'])
except ImportError:
    pass

try:
    from .groupnorm import groupnorm_int8_fixed_point
    __all__.extend(['groupnorm_int8_fixed_point'])
except ImportError:
    pass

try:
    from .rfft import rfft40_features_int8_fixed_point
    __all__.extend(['rfft40_features_int8_fixed_point'])
except ImportError:
    pass

try:
    from .rope import rope_apply_int8_q15, rope_precompute_sin_cos_q15
    __all__.extend(['rope_apply_int8_q15', 'rope_precompute_sin_cos_q15'])
except ImportError:
    pass

try:
    from .cross_attention import cross_attention_int8_hybrid
    __all__.extend(['cross_attention_int8_hybrid'])
except ImportError:
    pass

try:
    from .cross_attention_self_refine import cross_attention_with_self_refine_int8
    __all__.extend(['cross_attention_with_self_refine_int8'])
except ImportError:
    pass

try:
    from .classification_head import classification_head_with_mlp_int8
    __all__.extend(['classification_head_with_mlp_int8'])
except ImportError:
    pass

try:
    from .gelu import (
        gelu_int8, gelu_fp32, gelu_fp32_reference,
        gelu_int8_lut, get_builtin_gelu_lut,
        gelu_int8_ibert  # I-BERT polynomial approximation
    )
    __all__.extend([
        'gelu_int8', 'gelu_fp32', 'gelu_fp32_reference',
        'gelu_int8_lut', 'get_builtin_gelu_lut',
        'gelu_int8_ibert'
    ])
except ImportError:
    pass

try:
    from .transpose import transpose_int8, transpose_2d_batch_int8
    __all__.extend(['transpose_int8', 'transpose_2d_batch_int8'])
except ImportError:
    pass

try:
    from .flip import flip_sequence_int8, flip_sequence_fp32
    __all__.extend(['flip_sequence_int8', 'flip_sequence_fp32'])
except ImportError:
    pass

try:
    from .conv1d_depthwise import conv1d_depthwise_int8, conv1d_depthwise_fp32_reference
    __all__.extend(['conv1d_depthwise_int8', 'conv1d_depthwise_fp32_reference'])
except ImportError:
    pass

try:
    from .silu import silu_int8_lut, generate_silu_lut_q13
    __all__.extend(['silu_int8_lut', 'generate_silu_lut_q13'])
except ImportError:
    pass

try:
    from .swiglu import (
        swiglu_ffn_int8,
        swiglu_ffn_int8_fused,
        swiglu_ffn_fp32_reference
    )
    __all__.extend([
        'swiglu_ffn_int8',
        'swiglu_ffn_int8_fused',
        'swiglu_ffn_fp32_reference'
    ])
except ImportError:
    pass

try:
    from .llama_block import (
        LlamaBlockConfig,
        LlamaBlockWeights,
        llama_block_int8,
        llama_block_fp32_reference
    )
    __all__.extend([
        'LlamaBlockConfig',
        'LlamaBlockWeights',
        'llama_block_int8',
        'llama_block_fp32_reference'
    ])
except ImportError:
    pass

try:
    from .tanh import tanh_int8, tanh_int8_ibert, tanh_fp32_reference
    __all__.extend(['tanh_int8', 'tanh_int8_ibert', 'tanh_fp32_reference'])
except ImportError:
    pass

try:
    from .softmax import (
        softmax_int8, softmax_int8_lut, softmax_int8_ibert,
        softmax_int8_lut_pure_integer, softmax_fp32_reference,
        build_exp_lut, get_exp_lut
    )
    __all__.extend([
        'softmax_int8', 'softmax_int8_lut', 'softmax_int8_ibert',
        'softmax_int8_lut_pure_integer', 'softmax_fp32_reference',
        'build_exp_lut', 'get_exp_lut'
    ])
except ImportError:
    pass

try:
    from .ssm import (
        ssm_forward_int8,
        ssm_scan_q15,
        ssm_discretize_q15,
        ssm_gate_silu_q13,
        generate_exp_lut_q15,
        generate_phi1_lut_q15,
        generate_silu_gate_lut_q13
    )
    __all__.extend([
        'ssm_forward_int8',
        'ssm_scan_q15',
        'ssm_discretize_q15',
        'ssm_gate_silu_q13',
        'generate_exp_lut_q15',
        'generate_phi1_lut_q15',
        'generate_silu_gate_lut_q13'
    ])
except ImportError:
    pass


try:
    from .alternating_attention import (
        alternating_attention_int8,
        alternating_attention_fp32_reference,
        matmul_int8
    )
    __all__.extend([
        'alternating_attention_int8',
        'alternating_attention_fp32_reference',
        'matmul_int8'
    ])
except ImportError:
    pass
