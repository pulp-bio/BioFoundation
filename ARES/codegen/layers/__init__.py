# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Layer handler extraction modules for codegen decomposition."""

from .activation_handlers import handle_silu
from .attention_handlers import handle_alternating_attention
from .conv_handlers import handle_conv1d_depthwise, handle_quantconv2d
from .elementwise_handlers import (
    handle_add,
    handle_concatenate,
    handle_mean,
    handle_zeropad2d,
)
from .linear_handlers import handle_quantlinear
from .mamba_handlers import (
    handle_mambablock,
    handle_mambawrapper,
    handle_ssm,
)
from .embedding_handlers import (
    handle_embedding,
    handle_patchembed,
    handle_positionalembedding,
)
from .norm_activation_handlers import (
    handle_gelu,
    handle_groupnorm,
    handle_layernorm,
    handle_rmsnorm,
)
from .pool_handlers import (
    handle_adaptive_avgpool1d,
    handle_avgpool2d,
    handle_globalavgpool,
    handle_maxpool2d,
)
from .quant_handlers import handle_quant_identity, handle_quant_relu
from .reshape_handlers import (
    handle_flatten,
    handle_permute,
    handle_reshape,
    handle_squeeze,
)
from .signal_handlers import handle_rfft

__all__ = [
    "handle_embedding",
    "handle_patchembed",
    "handle_positionalembedding",
    "handle_gelu",
    "handle_groupnorm",
    "handle_layernorm",
    "handle_rmsnorm",
    "handle_squeeze",
    "handle_flatten",
    "handle_reshape",
    "handle_permute",
    "handle_quant_identity",
    "handle_quant_relu",
    "handle_silu",
    "handle_rfft",
    "handle_maxpool2d",
    "handle_avgpool2d",
    "handle_globalavgpool",
    "handle_adaptive_avgpool1d",
    "handle_zeropad2d",
    "handle_add",
    "handle_concatenate",
    "handle_mean",
    "handle_quantconv2d",
    "handle_conv1d_depthwise",
    "handle_quantlinear",
    "handle_alternating_attention",
    "handle_ssm",
    "handle_mambablock",
    "handle_mambawrapper",
]
