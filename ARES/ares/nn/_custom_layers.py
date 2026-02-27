# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Central import bridge for ARES custom layer implementations.

Why this indirection exists:
- It keeps `ares.nn` user imports stable while reusing the currently
  validated custom layer implementations already used by test networks.
- This is a temporary coupling to test infrastructure to avoid changing
  extraction/runtime behavior while usability APIs are introduced.

Planned direction:
- Move/duplicate these implementations into standalone `ares.nn` modules
  and keep this bridge only for backward compatibility during transition.
"""

from __future__ import annotations

try:
    from tests.test_networks.brevitas_custom_layers import (  # type: ignore
        QuantAlternatingAttention,
        QuantConv1dDepthwise,
        QuantCrossAttention,
        QuantMambaBlock,
        QuantMambaWrapper,
        QuantMultiHeadAttention,
        QuantPatchEmbed,
        QuantRoPESelfAttention,
        QuantSSM,
        QuantSelfAttention,
        QuantSiLU,
    )
except ImportError:
    from test_networks.brevitas_custom_layers import (  # type: ignore
        QuantAlternatingAttention,
        QuantConv1dDepthwise,
        QuantCrossAttention,
        QuantMambaBlock,
        QuantMambaWrapper,
        QuantMultiHeadAttention,
        QuantPatchEmbed,
        QuantRoPESelfAttention,
        QuantSSM,
        QuantSelfAttention,
        QuantSiLU,
    )

__all__ = [
    "QuantSelfAttention",
    "QuantMultiHeadAttention",
    "QuantRoPESelfAttention",
    "QuantCrossAttention",
    "QuantAlternatingAttention",
    "QuantConv1dDepthwise",
    "QuantSiLU",
    "QuantSSM",
    "QuantMambaWrapper",
    "QuantMambaBlock",
    "QuantPatchEmbed",
]
