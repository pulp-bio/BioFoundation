# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Test Network 43: Llama Block with SwiGLU FFN

This test validates the full Llama-style transformer block with:
- RMSNorm (instead of LayerNorm)
- Standard MHSA
- SwiGLU FFN (the key Llama innovation)
- Residual connections

Architecture:
- Input: [B, seq_len, dim] = [1, 16, 64]
- RMSNorm(64) - attention pre-norm
- MHSA(dim=64, num_heads=4)
- Residual add
- RMSNorm(64) - FFN pre-norm
- SwiGLU FFN (gate=W1, up=W3, down=W2)
- Residual add
- Final RMSNorm -> Mean pooling -> Classifier

Key goal: Validate SwiGLU FFN integration in the pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from brevitas.nn import QuantLinear, QuantIdentity
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat

from .brevitas_custom_layers import QuantAdd, QuantMultiHeadAttention


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        if hasattr(x, 'value'):
            x = x.value
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class QuantSwiGLU(nn.Module):
    """Quantized SwiGLU FFN (Llama-style).

    Formula: y = W2(silu(W1(x)) * W3(x))

    - W1: Gate projection (dim -> hidden_dim)
    - W3: Up projection (dim -> hidden_dim)
    - W2: Down projection (hidden_dim -> dim)
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()

        # Gate projection W1
        self.w1 = QuantLinear(
            dim,
            hidden_dim,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=None,
            bias=False,
            return_quant_tensor=False,
        )

        # Up projection W3
        self.w3 = QuantLinear(
            dim,
            hidden_dim,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=None,
            bias=False,
            return_quant_tensor=False,
        )

        # Quantize after gate and up projections
        self.gate_quant = QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )
        self.up_quant = QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

        # Down projection W2
        self.w2 = QuantLinear(
            hidden_dim,
            dim,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=None,
            bias=False,
            return_quant_tensor=False,
        )

        # Output quantization
        self.out_quant = QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

    def forward(self, x):
        # Extract value if QuantTensor
        if hasattr(x, 'value'):
            x_val = x.value
        else:
            x_val = x

        # Gate path: W1(x) -> SiLU
        gate = self.w1(x_val)
        gate = F.silu(gate)
        gate = self.gate_quant(gate)

        # Up path: W3(x)
        up = self.w3(x_val)
        up = self.up_quant(up)

        # Extract values for multiply
        if hasattr(gate, 'value'):
            gate = gate.value
        if hasattr(up, 'value'):
            up = up.value

        # Element-wise multiply (gating)
        hidden = gate * up

        # Down path: W2(hidden)
        out = self.w2(hidden)
        out = self.out_quant(out)

        return out


class LlamaBlockSwiGLU(nn.Module):
    """Llama-style transformer block with SwiGLU FFN."""

    def __init__(self, dim: int, seq_len: int, num_heads: int, hidden_dim: int):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len

        # Attention pre-norm
        self.attn_norm = RMSNorm(dim)
        self.attn_norm_quant = QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

        # Multi-head self-attention
        self.attn = QuantMultiHeadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            seq_len=seq_len,
        )

        # Attention residual
        self.attn_add = QuantAdd(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

        # FFN pre-norm
        self.ffn_norm = RMSNorm(dim)
        self.ffn_norm_quant = QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

        # SwiGLU FFN
        self.ffn = QuantSwiGLU(dim, hidden_dim)

        # FFN residual
        self.ffn_add = QuantAdd(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

    def forward(self, x):
        # Attention branch
        h = self.attn_norm(x)
        h = self.attn_norm_quant(h)
        h = self.attn(h)
        x = self.attn_add(x, h)

        # FFN branch with SwiGLU
        h = self.ffn_norm(x)
        h = self.ffn_norm_quant(h)
        h = self.ffn(h)
        x = self.ffn_add(x, h)

        return x


class LlamaSwiGLU(nn.Module):
    """Llama-style transformer with SwiGLU FFN.

    Input: [batch, seq_len, dim]
    Output: [batch, num_classes]
    """

    def __init__(
        self,
        dim: int = 64,
        seq_len: int = 16,
        num_heads: int = 4,
        hidden_dim: int = 172,  # ~2.67x dim (typical Llama ratio)
        num_classes: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len

        # Input quantization
        self.input_quant = QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

        # Single Llama block with SwiGLU
        self.block = LlamaBlockSwiGLU(dim, seq_len, num_heads, hidden_dim)

        # Final RMSNorm
        self.final_norm = RMSNorm(dim)
        self.final_norm_quant = QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

        # Mean pooling quantization
        self.pool_quant = QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

        # Classifier
        self.classifier = QuantLinear(
            dim,
            num_classes,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=None,
            return_quant_tensor=False,
        )

    def forward(self, x):
        # Input quantization
        x = self.input_quant(x)

        # Llama block with SwiGLU
        x = self.block(x)

        # Final norm
        x = self.final_norm(x)
        x = self.final_norm_quant(x)

        # Extract value
        if hasattr(x, 'value'):
            x = x.value

        # Mean pooling over sequence
        x = x.mean(dim=1)
        x = self.pool_quant(x)

        # Classifier
        x = self.classifier(x)

        return x


def get_network():
    """Return the test network."""
    return LlamaSwiGLU(
        dim=64,
        seq_len=16,
        num_heads=4,
        hidden_dim=172,  # ~2.67x dim
        num_classes=4,
    )


def get_input_shape():
    """Return the expected input shape (without batch dimension)."""
    return (16, 64)  # [seq_len, dim]


def get_input_range():
    """Return the expected input value range."""
    return (-1.0, 1.0)


if __name__ == "__main__":
    model = get_network()
    model.eval()

    batch_size = 1
    seq_len, dim = get_input_shape()
    x = torch.randn(batch_size, seq_len, dim)

    with torch.no_grad():
        y = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output: {y}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
