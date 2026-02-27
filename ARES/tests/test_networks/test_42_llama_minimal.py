# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Test Network 42: Minimal Llama Block (RMSNorm + MHSA + RMSNorm + Linear FFN)

This is a minimal test for Llama-style components, focusing on:
- RMSNorm (root mean square normalization) instead of LayerNorm
- Standard MHSA (no GQA initially for simplicity)
- Simple linear FFN (instead of SwiGLU for initial testing)

Architecture:
- Input: [B, seq_len, dim] = [1, 32, 64]
- RMSNorm(64) - attention pre-norm
- MHSA(dim=64, num_heads=4) - standard multi-head attention
- Residual add
- RMSNorm(64) - FFN pre-norm
- Linear(64, 256) - up projection
- GELU activation
- Linear(256, 64) - down projection
- Residual add
- Final QuantLinear(64, 4) - classifier

Key goal: Validate RMSNorm integration in the pipeline.
"""

import torch
import torch.nn as nn
from brevitas.nn import QuantLinear, QuantIdentity
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat

from .brevitas_custom_layers import QuantAdd, QuantMultiHeadAttention


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    RMSNorm is simpler than LayerNorm - it normalizes by the root mean square
    without subtracting the mean. Used in Llama and other modern LLMs.

    Formula: y = x / sqrt(mean(x^2) + eps) * weight
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Extract value from QuantTensor if needed
        if hasattr(x, 'value'):
            x = x.value
        # Compute root mean square
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return (x / rms) * self.weight


class QuantMLP(nn.Module):
    """Quantized MLP with GELU activation (simplified FFN for testing)."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = QuantLinear(
            dim,
            hidden_dim,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=None,
            return_quant_tensor=False,
        )
        self.gelu = nn.GELU()
        self.gelu_quant = QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )
        self.fc2 = QuantLinear(
            hidden_dim,
            dim,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=None,
            return_quant_tensor=False,
        )
        self.out_quant = QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.gelu_quant(x)
        x = self.fc2(x)
        x = self.out_quant(x)
        return x


class LlamaBlock(nn.Module):
    """Minimal Llama-style transformer block.

    Structure:
    - RMSNorm + MHSA + Residual
    - RMSNorm + MLP + Residual
    """

    def __init__(self, dim: int, seq_len: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        hidden_dim = int(dim * mlp_ratio)

        # Attention pre-norm (RMSNorm instead of LayerNorm)
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

        # FFN pre-norm (RMSNorm instead of LayerNorm)
        self.ffn_norm = RMSNorm(dim)
        self.ffn_norm_quant = QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

        # Feed-forward network
        self.mlp = QuantMLP(dim, hidden_dim)

        # FFN residual
        self.ffn_add = QuantAdd(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

    def forward(self, x):
        # Attention branch: RMSNorm -> MHSA -> Add
        h = self.attn_norm(x)
        h = self.attn_norm_quant(h)
        h = self.attn(h)
        x = self.attn_add(x, h)

        # FFN branch: RMSNorm -> MLP -> Add
        h = self.ffn_norm(x)
        h = self.ffn_norm_quant(h)
        h = self.mlp(h)
        x = self.ffn_add(x, h)

        return x


class MinimalLlama(nn.Module):
    """Minimal Llama-style transformer for testing RMSNorm integration.

    Input: [batch, seq_len, dim]
    Output: [batch, num_classes]
    """

    def __init__(
        self,
        dim: int = 64,
        seq_len: int = 32,
        num_heads: int = 4,
        num_classes: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len

        # Input quantization
        self.input_quant = QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

        # Single Llama block (for minimal testing)
        self.block = LlamaBlock(dim, seq_len, num_heads, mlp_ratio)

        # Final RMSNorm
        self.final_norm = RMSNorm(dim)
        self.final_norm_quant = QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

        # Global average pooling over sequence
        # (Manual implementation for compatibility)
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

        # Llama block
        x = self.block(x)

        # Final norm
        x = self.final_norm(x)
        x = self.final_norm_quant(x)

        # Extract value from QuantTensor if needed
        if hasattr(x, 'value'):
            x = x.value

        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # [B, seq_len, dim] -> [B, dim]
        x = self.pool_quant(x)

        # Classifier
        x = self.classifier(x)

        return x


# Network configuration for test harness
def get_network():
    """Return the test network."""
    return MinimalLlama(
        dim=64,
        seq_len=32,
        num_heads=4,
        num_classes=4,
        mlp_ratio=4.0,
    )


def get_input_shape():
    """Return the expected input shape (without batch dimension)."""
    return (32, 64)  # [seq_len, dim]


def get_input_range():
    """Return the expected input value range (for random data generation)."""
    return (-1.0, 1.0)


# For standalone testing
if __name__ == "__main__":
    model = get_network()
    model.eval()

    # Test forward pass
    batch_size = 1
    seq_len, dim = get_input_shape()
    x = torch.randn(batch_size, seq_len, dim)

    with torch.no_grad():
        y = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output: {y}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
