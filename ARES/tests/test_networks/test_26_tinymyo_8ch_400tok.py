# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Test Network 36: TinyMyo (8 channels, 400 tokens)

This is a scaled-down TinyMyo variant intended to mirror the production
TinyMyo architecture while reducing the input channel count and the token
context length:

- Input: [B, 1, 8, 1000]
- PatchEmbed: Conv2d(1, 192, kernel=(1, 20), stride=(1, 20))
  → 50 patches per channel x 8 channels = 400 tokens
  → Output tokens: [B, 400, 192]
- 8 x Transformer blocks:
  - LayerNorm(192)
  - Multi-head self-attention (3 heads) via QuantMultiHeadAttention (seq_len=400)
  - Residual add
  - LayerNorm(192)
  - MLP (192 → 768 → 192) with GELU
  - Residual add
- Final LayerNorm(192)
- Global average pooling over tokens → QuantLinear(192, 7)

Key goal: seq_len=400 enables MHSA L1 tiling under the current GAP9 tiler
(persistent K/V becomes feasible for head_dim=64).
"""

import torch
import torch.nn as nn
from brevitas.nn import QuantConv2d, QuantIdentity, QuantLinear
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat

from .brevitas_custom_layers import QuantMultiHeadAttention


class Permute(nn.Module):
    """Permute layer to rearrange tensor dimensions."""

    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class Reshape(nn.Module):
    """Reshape layer to change tensor shape."""

    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        batch = x.shape[0]
        return x.reshape(batch, *self.shape)


class QuantPatchEmbed8Ch(nn.Module):
    """Quantized 1D patch embedding for 8-channel EMG signals."""

    def __init__(
        self,
        img_size: int = 1000,
        patch_size: int = 20,
        in_chans: int = 8,
        embed_dim: int = 192,
        bias: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_patches = (self.img_size // self.patch_size) * self.in_chans  # 400

        self.proj = QuantConv2d(
            1,
            embed_dim,
            kernel_size=(1, self.patch_size),
            stride=(1, self.patch_size),
            bias=bias,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=None,
            return_quant_tensor=False,
        )
        self.reshape = Reshape(self.embed_dim, self.num_patches)
        self.permute = Permute(0, 2, 1)
        self.quant = QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.reshape(x)
        x = self.permute(x)
        x = self.quant(x)
        return x


class QuantMLP(nn.Module):
    """Quantized MLP with GELU activation."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int = None):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = QuantLinear(
            in_features,
            hidden_features,
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
            hidden_features,
            out_features,
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


class QuantTransformerBlock(nn.Module):
    """Quantized transformer block with pre-norm and residual connections."""

    def __init__(
        self,
        dim: int,
        seq_len: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = QuantMultiHeadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            seq_len=seq_len,
            bit_width=8,
            return_quant_tensor=True,
            pool_sequence="none",
            use_integer_softmax=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = QuantMLP(in_features=dim, hidden_features=hidden_features)
        self.qkv_bias = qkv_bias

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm)
        if hasattr(attn_out, "value"):
            attn_out = attn_out.value
        x = x + attn_out

        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        if hasattr(mlp_out, "value"):
            mlp_out = mlp_out.value
        x = x + mlp_out
        return x


class TinyMyo8Ch400Quant(nn.Module):
    """8-block TinyMyo variant with 8 channels and 400 tokens."""

    def __init__(
        self,
        img_size: int = 1000,
        patch_size: int = 20,
        in_chans: int = 8,
        embed_dim: int = 192,
        n_layer: int = 8,
        n_head: int = 3,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        num_classes: int = 7,
        conv_bias: bool = True,
    ):
        super().__init__()
        self.n_layer = n_layer
        self.n_head = n_head
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) * in_chans  # 400

        self.input_quant = QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )
        self.patch_embedding = QuantPatchEmbed8Ch(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=conv_bias,
        )

        self.blocks = nn.ModuleList(
            [
                QuantTransformerBlock(
                    dim=embed_dim,
                    seq_len=self.num_patches,
                    num_heads=n_head,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                )
                for _ in range(n_layer)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.pool_quant = QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )
        self.classifier = QuantLinear(
            embed_dim,
            num_classes,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=None,
            return_quant_tensor=False,
        )

    def forward(self, x):
        x = self.input_quant(x)
        x = self.patch_embedding(x)
        if hasattr(x, "value"):
            x = x.value
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.pool_quant(x)
        logits = self.classifier(x)
        return logits


def create_tinymyo_8ch_400tok_quant() -> TinyMyo8Ch400Quant:
    """Factory for the 8-channel / 400-token TinyMyo variant."""
    return TinyMyo8Ch400Quant(
        img_size=1000,
        patch_size=20,
        in_chans=8,
        embed_dim=192,
        n_layer=8,
        n_head=3,
        mlp_ratio=4,
        qkv_bias=True,
        num_classes=7,
        conv_bias=True,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("TinyMyo 8ch / 400tok Quantized Model")
    print("=" * 60)

    model = create_tinymyo_8ch_400tok_quant()
    model.eval()

    print("\nModel Configuration:")
    print(f"  Transformer blocks: {model.n_layer}")
    print(f"  Hidden dimension:   {model.embed_dim}")
    print(f"  Attention heads:    {model.n_head}")
    print(f"  Sequence length:    {model.num_patches}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {total_params:,}")

    test_input = torch.randn(1, 1, 8, 1000)
    with torch.no_grad():
        output = model(test_input)
    print(f"\nOutput shape: {output.shape}")

