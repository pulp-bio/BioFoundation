# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Test Network 23: TinyMyo Tiny - Minimal EMG Transformer for fast GVSOC

Reduced version for quick simulator validation:
- 1 transformer block
- 50 seq_len (vs 800 in full TinyMyo)
- Same architecture otherwise (192 embed_dim, 3 heads, GELU MLP)

Uses QuantMultiHeadAttention for proper MHSA detection by the extractor.

Purpose: Fast cycle counting and correctness validation on GVSOC.
"""

import torch
import torch.nn as nn
from brevitas.nn import QuantLinear, QuantIdentity, QuantConv2d
from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat

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


class QuantPatchEmbedTiny(nn.Module):
    """Quantized 1D Patch Embedding for small EMG signals."""
    def __init__(
        self,
        img_size: int = 200,
        patch_size: int = 20,
        in_chans: int = 5,
        embed_dim: int = 192,
        bias: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_patches = (self.img_size // self.patch_size) * self.in_chans  # 50

        self.proj = QuantConv2d(
            1, embed_dim,
            kernel_size=(1, self.patch_size),
            stride=(1, self.patch_size),
            bias=bias,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=None,
            return_quant_tensor=False
        )
        self.reshape = Reshape(self.embed_dim, self.num_patches)
        self.permute = Permute(0, 2, 1)
        self.quant = QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.reshape(x)
        x = self.permute(x)
        x = self.quant(x)
        return x


class QuantMLPTiny(nn.Module):
    """Quantized MLP with GELU activation."""
    def __init__(self, in_features: int, hidden_features: int, out_features: int = None):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = QuantLinear(in_features, hidden_features,
                               weight_quant=Int8WeightPerTensorFloat,
                               bias_quant=None, return_quant_tensor=False)
        self.gelu = nn.GELU()
        self.gelu_quant = QuantIdentity(act_quant=Int8ActPerTensorFloat,
                                        return_quant_tensor=True)
        self.fc2 = QuantLinear(hidden_features, out_features,
                               weight_quant=Int8WeightPerTensorFloat,
                               bias_quant=None, return_quant_tensor=False)
        self.out_quant = QuantIdentity(act_quant=Int8ActPerTensorFloat,
                                       return_quant_tensor=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.gelu_quant(x)
        x = self.fc2(x)
        x = self.out_quant(x)
        return x


class QuantTransformerBlockTiny(nn.Module):
    """Quantized Transformer Block with pre-norm and residual connections.

    Uses QuantMultiHeadAttention for proper MHSA detection.
    """
    def __init__(self, dim: int, seq_len: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        # Use QuantMultiHeadAttention for proper detection
        self.attn = QuantMultiHeadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            seq_len=seq_len,
            bit_width=8,
            return_quant_tensor=True,
            pool_sequence="none",  # Don't pool - we need full sequence for residual
            use_integer_softmax=True,  # Use integer softmax for bit-exact matching
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = QuantMLPTiny(in_features=dim, hidden_features=hidden_features)

    def forward(self, x):
        # Pre-norm + attention + residual
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm)
        # Extract value if QuantTensor
        if hasattr(attn_out, 'value'):
            attn_out = attn_out.value
        x = x + attn_out

        # Pre-norm + MLP + residual
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        # Extract value if QuantTensor
        if hasattr(mlp_out, 'value'):
            mlp_out = mlp_out.value
        x = x + mlp_out
        return x


class TinyMyoTinyQuant(nn.Module):
    """
    Tiny TinyMyo for fast GVSOC validation.
    - 1 block, 50 seq_len, 192 embed_dim, 3 heads

    Uses QuantMultiHeadAttention for proper MHSA detection by the code generator.
    """
    def __init__(
        self,
        img_size: int = 200,
        patch_size: int = 20,
        in_chans: int = 5,
        embed_dim: int = 192,
        n_head: int = 3,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        num_classes: int = 7,
        conv_bias: bool = True,
    ):
        super().__init__()
        self.n_layer = 1
        self.n_head = n_head
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) * in_chans  # 50

        self.input_quant = QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )
        self.patch_embedding = QuantPatchEmbedTiny(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=conv_bias,
        )

        self.blocks = nn.ModuleList([
            QuantTransformerBlockTiny(
                dim=embed_dim,
                seq_len=self.num_patches,
                num_heads=n_head,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
            )
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.pool_quant = QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )
        self.classifier = QuantLinear(
            embed_dim, num_classes,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=None,
            return_quant_tensor=False
        )

    def forward(self, x):
        x = self.input_quant(x)
        x = self.patch_embedding(x)
        # Extract value for transformer blocks
        if hasattr(x, 'value'):
            x = x.value
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.pool_quant(x)
        logits = self.classifier(x)
        return logits


def create_tinymyo_tiny_quant():
    """Factory function to create tiny TinyMyo."""
    model = TinyMyoTinyQuant(
        img_size=200,
        patch_size=20,
        in_chans=5,
        embed_dim=192,
        n_head=3,
        mlp_ratio=4,
        qkv_bias=True,
        num_classes=7,
        conv_bias=True,
    )
    return model


if __name__ == "__main__":
    print("="*60)
    print("TinyMyo Tiny Quantized Model")
    print("="*60)

    model = create_tinymyo_tiny_quant()
    model.eval()

    print(f"\nModel Configuration:")
    print(f"  Transformer blocks: {model.n_layer}")
    print(f"  Hidden dimension: {model.embed_dim}")
    print(f"  Attention heads: {model.n_head}")
    print(f"  Sequence length: {model.num_patches}")
    print(f"  MLP ratio: 4")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {total_params:,}")

    print(f"\nTesting forward pass...")
    test_input = torch.randn(1, 1, 5, 200)  # [B, 1, C, T]
    print(f"  Input shape: {test_input.shape}")

    with torch.no_grad():
        output = model(test_input)
        print(f"  Output shape: {output.shape}")
        print(f"  Predicted class: {output.argmax(dim=1).item()}")

    print("\n[OK] Model test completed!")
