# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Simplified Transformer Test Network

Tests the complete transformer architecture pipeline:
- PatchEmbed (Conv2d projection)
- Transformer Block:
  - LayerNorm -> MHSA -> Add (residual)
  - LayerNorm -> MLP (Linear -> GELU -> Linear) -> Add (residual)
- Final LayerNorm
- Linear classifier

This validates LayerNorm, GELU, MHSA, and Add working together
before scaling up to the full 8-block EMG transformer.
"""

import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat

# Import custom Brevitas layers
try:
    from brevitas_custom_layers import QuantAdd
except ImportError:
    from tests.test_networks.brevitas_custom_layers import QuantAdd


class Permute(nn.Module):
    """Permute layer to rearrange tensor dimensions."""
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class Squeeze(nn.Module):
    """Squeeze layer to remove singleton dimensions."""
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)


class SimpleTransformerBlock(nn.Module):
    """
    Single transformer block with:
    - LayerNorm -> MHSA -> Add
    - LayerNorm -> MLP -> Add
    """
    def __init__(self, embed_dim=64, num_heads=2, mlp_ratio=2):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = embed_dim * mlp_ratio

        # First sub-block: LayerNorm -> MHSA -> Add
        self.norm1 = nn.LayerNorm(embed_dim)
        self.post_norm1_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )

        # MHSA components (using separate Q/K/V projections like our existing MHSA)
        try:
            from brevitas_custom_layers import QuantMultiHeadAttention
        except ImportError:
            from tests.test_networks.brevitas_custom_layers import QuantMultiHeadAttention

        self.mhsa = QuantMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            seq_len=196,  # Will match MNIST reshaped tokens
            pool_sequence='none'  # Don't pool - preserve sequence for residual
        )

        self.add1 = QuantAdd()  # Residual connection

        # Second sub-block: LayerNorm -> MLP -> Add
        self.norm2 = nn.LayerNorm(embed_dim)
        self.post_norm2_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )

        # MLP: Linear -> GELU -> Linear
        self.mlp_fc1 = qnn.QuantLinear(
            embed_dim, self.mlp_dim,
            bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=None,
            return_quant_tensor=True
        )

        self.mlp_gelu = nn.GELU()
        self.post_gelu_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )

        self.mlp_fc2 = qnn.QuantLinear(
            self.mlp_dim, embed_dim,
            bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=None,
            return_quant_tensor=True
        )

        self.add2 = QuantAdd()  # Residual connection

    def forward(self, x):
        # x: (B, seq_len, embed_dim)

        # First sub-block: Attention with residual
        residual1 = x
        x = self.norm1(x)
        x = self.post_norm1_quant(x)
        x = self.mhsa(x)
        x = self.add1(x, residual1)

        # Second sub-block: MLP with residual
        residual2 = x
        x = self.norm2(x)
        x = self.post_norm2_quant(x)
        x = self.mlp_fc1(x)
        x = self.mlp_gelu(x)
        x = self.post_gelu_quant(x)
        x = self.mlp_fc2(x)
        x = self.add2(x, residual2)

        return x


class SimpleTransformer(nn.Module):
    """
    Simplified transformer for testing:
    - PatchEmbed (Conv2d)
    - 1 Transformer block
    - Final LayerNorm
    - Pooling + Linear classifier
    """
    def __init__(self, in_channels=4, seq_len=32, patch_size=4,
                 embed_dim=64, num_heads=2, num_classes=10):
        super().__init__()

        self.in_channels = in_channels
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (seq_len // patch_size) * in_channels

        # Input quantization
        self.input_quant_conv = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )

        # Patch embedding: Conv2d projects patches to embeddings
        # Input: (B, 1, in_channels, seq_len)
        # Output: (B, embed_dim, in_channels, seq_len/patch_size)
        self.patch_embed = qnn.QuantConv2d(
            1, embed_dim,
            kernel_size=(1, patch_size),
            stride=(1, patch_size),
            padding=0,
            bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=None,
            return_quant_tensor=True
        )

        # Flatten and permute to get (B, num_patches, embed_dim)
        self.flatten = nn.Flatten(start_dim=2)  # Flatten spatial dims
        self.permute = Permute(0, 2, 1)  # (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)

        self.input_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )

        # Transformer block
        self.transformer_block = SimpleTransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=2
        )

        # Final LayerNorm
        self.final_norm = nn.LayerNorm(embed_dim)
        self.post_final_norm_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )

        # Global average pooling over sequence
        self.permute_for_pool = Permute(0, 2, 1)  # (B, num_patches, embed_dim) -> (B, embed_dim, num_patches)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.squeeze_pool = Squeeze(-1)  # (B, embed_dim, 1) -> (B, embed_dim)

        # Classifier
        self.classifier = qnn.QuantLinear(
            embed_dim, num_classes,
            bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=None,
            return_quant_tensor=False
        )

    def forward(self, x):
        # Handle MNIST input: (B, 1, 28, 28) -> (B, 4, 196)
        # Or general 3D input: (B, in_channels, seq_len)
        if x.dim() == 4:
            # MNIST format: (B, 1, H, W) -> (B, C, T)
            B, _, H, W = x.shape
            # Reshape to (B, in_channels, seq_len)
            # Split H into in_channels groups
            x = x.view(B, 1, H, W)
            # Reshape: treat each row as a channel
            x = x.view(B, self.in_channels, (H * W) // self.in_channels)

        # x: (B, in_channels, seq_len)
        B, C, T = x.shape

        # Add channel dim for Conv2d: (B, 1, C, T)
        x = x.unsqueeze(1)

        # Quantize input
        x = self.input_quant_conv(x)

        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, C, T/patch_size)

        # Flatten and reshape to (B, num_patches, embed_dim)
        x = self.flatten(x)  # (B, embed_dim, C * T/patch_size)
        x = self.permute(x)  # (B, num_patches, embed_dim)

        x = self.input_quant(x)

        # Transformer block
        x = self.transformer_block(x)

        # Final norm
        x = self.final_norm(x)
        x = self.post_final_norm_quant(x)

        # Pool over sequence: (B, num_patches, embed_dim) -> (B, embed_dim)
        x = self.permute_for_pool(x)  # (B, embed_dim, num_patches)
        x = self.global_pool(x)  # (B, embed_dim, 1)
        x = self.squeeze_pool(x)  # (B, embed_dim)

        # Classify
        x = self.classifier(x)

        return x


def create_model():
    """Create the simplified transformer model"""
    return SimpleTransformer(
        in_channels=4,
        seq_len=32,
        patch_size=4,
        embed_dim=64,
        num_heads=2,
        num_classes=10
    )


def get_sample_input():
    """Get sample input for testing"""
    return torch.randn(1, 4, 32)


if __name__ == "__main__":
    model = create_model()
    model.eval()

    x = get_sample_input()
    print(f"Model: SimpleTransformer")
    print(f"Input shape: {x.shape}")

    with torch.no_grad():
        output = model(x)

    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")

    # Print model architecture
    print("\nModel Architecture:")
    for name, module in model.named_children():
        print(f"  {name}: {module.__class__.__name__}")
