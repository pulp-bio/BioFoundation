# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Test Network 31: LUNA Full (EEG foundation model) - Full Architecture

This implements the full LUNA model matching the original LUNA.py architecture:
- CNN-based PatchEmbedNetwork (3x Conv2D + GroupNorm + GELU)
- RFFT FrequencyFeatureEmbedder (magnitude + phase features + MLP)
- Pre-computed NeRF channel embeddings (computed offline, stored as weights)
- CrossAttentionBlock with 3-layer self-attention refinement
- 8x RoPE Transformer blocks
- ClassificationHeadWithQueries (learned aggregation + MLP)

Input:  (B, C, T) = (1, 22, 1280)
Config: patch_size=40, embed_dim=64, num_heads=2, depth=8, num_queries=4

Architecture:
    Input [1, 22, 1280]
        ↓
    +-- CNN PatchEmbed (3x Conv2D + GN + GELU) → [1, 704, 64]
    |
    +-- RFFT + MLP (42 → 168 → 64) → [1, 704, 64]
        ↓ (add)
    Patch tokens [1, 704, 64]
        ↓
    + Pre-computed NeRF channel embeddings [22, 64]
        ↓
    Reshape to [32, 22, 64] per patch
        ↓
    CrossAttn + 3-layer self-attn → [32, 4, 64]
        ↓
    Reshape to [1, 32, 256]
        ↓
    8x RoPE Transformer blocks
        ↓
    ClassificationHead (cross-attn pool + MLP)
        ↓
    Output [1, num_classes]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat

from brevitas.nn import QuantConv2d, QuantLinear, QuantIdentity

try:
    from brevitas_custom_layers import (
        QuantAdd,
        QuantRoPESelfAttention,
        QuantCrossAttentionWithSelfRefine,
        QuantClassificationHeadWithMLP,
        precompute_nerf_channel_embeddings,
        DEFAULT_CHANNEL_LOCATIONS_22,
    )
except ImportError:
    from .brevitas_custom_layers import (
        QuantAdd,
        QuantRoPESelfAttention,
        QuantCrossAttentionWithSelfRefine,
        QuantClassificationHeadWithMLP,
        precompute_nerf_channel_embeddings,
        DEFAULT_CHANNEL_LOCATIONS_22,
    )


class Permute(nn.Module):
    """Permute layer to rearrange tensor dimensions."""

    def __init__(self, *dims: int):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(x, "value"):
            x = x.value
        return x.permute(*self.dims)


class Reshape(nn.Module):
    """Reshape layer to change tensor shape (batch preserved)."""

    def __init__(self, *shape: int):
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(x, "value"):
            x = x.value
        batch = x.shape[0]
        return x.reshape(batch, *self.shape)


class Squeeze(nn.Module):
    """Squeeze layer to remove singleton dimensions."""

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(x, "value"):
            x = x.value
        return x.squeeze(self.dim)


class Unsqueeze(nn.Module):
    """Unsqueeze layer to add singleton dimensions."""

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(x, "value"):
            x = x.value
        return x.unsqueeze(self.dim)


class RFFT(nn.Module):
    """
    RFFT module for frequency feature extraction.

    Computes real FFT on patches and returns concatenated magnitude + phase features.
    Output size: 2 * (patch_size // 2 + 1) = 42 for patch_size=40.
    """

    def __init__(self, patch_size: int = 40):
        super().__init__()
        self.patch_size = int(patch_size)
        self.out_features = 2 * (self.patch_size // 2 + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(x, "value"):
            x = x.value
        # x: [B, C, S, P] where P = patch_size
        B, C, S, P = x.shape

        # Compute real FFT
        freq = torch.fft.rfft(x, dim=-1)  # [B, C, S, P//2+1]

        # Extract magnitude and phase
        magnitude = torch.abs(freq)
        phase = torch.angle(freq)

        # Concatenate: [B, C, S, 2*(P//2+1)]
        features = torch.cat([magnitude, phase], dim=-1)
        return features


class RotaryTransformerBlock(nn.Module):
    """RoPE transformer block: LN -> RoPE-MHSA -> Add, LN -> MLP -> Add."""

    def __init__(self, dim: int, num_heads: int, seq_len: int, mlp_ratio: int = 4):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.seq_len = int(seq_len)
        self.mlp_dim = int(dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(self.dim)
        self.post_norm1_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )
        self.attn = QuantRoPESelfAttention(
            embed_dim=self.dim,
            num_heads=self.num_heads,
            seq_len=self.seq_len,
            pool_sequence="none",
            use_integer_softmax=True,
            return_quant_tensor=True,
        )
        self.add1 = QuantAdd(return_quant_tensor=True)

        self.norm2 = nn.LayerNorm(self.dim)
        self.post_norm2_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )
        self.mlp_fc1 = qnn.QuantLinear(
            self.dim,
            self.mlp_dim,
            bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=None,
            return_quant_tensor=False,
        )
        self.mlp_gelu = nn.GELU()
        self.post_gelu_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )
        self.mlp_fc2 = qnn.QuantLinear(
            self.mlp_dim,
            self.dim,
            bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=None,
            return_quant_tensor=False,
        )
        self.add2 = QuantAdd(return_quant_tensor=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(x, "value"):
            x = x.value
        residual = x
        x = self.norm1(x)
        x = self.post_norm1_quant(x)
        x = self.attn(x)
        x = self.add1(x, residual)

        residual = x
        if hasattr(x, "value"):
            x = x.value
        x = self.norm2(x)
        x = self.post_norm2_quant(x)
        if hasattr(x, "value"):
            x = x.value
        x = self.mlp_fc1(x)
        x = self.mlp_gelu(x)
        x = self.post_gelu_quant(x)
        if hasattr(x, "value"):
            x = x.value
        x = self.mlp_fc2(x)
        x = self.add2(x, residual)
        return x


class LUNAFullQuant(nn.Module):
    """
    Full LUNA model with quantization for ARES/GAP9 deployment.

    This matches the original LUNA.py architecture with the following adaptations:
    - NeRF channel embeddings are pre-computed offline and stored as weights
    - All operations use INT8 quantization for GAP9 execution
    - No dropout (inference only)
    - CNN patch embedding is flattened into explicit layers for ARES codegen
    """

    def __init__(
        self,
        num_channels: int = 22,
        seq_len: int = 1280,
        patch_size: int = 40,
        embed_dim: int = 64,
        num_queries: int = 4,
        depth: int = 8,
        num_heads: int = 2,
        num_classes: int = 2,
        channel_locations: torch.Tensor = None,
    ):
        super().__init__()
        assert seq_len % patch_size == 0, "seq_len must be divisible by patch_size"
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_channels = int(num_channels)
        self.seq_len = int(seq_len)
        self.patch_size = int(patch_size)
        self.num_patches = int(seq_len // patch_size)
        self.num_tokens = self.num_channels * self.num_patches  # 704
        self.embed_dim = int(embed_dim)
        self.num_queries = int(num_queries)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.hidden_dim = int(embed_dim * num_queries)  # 256
        self.hidden_heads = int(num_heads * num_queries)  # 8
        self.num_classes = int(num_classes)

        # CNN output dimensions
        self.cnn_out_channels = embed_dim // 4  # 16
        self.cnn_out_width = 4  # After stride=10 on patch_size=40

        # Use default channel locations if not provided
        if channel_locations is None:
            channel_locations = DEFAULT_CHANNEL_LOCATIONS_22

        # Input quantization
        self.input_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

        # === CNN Patch Embedding Branch (flattened for ARES codegen) ===
        # Reshape: [B, C, T] -> [B, C, S, P] -> [B, C*S, P] -> [B, 1, C*S, P]
        self.cnn_reshape1 = Reshape(self.num_channels, self.num_patches, self.patch_size)
        self.cnn_reshape2 = Reshape(self.num_tokens, self.patch_size)
        self.cnn_reshape_4d = Reshape(1, self.num_tokens, self.patch_size)  # Add channel dim

        # CNN parameters (matching original LUNA PatchEmbedNetwork)
        cnn_groups = 4
        kernel_size = self.patch_size // 2  # 20 for patch_size=40

        # Conv1: Large kernel for initial feature extraction
        # Input: [B, 1, 704, 40] -> Output: [B, 16, 704, 4]
        self.cnn_conv1 = QuantConv2d(
            in_channels=1,
            out_channels=self.cnn_out_channels,
            kernel_size=(1, kernel_size - 1),  # (1, 19)
            stride=(1, kernel_size // 2),       # (1, 10)
            padding=(0, kernel_size // 2 - 1),  # (0, 9)
            bias=True,
            weight_bit_width=8,
            return_quant_tensor=False,
        )
        self.cnn_gn1 = nn.GroupNorm(cnn_groups, self.cnn_out_channels)
        self.cnn_gelu1 = nn.GELU()
        self.cnn_post_gelu1_quant = QuantIdentity(bit_width=8, return_quant_tensor=True)

        # Conv2: Refine features
        self.cnn_conv2 = QuantConv2d(
            in_channels=self.cnn_out_channels,
            out_channels=self.cnn_out_channels,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1),
            bias=True,
            weight_bit_width=8,
            return_quant_tensor=False,
        )
        self.cnn_gn2 = nn.GroupNorm(cnn_groups, self.cnn_out_channels)
        self.cnn_gelu2 = nn.GELU()
        self.cnn_post_gelu2_quant = QuantIdentity(bit_width=8, return_quant_tensor=True)

        # Conv3: Final refinement
        self.cnn_conv3 = QuantConv2d(
            in_channels=self.cnn_out_channels,
            out_channels=self.cnn_out_channels,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1),
            bias=True,
            weight_bit_width=8,
            return_quant_tensor=False,
        )
        self.cnn_gn3 = nn.GroupNorm(cnn_groups, self.cnn_out_channels)
        self.cnn_gelu3 = nn.GELU()
        self.cnn_post_gelu3_quant = QuantIdentity(bit_width=8, return_quant_tensor=True)

        # Reshape CNN output: [B, 16, 704, 4] -> [B, 704, 64]
        self.cnn_permute = Permute(0, 2, 3, 1)  # [B, 704, 4, 16]
        self.cnn_reshape_out = Reshape(self.num_tokens, self.cnn_out_width * self.cnn_out_channels)

        # === RFFT Frequency Embedding Branch ===
        self.rfft_reshape = Reshape(self.num_channels, self.num_patches, self.patch_size)
        self.rfft_op = RFFT(patch_size=self.patch_size)
        self.rfft_post_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )
        self.rfft_reshape_tokens = Reshape(
            self.num_tokens,
            2 * (self.patch_size // 2 + 1)
        )

        # Frequency embedding MLP: 42 -> 168 -> 64
        in_features = 2 * (self.patch_size // 2 + 1)
        hidden_features = 4 * in_features
        self.rfft_fc1 = qnn.QuantLinear(
            in_features,
            hidden_features,
            bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=None,
            return_quant_tensor=False,
        )
        self.rfft_gelu = nn.GELU()
        self.rfft_post_gelu_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )
        self.rfft_fc2 = qnn.QuantLinear(
            hidden_features,
            self.embed_dim,
            bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=None,
            return_quant_tensor=False,
        )
        self.rfft_post_out_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

        # Add CNN + RFFT embeddings
        self.add_cnn_rfft = QuantAdd(return_quant_tensor=True)

        # Pre-computed NeRF channel embeddings (stored as weight)
        # Shape: [num_channels, embed_dim] = [22, 64]
        nerf_embed = precompute_nerf_channel_embeddings(channel_locations, self.embed_dim)
        self.register_buffer("channel_nerf_embed", nerf_embed, persistent=True)
        self.channel_nerf_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

        # Add channel embeddings to patch tokens
        self.add_channel_embed = QuantAdd(return_quant_tensor=True)

        # Reshape for per-patch cross-attention: [B, 704, 64] -> [B*32, 22, 64]
        # We need to do this carefully with explicit reshapes

        # Cross-attention with 3-layer self-attention refinement
        self.cross_attn_unify = QuantCrossAttentionWithSelfRefine(
            num_queries=self.num_queries,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=4 * self.embed_dim,
            kv_len=self.num_channels,
            use_integer_softmax=True,
            return_quant_tensor=True,
        )

        # Post cross-attention quantization
        self.post_unify_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

        # Temporal encoder: 8x RoPE transformer blocks
        self.blocks = nn.ModuleList([
            RotaryTransformerBlock(
                dim=self.hidden_dim,
                num_heads=self.hidden_heads,
                seq_len=self.num_patches,
                mlp_ratio=4,
            )
            for _ in range(self.depth)
        ])
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.post_final_norm_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

        # Classification head with MLP
        self.classifier = QuantClassificationHeadWithMLP(
            embed_dim=self.embed_dim,
            num_queries=self.num_queries,
            num_heads=self.hidden_heads,
            num_classes=self.num_classes,
            use_integer_softmax=True,
            return_quant_tensor=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, C, T] e.g., [1, 22, 1280]

        Returns:
            Output logits [B, num_classes] e.g., [1, 2]
        """
        B = x.shape[0]

        # Input quantization
        x = self.input_quant(x)

        # === Branch 1: CNN Patch Embedding (flattened) ===
        if hasattr(x, "value"):
            x_cnn = x.value
        else:
            x_cnn = x
        x_cnn = self.cnn_reshape1(x_cnn)  # [B, 22, 32, 40]
        x_cnn = self.cnn_reshape2(x_cnn)  # [B, 704, 40]
        x_cnn = self.cnn_reshape_4d(x_cnn)  # [B, 1, 704, 40]

        # Conv1 + GN + GELU
        x_cnn = self.cnn_conv1(x_cnn)
        if hasattr(x_cnn, "value"):
            x_cnn = x_cnn.value
        x_cnn = self.cnn_gn1(x_cnn)
        x_cnn = self.cnn_gelu1(x_cnn)
        x_cnn = self.cnn_post_gelu1_quant(x_cnn)
        if hasattr(x_cnn, "value"):
            x_cnn = x_cnn.value

        # Conv2 + GN + GELU
        x_cnn = self.cnn_conv2(x_cnn)
        if hasattr(x_cnn, "value"):
            x_cnn = x_cnn.value
        x_cnn = self.cnn_gn2(x_cnn)
        x_cnn = self.cnn_gelu2(x_cnn)
        x_cnn = self.cnn_post_gelu2_quant(x_cnn)
        if hasattr(x_cnn, "value"):
            x_cnn = x_cnn.value

        # Conv3 + GN + GELU
        x_cnn = self.cnn_conv3(x_cnn)
        if hasattr(x_cnn, "value"):
            x_cnn = x_cnn.value
        x_cnn = self.cnn_gn3(x_cnn)
        x_cnn = self.cnn_gelu3(x_cnn)
        x_cnn = self.cnn_post_gelu3_quant(x_cnn)  # [B, 16, 704, 4]
        if hasattr(x_cnn, "value"):
            x_cnn = x_cnn.value

        x_cnn = self.cnn_permute(x_cnn)  # [B, 704, 4, 16]
        x_cnn = self.cnn_reshape_out(x_cnn)  # [B, 704, 64]

        # === Branch 2: RFFT Frequency Embedding ===
        if hasattr(x, "value"):
            x_freq = x.value
        else:
            x_freq = x
        x_freq = self.rfft_reshape(x_freq)  # [B, 22, 32, 40]
        x_freq = self.rfft_op(x_freq)  # [B, 22, 32, 42]
        x_freq = self.rfft_post_quant(x_freq)
        x_freq = self.rfft_reshape_tokens(x_freq)  # [B, 704, 42]

        # Frequency MLP
        if hasattr(x_freq, "value"):
            x_freq = x_freq.value
        x_freq = self.rfft_fc1(x_freq)
        x_freq = self.rfft_gelu(x_freq)
        x_freq = self.rfft_post_gelu_quant(x_freq)
        if hasattr(x_freq, "value"):
            x_freq = x_freq.value
        x_freq = self.rfft_fc2(x_freq)
        x_freq = self.rfft_post_out_quant(x_freq)  # [B, 704, 64]

        # === Add CNN + RFFT features ===
        x = self.add_cnn_rfft(x_cnn, x_freq)  # [B, 704, 64]

        # === Add pre-computed NeRF channel embeddings ===
        # Expand channel embeddings for all patches: [22, 64] -> [B, 704, 64]
        channel_embed = self.channel_nerf_embed.unsqueeze(0)  # [1, 22, 64]
        channel_embed = channel_embed.repeat(B, self.num_patches, 1)  # [B, 22*32, 64] = [B, 704, 64]
        channel_embed = self.channel_nerf_quant(channel_embed)
        x = self.add_channel_embed(x, channel_embed)  # [B, 704, 64]

        # === Cross-attention per patch ===
        # Reshape to [B*num_patches, num_channels, embed_dim]
        if hasattr(x, "value"):
            x = x.value
        x = x.view(B, self.num_channels, self.num_patches, self.embed_dim)
        x = x.permute(0, 2, 1, 3)  # [B, 32, 22, 64]
        x = x.reshape(B * self.num_patches, self.num_channels, self.embed_dim)  # [B*32, 22, 64]

        # Cross-attention with self-refinement
        x = self.cross_attn_unify(x)  # [B*32, 4, 64]

        # Reshape to encoder input: [B*32, 4, 64] -> [B, 32, 256]
        if hasattr(x, "value"):
            x = x.value
        x = x.view(B, self.num_patches, self.num_queries * self.embed_dim)  # [B, 32, 256]
        x = self.post_unify_quant(x)

        # === Temporal encoder ===
        for block in self.blocks:
            x = block(x)

        # Final normalization
        if hasattr(x, "value"):
            x = x.value
        x = self.final_norm(x)
        x = self.post_final_norm_quant(x)

        # === Classification ===
        x = self.classifier(x)  # [B, num_classes]

        return x


class LUNAFullTest(LUNAFullQuant):
    """
    Alias class used by `tests/generate_all_tests.py`.

    Uses default LUNA_base configuration.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__(
            num_channels=22,
            seq_len=1280,
            patch_size=40,
            embed_dim=64,
            num_queries=4,
            depth=8,
            num_heads=2,
            num_classes=num_classes,
        )


if __name__ == "__main__":
    print("Testing LUNAFullQuant...")

    # Create model
    model = LUNAFullQuant(
        num_channels=22,
        seq_len=1280,
        patch_size=40,
        embed_dim=64,
        num_queries=4,
        depth=8,
        num_heads=2,
        num_classes=2,
    )
    model.eval()

    # Test forward pass
    x = torch.randn(1, 22, 1280)
    print(f"Input shape: {x.shape}")

    with torch.no_grad():
        out = model(x)

    if hasattr(out, "value"):
        out = out.value

    print(f"Output shape: {out.shape}")
    print(f"Output: {out}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Print architecture summary
    print("\nArchitecture summary:")
    print(f"  Input: [B, {model.num_channels}, {model.seq_len}]")
    print(f"  Patches: {model.num_patches} patches of size {model.patch_size}")
    print(f"  Tokens after patchify: [B, {model.num_tokens}, {model.embed_dim}]")
    print(f"  After cross-attn: [B, {model.num_patches}, {model.hidden_dim}]")
    print(f"  Transformer: {model.depth} blocks, {model.hidden_heads} heads")
    print(f"  Output: [B, {model.num_classes}]")

    print("\n[PASS] LUNAFullQuant test passed!")
