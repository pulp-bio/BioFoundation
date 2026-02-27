# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Test Network 41: LUNA_base (EEG foundation model) — ARES bring-up

Goal:
- Exercise the 5 new ops needed for LUNA_base deployment on GAP9:
  1) Embedding
  2) GroupNorm
  3) RFFT
  4) RoPE (via RoPE-enabled MHSA)
  5) Cross-Attention

Input:  (B, C, T) = (1, 22, 1280)
Patch:  patch_size=40 -> num_patches=32

High-level structure (ARES-friendly, sequential execution):
- Patch features: Reshape -> RFFT -> MLP(42->168->64) -> GroupNorm -> GELU
- Channel embedding: Embedding(22->64) + Add to patch features
- Channel unification: CrossAttention over channels per patch (B*S, C, 64) -> (1, S, 256)
- Temporal encoder: 8x RoPE MHSA blocks at (1, 32, 256)
- Classification: CrossAttention pooling (1 query) -> Linear -> logits
"""

from __future__ import annotations

import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat

try:
    from brevitas_custom_layers import QuantAdd, QuantRoPESelfAttention, QuantCrossAttention
except ImportError:
    from .brevitas_custom_layers import QuantAdd, QuantRoPESelfAttention, QuantCrossAttention


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


class RFFT(nn.Module):
    """
    Placeholder RFFT module for extractor/codegen.

    ARES replaces this with an INT8 fixed-point RFFT feature kernel that outputs
    concatenated [magnitude, phase] features of size 2*(P//2+1) = 42 for P=40.
    """

    def __init__(self, patch_size: int = 40):
        super().__init__()
        self.patch_size = int(patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(x, "value"):
            x = x.value
        # x: [B, C, S, P]
        B, C, S, P = x.shape
        if P != self.patch_size:
            raise ValueError(f"RFFT expects patch_size={self.patch_size}, got {P}")
        out_features = 2 * (self.patch_size // 2 + 1)
        return torch.zeros((B, C, S, out_features), dtype=x.dtype, device=x.device)


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


class LUNABaseQuant(nn.Module):
    """
    LUNA_base-like quantized network for ARES.

    This is a bring-up model: it matches the LUNA tokenization and encoder shapes,
    and covers all required new ops, but keeps the patch feature extractor
    sequential (ARES execution model).
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
        num_classes: int = 5,
    ):
        super().__init__()
        assert seq_len % patch_size == 0, "seq_len must be divisible by patch_size"
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_channels = int(num_channels)
        self.seq_len = int(seq_len)
        self.patch_size = int(patch_size)
        self.num_patches = int(seq_len // patch_size)
        self.embed_dim = int(embed_dim)
        self.num_queries = int(num_queries)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.hidden_dim = int(embed_dim * num_queries)
        self.hidden_heads = int(num_heads * num_queries)
        self.num_classes = int(num_classes)

        # Input quantization (FP32 -> INT8 scale reference for ARES)
        self.input_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

        # Patchify: (1, 22, 1280) -> (1, 22, 32, 40)
        self.reshape_to_patches = Reshape(self.num_channels, self.num_patches, self.patch_size)

        # Frequency features: RFFT -> (1, 22, 32, 42) -> reshape to tokens (1, 704, 42)
        self.rfft = RFFT(patch_size=self.patch_size)
        self.post_rfft_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )
        self.reshape_rfft_to_tokens = Reshape(self.num_channels * self.num_patches, 2 * (self.patch_size // 2 + 1))

        # Frequency embedding MLP: 42 -> 168 -> 64
        in_features = 2 * (self.patch_size // 2 + 1)
        hidden_features = 4 * in_features
        self.freq_fc1 = qnn.QuantLinear(
            in_features,
            hidden_features,
            bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=None,
            return_quant_tensor=False,
        )
        self.post_freq_fc1_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )
        self.freq_gelu = nn.GELU()
        self.post_freq_gelu_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )
        self.freq_fc2 = qnn.QuantLinear(
            hidden_features,
            self.embed_dim,
            bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=None,
            return_quant_tensor=False,
        )
        self.post_freq_out_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

        # GroupNorm over embed_dim (treat embed_dim as channels)
        self.permute_for_gn = Permute(0, 2, 1)  # (B, tokens, D) -> (B, D, tokens)
        self.groupnorm = nn.GroupNorm(num_groups=4, num_channels=self.embed_dim)
        self.post_gn_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )
        self.gn_gelu = nn.GELU()
        self.post_gn_gelu_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )
        self.permute_after_gn = Permute(0, 2, 1)  # back to (B, tokens, D)

        # Channel embeddings (Embedding lookup) + Add to patch tokens
        self.channel_embed = nn.Embedding(self.num_channels, self.embed_dim)
        channel_ids = torch.arange(self.num_channels, dtype=torch.long).repeat_interleave(self.num_patches)
        self.register_buffer("channel_patch_indices", channel_ids.unsqueeze(0), persistent=False)  # [1, 704]
        self.add_channel_embed = QuantAdd(return_quant_tensor=True)

        # Channel-unification Cross-Attention:
        # tokens (1, 704, 64) -> CrossAttn(num_queries=32*4) -> (1, 128, 64) -> reshape -> (1, 32, 256)
        self.cross_attn_unify = QuantCrossAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_queries=self.num_patches * self.num_queries,
            kv_len=self.num_channels * self.num_patches,
            use_integer_softmax=True,
            return_quant_tensor=True,
        )
        self.reshape_to_encoder = Reshape(self.num_patches, self.hidden_dim)
        self.post_unify_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

        # Temporal encoder: 8x RoPE transformer blocks at (1, 32, 256)
        self.blocks = nn.ModuleList(
            [
                RotaryTransformerBlock(
                    dim=self.hidden_dim,
                    num_heads=self.hidden_heads,
                    seq_len=self.num_patches,
                    mlp_ratio=4,
                )
                for _ in range(self.depth)
            ]
        )
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.post_final_norm_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

        # Classification head: Cross-Attention pooling (1 query) -> Linear
        self.cross_attn_pool = QuantCrossAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.hidden_heads,
            num_queries=1,
            kv_len=self.num_patches,
            use_integer_softmax=True,
            return_quant_tensor=True,
        )
        self.squeeze_pool = Squeeze(dim=1)
        self.classifier = qnn.QuantLinear(
            self.hidden_dim,
            self.num_classes,
            bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=None,
            return_quant_tensor=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [B, C, T]
        x = self.input_quant(x)

        # Patchify + RFFT features
        x = self.reshape_to_patches(x)  # [1, 22, 32, 40]
        x = self.rfft(x)  # [1, 22, 32, 42]
        x = self.post_rfft_quant(x)
        x = self.reshape_rfft_to_tokens(x)  # [1, 704, 42]

        # Frequency embedding MLP
        if hasattr(x, "value"):
            x = x.value
        x = self.freq_fc1(x)
        x = self.post_freq_fc1_quant(x)
        if hasattr(x, "value"):
            x = x.value
        x = self.freq_gelu(x)
        x = self.post_freq_gelu_quant(x)
        if hasattr(x, "value"):
            x = x.value
        x = self.freq_fc2(x)
        x = self.post_freq_out_quant(x)  # [1, 704, 64]

        # GroupNorm over embed_dim (as channels)
        x = self.permute_for_gn(x)  # [1, 64, 704]
        x = self.groupnorm(x)
        x = self.post_gn_quant(x)
        if hasattr(x, "value"):
            x = x.value
        x = self.gn_gelu(x)
        x = self.post_gn_gelu_quant(x)
        x = self.permute_after_gn(x)  # [1, 704, 64]

        # Add channel embeddings
        channel_embed = self.channel_embed(self.channel_patch_indices)  # [1, 704, 64]
        x = self.add_channel_embed(x, channel_embed)

        # Cross-attention channel unification (batch preserved)
        x = self.cross_attn_unify(x)   # [1, 128, 64]
        x = self.reshape_to_encoder(x)  # [1, 32, 256]
        x = self.post_unify_quant(x)

        # Temporal encoder blocks
        for block in self.blocks:
            x = block(x)

        if hasattr(x, "value"):
            x = x.value
        x = self.final_norm(x)
        x = self.post_final_norm_quant(x)

        # Pool + classifier
        x = self.cross_attn_pool(x)  # [1, 1, 256]
        x = self.squeeze_pool(x)     # [1, 256]
        x = self.classifier(x)       # [1, num_classes]
        return x


class LUNABaseTest(LUNABaseQuant):
    """
    Alias class used by `tests/generate_all_tests.py`.

    Important: This must not wrap the model in a nested submodule, because the codegen
    expects the input quant layer to be named `input_quant` (not `model.input_quant`).
    """

    def __init__(self, num_classes: int = 5):
        super().__init__(num_classes=num_classes)
