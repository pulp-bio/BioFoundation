# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Cerebro Original - Full-sized Cerebro Transformer with L3 Weight Streaming

This is the Cerebro transformer matching the original paper specs:
- 6 transformer blocks with alternating channel/temporal attention
- Embedding dimension 180 with 5 attention heads (head_dim=36)
- Designed for EEG signal processing

Since the full model weights (~2.3MB) exceed GAP9 L2 (~1MB), this uses
L3 weight streaming - weights are stored in L3 (HyperRAM) and loaded
on-demand during inference, similar to ResNet-18.

Original Cerebro Configuration:
    - embed_dim=180
    - num_heads=5
    - num_blocks=6
    - num_channels=64 (EEG electrodes, reduced here for manageable seq_len)
    - temporal_len=30 (time samples, reduced here for manageable seq_len)
    - mlp_ratio=4

For this test, we use reduced spatial dimensions but full model depth:
    - num_channels=14 (reduced from 64)
    - temporal_len=8 (reduced from 30)
    - seq_len = 14 * 8 = 112 tokens

This keeps the transformer architecture full-sized while fitting in memory.

Note this is a bit of a WIP, full full model will come later.
"""

import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat

try:
    from brevitas_custom_layers import QuantAdd, QuantAlternatingAttention, QuantMean
except ImportError:
    from tests.test_networks.brevitas_custom_layers import QuantAdd, QuantAlternatingAttention, QuantMean


class CerebroOriginalBlock(nn.Module):
    """
    Full-sized Cerebro transformer block with alternating attention.

    Structure:
    - LayerNorm -> AlternatingAttention -> Add (residual)
    - LayerNorm -> MLP (Linear -> GELU -> Linear) -> Add (residual)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_channels: int,
        temporal_len: int,
        block_idx: int,
        mlp_ratio: int = 4
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp_dim = embed_dim * mlp_ratio
        self.block_idx = block_idx

        # First sub-block: LayerNorm -> AlternatingAttention -> Add
        self.norm1 = nn.LayerNorm(embed_dim)
        self.post_norm1_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )

        self.attn = QuantAlternatingAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_channels=num_channels,
            temporal_len=temporal_len,
            block_idx=block_idx,
            use_integer_softmax=True
        )

        self.add1 = QuantAdd()

        # Second sub-block: LayerNorm -> MLP -> Add
        self.norm2 = nn.LayerNorm(embed_dim)
        self.post_norm2_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )

        self.mlp_fc1 = qnn.QuantLinear(
            embed_dim, self.mlp_dim,
            bias=True,
            input_quant=Int8ActPerTensorFloat,
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
            input_quant=Int8ActPerTensorFloat,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=None,
            return_quant_tensor=True
        )

        self.add2 = QuantAdd()

    def forward(self, x):
        # x: (B, seq_len, embed_dim)

        # First sub-block: Attention with residual
        residual1 = x
        x = self.norm1(x)
        x = self.post_norm1_quant(x)
        x = self.attn(x)
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


class CerebroOriginal(nn.Module):
    """
    Full-sized Cerebro transformer with alternating channel/temporal attention.

    Input shape: [B, num_channels, temporal_len]
    Output shape: [B, num_classes]
    """

    def __init__(
        self,
        num_channels: int = 14,
        temporal_len: int = 8,
        embed_dim: int = 180,
        num_heads: int = 5,
        num_blocks: int = 6,
        num_classes: int = 4,
        mlp_ratio: int = 4
    ):
        super().__init__()
        self.num_channels = num_channels
        self.temporal_len = temporal_len
        self.embed_dim = embed_dim
        self.seq_len = num_channels * temporal_len

        # Input quantization
        self.input_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )

        # Linear projection: (num_channels) -> embed_dim
        self.proj = qnn.QuantLinear(
            num_channels, embed_dim,
            bias=True,
            input_quant=Int8ActPerTensorFloat,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=None,
            return_quant_tensor=True
        )

        # Transformer blocks with alternating attention
        self.blocks = nn.ModuleList([
            CerebroOriginalBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_channels=num_channels,
                temporal_len=temporal_len,
                block_idx=i,
                mlp_ratio=mlp_ratio
            )
            for i in range(num_blocks)
        ])

        # Final LayerNorm
        self.final_norm = nn.LayerNorm(embed_dim)
        self.post_final_norm_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )

        # Mean pooling over sequence dimension
        self.mean_pool = QuantMean(dim=1, keepdim=False)

        # Classifier
        self.classifier = qnn.QuantLinear(
            embed_dim, num_classes,
            bias=True,
            input_quant=Int8ActPerTensorFloat,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=None,
            return_quant_tensor=False
        )

    def forward(self, x):
        # x: [B, num_channels, temporal_len]
        B, C, T = x.shape

        # Quantize input
        x = self.input_quant(x)

        # Transpose to [B, temporal_len, num_channels] for linear projection
        if hasattr(x, 'value'):
            x = x.value
        x = x.transpose(1, 2).contiguous()  # [B, T, C]

        # Project channels to embed_dim: [B, T, embed_dim]
        x = self.proj(x)

        # Reshape to sequence format for alternating attention
        if hasattr(x, 'value'):
            x = x.value

        # Tile temporal features across channels for alternating attention
        # x: [B, T, D] -> [B, C, T, D] -> [B, C*T, D]
        x = x.unsqueeze(1).expand(-1, self.num_channels, -1, -1)  # [B, C, T, D]
        x = x.reshape(B, self.seq_len, self.embed_dim)  # [B, C*T, D]

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Extract tensor value if needed
        if hasattr(x, 'value'):
            x = x.value

        # Final norm
        x = self.final_norm(x)
        x = self.post_final_norm_quant(x)

        # Mean pool over sequence: [B, C*T, D] -> [B, D]
        x = self.mean_pool(x)

        # Classify
        x = self.classifier(x)

        return x


def create_model():
    """Create the original full-sized Cerebro model"""
    return CerebroOriginal(
        num_channels=14,      # Reduced from 64 for memory
        temporal_len=8,       # Reduced from 30 for memory
        embed_dim=180,        # ORIGINAL: full 180 embed_dim
        num_heads=5,          # ORIGINAL: 5 heads
        num_blocks=6,         # ORIGINAL: 6 blocks
        num_classes=4,
        mlp_ratio=4           # ORIGINAL: 4x MLP expansion
    )


def get_sample_input():
    """Get sample input for testing"""
    return torch.randn(1, 14, 8)


if __name__ == "__main__":
    model = create_model()
    model.eval()

    x = get_sample_input()
    print(f"Model: CerebroOriginal (Full-sized)")
    print(f"Input shape: {x.shape}")

    with torch.no_grad():
        output = model(x)

    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {num_params:,}")

    # Print model architecture
    print("\nModel Architecture (ORIGINAL CEREBRO SPECS):")
    print(f"  - Input: [B, {model.num_channels}, {model.temporal_len}]")
    print(f"  - Projection: {model.num_channels} -> {model.embed_dim}")
    print(f"  - Sequence length: {model.seq_len}")
    print(f"  - Number of blocks: {len(model.blocks)}")
    print(f"  - Attention heads: {model.blocks[0].attn.num_heads}")
    print(f"  - Head dimension: {model.blocks[0].attn.head_dim}")
    print(f"  - MLP hidden dim: {model.blocks[0].mlp_dim}")

    print("\nAlternating Attention Pattern:")
    for i, block in enumerate(model.blocks):
        attn_type = "channel" if block.block_idx % 2 == 0 else "temporal"
        print(f"  Block {i}: {attn_type} attention")

    # Estimate memory usage
    print("\nEstimated Memory Usage:")
    weight_bytes = num_params  # INT8 weights
    print(f"  Weights: {weight_bytes:,} bytes ({weight_bytes/1024:.1f} KB, {weight_bytes/1024/1024:.2f} MB)")
    act_bytes = model.seq_len * model.embed_dim * 1  # INT8
    print(f"  Per-layer activations: {act_bytes:,} bytes ({act_bytes/1024:.1f} KB)")
    mlp_bytes = model.seq_len * model.blocks[0].mlp_dim * 1
    print(f"  MLP hidden: {mlp_bytes:,} bytes ({mlp_bytes/1024:.1f} KB)")

    # Per-block parameter breakdown
    print("\nPer-block parameter count:")
    for i, block in enumerate(model.blocks):
        block_params = sum(p.numel() for p in block.parameters())
        print(f"  Block {i}: {block_params:,} params")

    # MLP layer sizes (these are the big ones)
    print("\nMLP layer sizes (per block):")
    mlp_fc1_params = model.embed_dim * model.blocks[0].mlp_dim
    mlp_fc2_params = model.blocks[0].mlp_dim * model.embed_dim
    print(f"  mlp_fc1: {model.embed_dim} x {model.blocks[0].mlp_dim} = {mlp_fc1_params:,} params")
    print(f"  mlp_fc2: {model.blocks[0].mlp_dim} x {model.embed_dim} = {mlp_fc2_params:,} params")
    print(f"  Total MLP per block: {mlp_fc1_params + mlp_fc2_params:,} params")
    print(f"  Total MLP all blocks: {(mlp_fc1_params + mlp_fc2_params) * len(model.blocks):,} params")
