# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Test 31: Full FEMBA Architecture Test Network

Complete FEMBA (Frequency-Enhanced Mamba) architecture with:
  - Patch Embedding (Conv2D + reshape + permute)
  - Positional Embedding (learnable parameter)
  - Encoder blocks: BiMamba + Residual + LayerNorm
  - Classification head

Pipeline:
  INPUT (1, 1, 10, 32) - [batch, channels, freq_bins, time_steps]
  -> QuantIdentity (input quantization)
  -> QuantPatchEmbed (patch_size=2, stride=2, embed_dim=35)
  -> [B, 16, 175] - [batch, seq_len, d_model]
  -> + PositionalEmbedding (learnable, same shape)
  -> Encoder Block x N:
     -> BiMamba (MambaWrapper)
     -> + Residual connection
     -> LayerNorm
     -> QuantIdentity
  -> Global Average Pool
  -> QuantLinear classifier
"""

import torch
import torch.nn as nn
from brevitas import nn as qnn
from brevitas.quant import Int8ActPerTensorFloat

from .brevitas_custom_layers import QuantPatchEmbed, QuantMambaWrapper


class FEMBAFullTest(nn.Module):
    """
    Full FEMBA architecture with positional embedding and residual connections.

    Args:
        inp_size: Input spatial size (height, width) - default (10, 32)
        patch_size: Patch size for embedding - default 2
        stride: Stride for patch embedding - default 2
        in_chans: Number of input channels - default 1
        embed_dim: Embedding dimension per patch row - default 35
        d_state: SSM state dimension - default 4
        conv_kernel: Conv1d kernel size in MAMBA - default 4
        num_blocks: Number of encoder blocks - default 2
        num_classes: Number of output classes - default 10
    """

    def __init__(
        self,
        inp_size=(10, 32),
        patch_size=2,
        stride=2,
        in_chans=1,
        embed_dim=35,
        d_state=4,
        conv_kernel=4,
        num_blocks=2,
        num_classes=10
    ):
        super().__init__()
        self.inp_size = inp_size
        self.patch_size = patch_size
        self.stride = stride
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        bit_width = 8

        # Calculate dimensions after patch embedding
        H, W = inp_size
        self.grid_h = (H - patch_size) // stride + 1
        self.grid_w = (W - patch_size) // stride + 1
        self.seq_len = self.grid_w  # Sequence length
        self.d_model = self.grid_h * embed_dim  # Model dimension

        # d_inner for MAMBA (cap to manage memory)
        self.d_inner = min(2 * self.d_model, 256)

        # Input quantization
        self.input_quant = qnn.QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )

        # Patch embedding: [B, 1, H, W] -> [B, seq_len, d_model]
        self.patch_embed = QuantPatchEmbed(
            inp_size=inp_size,
            patch_size=patch_size,
            stride=stride,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bit_width=bit_width,
            return_quant_tensor=True
        )

        # Positional embedding (learnable parameter)
        # Shape: [1, seq_len, d_model] - broadcasts over batch
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.seq_len, self.d_model)
        )
        # Initialize with small random values
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Quantization for positional embedding
        self.pos_quant = qnn.QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )

        # Scale equalizer for pos + x addition (ensures same scale)
        self.scale_equalizer = qnn.QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )

        # Encoder blocks: BiMamba + Residual + LayerNorm
        self.mamba_blocks = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.post_norm_quants = nn.ModuleList()

        for i in range(num_blocks):
            # Bi-Mamba block
            self.mamba_blocks.append(
                QuantMambaWrapper(
                    d_model=self.d_model,
                    d_inner=self.d_inner,
                    d_state=d_state,
                    conv_kernel=conv_kernel,
                    bidirectional_strategy="add",
                    bit_width=bit_width,
                    return_quant_tensor=True
                )
            )

            # LayerNorm after residual addition
            self.norm_layers.append(
                nn.LayerNorm(self.d_model)
            )

            # Post-norm quantization
            self.post_norm_quants.append(
                qnn.QuantIdentity(
                    bit_width=bit_width,
                    return_quant_tensor=True
                )
            )

        # Global average pool over sequence
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Pre-classifier quantization
        self.pre_classifier_quant = qnn.QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )

        # Final classifier
        self.classifier = qnn.QuantLinear(
            self.d_model,
            num_classes,
            bias=True,
            weight_bit_width=bit_width,
            return_quant_tensor=False  # Output FP32 logits
        )

    def forward(self, x):
        """
        Forward pass following FEMBA architecture.

        Args:
            x: Input tensor [batch, channels, height, width]

        Returns:
            Logits tensor [batch, num_classes]
        """
        # Quantize input
        x = self.input_quant(x)

        # Extract value for patch embedding
        if hasattr(x, 'value'):
            x = x.value

        # Patch embedding: [B, C, H, W] -> [B, seq_len, d_model]
        x = self.patch_embed(x)

        # Add positional embedding
        # Quantize pos_embed
        pos = self.pos_quant(self.pos_embed)

        # Scale equalization for addition
        # Extract values for scale equalizer
        if hasattr(x, 'value'):
            x_val = x.value
        else:
            x_val = x
        if hasattr(pos, 'value'):
            pos_val = pos.value
        else:
            pos_val = pos

        # Pass through scale equalizer to ensure same scale
        x = self.scale_equalizer(x_val)
        pos = self.scale_equalizer(pos_val)

        # Add positional embedding
        if hasattr(x, 'value'):
            x_val = x.value
        else:
            x_val = x
        if hasattr(pos, 'value'):
            pos_val = pos.value
        else:
            pos_val = pos

        x = x_val + pos_val

        # Re-quantize after addition
        x = self.scale_equalizer(x)

        # Encoder blocks: BiMamba + Residual + LayerNorm
        for mamba_block, norm_layer, post_norm_quant in zip(
            self.mamba_blocks, self.norm_layers, self.post_norm_quants
        ):
            # Store residual
            if hasattr(x, 'value'):
                res = x.value
            else:
                res = x

            # BiMamba forward
            x = mamba_block(x)

            # Extract tensor for residual addition
            if hasattr(x, 'value'):
                x_val = x.value
            else:
                x_val = x

            # Residual connection (FP32 addition)
            x = res + x_val

            # LayerNorm
            x = norm_layer(x)

            # Quantize output
            x = post_norm_quant(x)

        # Extract value for pooling
        if hasattr(x, 'value'):
            x = x.value

        # Global average pool: [B, L, D] -> [B, D]
        # Transpose for AdaptiveAvgPool1d: [B, L, D] -> [B, D, L]
        x = x.transpose(1, 2)  # [B, D, L]
        x = self.global_pool(x)  # [B, D, 1]
        x = x.squeeze(-1)  # [B, D]

        # Quantize before classifier
        x = self.pre_classifier_quant(x)

        # Classifier
        x = self.classifier(x)
        return x


def get_sample_input(batch_size=1, in_chans=1, inp_size=(10, 32)):
    """Generate a sample input tensor for testing."""
    return torch.randn(batch_size, in_chans, inp_size[0], inp_size[1])


def test_model():
    """Quick sanity test of the model."""
    print("=" * 60)
    print("Test 31: Full FEMBA Architecture")
    print("=" * 60)

    inp_size = (10, 32)
    patch_size = 2
    stride = 2
    in_chans = 1
    embed_dim = 35
    d_state = 4
    conv_kernel = 4
    num_blocks = 2
    num_classes = 10

    model = FEMBAFullTest(
        inp_size=inp_size,
        patch_size=patch_size,
        stride=stride,
        in_chans=in_chans,
        embed_dim=embed_dim,
        d_state=d_state,
        conv_kernel=conv_kernel,
        num_blocks=num_blocks,
        num_classes=num_classes
    )
    model.eval()

    # Generate sample input
    x = get_sample_input(batch_size=2, in_chans=in_chans, inp_size=inp_size)
    print(f"Input shape: {x.shape}")

    # Forward pass
    with torch.no_grad():
        output = model(x)

    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    print(f"Predicted classes: {output.argmax(dim=1).tolist()}")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params}")

    # Print architecture details
    print(f"\nFEMBA Full Configuration:")
    print(f"  Input size: {model.inp_size}")
    print(f"  Patch size: {model.patch_size}")
    print(f"  Stride: {model.stride}")
    print(f"  Embed dim: {model.embed_dim}")
    print(f"  Grid: {model.grid_h} x {model.grid_w}")
    print(f"  Sequence length: {model.seq_len}")
    print(f"  Model dimension (d_model): {model.d_model}")
    print(f"  Inner dimension (d_inner): {model.d_inner}")
    print(f"  Num encoder blocks: {model.num_blocks}")
    print(f"  Positional embedding shape: {model.pos_embed.shape}")

    print("=" * 60)
    print("Test PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_model()
