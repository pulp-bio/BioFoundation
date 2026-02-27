# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Test 30: FEMBA Patch Embedder Test Network

Pipeline:
  INPUT (1, 1, 10, 32) - [batch, channels, freq_bins, time_steps]
  -> QuantIdentity (input quantization)
  -> QuantPatchEmbed (patch_size=2, stride=2, embed_dim=35)
  -> [B, 16, 175] - [batch, seq_len, d_model]
  -> QuantMambaWrapper #1 (bidirectional)
  -> QuantIdentity (inter-block quantization)
  -> QuantMambaWrapper #2 (bidirectional)
  -> QuantIdentity (post-block quantization)
  -> Global Average Pooling (over sequence)
  -> QuantLinear classifier (10 classes)

Purpose: Validate the full FEMBA-style architecture with:
  - Patch embedding (Conv2D + reshape)
  - Stacked bidirectional MAMBA blocks
  - Classification head

Architecture:
    EMG_input -> PatchEmbed -> BiMamba_1 -> BiMamba_2 -> pool -> classifier

Data Format:
- Input: [batch, 1, 10, 32] (EMG spectrogram: 1 channel, 10 freq bins, 32 time steps)
- After PatchEmbed: [batch, 16, 175] (seq_len=16, d_model=5*35=175)
- Output: [batch, num_classes]
"""

import torch
import torch.nn as nn
from brevitas import nn as qnn

from .brevitas_custom_layers import QuantPatchEmbed, QuantMambaWrapper


class FEMBAPatchEmbedderTest(nn.Module):
    """
    Test network with Patch Embedding and Bidirectional MAMBA blocks.

    This validates the FEMBA-style architecture:
    - Patch embedding converts 2D input to sequence
    - Bidirectional MAMBA processes the sequence
    - Global pooling and classifier produce output

    Args:
        inp_size: Input spatial size (height, width) - default (10, 32)
        patch_size: Patch size for embedding - default 2
        stride: Stride for patch embedding - default 2
        in_chans: Number of input channels - default 1
        embed_dim: Embedding dimension per patch row - default 35
        d_state: SSM state dimension - default 4
        conv_kernel: Conv1d kernel size in MAMBA - default 4
        num_blocks: Number of Bi-Mamba blocks - default 2
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

        # d_inner for MAMBA (typically 2x d_model, but we keep it smaller for memory)
        self.d_inner = min(2 * self.d_model, 256)  # Cap at 256 to manage memory

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

        # Stack of Bi-Mamba blocks with inter-block quantization
        self.blocks = nn.ModuleList()
        self.inter_quants = nn.ModuleList()

        for i in range(num_blocks):
            # Bi-Mamba block
            self.blocks.append(
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

            # Inter-block quantization
            self.inter_quants.append(
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
        Forward pass.

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

        # Pass through stacked Bi-Mamba blocks
        for i, (block, inter_quant) in enumerate(zip(self.blocks, self.inter_quants)):
            x = block(x)
            x = inter_quant(x)

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
    print("Test 30: FEMBA Patch Embedder Network")
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

    model = FEMBAPatchEmbedderTest(
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
    print(f"\nFEMBA Patch Embedder Configuration:")
    print(f"  Input size: {model.inp_size}")
    print(f"  Patch size: {model.patch_size}")
    print(f"  Stride: {model.stride}")
    print(f"  Embed dim: {model.embed_dim}")
    print(f"  Grid: {model.grid_h} x {model.grid_w}")
    print(f"  Sequence length: {model.seq_len}")
    print(f"  Model dimension (d_model): {model.d_model}")
    print(f"  Inner dimension (d_inner): {model.d_inner}")
    print(f"  Num blocks: {model.num_blocks}")

    # Per-block info
    print(f"\nPer Bi-Mamba block:")
    print(f"  in_proj: {model.d_model} -> {2 * model.d_inner}")
    print(f"  conv1d: {model.d_inner} channels, kernel={conv_kernel}")
    print(f"  ssm: d_inner={model.d_inner}, d_state={d_state}")
    print(f"  out_proj: {model.d_inner} -> {model.d_model}")

    # Trace through layers
    print(f"\nLayer-by-layer trace:")
    with torch.no_grad():
        trace_x = model.input_quant(x)
        if hasattr(trace_x, 'value'):
            trace_x = trace_x.value

        trace_x = model.patch_embed(trace_x)
        trace_val = trace_x.value if hasattr(trace_x, 'value') else trace_x
        print(f"  PatchEmbed output: shape={trace_val.shape}, "
              f"norm={trace_val.norm().item():.4f}")

        for i, (block, inter_quant) in enumerate(zip(model.blocks, model.inter_quants)):
            trace_x = block(trace_x)
            trace_val = trace_x.value if hasattr(trace_x, 'value') else trace_x
            print(f"  Block {i+1} output: norm={trace_val.norm().item():.4f}, "
                  f"range=[{trace_val.min().item():.2f}, {trace_val.max().item():.2f}]")
            trace_x = inter_quant(trace_x)

    print("=" * 60)
    print("Test PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_model()
