# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Test 27: Stacked MAMBA Blocks Test Network

Pipeline:
  INPUT (1, 16, 32) - [batch, seq_len, d_model]
  -> QuantIdentity
  -> QuantMambaBlock x N (stacked blocks)
  -> QuantIdentity
  -> Global Average Pooling (over sequence)
  -> QuantLinear classifier (10 classes)

Purpose: Validate multiple stacked MAMBA blocks to measure:
- Cycle scaling with depth
- Memory usage patterns
- Accuracy preservation through multiple blocks

Architecture:
    x_in -> [MambaBlock_0] -> [MambaBlock_1] -> ... -> [MambaBlock_N-1] -> pool -> classifier

Each MambaBlock:
    x -> in_proj -> [x, z] split
                     |
                     v
               conv1d -> SiLU -> SSM
                     |              |
                     +-> SiLU(z) <--+  (gating)
                            |
                            v
                       out_proj -> x_out
"""

import torch
import torch.nn as nn
from brevitas import nn as qnn

from .brevitas_custom_layers import QuantMambaBlock


class MambaStackedTest(nn.Module):
    """
    Test network with multiple stacked MAMBA blocks.

    Args:
        num_blocks: Number of MAMBA blocks to stack (default: 3)
        d_model: Model dimension (default: 32)
        d_inner: Inner dimension (default: 64, typically 2*d_model)
        d_state: SSM state dimension (default: 4)
        conv_kernel: Conv1d kernel size (default: 4)
        seq_len: Sequence length (default: 16)
        num_classes: Number of output classes (default: 10)
    """

    def __init__(
        self,
        num_blocks=3,
        d_model=32,
        d_inner=64,
        d_state=4,
        conv_kernel=4,
        seq_len=16,
        num_classes=10
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.seq_len = seq_len
        bit_width = 8

        # Input quantization
        self.input_quant = qnn.QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )

        # Stacked MAMBA blocks
        self.mamba_blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = QuantMambaBlock(
                d_model=d_model,
                d_inner=d_inner,
                d_state=d_state,
                conv_kernel=conv_kernel,
                bit_width=bit_width,
                return_quant_tensor=True
            )
            self.mamba_blocks.append(block)

        # Inter-block quantization layers
        self.block_quants = nn.ModuleList()
        for i in range(num_blocks):
            quant = qnn.QuantIdentity(
                bit_width=bit_width,
                return_quant_tensor=True
            )
            self.block_quants.append(quant)

        # Global average pool over sequence
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Pre-classifier quantization
        self.pre_classifier_quant = qnn.QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )

        # Final classifier
        self.classifier = qnn.QuantLinear(
            d_model,
            num_classes,
            bias=True,
            weight_bit_width=bit_width,
            return_quant_tensor=False  # Output FP32 logits
        )

    def forward(self, x):
        """
        Forward pass through stacked MAMBA blocks.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Logits tensor [batch, num_classes]
        """
        # Quantize input
        x = self.input_quant(x)

        # Pass through each MAMBA block
        for i, (block, quant) in enumerate(zip(self.mamba_blocks, self.block_quants)):
            x = block(x)
            x = quant(x)

        # Extract value for pooling
        if hasattr(x, 'value'):
            x = x.value

        # Global average pool: [B, L, M] -> [B, M]
        x = x.transpose(1, 2)  # [B, M, L]
        x = self.global_pool(x)  # [B, M, 1]
        x = x.squeeze(-1)  # [B, M]

        # Quantize before classifier
        x = self.pre_classifier_quant(x)

        # Classifier
        x = self.classifier(x)
        return x


def get_sample_input(batch_size=1, seq_len=16, d_model=32):
    """Generate a sample input tensor for testing."""
    return torch.randn(batch_size, seq_len, d_model)


def test_model():
    """Quick sanity test of the model."""
    print("=" * 60)
    print("Test 27: Stacked MAMBA Blocks Network")
    print("=" * 60)

    num_blocks = 3
    d_model = 32
    d_inner = 64
    d_state = 4
    conv_kernel = 4
    seq_len = 16
    num_classes = 10

    model = MambaStackedTest(
        num_blocks=num_blocks,
        d_model=d_model,
        d_inner=d_inner,
        d_state=d_state,
        conv_kernel=conv_kernel,
        seq_len=seq_len,
        num_classes=num_classes
    )
    model.eval()

    # Generate sample input
    x = get_sample_input(batch_size=2, seq_len=seq_len, d_model=d_model)
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
    print(f"\nStacked MAMBA Configuration:")
    print(f"  num_blocks: {num_blocks}")
    print(f"  d_model: {model.d_model}")
    print(f"  d_inner: {model.d_inner}")
    print(f"  d_state: {model.d_state}")
    print(f"  conv_kernel: {conv_kernel}")
    print(f"  seq_len: {seq_len}")

    # Estimate MACs per block
    macs_per_block = (
        d_model * 2 * d_inner +  # in_proj
        d_inner * conv_kernel * seq_len +  # conv1d
        d_inner * (4 + 2 * d_state) * seq_len +  # x_proj
        d_inner * 4 * seq_len +  # dt_proj (dt_rank=4)
        d_inner * d_state * seq_len * 3 +  # SSM scan
        d_inner * seq_len +  # gating
        d_inner * d_model  # out_proj
    )
    print(f"\nEstimated MACs per block: ~{macs_per_block:,}")
    print(f"Total estimated MACs: ~{macs_per_block * num_blocks:,}")

    print("=" * 60)
    print("Test PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_model()
