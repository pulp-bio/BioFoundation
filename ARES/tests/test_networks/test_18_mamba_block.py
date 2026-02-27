# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Test 26: Full MAMBA Block Test Network

Pipeline:
  INPUT (1, 16, 32) - [batch, seq_len, d_model]
  -> QuantIdentity
  -> QuantMambaBlock (full block with all components)
      - in_proj: d_model -> 2*d_inner (split into x and z branches)
      - x branch: conv1d -> SiLU -> SSM core
      - z branch: SiLU gate
      - output: x * z -> out_proj
  -> QuantIdentity
  -> Global Average Pooling (over sequence)
  -> QuantLinear classifier (10 classes)

Purpose: Validate complete MAMBA block combining:
- Input projection with split
- Depthwise causal Conv1d
- SiLU activation
- SSM (State Space Model) core
- Gated output mechanism
- Output projection

Architecture (MAMBA block internals):
    x_in -> in_proj -> [x, z] split
                        |
                        v
                  conv1d -> SiLU -> SSM
                        |              |
                        +-> SiLU(z) <--+  (gating)
                               |
                               v
                          out_proj -> x_out

Data Format:
- Input: [batch, seq_len, d_model]
- Output: [batch, num_classes]
"""

import torch
import torch.nn as nn
from brevitas import nn as qnn

from .brevitas_custom_layers import QuantMambaBlock


class MambaBlockTest(nn.Module):
    """
    Test network for complete MAMBA block.

    This network validates the full MAMBA architecture including:
    - Input/output projections
    - Conv1d with causal padding
    - SiLU activations
    - SSM core (discretization, scan, gating)
    - Gated output mechanism

    Args:
        d_model: Model dimension (default: 32)
        d_inner: Inner dimension (default: 64, typically 2*d_model)
        d_state: SSM state dimension (default: 4)
        conv_kernel: Conv1d kernel size (default: 4)
        seq_len: Sequence length (default: 16)
        num_classes: Number of output classes (default: 10)
    """

    def __init__(
        self,
        d_model=32,
        d_inner=64,
        d_state=4,
        conv_kernel=4,
        seq_len=16,
        num_classes=10
    ):
        super().__init__()
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

        # Full MAMBA block
        self.mamba_block = QuantMambaBlock(
            d_model=d_model,
            d_inner=d_inner,
            d_state=d_state,
            conv_kernel=conv_kernel,
            bit_width=bit_width,
            return_quant_tensor=True
        )

        # Post-block quantization
        self.block_quant = qnn.QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
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
            d_model,  # Output of mamba block is d_model
            num_classes,
            bias=True,
            weight_bit_width=bit_width,
            return_quant_tensor=False  # Output FP32 logits
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Logits tensor [batch, num_classes]
        """
        # Quantize input
        x = self.input_quant(x)

        # MAMBA block forward
        x = self.mamba_block(x)
        x = self.block_quant(x)

        # Extract value for pooling
        if hasattr(x, 'value'):
            x = x.value

        # Global average pool: [B, L, M] -> [B, M]
        # Transpose for AdaptiveAvgPool1d: [B, L, M] -> [B, M, L]
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
    print("Test 26: Full MAMBA Block Network")
    print("=" * 60)

    d_model = 32
    d_inner = 64
    d_state = 4
    conv_kernel = 4
    seq_len = 16
    num_classes = 10

    model = MambaBlockTest(
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
    print(f"\nMAMBA Block Configuration:")
    print(f"  d_model: {model.d_model}")
    print(f"  d_inner: {model.d_inner}")
    print(f"  d_state: {model.d_state}")
    print(f"  conv_kernel: {conv_kernel}")
    print(f"  seq_len: {seq_len}")

    # Show subcomponents
    print(f"\nSubcomponents:")
    print(f"  in_proj: {d_model} -> {2 * d_inner}")
    print(f"  conv1d: {d_inner} channels, kernel={conv_kernel}")
    print(f"  ssm: d_inner={d_inner}, d_state={d_state}")
    print(f"  out_proj: {d_inner} -> {d_model}")

    print("=" * 60)
    print("Test PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_model()
