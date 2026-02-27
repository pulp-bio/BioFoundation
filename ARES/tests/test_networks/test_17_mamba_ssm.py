# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Test 25: MAMBA SSM (State Space Model) Test Network

Pipeline:
  INPUT (1, 16, 32) - [batch, seq_len, d_inner]
  -> QuantIdentity
  -> QuantSSM (state space model core)
  -> QuantIdentity
  -> Global Average Pooling (over sequence)
  -> QuantLinear classifier (10 classes)

Purpose: Validate MAMBA SSM components including:
- Softplus activation on dt (delta time)
- Discretization of A and B matrices
- Sequential state recurrence (scan)
- Element-wise multiplication for gating

Key Features:
- SSM core with learnable A, D parameters
- Input-dependent B, C projections
- dt projection with softplus
- Output gating with SiLU

Data Format:
- Input: [batch, seq_len, d_inner] (sequence-first for SSM)
- Output: [batch, num_classes]
"""

import torch
import torch.nn as nn
from brevitas import nn as qnn

from .brevitas_custom_layers import QuantSSM


class MambaSSMTest(nn.Module):
    """
    Simple test network for MAMBA SSM component.

    This network isolates the SSM computation to validate:
    - Softplus on dt
    - Discretization (dA, dB')
    - Sequential scan
    - Output projection

    Args:
        d_inner: Inner dimension / channels (default: 32)
        d_state: SSM state dimension (default: 4)
        seq_len: Sequence length (default: 16)
        num_classes: Number of output classes (default: 10)
    """

    def __init__(self, d_inner=32, d_state=4, seq_len=16, num_classes=10):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.seq_len = seq_len
        bit_width = 8

        # Input quantization
        self.input_quant = qnn.QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )

        # SSM core (without gating for simplicity)
        self.ssm = QuantSSM(
            d_inner=d_inner,
            d_state=d_state,
            bit_width=bit_width,
            return_quant_tensor=True
        )

        # Post-SSM quantization
        self.ssm_quant = qnn.QuantIdentity(
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
            d_inner,
            num_classes,
            bias=True,
            weight_bit_width=bit_width,
            return_quant_tensor=False  # Output FP32 logits
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, d_inner]

        Returns:
            Logits tensor [batch, num_classes]
        """
        # Quantize input
        x = self.input_quant(x)

        # SSM forward (no gating z)
        x = self.ssm(x, z=None)
        x = self.ssm_quant(x)

        # Extract value for pooling
        if hasattr(x, 'value'):
            x = x.value

        # Global average pool: [B, L, M] -> [B, M]
        # Need to transpose for AdaptiveAvgPool1d: [B, L, M] -> [B, M, L]
        x = x.transpose(1, 2)  # [B, M, L]
        x = self.global_pool(x)  # [B, M, 1]
        x = x.squeeze(-1)  # [B, M]

        # Quantize before classifier
        x = self.pre_classifier_quant(x)

        # Classifier
        x = self.classifier(x)
        return x


def get_sample_input(batch_size=1, seq_len=16, d_inner=32):
    """Generate a sample input tensor for testing."""
    return torch.randn(batch_size, seq_len, d_inner)


def test_model():
    """Quick sanity test of the model."""
    print("=" * 60)
    print("Test 25: MAMBA SSM Network")
    print("=" * 60)

    d_inner = 32
    d_state = 4
    seq_len = 16
    num_classes = 10

    model = MambaSSMTest(
        d_inner=d_inner,
        d_state=d_state,
        seq_len=seq_len,
        num_classes=num_classes
    )
    model.eval()

    # Generate sample input
    x = get_sample_input(batch_size=2, seq_len=seq_len, d_inner=d_inner)
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

    # Print SSM details
    print(f"\nSSM Configuration:")
    print(f"  d_inner: {model.ssm.d_inner}")
    print(f"  d_state: {model.ssm.d_state}")
    print(f"  dt_rank: {model.ssm.dt_rank}")

    print("=" * 60)
    print("Test PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_model()
