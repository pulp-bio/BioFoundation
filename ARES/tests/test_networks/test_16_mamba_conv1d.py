# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Test 24: MAMBA Conv1D and SiLU Test Network

Pipeline:
  INPUT (1, 32, 64) - [batch, channels, length]
  -> QuantIdentity
  -> QuantConv1dDepthwise (causal, kernel=4)
  -> QuantIdentity
  -> QuantSiLU
  -> QuantIdentity
  -> Global Average Pooling (over sequence)
  -> QuantLinear classifier (10 classes)

Purpose: Validate MAMBA-style depthwise 1D convolution and SiLU activation
before integrating the full SSM block.

Key Features:
- Depthwise 1D convolution (groups=channels)
- Causal padding (left-only for autoregressive)
- SiLU (Swish) activation with 256-entry LUT

Data Format:
- Input: [batch, channels, length] (channel-first, like PyTorch Conv1d)
- Output: [batch, num_classes]
"""

import torch
import torch.nn as nn
from brevitas import nn as qnn

from .brevitas_custom_layers import QuantConv1dDepthwise, QuantSiLU


class MambaConv1DTest(nn.Module):
    """
    Simple test network for MAMBA Conv1D and SiLU components.

    Args:
        d_inner: Hidden dimension (channels) (default: 32)
        seq_len: Sequence length (default: 64)
        kernel_size: Conv1D kernel size (default: 4)
        num_classes: Number of output classes (default: 10)
    """

    def __init__(self, d_inner=32, seq_len=64, kernel_size=4, num_classes=10):
        super().__init__()
        self.d_inner = d_inner
        self.seq_len = seq_len
        self.kernel_size = kernel_size
        bit_width = 8

        # Input quantization
        self.input_quant = qnn.QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )

        # Depthwise 1D convolution (MAMBA style)
        self.conv1d = QuantConv1dDepthwise(
            in_channels=d_inner,
            kernel_size=kernel_size,
            causal=True,
            bit_width=bit_width,
            return_quant_tensor=True
        )

        # Post-conv quantization
        self.conv_quant = qnn.QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )

        # SiLU activation (MAMBA uses SiLU after conv1d)
        self.silu = QuantSiLU(
            bit_width=bit_width,
            return_quant_tensor=True
        )

        # Post-SiLU quantization
        self.silu_quant = qnn.QuantIdentity(
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
            x: Input tensor [batch, channels, length]

        Returns:
            Logits tensor [batch, num_classes]
        """
        # Quantize input
        x = self.input_quant(x)

        # Depthwise conv1d (causal)
        x = self.conv1d(x)
        x = self.conv_quant(x)

        # SiLU activation
        x = self.silu(x)
        x = self.silu_quant(x)

        # Global average pool: [B, C, L] -> [B, C, 1]
        x = self.global_pool(x)

        # Squeeze and quantize: [B, C, 1] -> [B, C]
        x = x.squeeze(-1)
        x = self.pre_classifier_quant(x)

        # Classifier
        x = self.classifier(x)
        return x


def get_sample_input(batch_size=1, d_inner=32, seq_len=64):
    """Generate a sample input tensor for testing."""
    return torch.randn(batch_size, d_inner, seq_len)


def test_model():
    """Quick sanity test of the model."""
    print("=" * 60)
    print("Test 24: MAMBA Conv1D + SiLU Network")
    print("=" * 60)

    d_inner = 32
    seq_len = 64
    kernel_size = 4
    num_classes = 10

    model = MambaConv1DTest(
        d_inner=d_inner,
        seq_len=seq_len,
        kernel_size=kernel_size,
        num_classes=num_classes
    )
    model.eval()

    # Generate sample input
    x = get_sample_input(batch_size=2, d_inner=d_inner, seq_len=seq_len)
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

    print("=" * 60)
    print("Test PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_model()
