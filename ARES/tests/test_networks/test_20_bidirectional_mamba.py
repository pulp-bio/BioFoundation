# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Test 28: Bidirectional MAMBA Block Test Network

Pipeline:
  INPUT (1, 16, 32) - [batch, seq_len, d_model]
  -> QuantIdentity
  -> QuantMambaWrapper (bidirectional)
      - mamba_fwd: forward MAMBA block
      - flip -> mamba_rev -> flip back
      - scale_equalizer -> add
  -> QuantIdentity
  -> Global Average Pooling (over sequence)
  -> QuantLinear classifier (10 classes)

Purpose: Validate bidirectional MAMBA wrapper combining:
- Forward MAMBA block processing
- Sequence flip for reverse processing
- Reverse MAMBA block processing
- Flip back and scale-equalized addition

This is the core building block for FEMBA-style bidirectional models.

Architecture:
    x_in ────┬──────> mamba_fwd ──────────────┬──> scale_eq ──┐
             │                                 │               │
             └──> flip ──> mamba_rev ──> flip ─┘               ├──> add ──> out
                                                               │
                                                               └──────────┘

Data Format:
- Input: [batch, seq_len, d_model]
- Output: [batch, num_classes]
"""

import torch
import torch.nn as nn
from brevitas import nn as qnn

from .brevitas_custom_layers import QuantMambaWrapper


class BidirectionalMambaTest(nn.Module):
    """
    Test network for bidirectional MAMBA wrapper.

    This network validates the FEMBA-style bidirectional processing:
    - Forward and reverse MAMBA blocks
    - Sequence flip operations
    - Scale-equalized output combination

    Args:
        d_model: Model dimension (default: 32)
        d_inner: Inner dimension (default: 64, typically 2*d_model)
        d_state: SSM state dimension (default: 4)
        conv_kernel: Conv1d kernel size (default: 4)
        seq_len: Sequence length (default: 16)
        num_classes: Number of output classes (default: 10)
        bidirectional_strategy: How to combine fwd/rev ("add" or "concat")
    """

    def __init__(
        self,
        d_model=32,
        d_inner=64,
        d_state=4,
        conv_kernel=4,
        seq_len=16,
        num_classes=10,
        bidirectional_strategy="add"
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.seq_len = seq_len
        self.bidirectional_strategy = bidirectional_strategy
        bit_width = 8

        # Input quantization
        self.input_quant = qnn.QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )

        # Bidirectional MAMBA wrapper
        self.bi_mamba = QuantMambaWrapper(
            d_model=d_model,
            d_inner=d_inner,
            d_state=d_state,
            conv_kernel=conv_kernel,
            bidirectional_strategy=bidirectional_strategy,
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
        # Output dimension depends on strategy
        classifier_in_dim = d_model if bidirectional_strategy == "add" else 2 * d_model
        self.classifier = qnn.QuantLinear(
            classifier_in_dim,
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

        # Bidirectional MAMBA forward
        x = self.bi_mamba(x)
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
    print("Test 28: Bidirectional MAMBA Network")
    print("=" * 60)

    d_model = 32
    d_inner = 64
    d_state = 4
    conv_kernel = 4
    seq_len = 16
    num_classes = 10

    model = BidirectionalMambaTest(
        d_model=d_model,
        d_inner=d_inner,
        d_state=d_state,
        conv_kernel=conv_kernel,
        seq_len=seq_len,
        num_classes=num_classes,
        bidirectional_strategy="add"
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
    print(f"\nBidirectional MAMBA Configuration:")
    print(f"  d_model: {model.d_model}")
    print(f"  d_inner: {model.d_inner}")
    print(f"  d_state: {model.d_state}")
    print(f"  conv_kernel: {conv_kernel}")
    print(f"  seq_len: {seq_len}")
    print(f"  bidirectional_strategy: {model.bidirectional_strategy}")

    # Show subcomponents
    print(f"\nSubcomponents (per direction):")
    print(f"  in_proj: {d_model} -> {2 * d_inner}")
    print(f"  conv1d: {d_inner} channels, kernel={conv_kernel}")
    print(f"  ssm: d_inner={d_inner}, d_state={d_state}")
    print(f"  out_proj: {d_inner} -> {d_model}")

    # Verify bidirectionality
    print(f"\nBidirectional verification:")
    with torch.no_grad():
        # Get forward output only
        x_quant = model.input_quant(x)
        out_fwd = model.bi_mamba.mamba_fwd(x_quant)
        out_fwd_val = out_fwd.value if hasattr(out_fwd, 'value') else out_fwd

        # Get reverse output
        x_flipped = torch.flip(x, dims=[1])
        x_flipped_quant = model.bi_mamba.post_flip_quant(x_flipped)
        out_rev = model.bi_mamba.mamba_rev(x_flipped_quant)
        out_rev_val = out_rev.value if hasattr(out_rev, 'value') else out_rev

        print(f"  Forward output norm: {out_fwd_val.norm().item():.4f}")
        print(f"  Reverse output norm: {out_rev_val.norm().item():.4f}")
        print(f"  Outputs are different: {not torch.allclose(out_fwd_val, out_rev_val)}")

    print("=" * 60)
    print("Test PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_model()
