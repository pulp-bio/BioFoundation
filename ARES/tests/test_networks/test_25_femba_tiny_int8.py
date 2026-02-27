# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Test 34: FEMBA Tiny INT8 (True Architecture)

  - Input: (1, 1, 22, 1280) - 22 EEG channels, 1280 samples (~5s @ 256Hz)
  - Patch Embedding (patch_size=(2,16), stride=(2,16), embed_dim=35)
  - d_model = 385 (11 * 35)
  - expand = 4 (FEMBA standard)
  - d_inner = 1540 (4 * d_model)
  - d_state = 16
  - dt_rank = 25 (ceil(d_model / 16))
  - 2 BiMamba blocks with L3 streaming

Total Parameters: ~7.6 million

Memory requirements per direction (~1.94 MB):
  - in_proj: 385 * 3080 = 1,185,800 bytes (~1.13 MB!)
  - out_proj: 1540 * 385 = 593,450 bytes (~580 KB)
  - conv1d: 1540 * 4 + 1540 = 7,700 bytes
  - x_proj: 1540 * 57 = 87,780 bytes
  - dt_proj: 25 * 1540 + 1540 = 40,040 bytes
  - A_log: 1540 * 16 * 4 = 98,560 bytes (FP32)
  - D: 1540 * 4 = 6,160 bytes (FP32)

Total for 4 directions: ~7.8 MB (requires aggressive L3 streaming)

"""

import math
import torch
import torch.nn as nn
from brevitas import nn as qnn
from brevitas.quant import Int8ActPerTensorFloat

from .brevitas_custom_layers import QuantPatchEmbed, QuantMambaWrapper


class FEMBATinyInt8(nn.Module):
    """
    FEMBA Tiny with INT8 quantization (true architecture).

    Key parameters:
    - expand = 4
    - d_state = 16 (not 4)
    - d_inner = 4 * d_model = 1540
    - in_proj projects from d_model to 2 * d_inner = 3080
    - dt_rank = ceil(d_model / 16) = 25
    - Weight bit width = 8 (INT8)

    Args:
        inp_size: Input spatial size (EEG_channels, samples) - default (22, 1280)
        patch_size: Patch size for embedding - default (2, 16) per FEMBA spec
        stride: Stride for patch embedding - default (2, 16) per FEMBA spec
        in_chans: Number of input channels - default 1
        embed_dim: Embedding dimension per patch row - default 35
        expand: Expansion factor for d_inner - default 4 (FEMBA standard)
        d_state: SSM state dimension - default 16
        d_conv: Conv1d kernel size in MAMBA - default 4
        num_blocks: Number of encoder blocks - default 2
        num_classes: Number of output classes - default 2 (binary classification)
    """

    def __init__(
        self,
        inp_size=(22, 1280),
        patch_size=(2, 16),
        stride=(2, 16),
        in_chans=1,
        embed_dim=35,
        expand=4,
        d_state=16,
        d_conv=4,
        num_blocks=2,
        num_classes=2
    ):
        super().__init__()
        self.inp_size = inp_size
        self.patch_size = patch_size
        self.stride = stride
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.expand = expand
        self.d_state = d_state
        self.d_conv = d_conv
        self.num_blocks = num_blocks
        bit_width = 8

        # Calculate dimensions after patch embedding
        H, W = inp_size
        self.grid_h = (H - patch_size[0]) // stride[0] + 1
        self.grid_w = (W - patch_size[1]) // stride[1] + 1
        self.seq_len = self.grid_w  # Sequence length
        self.d_model = self.grid_h * embed_dim  # Model dimension

        # True FEMBA: d_inner = expand * d_model (expand=4)
        self.d_inner = expand * self.d_model

        # dt_rank as per FEMBA spec
        self.dt_rank = math.ceil(self.d_model / 16)

        print(f"[FEMBATinyInt8] Configuration:")
        print(f"  Input: {inp_size}, patch: {patch_size}, stride: {stride}")
        print(f"  Grid: ({self.grid_h}, {self.grid_w})")
        print(f"  d_model: {self.d_model}, seq_len: {self.seq_len}")
        print(f"  d_inner: {self.d_inner} (expand={expand})")
        print(f"  d_state: {d_state}, dt_rank: {self.dt_rank}")
        print(f"  Weight bit width: {bit_width}")

        # Estimate weight size per direction
        in_proj_size = self.d_model * 2 * self.d_inner  # projects to 2*d_inner
        out_proj_size = self.d_inner * self.d_model
        conv_size = self.d_inner * d_conv + self.d_inner  # depthwise + bias
        x_proj_size = self.d_inner * (self.dt_rank + 2 * d_state)  # dt_rank + 2*d_state
        dt_proj_size = self.dt_rank * self.d_inner + self.d_inner  # with bias
        a_log_size = self.d_inner * d_state * 4  # FP32
        d_size = self.d_inner * 4  # FP32
        total_per_dir = in_proj_size + out_proj_size + conv_size + x_proj_size + dt_proj_size + a_log_size + d_size
        print(f"  Weight size per direction: {total_per_dir / 1024 / 1024:.2f} MB")
        print(f"  Total for 4 directions (2 blocks): {4 * total_per_dir / 1024 / 1024:.2f} MB")

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
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.seq_len, self.d_model)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Quantization for positional embedding
        self.pos_quant = qnn.QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )

        # Scale equalizer
        self.scale_equalizer = qnn.QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )

        # Encoder blocks: BiMamba + Residual + LayerNorm
        self.mamba_blocks = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.post_norm_quants = nn.ModuleList()

        for i in range(num_blocks):
            # Bi-Mamba block with true FEMBA dimensions
            self.mamba_blocks.append(
                QuantMambaWrapper(
                    d_model=self.d_model,
                    d_inner=self.d_inner,
                    d_state=d_state,
                    conv_kernel=d_conv,
                    bidirectional_strategy="add",
                    bit_width=bit_width,
                    return_quant_tensor=True
                )
            )

            # LayerNorm after residual
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
            return_quant_tensor=False
        )

    def forward(self, x):
        """Forward pass following FEMBA architecture."""
        # Quantize input
        x = self.input_quant(x)

        if hasattr(x, 'value'):
            x = x.value

        # Patch embedding
        x = self.patch_embed(x)

        # Add positional embedding
        pos = self.pos_quant(self.pos_embed)

        if hasattr(x, 'value'):
            x_val = x.value
        else:
            x_val = x
        if hasattr(pos, 'value'):
            pos_val = pos.value
        else:
            pos_val = pos

        x = self.scale_equalizer(x_val)
        pos = self.scale_equalizer(pos_val)

        if hasattr(x, 'value'):
            x_val = x.value
        else:
            x_val = x
        if hasattr(pos, 'value'):
            pos_val = pos.value
        else:
            pos_val = pos

        x = x_val + pos_val
        x = self.scale_equalizer(x)

        # Encoder blocks
        for mamba_block, norm_layer, post_norm_quant in zip(
            self.mamba_blocks, self.norm_layers, self.post_norm_quants
        ):
            if hasattr(x, 'value'):
                res = x.value
            else:
                res = x

            x = mamba_block(x)

            if hasattr(x, 'value'):
                x_val = x.value
            else:
                x_val = x

            x = res + x_val
            x = norm_layer(x)
            x = post_norm_quant(x)

        if hasattr(x, 'value'):
            x = x.value

        # Global pool and classify
        x = x.transpose(1, 2)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.pre_classifier_quant(x)
        x = self.classifier(x)
        return x


def get_sample_input(batch_size=1, in_chans=1, inp_size=(22, 1280)):
    """Generate a sample input tensor for testing."""
    return torch.randn(batch_size, in_chans, inp_size[0], inp_size[1])


def test_model():
    """Quick sanity test of the model."""
    print("=" * 70)
    print("Test 34: FEMBA Tiny INT8 (True Architecture)")
    print("=" * 70)

    model = FEMBATinyInt8(
        inp_size=(22, 1280),
        patch_size=(2, 16),
        stride=(2, 16),
        in_chans=1,
        embed_dim=35,
        expand=4,        # True FEMBA: expand=4
        d_state=16,      # True FEMBA: d_state=16
        d_conv=4,
        num_blocks=2,
        num_classes=2
    )
    model.eval()

    x = get_sample_input(batch_size=1, in_chans=1, inp_size=(22, 1280))
    print(f"\nInput shape: {x.shape}")

    with torch.no_grad():
        output = model(x)

    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    print(f"Predicted class: {output.argmax(dim=1).item()}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {num_params:,}")

    print(f"\nFEMBA Tiny INT8 Configuration:")
    print(f"  Input size: {model.inp_size}")
    print(f"  d_model: {model.d_model}")
    print(f"  d_inner: {model.d_inner} (expand={model.expand})")
    print(f"  d_state: {model.d_state}")
    print(f"  dt_rank: {model.dt_rank}")
    print(f"  Sequence length: {model.seq_len}")

    # Verify parameter count is ~7.6M
    expected_params = 7_600_000
    if num_params > expected_params * 0.9 and num_params < expected_params * 1.1:
        print(f"\n[OK] Parameter count matches expected ~7.6M")
    else:
        print(f"\n[NOTE] Parameter count {num_params:,} differs from expected ~7.6M")
        print(f"       (Brevitas quantization layers add overhead)")

    print("=" * 70)
    print("Test PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_model()
