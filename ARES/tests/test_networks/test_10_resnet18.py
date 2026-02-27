# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
test_14_resnet18.py - ResNet-18 adapted for MNIST (Large Model Validation)

ResNet-18 architecture adapted for 28x28 MNIST input while maintaining the core
ResNet design principles. This test validates:
- Large model support (~100K+ parameters)
- Multiple skip connections throughout the network
- L3 staging under memory pressure
- Deep network (18 weight layers)
- Bottleneck pattern at scale

Original ResNet-18 (ImageNet):
- Input: 224x224x3
- Parameters: ~11M
- Stages: [2, 2, 2, 2] blocks

MNIST-Adapted ResNet-18:
- Input: 28x28x1
- Parameters: ~100K (scaled down)
- Stages: [2, 2, 2, 2] blocks (maintains structure)
- Channel progression: 16 → 32 → 64 → 128

Architecture:
- Initial conv: 3x3 (1→16)
- Stage 1: 2 residual blocks (16→16)
- Stage 2: 2 residual blocks (16→32, downsample)
- Stage 3: 2 residual blocks (32→64, downsample)
- Stage 4: 2 residual blocks (64→128, downsample)
- Global average pooling
- Fully connected classifier (128→10)

Total: 1 initial + 8 residual blocks (16 conv layers) + 1 fc = 18 layers
"""

import sys
from pathlib import Path

# Add parent directory to path for imports when running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat
from test_networks.brevitas_custom_layers import QuantAdd


class BasicBlock(nn.Module):
    """
    ResNet Basic Block with quantization.

    Structure:
    - Conv 3x3 → ReLU → Quant
    - Conv 3x3 → Add (with skip) → ReLU → Quant

    If downsample is needed (stride=2 or channel mismatch):
    - Skip connection uses 1x1 conv to match dimensions
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        # First conv layer (may downsample)
        self.conv1 = qnn.QuantConv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1,
            weight_quant=Int8WeightPerTensorFloat,
            bias=True,
            return_quant_tensor=True
        )
        self.relu1 = qnn.QuantReLU(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )
        self.quant1 = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )

        # Second conv layer (stride=1 always)
        self.conv2 = qnn.QuantConv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1,
            weight_quant=Int8WeightPerTensorFloat,
            bias=True,
            return_quant_tensor=True
        )

        # Skip connection (identity or 1x1 projection)
        self.needs_projection = (stride != 1) or (in_channels != out_channels)
        if self.needs_projection:
            self.shortcut = qnn.QuantConv2d(
                in_channels, out_channels,
                kernel_size=1, stride=stride, padding=0,
                weight_quant=Int8WeightPerTensorFloat,
                bias=True,
                return_quant_tensor=True
            )
            self.shortcut_quant = qnn.QuantIdentity(
                act_quant=Int8ActPerTensorFloat,
                return_quant_tensor=True
            )

        # Add operation (combines conv2 output + skip)
        # Using custom QuantAdd layer (defined in brevitas_custom_layers.py)
        self.add = QuantAdd(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )

        # Post-add activation
        self.relu2 = qnn.QuantReLU(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )
        self.quant2 = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )

    def forward(self, x):
        # Main path
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.quant1(out)

        out = self.conv2(out)

        # Skip connection
        if self.needs_projection:
            skip = self.shortcut(x)
            skip = self.shortcut_quant(skip)
        else:
            skip = x

        # Add and post-activation
        out = self.add(out, skip)
        out = self.relu2(out)
        out = self.quant2(out)

        return out


class ResNet18MNIST(nn.Module):
    """
    ResNet-18 adapted for MNIST (28x28 input).

    Architecture:
    - Input quantization
    - Initial conv: 3x3 (1→16)
    - Layer 1: 2 blocks (16→16)
    - Layer 2: 2 blocks (16→32, stride=2)
    - Layer 3: 2 blocks (32→64, stride=2)
    - Layer 4: 2 blocks (64→128, stride=2)
    - Global average pooling
    - Fully connected (128→10)

    Output dimension progression:
    - Input: 28x28
    - After initial conv: 28x28 (stride=1, padding=1)
    - After layer1: 28x28
    - After layer2: 14x14 (stride=2 downsample)
    - After layer3: 7x7 (stride=2 downsample)
    - After layer4: 3x3 (stride=2 downsample, floor(7/2)=3)
    - After global pool: 1x1
    """
    def __init__(self, num_classes=10):
        super(ResNet18MNIST, self).__init__()

        # Input quantization
        self.input_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )

        # Initial convolution (1→16, no downsampling)
        self.conv1 = qnn.QuantConv2d(
            1, 16,
            kernel_size=3, stride=1, padding=1,
            weight_quant=Int8WeightPerTensorFloat,
            bias=True,
            return_quant_tensor=True
        )
        self.relu1 = qnn.QuantReLU(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )
        self.quant1 = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )

        # ResNet layers (4 stages, 2 blocks each)
        self.layer1 = self._make_layer(16, 16, blocks=2, stride=1)
        self.layer2 = self._make_layer(16, 32, blocks=2, stride=2)
        self.layer3 = self._make_layer(32, 64, blocks=2, stride=2)
        self.layer4 = self._make_layer(64, 128, blocks=2, stride=2)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.pool_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )

        # Classifier
        self.classifier = qnn.QuantLinear(
            128, num_classes,
            weight_quant=Int8WeightPerTensorFloat,
            bias=True,
            return_quant_tensor=False
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        """
        Create a ResNet layer consisting of multiple BasicBlocks.

        Args:
            in_channels: Input channel count
            out_channels: Output channel count
            blocks: Number of BasicBlocks in this layer
            stride: Stride for first block (1 or 2)

        Returns:
            nn.Sequential of BasicBlocks
        """
        layers = []

        # First block (may downsample with stride=2)
        layers.append(BasicBlock(in_channels, out_channels, stride))

        # Remaining blocks (stride=1, channels match)
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Input quantization
        x = self.input_quant(x)

        # Initial convolution
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.quant1(x)

        # ResNet stages
        x = self.layer1(x)  # 28x28
        x = self.layer2(x)  # 14x14
        x = self.layer3(x)  # 7x7
        x = self.layer4(x)  # 3x3

        # Global pooling and classifier
        x = self.global_pool(x)
        x = self.pool_quant(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


def create_network():
    """Factory function for test harness."""
    return ResNet18MNIST(num_classes=10)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test network creation and forward pass
    model = create_network()
    model.eval()

    # Print architecture summary
    print("=" * 60)
    print("ResNet-18 (MNIST-Adapted) Architecture")
    print("=" * 60)
    print(f"Total parameters: {count_parameters(model):,}")
    print()

    # Test forward pass
    x = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        out = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output logits: {out.squeeze().numpy()}")
    print()
    print("Network structure:")
    print(f"  - Input quantization")
    print(f"  - Initial conv (1→16)")
    print(f"  - Layer 1: 2 blocks (16→16, 28x28)")
    print(f"  - Layer 2: 2 blocks (16→32, 28→14)")
    print(f"  - Layer 3: 2 blocks (32→64, 14→7)")
    print(f"  - Layer 4: 2 blocks (64→128, 7→3)")
    print(f"  - Global avg pool (3x3→1x1)")
    print(f"  - Classifier (128→10)")
    print()
    print("Expected layer count in extraction:")
    print("  - 1 input_quant")
    print("  - 1 initial conv + relu + quant")
    print("  - 8 residual blocks x (2 conv + skip projection + add + relu + quant)")
    print("  - 1 global_pool + quant")
    print("  - 1 classifier")
    print("  Total: ~40-50 layers (depending on skip projections)")
    print("=" * 60)
