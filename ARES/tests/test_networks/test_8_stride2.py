# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
test_11_stride2.py - Edge case test for stride-2 convolutions

Tests stride-2 convolutions used for downsampling (alternative to pooling).
Common in modern architectures like ResNet, EfficientNet for spatial reduction.

Architecture:
- conv1: 3x3 stride-1 (baseline)
- conv2: 3x3 stride-2 (downsample 28→14)
- conv3: 3x3 stride-1 (process 14x14)
- conv4: 3x3 stride-2 (downsample 14→7)
- global_pool: 7x7→1x1
- classifier: fully connected

Edge cases tested:
- Stride-2 with 3x3 kernel (most common)
- Multiple stride-2 layers
- Odd output dimensions (14x14, 7x7)
- Interaction with padding
"""

import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat

class Stride2Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Stride2Net, self).__init__()

        # Input quantization
        self.input_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )

        # Conv1: 3x3 stride-1 (1→16)
        self.conv1 = qnn.QuantConv2d(
            1, 16, kernel_size=3, stride=1, padding=1,
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

        # Conv2: 3x3 stride-2 downsample (16→32, 28→14)
        self.conv2 = qnn.QuantConv2d(
            16, 32, kernel_size=3, stride=2, padding=1,
            weight_quant=Int8WeightPerTensorFloat,
            bias=True,
            return_quant_tensor=True
        )
        self.relu2 = qnn.QuantReLU(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )
        self.quant2 = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )

        # Conv3: 3x3 stride-1 (32→32, 14x14)
        self.conv3 = qnn.QuantConv2d(
            32, 32, kernel_size=3, stride=1, padding=1,
            weight_quant=Int8WeightPerTensorFloat,
            bias=True,
            return_quant_tensor=True
        )
        self.relu3 = qnn.QuantReLU(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )
        self.quant3 = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )

        # Conv4: 3x3 stride-2 downsample (32→16, 14→7)
        self.conv4 = qnn.QuantConv2d(
            32, 16, kernel_size=3, stride=2, padding=1,
            weight_quant=Int8WeightPerTensorFloat,
            bias=True,
            return_quant_tensor=True
        )
        self.relu4 = qnn.QuantReLU(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )
        self.quant4 = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.pool_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )

        # Classifier
        self.classifier = qnn.QuantLinear(
            16, num_classes,
            weight_quant=Int8WeightPerTensorFloat,
            bias=True,
            return_quant_tensor=False
        )

    def forward(self, x):
        x = self.input_quant(x)

        # Conv1: stride-1 (28x28)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.quant1(x)

        # Conv2: stride-2 downsample (28→14)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.quant2(x)

        # Conv3: stride-1 (14x14)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.quant3(x)

        # Conv4: stride-2 downsample (14→7)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.quant4(x)

        # Global pooling and classifier
        x = self.global_pool(x)
        x = self.pool_quant(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

def create_network():
    return Stride2Net(num_classes=10)
