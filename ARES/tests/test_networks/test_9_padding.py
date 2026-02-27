# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
test_12_padding.py - Edge case test for various padding values

Tests different padding configurations to ensure correct boundary handling.
Padding is critical for maintaining spatial dimensions and preventing
information loss at boundaries.

Architecture:
- conv1: 3x3 padding=0 (no padding, 28→26)
- conv2: 3x3 padding=1 (same padding, 26→26)
- conv3: 5x5 padding=2 (same padding, 26→26)
- pool1: 2x2 stride=2 (26→13)
- conv4: 3x3 padding=0 (13→11)
- global_pool: 11x11→1x1
- classifier: fully connected

Edge cases tested:
- Zero padding (dimension reduction)
- Standard padding=1 for 3x3 (same dimensions)
- Larger padding=2 for 5x5 (same dimensions)
- Interaction with pooling
- Non-standard output dimensions (26x26, 13x13, 11x11)
"""

import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat

class PaddingNet(nn.Module):
    def __init__(self, num_classes=10):
        super(PaddingNet, self).__init__()

        # Input quantization
        self.input_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )

        # Conv1: 3x3 padding=0 (1→16, 28→26)
        self.conv1 = qnn.QuantConv2d(
            1, 16, kernel_size=3, stride=1, padding=0,
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

        # Conv2: 3x3 padding=1 (16→32, 26→26)
        self.conv2 = qnn.QuantConv2d(
            16, 32, kernel_size=3, stride=1, padding=1,
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

        # Conv3: 5x5 padding=2 (32→32, 26→26)
        self.conv3 = qnn.QuantConv2d(
            32, 32, kernel_size=5, stride=1, padding=2,
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

        # Pool1: 2x2 stride=2 (26→13)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )

        # Conv4: 3x3 padding=0 (32→16, 13→11)
        self.conv4 = qnn.QuantConv2d(
            32, 16, kernel_size=3, stride=1, padding=0,
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

        # Conv1: padding=0 (28→26)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.quant1(x)

        # Conv2: padding=1 (26→26)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.quant2(x)

        # Conv3: padding=2 (26→26)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.quant3(x)

        # Pool1 (26→13)
        x = self.pool1(x)
        x = self.pool1_quant(x)

        # Conv4: padding=0 (13→11)
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
    return PaddingNet(num_classes=10)
