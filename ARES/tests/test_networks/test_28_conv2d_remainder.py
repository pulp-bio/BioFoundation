# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Test 38: Conv2D im2col remainder (col_size % 4) benchmark

Purpose:
- Exercise Conv2D shapes where `in_ch * kH * kW` is NOT a multiple of 4.
- This hits the im2col+SIMD tail path (`remainder != 0`) and is useful to tune:
  - output-channel unrolling (B)
  - SIMD tail handling (C)

Architecture (MNIST-compatible, 1 input channel):
- conv_expand: 1x1 (1 -> 3)  (creates a 3-channel feature map)
- conv7x7:     7x7 (3 -> 32, stride=2) (col_size = 3*7*7 = 147, remainder = 3)
- global_pool
- classifier
"""

import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat


class Conv2DRemainderNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.input_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

        # Expand 1-channel MNIST input into 3 channels (keeps test MNIST-compatible).
        self.conv_expand = qnn.QuantConv2d(
            1, 3, kernel_size=1, stride=1, padding=0,
            weight_quant=Int8WeightPerTensorFloat,
            bias=True,
            return_quant_tensor=True,
        )
        self.relu_expand = qnn.QuantReLU(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )
        self.quant_expand = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

        # Main target: 7x7 over 3 channels -> col_size=147 (remainder=3)
        self.conv7x7 = qnn.QuantConv2d(
            3, 32, kernel_size=7, stride=2, padding=3,
            weight_quant=Int8WeightPerTensorFloat,
            bias=True,
            return_quant_tensor=True,
        )
        self.relu7x7 = qnn.QuantReLU(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )
        self.quant7x7 = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.pool_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

        self.classifier = qnn.QuantLinear(
            32, num_classes,
            weight_quant=Int8WeightPerTensorFloat,
            bias=True,
            return_quant_tensor=False,
        )

    def forward(self, x):
        x = self.input_quant(x)
        x = self.conv_expand(x)
        x = self.relu_expand(x)
        x = self.quant_expand(x)

        x = self.conv7x7(x)
        x = self.relu7x7(x)
        x = self.quant7x7(x)

        x = self.global_pool(x)
        x = self.pool_quant(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
