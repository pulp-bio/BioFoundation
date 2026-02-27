# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Test 36: BioCAS Drowsiness Detection - EEG+PPG Sensor Fusion Network

Pipeline:
  EEG Input (1, 1, 8, 2200)           PPG Input (1, 1, 2, 2200)
         |                                   |
    [EEG Encoder]                       [PPG Encoder]
         |                                   |
      (B, 16)                             (B, 32)
         |_______________    _______________|
                        |  |
                     Concatenate
                        |
                     (B, 48)
                        |
                   Linear(48, 2)
                        |
                   Output (B, 2)

Purpose: Stress test ARES with dual-input sensor fusion network.
Target benchmark: reference implementation achieves 14.19ms @ 370MHz (1.04 MACs/cycle)

Data Format:
- EEG Input: [batch, 1, 8, 2200] - 8 EEG channels, 2200 samples @ 500Hz
- PPG Input: [batch, 1, 2, 2200] - 2 PPG channels, 2200 samples @ 500Hz
- Output: [batch, 2] - Binary classification (Alert vs Drowsy)
"""

import torch
import torch.nn as nn
from brevitas import nn as qnn
from tests.test_networks.brevitas_custom_layers import QuantConcatenate


class DrowsinessFusionNetwork(nn.Module):
    """
    Quantized EEG+PPG Sensor Fusion Network for Drowsiness Detection.

    Based on BioCAS'23 EpiDeNet architecture with explicit padding
    to handle asymmetric 'same' padding requirements.
    """

    def __init__(
        self,
        eeg_out_channels=16,
        ppg_out_channels=32,
        num_classes=2,
        bit_width=8
    ):
        super().__init__()
        self.eeg_out_channels = eeg_out_channels
        self.ppg_out_channels = ppg_out_channels
        self.num_classes = num_classes

        # EEG Encoder
        # Input: (B, 1, 8, 2200) -> Output: (B, 16)

        self.eeg_input_quant = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Conv1: (1,4) kernel, need total_pad_w=3 -> (1,2)
        self.eeg_pad1 = nn.ZeroPad2d((1, 2, 0, 0))
        self.eeg_conv1 = qnn.QuantConv2d(1, 4, (1, 4), padding=0, weight_bit_width=bit_width, return_quant_tensor=True)
        self.eeg_bn1 = nn.BatchNorm2d(4)
        self.eeg_relu1 = qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.eeg_pool1 = nn.MaxPool2d((1, 8))

        # Conv2: (1,16) kernel, need total_pad_w=15 -> (7,8)
        self.eeg_pad2 = nn.ZeroPad2d((7, 8, 0, 0))
        self.eeg_conv2 = qnn.QuantConv2d(4, 16, (1, 16), padding=0, weight_bit_width=bit_width, return_quant_tensor=True)
        self.eeg_bn2 = nn.BatchNorm2d(16)
        self.eeg_relu2 = qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.eeg_pool2 = nn.MaxPool2d((1, 4))

        # Conv3: (1,8) kernel, need total_pad_w=7 -> (3,4)
        self.eeg_pad3 = nn.ZeroPad2d((3, 4, 0, 0))
        self.eeg_conv3 = qnn.QuantConv2d(16, 16, (1, 8), padding=0, weight_bit_width=bit_width, return_quant_tensor=True)
        self.eeg_bn3 = nn.BatchNorm2d(16)
        self.eeg_relu3 = qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.eeg_pool3 = nn.MaxPool2d((1, 4))

        # Conv4: (16,1) kernel, need total_pad_h=15 -> (7,8)
        self.eeg_pad4 = nn.ZeroPad2d((0, 0, 7, 8))
        self.eeg_conv4 = qnn.QuantConv2d(16, 16, (16, 1), padding=0, weight_bit_width=bit_width, return_quant_tensor=True)
        self.eeg_bn4 = nn.BatchNorm2d(16)
        self.eeg_relu4 = qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.eeg_pool4 = nn.MaxPool2d((4, 1))

        # Conv5: (8,1) kernel on height 2, need total_pad_h=7 -> (3,4)
        self.eeg_pad5 = nn.ZeroPad2d((0, 0, 3, 4))
        self.eeg_conv5 = qnn.QuantConv2d(16, eeg_out_channels, (8, 1), padding=0, weight_bit_width=bit_width, return_quant_tensor=True)
        self.eeg_bn5 = nn.BatchNorm2d(eeg_out_channels)
        self.eeg_relu5 = qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.eeg_avgpool = nn.AvgPool2d((2, 17))

        # PPG Encoder
        # Input: (B, 1, 2, 2200) -> Output: (B, 32)

        self.ppg_input_quant = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Conv1: (1,4) kernel, need total_pad_w=3 -> (1,2)
        self.ppg_pad1 = nn.ZeroPad2d((1, 2, 0, 0))
        self.ppg_conv1 = qnn.QuantConv2d(1, 4, (1, 4), padding=0, weight_bit_width=bit_width, return_quant_tensor=True)
        self.ppg_bn1 = nn.BatchNorm2d(4)
        self.ppg_relu1 = qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.ppg_pool1 = nn.MaxPool2d((1, 4))

        # Conv2: (1,4) kernel, need total_pad_w=3 -> (1,2)
        self.ppg_pad2 = nn.ZeroPad2d((1, 2, 0, 0))
        self.ppg_conv2 = qnn.QuantConv2d(4, 8, (1, 4), padding=0, weight_bit_width=bit_width, return_quant_tensor=True)
        self.ppg_bn2 = nn.BatchNorm2d(8)
        self.ppg_relu2 = qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.ppg_pool2 = nn.MaxPool2d((1, 4))

        # Conv3: (1,4) kernel, need total_pad_w=3 -> (1,2)
        self.ppg_pad3 = nn.ZeroPad2d((1, 2, 0, 0))
        self.ppg_conv3 = qnn.QuantConv2d(8, 16, (1, 4), padding=0, weight_bit_width=bit_width, return_quant_tensor=True)
        self.ppg_bn3 = nn.BatchNorm2d(16)
        self.ppg_relu3 = qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.ppg_pool3 = nn.MaxPool2d((1, 4))

        # Conv4: (1,8) kernel, need total_pad_w=7 -> (3,4)
        self.ppg_pad4 = nn.ZeroPad2d((3, 4, 0, 0))
        self.ppg_conv4 = qnn.QuantConv2d(16, 32, (1, 8), padding=0, weight_bit_width=bit_width, return_quant_tensor=True)
        self.ppg_bn4 = nn.BatchNorm2d(32)
        self.ppg_relu4 = qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True)

        # Conv5: (1,16) kernel, need total_pad_w=15 -> (7,8)
        self.ppg_pad5 = nn.ZeroPad2d((7, 8, 0, 0))
        self.ppg_conv5 = qnn.QuantConv2d(32, ppg_out_channels, (1, 16), padding=0, weight_bit_width=bit_width, return_quant_tensor=True)
        self.ppg_bn5 = nn.BatchNorm2d(ppg_out_channels)
        self.ppg_relu5 = qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.ppg_avgpool = nn.AvgPool2d((2, 34))

        # Fusion + Classifier
        # Use QuantConcatenate for proper tracking in extractor
        self.concat = QuantConcatenate(bit_width=bit_width, return_quant_tensor=True, dim=1)
        self.classifier = qnn.QuantLinear(
            eeg_out_channels + ppg_out_channels, num_classes,
            bias=True, weight_bit_width=bit_width, return_quant_tensor=False
        )

    def forward_eeg(self, x):
        """Process EEG stream: (B, 1, 8, 2200) -> (B, 16)"""
        x = self.eeg_input_quant(x)

        x = self.eeg_pad1(x)
        x = self.eeg_conv1(x)
        x = self.eeg_bn1(x)
        x = self.eeg_relu1(x)
        x = self.eeg_pool1(x)

        x = self.eeg_pad2(x)
        x = self.eeg_conv2(x)
        x = self.eeg_bn2(x)
        x = self.eeg_relu2(x)
        x = self.eeg_pool2(x)

        x = self.eeg_pad3(x)
        x = self.eeg_conv3(x)
        x = self.eeg_bn3(x)
        x = self.eeg_relu3(x)
        x = self.eeg_pool3(x)

        x = self.eeg_pad4(x)
        x = self.eeg_conv4(x)
        x = self.eeg_bn4(x)
        x = self.eeg_relu4(x)
        x = self.eeg_pool4(x)

        x = self.eeg_pad5(x)
        x = self.eeg_conv5(x)
        x = self.eeg_bn5(x)
        x = self.eeg_relu5(x)
        x = self.eeg_avgpool(x)

        if hasattr(x, 'value'):
            x = x.value
        return x.view(x.size(0), -1)

    def forward_ppg(self, x):
        """Process PPG stream: (B, 1, 2, 2200) -> (B, 32)"""
        x = self.ppg_input_quant(x)

        x = self.ppg_pad1(x)
        x = self.ppg_conv1(x)
        x = self.ppg_bn1(x)
        x = self.ppg_relu1(x)
        x = self.ppg_pool1(x)

        x = self.ppg_pad2(x)
        x = self.ppg_conv2(x)
        x = self.ppg_bn2(x)
        x = self.ppg_relu2(x)
        x = self.ppg_pool2(x)

        x = self.ppg_pad3(x)
        x = self.ppg_conv3(x)
        x = self.ppg_bn3(x)
        x = self.ppg_relu3(x)
        x = self.ppg_pool3(x)

        x = self.ppg_pad4(x)
        x = self.ppg_conv4(x)
        x = self.ppg_bn4(x)
        x = self.ppg_relu4(x)

        x = self.ppg_pad5(x)
        x = self.ppg_conv5(x)
        x = self.ppg_bn5(x)
        x = self.ppg_relu5(x)
        x = self.ppg_avgpool(x)

        if hasattr(x, 'value'):
            x = x.value
        return x.view(x.size(0), -1)

    def forward(self, eeg_input, ppg_input):
        """Forward pass: (eeg, ppg) -> logits"""
        eeg_features = self.forward_eeg(eeg_input)
        ppg_features = self.forward_ppg(ppg_input)

        # Use QuantConcatenate module for proper extractor tracking
        fused = self.concat([eeg_features, ppg_features])
        return self.classifier(fused)


def get_model():
    """Factory function for test generation."""
    return DrowsinessFusionNetwork(eeg_out_channels=16, ppg_out_channels=32, num_classes=2, bit_width=8)


def get_input_shape():
    """Return expected input shapes as tuple of (eeg_shape, ppg_shape)."""
    return ((1, 1, 8, 2200), (1, 1, 2, 2200))


def get_num_inputs():
    """Return number of inputs for this network."""
    return 2


if __name__ == "__main__":
    model = get_model()
    model.eval()

    eeg_shape, ppg_shape = get_input_shape()
    eeg_input = torch.randn(eeg_shape)
    ppg_input = torch.randn(ppg_shape)

    print(f"EEG Input shape: {eeg_input.shape}")
    print(f"PPG Input shape: {ppg_input.shape}")

    with torch.no_grad():
        out = model(eeg_input, ppg_input)

    print(f"Output shape: {out.shape}")
    print(f"Output: {out}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print("\nTest PASSED!")
