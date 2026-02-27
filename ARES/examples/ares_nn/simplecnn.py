# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Minimal ARES-compatible CNN example built from ares.nn blocks."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ares.nn as ann


class ExampleSimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_quant = ann.QuantIdentity(bit_width=8, return_quant_tensor=True)
        self.conv1 = ann.Conv2d(1, 16, kernel_size=3, padding=1, bias=True)
        self.relu1 = ann.ReLU()
        self.pool1 = ann.MaxPool2d(2)
        self.pool1_quant = ann.QuantIdentity(bit_width=8, return_quant_tensor=True)

        self.conv2 = ann.Conv2d(16, 32, kernel_size=3, padding=1, bias=True)
        self.relu2 = ann.ReLU()
        self.pool2 = ann.MaxPool2d(2)
        self.pool2_quant = ann.QuantIdentity(bit_width=8, return_quant_tensor=True)

        self.flatten = ann.Flatten(start_dim=1)
        self.pre_fc_quant = ann.QuantIdentity(bit_width=8, return_quant_tensor=True)
        self.fc = ann.Linear(7 * 7 * 32, 10, bias=True, return_quant_tensor=False)

    def forward(self, x):
        x = self.input_quant(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.pool1_quant(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.pool2_quant(x)
        x = self.flatten(x)
        x = self.pre_fc_quant(x)
        return self.fc(x)


if __name__ == "__main__":
    model = ExampleSimpleCNN().eval()
    report = ann.check_compatibility(model, strict=False)
    print(f"Compatible: {report['compatible']}")

    x = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        y = model(x)
    print(f"Output shape: {tuple(y.shape)}")
