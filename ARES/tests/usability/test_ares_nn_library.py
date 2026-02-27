# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the additive ares.nn package."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

import ares.nn as ann


class _AresSimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_quant = ann.QuantIdentity(bit_width=8, return_quant_tensor=True)
        self.conv = ann.Conv2d(1, 8, kernel_size=3, padding=1, bias=True)
        self.relu = ann.ReLU()
        self.pool = ann.MaxPool2d(2)
        self.pool_quant = ann.QuantIdentity(bit_width=8, return_quant_tensor=True)
        self.flatten = ann.Flatten(start_dim=1)
        self.fc = ann.Linear(14 * 14 * 8, 10, return_quant_tensor=False)

    def forward(self, x):
        x = self.input_quant(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.pool_quant(x)
        x = self.flatten(x)
        return self.fc(x)


class _AresTransformerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_quant = ann.QuantIdentity(bit_width=8, return_quant_tensor=True)
        self.block = ann.TransformerBlock(embed_dim=64, num_heads=2, ff_dim=128, seq_len=32)
        self.norm = ann.LayerNorm(64)
        self.norm_quant = ann.QuantIdentity(bit_width=8, return_quant_tensor=True)
        self.permute = ann.Permute(0, 2, 1)
        self.pool = ann.AdaptiveAvgPool1d(1)
        self.squeeze = ann.Squeeze(-1)
        self.fc = ann.Linear(64, 5, return_quant_tensor=False)

    def forward(self, x):
        x = self.input_quant(x)
        x = self.block(x)
        x = self.norm(x)
        x = self.norm_quant(x)
        x = self.permute(x)
        x = self.pool(x)
        x = self.squeeze(x)
        return self.fc(x)


class AresNNLibraryTests(unittest.TestCase):
    def test_ares_simple_net_is_compatible(self):
        model = _AresSimpleNet().eval()
        report = ann.check_compatibility(model, strict=False)
        self.assertTrue(report["compatible"])

        with torch.no_grad():
            out = model(torch.randn(1, 1, 28, 28))
        self.assertEqual(tuple(out.shape), (1, 10))

    def test_ares_transformer_net_is_compatible(self):
        model = _AresTransformerNet().eval()
        report = ann.check_compatibility(model, strict=False)
        self.assertTrue(report["compatible"])

        with torch.no_grad():
            out = model(torch.randn(1, 32, 64))
        self.assertEqual(tuple(out.shape), (1, 5))

    def test_check_compatibility_raises_in_strict_mode(self):
        class Unsupported(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm2d(4)

            def forward(self, x):
                return self.bn(x)

        with self.assertRaises(ValueError):
            ann.check_compatibility(Unsupported().eval(), strict=True)


if __name__ == "__main__":
    unittest.main()
