# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tools/model_compatibility_core.py."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch.nn as nn
from brevitas import nn as qnn

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = REPO_ROOT / "tests"
for path in (REPO_ROOT, TESTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tools.model_compatibility_core import scan_model_modules


class _BatchNormModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_quant = qnn.QuantIdentity(bit_width=8, return_quant_tensor=True)
        self.conv = qnn.QuantConv2d(1, 8, kernel_size=3, padding=1, weight_bit_width=8)
        self.bn = nn.BatchNorm2d(8)
        self.relu = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)

    def forward(self, x):
        x = self.input_quant(x)
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class _DropoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_quant = qnn.QuantIdentity(bit_width=8, return_quant_tensor=True)
        self.fc = qnn.QuantLinear(16, 8, weight_bit_width=8, return_quant_tensor=True)
        self.drop = nn.Dropout(p=0.1)
        self.out = qnn.QuantLinear(8, 2, weight_bit_width=8, return_quant_tensor=False)

    def forward(self, x):
        x = self.input_quant(x)
        x = self.fc(x)
        x = self.drop(x)
        return self.out(x)


class ModelCompatibilityCheckerTests(unittest.TestCase):
    def test_simplecnn_is_compatible(self):
        from test_networks.test_1_simplecnn import SimpleCNN

        report = scan_model_modules(SimpleCNN().eval())
        self.assertTrue(report.compatible)
        self.assertEqual(len(report.unsupported), 0)

    def test_resnet_basic_is_compatible(self):
        from test_networks.test_4_resnet_basic import ResNetBasic

        report = scan_model_modules(ResNetBasic().eval())
        self.assertTrue(report.compatible)
        self.assertEqual(len(report.unsupported), 0)

    def test_transformer_simple_is_compatible(self):
        from test_networks.test_13_transformer_simple import SimpleTransformer

        report = scan_model_modules(SimpleTransformer().eval())
        self.assertTrue(report.compatible)
        self.assertEqual(len(report.unsupported), 0)

    def test_batchnorm_is_unsupported(self):
        report = scan_model_modules(_BatchNormModel().eval())
        self.assertFalse(report.compatible)
        bn_findings = [f for f in report.unsupported if f.class_name == "BatchNorm2d"]
        self.assertGreaterEqual(len(bn_findings), 1)

    def test_dropout_is_warning_not_unsupported(self):
        report = scan_model_modules(_DropoutModel().eval())
        self.assertTrue(report.compatible)
        self.assertEqual(len(report.unsupported), 0)
        drop_warnings = [f for f in report.warnings if f.class_name == "Dropout"]
        self.assertGreaterEqual(len(drop_warnings), 1)

    def test_composite_children_are_skipped(self):
        from test_networks.test_18_mamba_block import MambaBlockTest

        report = scan_model_modules(MambaBlockTest().eval())
        self.assertTrue(report.compatible)
        # Internal submodules of the composite block should not generate findings.
        composite_children = [
            f for f in report.findings if f.module_name.startswith("mamba_block.")
        ]
        self.assertEqual(composite_children, [])


if __name__ == "__main__":
    unittest.main()
