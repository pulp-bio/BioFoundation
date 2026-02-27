# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Minimal transformer-style example using ares.nn.TransformerBlock."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ares.nn as ann


class ExampleTransformer(nn.Module):
    def __init__(self, seq_len: int = 32, embed_dim: int = 64, num_heads: int = 2):
        super().__init__()
        self.input_quant = ann.QuantIdentity(bit_width=8, return_quant_tensor=True)
        self.block = ann.TransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=embed_dim * 2,
            seq_len=seq_len,
        )
        self.norm = ann.LayerNorm(embed_dim)
        self.norm_quant = ann.QuantIdentity(bit_width=8, return_quant_tensor=True)
        self.permute = ann.Permute(0, 2, 1)
        self.pool = ann.AdaptiveAvgPool1d(1)
        self.squeeze = ann.Squeeze(-1)
        self.head = ann.Linear(embed_dim, 10, return_quant_tensor=False)

    def forward(self, x):
        x = self.input_quant(x)
        x = self.block(x)
        x = self.norm(x)
        x = self.norm_quant(x)
        x = self.permute(x)
        x = self.pool(x)
        x = self.squeeze(x)
        return self.head(x)


if __name__ == "__main__":
    model = ExampleTransformer().eval()
    report = ann.check_compatibility(model, strict=False)
    print(f"Compatible: {report['compatible']}")

    x = torch.randn(1, 32, 64)
    with torch.no_grad():
        y = model(x)
    print(f"Output shape: {tuple(y.shape)}")
