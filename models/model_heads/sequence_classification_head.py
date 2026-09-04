#*----------------------------------------------------------------------------*
#* Copyright (C) 2026 ETH Zurich, Switzerland                                 *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Author:  BioFoundation Contributors                                        *
#*                                                                            *
#* Imported from the S-CEReBrO reference implementation (TimeFM).             *
#*----------------------------------------------------------------------------*

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_


class SequenceClassificationHead(nn.Module):
    """Per-epoch classification head for sequences of consecutive EEG epochs.

    Used for sleep staging (ISRUC), where a sample is a sequence of
    ``sequence_length`` consecutive epochs and every epoch carries its own label.
    The encoder sees the sequence flattened into the batch axis, so this head
    receives ``(batch * sequence_length, tokens_per_epoch, embed_dim)``. It regroups
    the sequence, compresses each epoch to a single vector, contextualises the epochs
    against one another with a small transformer encoder, and emits one prediction per
    epoch.

    Logits are returned flattened as ``(batch * sequence_length, num_classes)`` so
    they align with the labels flattened by the classification task.

    Args:
        sequence_length: Number of consecutive epochs per sample.
        num_channels: EEG channels per epoch.
        num_patches: Patches per channel per epoch.
        embed_dim: Token embedding dimension of the encoder.
        num_classes: Number of classes predicted per epoch.
        hidden_dim: Width of the per-epoch representation and the sequence encoder.
        num_layers: Transformer encoder layers over the epoch sequence.
        nhead: Attention heads in the sequence encoder.
        dim_feedforward: Feed-forward width in the sequence encoder.
        dropout: Dropout in the sequence encoder.
        norm_first: Use pre-norm ordering in the sequence encoder.
    """

    def __init__(
        self,
        sequence_length: int,
        num_channels: int,
        num_patches: int,
        embed_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        num_layers: int = 1,
        nhead: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        norm_first: bool = True,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.tokens_per_epoch = num_channels * num_patches
        self.feature_dim = self.tokens_per_epoch * embed_dim

        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.GELU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=F.gelu,
            batch_first=True,
            norm_first=norm_first,
        )
        self.sequence_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify every epoch in each sequence.

        Args:
            x: Token embeddings of shape ``(batch * sequence_length, tokens_per_epoch, embed_dim)``.

        Returns:
            Logits of shape ``(batch * sequence_length, num_classes)``.
        """
        flat_batch, num_tokens, embed_dim = x.shape
        if embed_dim != self.embed_dim:
            raise ValueError(f"Expected embed_dim {self.embed_dim}, got {embed_dim}")
        if num_tokens != self.tokens_per_epoch:
            raise ValueError(f"Expected {self.tokens_per_epoch} tokens per epoch, got {num_tokens}")
        if flat_batch % self.sequence_length != 0:
            raise ValueError(
                f"Flattened batch {flat_batch} is not divisible by sequence_length {self.sequence_length}"
            )

        batch = flat_batch // self.sequence_length
        x = x.reshape(batch, self.sequence_length, self.feature_dim)
        x = self.head(x)
        x = self.sequence_encoder(x)
        return self.classifier(x).reshape(flat_batch, self.num_classes)
