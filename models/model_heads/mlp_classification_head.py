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
#*----------------------------------------------------------------------------*

import warnings

import torch
import torch.nn as nn
from timm.layers import trunc_normal_

# Above this, a flatten classifier is almost certainly a configuration mistake.
FLATTEN_PARAMETER_WARNING = 100_000_000


class MlpClassificationHead(nn.Module):
    """Classification head over the encoder's token embeddings.

    An optional token-wise pooler is applied first. Tokens are then aggregated
    either by averaging (``pooling_method='mean'``) or by concatenation
    (``pooling_method='flatten'``), and a linear or MLP classifier produces one
    prediction per input window.

    Args:
        embed_dim: Token embedding dimension.
        num_classes: Number of output classes.
        pooling_method: ``'mean'`` or ``'flatten'``.
        dropout: Dropout used in the pooler and, for ``'flatten'``, the classifier.
        pooling: Whether to apply the token-wise pooler.
        num_channels: Channel count, required for ``'flatten'`` to size the classifier.
        num_patches: Patches per channel, required for ``'flatten'``.
    """

    def __init__(
        self,
        embed_dim: int = 200,
        num_classes: int = 2,
        pooling_method: str = "mean",
        dropout: float = 0.1,
        pooling: bool = True,
        num_channels: int = 32,
        num_patches: int = 10,
    ):
        super().__init__()
        if pooling_method not in {"mean", "flatten"}:
            raise ValueError(f"pooling_method must be 'mean' or 'flatten', got '{pooling_method}'")

        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.pooling_method = pooling_method
        self.pooling = pooling
        self.num_tokens = num_channels * num_patches

        if self.pooling:
            self.pooler = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
                nn.Dropout(dropout),
                nn.Tanh(),
            )

        if pooling_method == "mean":
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:
            # The first layer is quartic in the window: 64 channels, 10 patches and
            # embed_dim 400 is 1.02e9 parameters in one matrix.
            first_layer = self.num_tokens * embed_dim * num_patches * embed_dim
            if first_layer > FLATTEN_PARAMETER_WARNING:
                warnings.warn(
                    f"pooling_method='flatten' with num_channels={num_channels}, "
                    f"num_patches={num_patches} and embed_dim={embed_dim} builds a first "
                    f"layer of {first_layer:,} parameters. Use pooling_method='mean' "
                    "unless the flattened layout is specifically required.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.classifier = nn.Sequential(
                nn.Linear(self.num_tokens * embed_dim, num_patches * embed_dim),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(num_patches * embed_dim, embed_dim),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, num_classes),
            )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a batch of token sequences.

        Args:
            x: Token embeddings of shape ``(batch, num_tokens, embed_dim)``.

        Returns:
            Logits of shape ``(batch, num_classes)``.
        """
        if self.pooling:
            x = self.pooler(x)

        if self.pooling_method == "mean":
            # Unmasked: fine-tuning never pads. Padded fine-tuning would need a mask
            # here, or pad tokens would dilute the pooled vector.
            x = x.mean(dim=1)
        else:
            batch, num_tokens, embed_dim = x.shape
            if num_tokens != self.num_tokens or embed_dim != self.embed_dim:
                raise ValueError(
                    f"Expected ({self.num_tokens}, {self.embed_dim}) tokens, got ({num_tokens}, {embed_dim})"
                )
            x = x.reshape(batch, num_tokens * embed_dim)

        return self.classifier(x)
