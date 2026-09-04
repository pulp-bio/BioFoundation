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
from timm.layers import trunc_normal_


class PositionalEmbedding(nn.Module):
    """Learned temporal position embedding, shared across channels.

    A single table of ``max_patches`` vectors is broadcast over the channel axis, so
    two tokens at the same time index in different channels receive the same
    temporal embedding.
    """

    def __init__(self, max_patches: int, num_channels: int, embed_dim: int):
        super().__init__()
        self.max_patches = max_patches
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_patches, embed_dim))
        trunc_normal_(self.pos_embedding, std=0.02, a=-0.02, b=0.02)

    def forward(self) -> torch.Tensor:
        """Return the embedding table of shape ``(1, num_channels, max_patches, embed_dim)``."""
        return self.pos_embedding.unsqueeze(1).repeat(1, self.num_channels, 1, 1)


class ChannelEmbedding(nn.Module):
    """Channel embedding computed from 3D electrode coordinates.

    Every channel is described by two electrodes (a bipolar pair, or a scalp
    electrode and its reference). A shared MLP maps each electrode's 3D coordinate
    to ``embed_dim // 2`` features and the two halves are concatenated. Because the
    embedding is a function of geometry rather than of a channel index, montages
    with different channel counts and orderings share the same parameters.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(3, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, embed_dim // 2),
        )
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, channel_positions: torch.Tensor) -> torch.Tensor:
        """Embed electrode coordinates.

        Args:
            channel_positions: Tensor of shape ``(batch, num_channels, 2, 3)``.

        Returns:
            Tensor of shape ``(batch, num_channels, embed_dim)``.
        """
        batch, channels, electrodes, coords = channel_positions.shape
        if electrodes != 2:
            raise ValueError(f"Expected 2 electrodes per channel, got {electrodes}")
        if coords != 3:
            raise ValueError(f"Expected 3D electrode coordinates, got {coords}")

        embedded = self.mlp(channel_positions.reshape(-1, 3))
        return embedded.view(batch, channels, self.embed_dim)
