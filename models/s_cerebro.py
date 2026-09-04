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

from typing import Iterable, Optional

import torch
import torch.nn as nn
from timm.layers import trunc_normal_

from models.modules.attention import TransformerBlock
from models.modules.patching import PatchEmbedding
from models.modules.pos_embed import ChannelEmbedding, PositionalEmbedding


class SCerebroEncoder(nn.Module):
    """S-CEReBrO: a transformer encoder for multi-channel EEG with windowed alternating attention.

    A waveform of shape ``(batch, num_channels, num_patches, patch_size)`` is
    tokenised per channel-patch pair, giving ``num_channels * num_patches`` tokens.
    Each token receives a learned temporal position embedding plus a channel
    embedding derived from its electrode coordinates. The token sequence then passes
    through ``depth`` transformer blocks whose attention alternates between the
    channel (spatial) and time (temporal) axes, restricted to dilated and shifted
    windows on each axis.

    The temporal position table is sized by ``max_timesteps // patch_size`` and
    sliced to the number of patches actually present, so one pre-trained encoder can
    be fine-tuned on shorter recordings and on montages with fewer channels than
    ``max_channels``.

    Args:
        patch_size: Timesteps per patch.
        num_channels: Number of EEG channels in the input.
        embed_dim: Token embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Attention heads per block; must divide ``embed_dim``.
        mlp_ratio: Feed-forward hidden size as a multiple of ``embed_dim``.
        norm_layer: Normalisation layer constructor.
        attention_type: One of ``windowed-alternating``, ``alternating``, ``full``.
        drop_path: Stochastic depth rate.
        attn_drop: Dropout on attention weights.
        proj_drop: Dropout on projection and feed-forward outputs.
        max_channels: Channel capacity of the temporal position table.
        max_timesteps: Timestep capacity used to size the temporal position table.
        window_size_spatial: Channel-axis window size.
        window_size_temporal: Time-axis window size.
        dilation_cycle_spatial: Per-pair channel-axis dilations, cycled over blocks.
        dilation_cycle_temporal: Per-pair time-axis dilations, cycled over blocks.
        shift_cycle_spatial: Per-pair channel-axis window shifts, cycled over blocks.
        shift_cycle_temporal: Per-pair time-axis window shifts, cycled over blocks.
        use_axial_mode: Run all spatial blocks before all temporal blocks instead of
            alternating them.
    """

    def __init__(
        self,
        patch_size: int = 200,
        num_channels: int = 64,
        embed_dim: int = 200,
        depth: int = 12,
        num_heads: int = 10,
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = nn.LayerNorm,
        attention_type: str = "windowed-alternating",
        drop_path: float = 0.0,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
        max_channels: int = 64,
        max_timesteps: int = 6000,
        window_size_spatial: int = 5,
        window_size_temporal: int = 5,
        dilation_cycle_spatial: Iterable[int] = (1, 2, 4),
        dilation_cycle_temporal: Iterable[int] = (1, 2, 4),
        shift_cycle_spatial: Iterable[int] = (-1, 1, -2, 2),
        shift_cycle_temporal: Iterable[int] = (-1, 1, -2, 2),
        use_axial_mode: bool = False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.max_channels = max_channels
        self.max_timesteps = max_timesteps
        self.max_patches = max_timesteps // patch_size

        if num_channels > max_channels:
            raise ValueError(f"num_channels ({num_channels}) exceeds max_channels ({max_channels})")

        self.patch_embed = PatchEmbedding(patch_size=patch_size, embed_dim=embed_dim)
        self.positional_embedding = PositionalEmbedding(self.max_patches, max_channels, embed_dim)
        self.channel_embedding = ChannelEmbedding(embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pad_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer,
                attention_type=attention_type,
                block_idx=idx,
                total_blocks=depth,
                spatial_first=True,
                drop_path=drop_path,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                num_channels=num_channels,
                window_size_spatial=window_size_spatial,
                window_size_temporal=window_size_temporal,
                dilation_cycle_spatial=dilation_cycle_spatial,
                dilation_cycle_temporal=dilation_cycle_temporal,
                shift_cycle_spatial=shift_cycle_spatial,
                shift_cycle_temporal=shift_cycle_temporal,
                use_axial_mode=use_axial_mode,
            )
            for idx in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialise special tokens, then every submodule."""
        trunc_normal_(self.pad_token, std=0.02, a=-0.02, b=0.02)
        trunc_normal_(self.mask_token, std=0.02, a=-0.02, b=0.02)
        trunc_normal_(self.positional_embedding.pos_embedding, std=0.02, a=-0.02, b=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        channel_positions: torch.Tensor,
        directly_input_tokens: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a batch of patched EEG waveforms.

        Args:
            x: Patched waveform of shape ``(batch, num_channels, num_patches, patch_size)``,
                or token embeddings of shape ``(batch, num_channels, num_patches, embed_dim)``
                when ``directly_input_tokens`` is True.
            channel_positions: Electrode coordinates of shape ``(batch, num_channels, 2, 3)``.
            directly_input_tokens: Skip patch embedding and treat ``x`` as tokens. Used by
                pre-training, which masks tokens between embedding and encoding.
            attn_mask: Optional ``(batch, num_tokens)`` mask, 1 for real and 0 for padded tokens.

        Returns:
            Contextualised token embeddings of shape ``(batch, num_tokens, embed_dim)``.
        """
        batch, channels, patches = x.shape[0], x.shape[1], x.shape[2]

        if channels != self.num_channels:
            raise ValueError(
                f"Input has {channels} channels but the encoder was built for {self.num_channels}"
            )
        if patches > self.max_patches:
            raise ValueError(
                f"Input has {patches} patches per channel, exceeding the positional "
                f"embedding capacity of {self.max_patches}"
            )

        if not directly_input_tokens:
            x = self.patch_embed(x)

        x = x.view(batch, channels, patches, self.embed_dim)

        pos_embed = self.positional_embedding()[:, :channels, :patches, :]
        chan_embed = self.channel_embedding(channel_positions).unsqueeze(2)

        x = (x + pos_embed + chan_embed).reshape(batch, channels * patches, self.embed_dim)

        for block in self.blocks:
            x = block(x=x, attn_mask=attn_mask)

        return self.norm(x)
