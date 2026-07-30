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
#* Author:  Glenn Anta Bucagu                                                 *
#* Author:  BioFoundation Contributors                                        *
#*                                                                            *
#* Imported from the S-CEReBrO reference implementation (TimeFM).             *
#*----------------------------------------------------------------------------*

import torch
import torch.nn as nn
from einops import rearrange


def patchify(signal: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Split a multi-channel waveform into per-channel patches.

    Args:
        signal: Waveform of shape ``(batch, num_channels, num_timesteps)``.
        patch_size: Number of timesteps per patch; must divide ``num_timesteps``.

    Returns:
        Tensor of shape ``(batch, num_channels * num_patches, patch_size)``, with
        channel as the slower-varying index.
    """
    batch, channels, timesteps = signal.shape
    if timesteps % patch_size != 0:
        raise ValueError(f"num_timesteps ({timesteps}) must be divisible by patch_size ({patch_size})")
    patches = signal.reshape(batch, channels, timesteps // patch_size, patch_size)
    return rearrange(patches, "B C t p -> B (C t) p")


def unpatchify(patches: torch.Tensor, num_channels: int) -> torch.Tensor:
    """Reassemble per-channel patches into a multi-channel waveform.

    Args:
        patches: Tensor of shape ``(batch, num_channels * num_patches, patch_size)``.
        num_channels: Number of channels encoded in the token axis.

    Returns:
        Waveform of shape ``(batch, num_channels, num_timesteps)``.
    """
    return rearrange(patches, "B (C t) p -> B C (t p)", C=num_channels)


PATCH_SIZE = 200
CONV_OUT_CHANNELS = 8
CONV_OUTPUT_FEATURES = 200


class TemporalConvTokenizer(nn.Module):
    """Convolutional tokenizer mapping one waveform patch to one embedding.

    Each ``(channel, patch)`` pair is encoded independently by a stack of three
    strided 1D convolutions (applied as 2D convolutions over a flattened
    channel-patch axis), then projected to ``embed_dim``. The channel dimension is
    preserved, so the token count is ``num_channels * num_patches``.

    The convolution stack is defined for a patch of 200 timesteps, which at the
    project-wide 200 Hz sampling rate is one second per token. The first convolution
    strides by 8, so a 200-sample patch yields 25 positions of 8 features each and the
    projection consumes exactly 200 features. Other patch sizes are rejected rather
    than silently reshaped.
    """

    def __init__(
        self,
        patch_size: int = PATCH_SIZE,
        out_channels: int = CONV_OUT_CHANNELS,
        embed_dim: int = PATCH_SIZE,
    ):
        super().__init__()
        if patch_size != PATCH_SIZE:
            raise ValueError(
                f"TemporalConvTokenizer is defined for patch_size={PATCH_SIZE}, got {patch_size}"
            )
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2d(1, out_channels, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))
        self.gelu3 = nn.GELU()
        self.norm3 = nn.GroupNorm(4, out_channels)
        self.proj = nn.Linear(CONV_OUTPUT_FEATURES, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed patched waveforms.

        Args:
            x: Patched waveform of shape ``(batch, num_channels, num_patches, patch_size)``.

        Returns:
            Tensor of shape ``(batch, num_channels * num_patches, embed_dim)``.
        """
        x = rearrange(x, "B C P S -> B (C P) S").unsqueeze(1)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        x = rearrange(x, "B F N T -> B N (T F)")
        return self.proj(x)


class PatchEmbedding(nn.Module):
    """Waveform patch embedding used by S-CEReBrO.

    Thin wrapper around :class:`TemporalConvTokenizer`. The inner module is held in
    an attribute named ``patch_embed`` so that parameter keys remain
    ``patch_embed.patch_embed.*``, keeping checkpoints from earlier versions of
    this code loadable. Weight initialisation is delegated to the encoder, which
    applies it to the whole model tree.
    """

    def __init__(self, patch_size: int = PATCH_SIZE, embed_dim: int = PATCH_SIZE):
        super().__init__()
        self.patch_embed = TemporalConvTokenizer(patch_size=patch_size, embed_dim=embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed patched waveforms of shape ``(batch, channels, patches, patch_size)``."""
        return self.patch_embed(x)
