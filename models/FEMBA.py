#*----------------------------------------------------------------------------*
#* Copyright (C) 2025 ETH Zurich, Switzerland                                 *
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
#* Author:  Anna Tegon                                                        *
#* Author:  Thorir Mar Ingolfsson                                             *
#*----------------------------------------------------------------------------*

import torch
import torch.nn as nn
from typing import Optional, Tuple
from mamba_ssm import Mamba


class MambaWrapper(nn.Module):
    """
    Thin wrapper around Mamba to support bi-directionality.

    Args:
        d_model (int): Dimension of the model.
        bidirectional (bool): Whether to use bidirectional processing.
        bidirectional_strategy (str, optional): Strategy to combine forward and backward passes ("add", "ew_multiply").
        **mamba_kwargs: Additional arguments passed to Mamba.
    """
    def __init__(self, d_model: int, bidirectional: bool = True, bidirectional_strategy: Optional[str] = "add", **mamba_kwargs):
        super().__init__()
        if bidirectional and bidirectional_strategy is None:
            bidirectional_strategy = "add"
        if bidirectional and bidirectional_strategy not in ["add", "ew_multiply"]:
            raise NotImplementedError(f"{bidirectional_strategy} strategy for bi-directionality is not implemented!")
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.mamba_fwd = Mamba(d_model=d_model, **mamba_kwargs)
        if bidirectional:
            self.mamba_rev = Mamba(d_model=d_model, **mamba_kwargs)
        else:
            self.mamba_rev = None

    def forward(self, hidden_states, inference_params=None):
        out = self.mamba_fwd(hidden_states, inference_params=inference_params)
        if self.bidirectional:
            out_rev = self.mamba_rev(hidden_states.flip(dims=(1,)), inference_params=inference_params).flip(dims=(1,))
            if self.bidirectional_strategy == "add":
                out = out + out_rev
            elif self.bidirectional_strategy == "ew_multiply":
                out = out * out_rev
        return out


class PatchEmbed(nn.Module):
    """
    Converts input signal into patch embeddings using a convolutional layer.

    Args:
        inp_size (Tuple[int, int]): Input size (channels, sequence length).
        patch_size (Tuple[int, int]): Size of each patch.
        stride (Tuple[int, int]): Stride for patch extraction.
        in_chans (int): Number of input channels.
        embed_dim (int): Dimension of the embedding.
        kernel_1 (int): Kernel size (used to define padding).
        norm_layer (nn.Module, optional): Normalization layer.
    """
    def __init__(self, inp_size, patch_size, stride, in_chans, embed_dim, kernel_1: int = 64, norm_layer=None):
        super().__init__()
        self.inp_size = inp_size
        self.patch_size = patch_size
        self.kernel_1 = kernel_1 - 1
        self.grid_size = ((inp_size[0] - patch_size[0]) // stride[0] + 1,
                          (inp_size[1] - patch_size[1]) // stride[1] + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, channels, length)
        x = self.proj(x)  # (batch, embed_dim, grid_h, grid_w)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])  # (batch, embed_dim * grid_h, grid_w)
        x = x.permute(0, 2, 1)  # (batch, grid_w, embed_dim * grid_h)
        x = self.norm(x)
        return x


class MambaClassifier(nn.Module):
    """
    Classifier head using Mamba block for temporal processing.

    Args:
        embed_dim (int): Embedding dimension.
        grid_size (Tuple[int, int]): Grid size from patch embedding.
        num_classes (int): Number of output classes.
        num_channels (int): Number of input channels.
        classification_type (str): Classification strategy:
            - 'bc'  = Binary Classification (e.g. TUAB, TUAR)
            - 'ml'  = Multi-Label Classification (e.g. TUSL)
            - 'mc' = Multi-Label  Classification  for TUAR 
            - 'mcc' = Multi-Class Classification (e.g. TUAR )
            - 'mmc' = Multi-Class Multi-Output Classification (e.g. TUAR)
    """
    def __init__(self, embed_dim, grid_size, num_classes, num_channels, classification_type: str):
        super(MambaClassifier, self).__init__()
        self.grid_size = grid_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.classification_type = classification_type

        hidden_size1 = 256
        input_size = embed_dim * grid_size[0]

        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.activation1 = nn.GELU()
        self.mamba_1 = Mamba(d_model=hidden_size1, expand=2)

        if classification_type in ("bc", "mcc", "ml"):
            self.fc3 = nn.Linear(hidden_size1, num_classes)
        elif classification_type == "mc":
            self.fc3 = nn.Linear(hidden_size1, num_channels)
        elif classification_type == "mmc":
            self.fc3 = nn.Linear(hidden_size1, num_channels * num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.mamba_1(x)
        x = x.permute(0, 2, 1).contiguous()  # (batch, features, time)
        x = x.mean(dim=-1)  # Global average pooling over time
        x = self.fc3(x)

        if self.classification_type == "mmc":
            x = x.view(-1, self.num_channels, self.num_classes)

        return x


class Decoder(nn.Module):
    """
    Reconstructs original signal from encoded representation.

    Args:
        embed_dim (int): Embedding dimension.
        grid_size (Tuple[int, int]): Grid size from encoder.
        kernel_dec (Tuple[int, int]): Kernel size for decoding conv.
        patch_size (Tuple[int, int]): Patch size used in encoding.
        stride (Tuple[int, int]): Stride used in encoding.
    """
    def __init__(self, embed_dim: int, grid_size: Tuple[int, int], kernel_dec: Tuple[int, int], patch_size: Tuple[int, int], stride: Tuple[int, int]):
        super(Decoder, self).__init__()
        self.kernel_dec = kernel_dec
        self.embed_dim = embed_dim
        self.grid_size = grid_size

        self.dec_conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=self.kernel_dec,
            stride=1,
            padding=((self.kernel_dec[0] - 1) // 2, (self.kernel_dec[1] - 1) // 2),
            bias=False
        )

        self.unflatten = nn.Unflatten(
            dim=1,
            unflattened_size=(embed_dim, grid_size[0])
        )

        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=embed_dim,
            out_channels=1,
            kernel_size=patch_size,
            stride=stride
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, N=grid_size[0] * embed_dim, grid_size[1])
        x = self.dec_conv(x)  # (batch, 1, grid_size[0] * grid_size[1], embed_dim)
        x = x.squeeze(1)  # (batch, grid_size[0] * grid_size[1], embed_dim)
        x = x.transpose(1, 2)  # (batch, embed_dim, grid_size[0] * grid_size[1])
        x = self.unflatten(x)  # (batch, embed_dim, grid_size[0], grid_size[1])
        x = self.conv_transpose(x)  # (batch, 1, H, W)
        x_reconstructed = x.squeeze(1)  # (batch, H, W)
        return x_reconstructed


class FEMBA(nn.Module):
    """
    Foundational Encoder Model with Bidirectional Mamba (FEMBA).
    Can perform classification or masked signal reconstruction.

    Args:
        seq_length (int): Length of the input signal.
        num_channels (int): Number of input channels.
        num_classes (int): Number of output classes. Set to 0 for reconstruction.
        kernel_1 (int): First convolution kernel size.
        kernel_dec (Tuple[int, int]): Decoder kernel size.
        dropout (float): Dropout rate.
        exp (int): Expansion factor for Mamba.
        patch_size (Tuple[int, int]): Patch size for encoder.
        stride (Tuple[int, int]): Stride for encoder.
        embed_dim (int): Embedding dimension.
        num_blocks (int): Number of Mamba blocks.
        classification_type (str): Classification type (bc, ml, mc, mcc, mmc).
    """
    def __init__(self,
                seq_length: int = 1280,
                num_channels: int = 22,
                num_classes: int = 0,
                kernel_1: int = 64,
                kernel_dec: Tuple[int, int] = (31, 31),
                exp: int = 4, 
                patch_size: Tuple[int, int] = (2, 16),
                stride: Tuple[int, int] = (2, 16),
                embed_dim: int = 79,
                num_blocks: int= 1,
                classification_type: str = "bc"):

        super(FEMBA, self).__init__()
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.kernel_1 = kernel_1 - 1
        self.exp = exp
        self.inp_size = (num_channels, seq_length)
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.classification_type = classification_type

        self.patch_embed = PatchEmbed(
            inp_size=self.inp_size,
            patch_size=self.patch_size,
            stride=self.stride,
            in_chans=1,
            embed_dim=embed_dim
        )

        grid_size = self.patch_embed.grid_size
        self.pos_embed = nn.Parameter(torch.zeros(1, grid_size[1], grid_size[0] * self.embed_dim))

        self.mamba_blocks = nn.ModuleList([
            MambaWrapper(d_model=grid_size[0] * self.embed_dim, expand=self.exp)
            for _ in range(self.num_blocks)
        ])

        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(grid_size[0] * self.embed_dim)
            for _ in range(self.num_blocks)
        ])

        if num_classes == 0:
            self.classifier = None
            self.decoder = Decoder(embed_dim=self.embed_dim, grid_size=grid_size, kernel_dec=kernel_dec, patch_size=self.patch_size, stride=stride)
        else:
            self.classifier = MambaClassifier(embed_dim, grid_size, num_classes, num_channels, classification_type)

    def forward(self, x, mask):
        x_original = x
        x_masked = x.clone()
        x_masked[mask] = 0  # Apply mask
        x = self.patch_embed(x_masked)  # (B, T, D)
        x = x + self.pos_embed  # Add positional embedding

        for mamba_block, norm_layer in zip(self.mamba_blocks, self.norm_layers):
            res = x
            x = mamba_block(x)
            x = res + x
            x = norm_layer(x)

        if self.classifier is not None:
            x_classified = self.classifier(x)
            return x_classified, x_original
        else:
            x_reconstructed = self.decoder(x)
            return x_reconstructed, x_original
