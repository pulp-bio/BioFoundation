#*----------------------------------------------------------------------------*
#* Copyright (C) 2025-2026 ETH Zurich, Switzerland                            *
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
#* Author:  BioFoundation Contributors                                       *
#*----------------------------------------------------------------------------*

import math
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers.drop import DropPath
from timm.layers.mlp import Mlp
from timm.layers.weight_init import trunc_normal_ as __call_trunc_normal_


def trunc_normal_(tensor, mean=0.0, std=1.0):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


# https://docs.pytorch.org/torchtune/stable/_modules/torchtune/modules/position_embeddings.html#RotaryPositionalEmbeddings
@dataclass(eq=False)
class RotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verification)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim // num_heads``
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    dim: int
    max_seq_len: int = 4096
    base: int = 10_000

    def __post_init__(self):
        super().__init__()
        self.rope_init()

    def rope_init(self):
        """Initialize the RoPE embeddings and cache."""
        # ensure dim is int
        dim = int(self.dim)
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32)[: dim // 2] / dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        """Build the RoPE cache for positions up to max_seq_len."""
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(0, max_seq_len, dtype=torch.float32)  # type: ignore

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache: torch.Tensor = torch.stack(
            [torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1
        )
        self.register_buffer("cache", cache, persistent=False)

    def forward(
        self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                ``[b, s, n_h, h_d]``
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            torch.Tensor: output tensor with shape ``[b, s, n_h, h_d]``

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        if input_pos is None:
            rope_cache = self.cache[:seq_len]  # type: ignore
        else:
            input_pos = input_pos.to(torch.long)
            rope_cache = self.cache[input_pos]  # type: ignore

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.reshape(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)


@dataclass(eq=False)
class PatchEmbedWaveformKeepChans(nn.Module):
    """
    Patch embedding layer for waveform data that keeps channel information.

    This module embeds patches from waveform inputs while preserving the channel dimension.
    It uses a 2D convolution to project patches into an embedding space, and rearranges
    the output to flatten patches across channels and time.

    Args:
        img_size (int): The size of the input waveform in the time dimension.
        patch_size (int): The size of each patch in the time dimension.
        in_chans (int): Number of input channels.
        embed_dim (int): Dimensionality of the embedding space.
    """

    img_size: int
    patch_size: int
    in_chans: int
    embed_dim: int

    def __post_init__(self):
        super().__init__()
        self.num_patches = (self.img_size // self.patch_size) * self.in_chans
        self.proj = nn.Conv2d(
            1,
            self.embed_dim,
            kernel_size=(1, self.patch_size),
            stride=(1, self.patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input waveform tensor of shape (B, C, T)
        Returns:
            Patch embeddings of shape (B, N, D) where N is number of patches and D is embed_dim
        """
        # shared projection layer across channels
        x = self.proj(x.unsqueeze(1))
        x = rearrange(x, "B D C t -> B (C t) D")
        return x


@dataclass(eq=False)
class PatchingModule(nn.Module):
    """Image to Patch Embedding of choice according to the parameters given."""

    img_size: int
    patch_size: int
    in_chans: int
    embed_dim: int

    def __post_init__(self):
        super().__init__()
        self.patch_embed = PatchEmbedWaveformKeepChans(
            self.img_size, self.patch_size, self.in_chans, self.embed_dim
        )
        self.num_patches = self.patch_embed.num_patches
        self.init_patch_embed()

    def init_patch_embed(self):
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.patch_embed(x)


@dataclass(eq=False)
class RotarySelfAttentionBlock(nn.Module):
    """
    A self-attention block that incorporates rotary positional embeddings (RoPE) for enhanced positional awareness.

    This module implements multi-head self-attention with rotary positional embeddings applied to query and key tensors,
    followed by scaled dot-product attention. It is designed for transformer-based architectures, particularly in vision
    or sequence modeling tasks where positional information is crucial.

    Attributes:
        dim (int): The dimensionality of the input and output features.
        num_heads (int): The number of attention heads.
        rope (RotaryPositionalEmbeddings): The rotary positional embedding module.
        scale (float): The scaling factor for attention logits.
        qkv (nn.Linear): Linear layer for projecting input to query, key, and value.
        attn_drop_fn (nn.Dropout): Dropout layer for attention weights.
        proj (nn.Linear): Linear layer for projecting attention output.
        proj_drop (nn.Dropout): Dropout layer for projection output.
    """

    dim: int
    num_heads: int = 8
    qkv_bias: bool = False
    attn_drop: float = 0.0
    proj_drop: float = 0.0

    def __post_init__(self):
        super().__init__()
        head_dim = self.dim // self.num_heads
        self.rope = RotaryPositionalEmbeddings(
            dim=head_dim,
            max_seq_len=1024,
            base=10_000,
        )
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=self.qkv_bias)
        self.attn_drop_fn = nn.Dropout(self.attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.p_drop = nn.Dropout(self.proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        pos_ids: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass for rotary self-attention block."""
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )  # (K, B, H, N, D)
        q, k, v = qkv.unbind(0)  # each: (B, H, N, D)

        # Transpose to [B, N, H, D] for RoPE
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        q = self.rope(q, input_pos=pos_ids)
        k = self.rope(k, input_pos=pos_ids)

        # Transpose back to [B, H, N, D] for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,  # Mask out padded channels
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )

        x = x.transpose(2, 1).reshape(B, N, C)
        x = self.proj(x)
        x = self.p_drop(x)
        return x


@dataclass(eq=False)
class RotaryTransformerBlock(nn.Module):
    """
    A transformer block that incorporates rotary self-attention for enhanced positional encoding.

    This block applies layer normalization, rotary self-attention, and a multi-layer perceptron (MLP)
    in sequence, with drop paths for regularization. It follows a standard transformer
    architecture but uses rotary embeddings to improve handling of sequential data.

    Args:
        dim (int): The dimensionality of the input and output features.
        num_heads (int): The number of attention heads in the self-attention mechanism.
        mlp_ratio (float, optional): The ratio of hidden features in the MLP to the input dimension. Defaults to 4.0.
        qkv_bias (bool, optional): Whether to include bias terms in the query, key, and value projections. Defaults to False.
        drop (float, optional): Dropout rate for the MLP and attention projections. Defaults to 0.0.
        attn_drop (float, optional): Dropout rate specifically for the attention weights. Defaults to 0.0.
        drop_path (float, optional): Drop path rate for stochastic depth regularization. Defaults to 0.0.
    """

    dim: int
    num_heads: int
    mlp_ratio: float = 4.0
    qkv_bias: bool = False
    drop: float = 0.0
    attn_drop: float = 0.0
    drop_path: float = 0.0
    norm_layer = nn.LayerNorm

    def __post_init__(self):
        super().__init__()
        self.norm1 = self.norm_layer(self.dim)
        self.attn = RotarySelfAttentionBlock(
            dim=self.dim,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            attn_drop=self.attn_drop,
            proj_drop=self.drop,
        )
        self.drop_path1 = (
            DropPath(self.drop_path) if self.drop_path > 0.0 else nn.Identity()
        )
        self.drop_path2 = (
            DropPath(self.drop_path) if self.drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = self.norm_layer(self.dim)
        self.mlp = Mlp(
            in_features=self.dim,
            hidden_features=int(self.dim * self.mlp_ratio),
            act_layer=nn.GELU,
            drop=self.drop,
        )

    def forward(
        self,
        x: torch.Tensor,
        pos_ids: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        x = x + self.drop_path1(
            self.attn(self.norm1(x), pos_ids=pos_ids, attn_mask=attn_mask)
        )
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


@dataclass(eq=False)
class PatchReconstructionHead(nn.Module):
    """
    A neural network module for reconstructing image patches from token embeddings.

    This head takes token embeddings as input and projects them back to the pixel space
    for patch reconstruction. It is designed for use in vision transformer models where
    patches are embedded and then reconstructed.

        img_size (int): The size of the input image.
        patch_size (int): The size of each patch.
        in_chans (int): Number of input channels.
        embed_dim (int): Dimensionality of the embedding space.

    Attributes:
        in_chans (int): Number of input channels.
        img_size (int): The size of the input image.
        patch_size (int): The size of each patch.
        embed_dim (int): Dimensionality of the embedding space.
        reconstruction_shape (int): Shape of the reconstructed patch, equal to patch_size.
        decoder_pred (nn.Linear): Linear layer for projecting embeddings to pixel space.
    """

    img_size: int
    patch_size: int
    in_chans: int
    embed_dim: int

    def __post_init__(self):
        super().__init__()
        self.reconstruction_shape = self.patch_size

        # Projection from embed space to pixel space
        self.decoder_pred = nn.Linear(self.embed_dim, self.reconstruction_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        No cls token is expected
        Args:
            x: [B, num_tokens, embed_dim] - token embeddings
        """
        x = self.decoder_pred(x)
        return x


@dataclass(eq=False)
class EMGClassificationHead(nn.Module):
    """
    A classification head for EMG (Electromyography) data processing, designed to classify token embeddings into a specified number of classes.

    This module takes token embeddings as input, applies a reduction strategy (concatenation across channels),
    averages across patches, and then uses a linear classifier to produce logits for classification.

        embed_dim (int): Dimensionality of the token embeddings.
        num_classes (int): Number of output classes for classification.
        in_chans (int): Number of input channels (e.g., EMG channels).

    Attributes:
        classifier (nn.Linear): Linear layer for final classification, mapping from reduced feature dimension to num_classes.
    """

    embed_dim: int
    num_classes: int
    in_chans: int

    def __post_init__(self):
        super().__init__()
        feat_dim = self.in_chans * self.embed_dim

        self.classifier = nn.Linear(feat_dim, self.num_classes)

        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: token embeddings, shape (B, num_tokens, embed_dim)
        Returns:
            logits: (B, num_classes)
        """
        B, N, D = x.shape
        num_patches = N // self.in_chans

        # Reshape to (B, num_patches, embed_dim * in_chans)
        x = rearrange(x, "b (c p) d -> b p (c d)", c=self.in_chans, p=num_patches)
        x = x.mean(dim=1)

        # apply projection to get logits
        logits = self.classifier(x)
        return logits


@dataclass(eq=False)
class EMGRegressionHead(nn.Module):
    """
    A regression head for EMG (Electromyography) signals using convolutional layers.

    This module processes embedded features from a transformer model to perform
    regression, predicting output signals of a specified dimension and length upsampling to a target sequence length.

    Args:
        in_chans (int): Number of input channels (e.g., EMG channels).
        embed_dim (int): Dimension of the input embeddings.
        output_dim (int): Dimension of the output regression targets.
        hidden_dim (int): Hidden dimension for the convolutional layers. Defaults to 256.
        dropout (float): Dropout probability applied after the first convolution. Defaults to 0.1.
        target_length (int): Desired length of the output sequence. If the input length differs,
            linear interpolation is used to upsample. Defaults to 1000.
    """

    in_chans: int
    embed_dim: int
    output_dim: int
    hidden_dim: int = 256
    dropout: float = 0.1
    target_length: int = 1000

    def __post_init__(self):
        super().__init__()
        feat_dim = self.in_chans * self.embed_dim

        self.regressor = nn.Sequential(
            nn.Conv1d(feat_dim, self.hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            # depthwise 3x3 conv: groups=hidden_dim to hidden_dimx3 params
            nn.Conv1d(
                self.hidden_dim,
                self.hidden_dim,
                kernel_size=3,
                padding=1,
                groups=self.hidden_dim,
            ),
            nn.SiLU(),
            # pointwise 1x1 back to output_dim
            nn.Conv1d(self.hidden_dim, self.output_dim, kernel_size=1),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, num_tokens, token_dim), where B is batch size,
                              num_tokens is the number of tokens, and token_dim is the token dimension.

        Returns:
            torch.Tensor: Output tensor of shape (B, target_length, output_dim), where target_length
                          is the target sequence length and output_dim is the output dimension.
        """
        # x: (B, num_tokens, token_dim)
        x = rearrange(
            x, "b (c p) d -> b p (c d)", c=self.in_chans, p=x.size(1) // self.in_chans
        )

        # conv head expects (B, C, L)
        x = x.transpose(1, 2)  # (B, feat_dim, num_patches)
        x = self.regressor(x)  # (B, output_dim, num_patches)

        # now upsample to target length
        if x.size(-1) != self.target_length:
            x = F.interpolate(
                x, size=self.target_length, mode="linear", align_corners=False
            )

        # x: (B, output_dim, target_length)
        out = x.transpose(1, 2)  # (B, target_length, output_dim)
        return out


@dataclass(eq=False)
class TinyMyo(nn.Module):
    """
    TinyMyo is a bidirectional Transformer model based on the Vision Transformer (ViT) architecture, adapted for electromyography (EMG) signal processing.
    It supports multiple tasks including pretraining (reconstruction), classification, and regression.

    The model uses a patch-based embedding approach to process input signals, followed by a series of transformer blocks with rotary position embeddings.
    It includes a masking token for pretraining tasks and different heads for various downstream tasks.

    Args:
        img_size (int, optional): The size of the input signal (temporal dimension). Defaults to 1000.
        patch_size (int, optional): The size of each patch for embedding. Defaults to 20.
        in_chans (int, optional): Number of input channels (e.g., EMG channels). Defaults to 16.
        embed_dim (int, optional): Dimensionality of the embedding space. Defaults to 192.
        n_layer (int, optional): Number of transformer layers. Defaults to 8.
        n_head (int, optional): Number of attention heads. Defaults to 3.
        mlp_ratio (int, optional): Ratio for expanding the MLP hidden dimension. Defaults to 4.
        qkv_bias (bool, optional): Whether to include bias in QKV projections. Defaults to True.
        attn_drop (float, optional): Dropout rate for attention. Defaults to 0.1.
        proj_drop (float, optional): Dropout rate for projections. Defaults to 0.1.
        drop_path (float, optional): Stochastic depth drop path rate. Defaults to 0.1.
        norm_layer (nn.Module, optional): Normalization layer class. Defaults to nn.LayerNorm.
        task (str, optional): Task type, one of "pretraining", "classification", or "regression". Defaults to "classification".
        num_classes (int, optional): Number of classes for classification or output dimension for regression. Defaults to 53.
    """

    img_size: int = 1000
    patch_size: int = 20
    in_chans: int = 16
    embed_dim: int = 192
    n_layer: int = 8
    n_head: int = 3
    mlp_ratio: int = 4
    qkv_bias: bool = True
    attn_drop: float = 0.1
    proj_drop: float = 0.1
    drop_path: float = 0.1
    norm_layer = nn.LayerNorm
    task: Literal["pretraining", "classification", "regression"] = "classification"
    num_classes: int = 53

    def __post_init__(self):
        super().__init__()
        # Learnable mask token for pretraining (dimension: embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # Learnable channel IDs (always initialized to the maximum number of channels, but only the first C are used based on input)
        self.channel_embed = nn.Parameter(torch.zeros(1, 16, 1, self.embed_dim))

        self.patch_embedding = PatchingModule(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
        )
        self.num_patches = self.patch_embedding.num_patches

        self.blocks = nn.ModuleList(
            [
                RotaryTransformerBlock(
                    dim=self.embed_dim,
                    num_heads=self.n_head,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    drop=self.proj_drop,
                    attn_drop=self.attn_drop,
                    drop_path=self.drop_path,
                )
                for _ in range(self.n_layer)
            ]
        )
        self.norm = self.norm_layer(self.embed_dim)

        # reconstruction (pre-training)
        if self.task == "pretraining":
            self.model_head = PatchReconstructionHead(
                img_size=self.img_size,
                patch_size=self.patch_size,
                in_chans=self.in_chans,
                embed_dim=self.embed_dim,
            )
        # classification or regression tasks
        elif self.task == "classification":
            assert self.num_classes > 0, (
                "num_classes must be > 0 for classification task"
            )
            self.model_head = EMGClassificationHead(
                embed_dim=self.embed_dim,
                num_classes=self.num_classes,
                in_chans=self.in_chans,
            )
        elif self.task == "regression":
            assert self.num_classes > 0, (
                "num_classes must be > 0 for regression task (output dimension of regression targets)"
            )
            self.model_head = EMGRegressionHead(
                in_chans=self.in_chans,
                embed_dim=self.embed_dim,
                output_dim=self.num_classes,
                target_length=self.img_size,
            )
        else:
            raise ValueError(f"Unknown task type {self.task}")
        self.initialize_weights()

        # Some checks
        assert self.img_size % self.patch_size == 0, (
            f"img_size ({self.img_size}) must be divisible by patch_size ({self.patch_size})"
        )

    def initialize_weights(self):
        """Initializes the model weights."""
        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.channel_embed, std=0.02)

        self.apply(self._init_weights)
        self.fix_init_weight()

    def _init_weights(self, m):
        """Initializes the model weights."""
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def fix_init_weight(self):
        """Rescales the weights of attention and MLP layers to improve training stability."""

        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def prepare_tokens(
        self, x_signal: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        _, C, T = x_signal.shape

        x_patched = self.patch_embedding(x_signal)  # [B, N, D] where N = C * P
        P = T // self.patch_size

        # Unflatten to grid:[Batch, Channels, Patches, Dim]
        x_patched = rearrange(x_patched, "B (C P) D -> B C P D", C=C, P=P)

        if mask is not None:
            # Reduce independent signal mask (B, C, T) to patch mask (B, C, P)
            mask_p = rearrange(mask, "B C (P s) -> B C P s", s=self.patch_size).any(
                dim=-1
            )
            mask_exp = mask_p.unsqueeze(-1)  # (B, C, P, 1)

            # Apply Masking
            x_patched = torch.where(
                mask_exp, self.mask_token.view(1, 1, 1, -1), x_patched
            )

        # Apply Spatial Identity
        # Crop to C channels, fewer channels can be used
        x = x_patched + self.channel_embed[:, :C, :, :]

        return rearrange(x, "B C P D -> B (C P) D")

    def forward(
        self,
        x_signal: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        pad_mask_ch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, C, T = x_signal.shape
        x = self.prepare_tokens(x_signal, mask=mask)

        _, N, _ = x.shape
        # Number of patches per channel
        P = T // self.patch_size
        assert C * P == N, f"Token shape mismatch: expected {C * P} tokens, got {N}."

        # Cyclic Sequence for RoPE
        # Sequence resets for each channel:[0..P-1, 0..P-1, ...]
        pos_ids_single = torch.arange(P, device=x.device).repeat(C)
        pos_ids = pos_ids_single.unsqueeze(0).expand(B, -1)  # Shape (B, N)

        # Attention Masking for Padded Channels
        attn_mask = None
        if pad_mask_ch is not None:
            token_pad_mask = pad_mask_ch.repeat_interleave(P, dim=1)
            # Boolean mask for SDPA: True means keep, False means mask out
            attn_mask = (~token_pad_mask).reshape(B, 1, 1, N)

        # Pass through RoPE-Attention Blocks
        for blk in self.blocks:
            x = blk(x, pos_ids=pos_ids, attn_mask=attn_mask)

        # Final normalization
        x = self.norm(x)

        # Pass through task-specific head
        return self.model_head(x)


if __name__ == "__main__":
    model = TinyMyo()
    input_signal = torch.randn(1, 16, 1000)  # (B, C, T)
    with torch.no_grad():
        output = model(input_signal)
    print("Input shape:", input_signal.shape)
    print("Output shape:", output.shape)
