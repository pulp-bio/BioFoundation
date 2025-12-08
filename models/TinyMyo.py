import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath, Mlp
from timm.layers import trunc_normal_ as __call_trunc_normal_


def trunc_normal_(tensor, mean=0.0, std=1.0):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


# https://docs.pytorch.org/torchtune/stable/_modules/torchtune/modules/position_embeddings.html#RotaryPositionalEmbeddings
class RotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
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

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
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
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

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

    def __init__(
        self,
        img_size: int = 1000,
        patch_size: int = 20,
        in_chans: int = 16,
        embed_dim: int = 192,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_patches = (self.img_size // self.patch_size) * self.in_chans
        self.proj = nn.Conv2d(
            1,
            embed_dim,
            kernel_size=(1, self.patch_size),
            stride=(1, self.patch_size),
        )

    def forward(self, x):
        B, C, T = x.shape
        # shared projection layer across channels
        x = self.proj(x.unsqueeze(1))
        x = rearrange(x, "B D C t -> B (C t) D")
        return x


class PatchingModule(nn.Module):
    """Image to Patch Embedding of choice according to the parameters given."""

    def __init__(
        self,
        img_size: int = 1000,
        patch_size: int = 20,
        in_chans: int = 16,
        embed_dim: int = 192,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedWaveformKeepChans(
            img_size, patch_size, in_chans, embed_dim
        )
        self.num_patches = self.patch_embed.num_patches
        self.init_patch_embed()

    def init_patch_embed(self):
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def forward(self, x):
        return self.patch_embed(x)


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

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.rope = RotaryPositionalEmbeddings(
            dim=head_dim,
            base=10_000,
            max_seq_len=1024,
        )
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.attn_drop_fn = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )  # (K, B, H, N, D)
        q, k, v = qkv.unbind(0)  # each: (B, H, N, D)

        q = self.rope(q)
        k = self.rope(k)

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
            enable_gqa=False,
        )

        x = x.transpose(2, 1).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RotaryTransformerBlock(nn.Module):
    """
    A transformer block that incorporates rotary self-attention for enhanced positional encoding.

    This block applies layer normalization, rotary self-attention, and a multi-layer perceptron (MLP)
    in sequence, with optional drop paths for regularization. It follows a standard transformer
    architecture but uses rotary embeddings to improve handling of sequential data.

    Args:
        dim (int): The dimensionality of the input and output features.
        num_heads (int): The number of attention heads in the self-attention mechanism.
        mlp_ratio (float, optional): The ratio of hidden features in the MLP to the input dimension. Defaults to 4.0.
        qkv_bias (bool, optional): Whether to include bias terms in the query, key, and value projections. Defaults to False.
        drop (float, optional): Dropout rate for the MLP and attention projections. Defaults to 0.0.
        attn_drop (float, optional): Dropout rate specifically for the attention weights. Defaults to 0.0.
        drop_path (float, optional): Drop path rate for stochastic depth regularization. Defaults to 0.0.
        norm_layer (nn.Module, optional): Normalization layer to use. Defaults to nn.LayerNorm.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = RotarySelfAttentionBlock(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class PatchReconstructionHead(nn.Module):
    """
    A neural network module for reconstructing image patches from token embeddings.

    This head takes token embeddings as input and projects them back to the pixel space
    for patch reconstruction. It is designed for use in vision transformer models where
    patches are embedded and then reconstructed.

        img_size (int, optional): The size of the input image. Defaults to 1000.
        patch_size (int, optional): The size of each patch. Defaults to 20.
        in_chans (int, optional): Number of input channels. Defaults to 16.
        embed_dim (int, optional): Dimensionality of the embedding space. Defaults to 192.

    Attributes:
        in_chans (int): Number of input channels.
        img_size (int): The size of the input image.
        patch_size (int): The size of each patch.
        embed_dim (int): Dimensionality of the embedding space.
        reconstruction_shape (int): Shape of the reconstructed patch, equal to patch_size.
        decoder_pred (nn.Linear): Linear layer for projecting embeddings to pixel space.
    """

    def __init__(
        self,
        img_size: int = 1000,
        patch_size: int = 20,
        in_chans: int = 16,
        embed_dim: int = 192,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.reconstruction_shape = self.patch_size

        # Projection from embed space to pixel space
        self.decoder_pred = nn.Linear(
            self.embed_dim, self.reconstruction_shape, bias=True
        )

    def forward(self, x):
        """
        No cls token is expected
        Args:
            x: [B, num_tokens, embed_dim] - token embeddings
        """
        x = self.decoder_pred(x)
        return x


class EMGClassificationHead(nn.Module):
    """
    A classification head for EMG (Electromyography) data processing, designed to classify token embeddings into a specified number of classes.

    This module takes token embeddings as input, applies a reduction strategy (either mean or concatenation across channels),
    averages across patches, and then uses a linear classifier to produce logits for classification.

        embed_dim (int, optional): Dimensionality of the token embeddings. Defaults to 192.
        num_classes (int, optional): Number of output classes for classification. Defaults to 53.
        reduction (str, optional): Reduction strategy for combining channel features. Options are "mean" or "concat".
            - "mean": Averages across channels, resulting in feature dimension of embed_dim.
            - "concat": Concatenates across channels, resulting in feature dimension of in_chans * embed_dim. Defaults to "concat".
        in_chans (int, optional): Number of input channels (e.g., EMG channels). Defaults to 16.

    Attributes:
        classifier (nn.Linear): Linear layer for final classification, mapping from reduced feature dimension to num_classes.
    """

    def __init__(
        self,
        embed_dim: int = 192,
        num_classes: int = 53,
        reduction: str = "concat",
        in_chans: int = 16,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.reduction = reduction
        self.in_chans = in_chans

        # after reduction, feature_dim to either embed_dim or in_chans*embed_dim
        feat_dim = embed_dim if reduction == "mean" else in_chans * embed_dim

        self.classifier = nn.Linear(feat_dim, num_classes)

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

        if self.reduction == "mean":
            # Reshape to (B, in_chans, num_patches, embed_dim)
            x = rearrange(x, "b (c p) d -> b c p d", c=self.in_chans, p=num_patches)
            # Take mean across the channels (in_chans)
            x = x.mean(dim=1)  # (B, num_patches, embed_dim)
        elif self.reduction == "concat":
            # Reshape to (B, num_patches, embed_dim * in_chans)
            x = rearrange(x, "b (c p) d -> b p (c d)", c=self.in_chans, p=num_patches)

        # average across patches
        x = x.mean(dim=1)  # (B, feat_dim)

        # apply projection to get logits
        logits = self.classifier(x)
        return logits


class EMGRegressionHead(nn.Module):
    """
    A regression head for EMG (Electromyography) signals using convolutional layers.

    This module processes embedded features from a transformer model to perform
    regression, predicting output signals of a specified dimension and length. It supports
    different reduction methods for combining channel and patch features, followed by
    convolutional layers for regression, and optional upsampling to a target sequence length.

    Args:
        in_chans (int): Number of input channels (e.g., EMG channels).
        embed_dim (int): Dimension of the input embeddings.
        output_dim (int): Dimension of the output regression targets.
        reduction (str, optional): Method to reduce features across channels.
            "mean" averages embeddings, "concat" concatenates them. Defaults to "concat".
        hidden_dim (int, optional): Hidden dimension for the convolutional layers. Defaults to 256.
        dropout (float, optional): Dropout probability applied after the first convolution. Defaults to 0.1.
        target_length (int, optional): Desired length of the output sequence. If the input length differs,
            linear interpolation is used to upsample. Defaults to 500.

    Attributes:
        in_chans (int): Number of input channels.
        embed_dim (int): Dimension of the embeddings.
        output_dim (int): Dimension of the output.
        reduction (str): Reduction method used.
        dropout (float): Dropout rate.
        target_length (int): Target output sequence length.
    """

    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        output_dim: int,
        reduction: str = "concat",
        hidden_dim: int = 256,
        dropout: float = 0.1,
        target_length: int = 500,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.reduction = reduction
        self.dropout = dropout
        self.target_length = target_length

        feat_dim = embed_dim if reduction == "mean" else in_chans * embed_dim

        self.regressor = nn.Sequential(
            nn.Conv1d(feat_dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Dropout(dropout),
            # depthwise 3x3 conv: groups=hidden_dim to hidden_dimx3 params
            nn.Conv1d(
                hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim
            ),
            nn.SiLU(),
            # pointwise 1x1 back to output_dim
            nn.Conv1d(hidden_dim, output_dim, kernel_size=1),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_tokens, token_dim)
        if self.reduction == "mean":
            x = rearrange(x, "b (c p) d -> b p d", c=self.in_chans)
        else:  # concat
            x = rearrange(x, "b (c p) d -> b p (c d)", c=self.in_chans)

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
        classification_type (str, optional): Type of classification (e.g., "ml" for multi-label). Defaults to "ml".
        num_classes (int, optional): Number of classes for classification or output dimension for regression. Defaults to 53.
        reg_target_len (int, optional): Target length for regression output. Defaults to 500.
    """

    def __init__(
        self,
        img_size: int = 1000,
        patch_size: int = 20,
        in_chans: int = 16,
        embed_dim: int = 192,
        n_layer: int = 8,
        n_head: int = 3,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
        drop_path: float = 0.1,
        norm_layer=nn.LayerNorm,
        task: str = "classification",
        classification_type: str = "ml",
        num_classes: int = 53,
        reg_target_len: int = 500,
    ):
        super().__init__()

        # MAE encoder
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.n_layer = n_layer
        self.n_head = n_head
        self.embed_dim = embed_dim
        self.task = task
        self.classification_type = classification_type
        self.num_classes = num_classes

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.patch_embedding = PatchingModule(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.num_patches = self.patch_embedding.num_patches

        self.blocks = nn.ModuleList(
            [
                RotaryTransformerBlock(
                    dim=embed_dim,
                    num_heads=n_head,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop,
                    drop=proj_drop,
                    drop_path=drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(n_layer)
            ]
        )
        self.norm = norm_layer(embed_dim)

        if (
            self.task == "pretraining" or num_classes == 0
        ):  # reconstruction (pre-training)
            self.model_head = PatchReconstructionHead(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        elif self.task == "classification" and num_classes > 0:
            self.model_head = EMGClassificationHead(
                embed_dim=embed_dim,
                num_classes=num_classes,
                reduction="concat",
                in_chans=in_chans,
            )
        elif self.task == "regression":
            self.model_head = EMGRegressionHead(
                in_chans=in_chans,
                embed_dim=embed_dim,
                output_dim=num_classes,
                reduction="concat",
                target_length=reg_target_len,
            )
        else:
            raise ValueError(f"Unknown task type {self.task}")
        self.initialize_weights()

        # Some checks
        assert (
            img_size % patch_size == 0
        ), f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"

    def initialize_weights(self):
        """Initializes the model weights."""
        # Encodings Initializations code taken from the LaBraM paper
        trunc_normal_(self.mask_token, std=0.02)

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

    def forward(self, x, directly_input_tokens: bool = False):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor. If directly_input_tokens is False, expected shape is (B, C, T) where B is batch size, C is channels, T is time steps. If True, expected shape is (B, N, D) where N is number of patches, D is embedding dimension.
            directly_input_tokens (bool, optional): If True, skips patch embedding and assumes x is already tokenized. Defaults to False.

        Returns:
            tuple: A tuple containing:
                - If self.num_classes == 0 (reconstruction mode): (x_reconstructed, x_original) where x_reconstructed is the reconstructed output of shape (B, N, patch_size), and x_original is the original input.
                - Otherwise (classification or regression mode): (x_out, x_original) where x_out is the model output of shape (B, Out), and x_original is the original input.
        """
        # x_signal: (B, C, T)
        x_original = x.clone()
        if not directly_input_tokens:
            x = self.patch_embedding(x)  # (B, N, D)

        # forward pass through transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x_latent = self.norm(x)  # [B, N, D]

        if self.num_classes == 0:  # reconstruction
            x_reconstructed = self.model_head(x_latent)  # [B, N, patch_size]
            return x_reconstructed, x_original

        else:  # classification or regression
            x_out = self.model_head(x_latent)  # [B, Out]
            return x_out, x_original
