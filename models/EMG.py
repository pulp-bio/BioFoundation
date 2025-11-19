import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from timm.layers import DropPath, Mlp
from timm.layers import trunc_normal_ as __call_trunc_normal_


def trunc_normal_(tensor, mean=0.0, std=1.0):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class PatchEmbedWaveformKeepChans(nn.Module):
    """Waveform to embedding that maintain the channel dimension"""

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_chans: int = 23,
        embed_dim: int = 1024,
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
        img_size: int = 64,
        patch_size: int = 8,
        in_chans: int = 23,
        embed_dim: int = 1024,
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
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.rotary_emb = RotaryEmbedding(
            dim=head_dim,
            learned_freq=False,
            theta=10_000,
            cache_if_possible=True,
            cache_max_seq_len=1024,
        )

        self.scale = qk_scale or head_dim**-0.5

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
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

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
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = RotarySelfAttentionBlock(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
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
    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_chans: int = 23,
        embed_dim: int = 768,
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


class EMG(nn.Module):
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
        num_classes: int = 53,
        classification_type: str = "ml",
    ):
        super().__init__()

        # MAE encoder
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.n_layer = n_layer
        self.n_head = n_head
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.classification_type = classification_type

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

        if num_classes == 0:  # reconstruction (pre-training)
            self.model_head = PatchReconstructionHead(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        else:  # classification (fine-tuning)
            self.model_head = EMGClassificationHead(
                embed_dim=embed_dim,
                num_classes=num_classes,
                reduction="concat",
                in_chans=in_chans,
            )
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
        # x_signal: (B, C, T)
        x_original = x.clone()
        if not directly_input_tokens:
            x = self.patch_embedding(x)  # (B, N, D)

        # forward pass through transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x_latent = self.norm(x)  # [B, N, D]

        if self.num_classes > 0:
            x_classified = self.model_head(x_latent)  # [B, Out]
            return x_classified, x_original

        else:
            x_reconstructed = self.model_head(x_latent)  # [B, N, patch_size]
            return x_reconstructed, x_original
