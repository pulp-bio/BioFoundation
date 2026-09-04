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

from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath, Mlp


def build_window_indices(
    length: int,
    window_size: int,
    dilation: int,
    include_self: bool,
    shift: int,
    device: torch.device,
) -> torch.Tensor:
    """Build clamped, dilated, shifted window indices for every position along an axis.

    Args:
        length: Number of positions along the axis.
        window_size: Requested window size; clamped to ``length``.
        dilation: Spacing between consecutive window offsets.
        include_self: Whether a position attends to itself.
        shift: Constant offset applied to the whole window.
        device: Device of the returned index tensor.

    Returns:
        Long tensor of shape ``(length, effective_window_size)``.
    """
    step = max(int(dilation), 1)
    effective_size = max(1, min(int(window_size), int(length)))
    offset = int(shift)

    if include_self:
        left = (effective_size - 1) // 2
        right = effective_size - 1 - left
        offsets = torch.cat([
            torch.arange(-left, 0, device=device),
            torch.zeros(1, device=device, dtype=torch.long),
            torch.arange(1, right + 1, device=device),
        ]) * step + offset
    else:
        left = effective_size // 2
        right = effective_size - left
        offsets = torch.cat([
            torch.arange(-left, 0, device=device),
            torch.arange(1, right + 1, device=device),
        ]) * step + offset

    base = torch.arange(length, device=device)[:, None]
    return (base + offsets[None, :]).clamp_(0, length - 1).to(torch.long)


class WindowedAlternatingAttention(nn.Module):
    """Windowed attention that alternates between the spatial and temporal axes.

    Tokens are laid out as ``num_channels * num_patches`` per sample. Even-indexed
    blocks attend across channels at a fixed time index (spatial pass); odd-indexed
    blocks attend across time within a fixed channel (temporal pass). Each pass is
    restricted to a dilated, optionally shifted window, so cost per block is
    ``O(batch * channels * patches * window_size)`` instead of quadratic in the
    full token count.

    Setting ``use_axial_mode`` replaces the alternating schedule with an axial one:
    the first half of the blocks perform spatial attention and the second half
    temporal attention.

    Padded keys are masked to ``-inf`` before the softmax and padded queries are
    zeroed on output, so variable channel counts share one batch safely.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        *,
        num_channels: int,
        block_idx: int = 0,
        total_blocks: Optional[int] = None,
        use_axial_mode: bool = False,
        spatial_first: bool = True,
        window_size_spatial: int = 7,
        window_size_temporal: int = 5,
        dilation_spatial: int = 1,
        dilation_temporal: int = 1,
        include_self: bool = True,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        normalize_qk: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.num_channels = int(num_channels)
        self.block_idx = int(block_idx)
        self.use_axial_mode = bool(use_axial_mode)

        if self.use_axial_mode:
            if total_blocks is None:
                raise ValueError("total_blocks is required when use_axial_mode is True")
            half = total_blocks // 2
            self.spatial_pass = block_idx < half if spatial_first else block_idx >= half
        else:
            self.spatial_pass = (block_idx % 2 == 0)

        self.window_size_spatial = int(window_size_spatial)
        self.window_size_temporal = int(window_size_temporal)
        self.dilation_spatial = int(dilation_spatial)
        self.dilation_temporal = int(dilation_temporal)
        self.include_self = bool(include_self)
        self.normalize_qk = bool(normalize_qk)

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self._index_cache: Dict[Tuple[str, int, int], torch.Tensor] = {}

    def _window_indices(
        self, axis: str, length: int, window_size: int, dilation: int, shift: int, device: torch.device
    ) -> torch.Tensor:
        """Return cached window indices for one axis, rebuilding on device change."""
        key = (axis, length, shift)
        cached = self._index_cache.get(key)
        if cached is None or cached.device != device:
            cached = build_window_indices(
                length=length,
                window_size=window_size,
                dilation=dilation,
                include_self=self.include_self,
                shift=shift,
                device=device,
            )
            self._index_cache[key] = cached
        return cached

    def _windowed_softmax_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_mask: Optional[torch.Tensor],
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Attend from each position to its window along dim 1.

        Args:
            q, k, v: Tensors of shape ``(flat_batch, length, heads, head_dim)``.
            key_mask: Optional ``(flat_batch, length)`` mask, 1 for real and 0 for padded keys.
            indices: ``(length, window)`` key indices per query.

        Returns:
            Tensor of shape ``(flat_batch, length, heads, head_dim)``.
        """
        length, window = indices.shape
        gather = indices.view(1, length, window, 1, 1)

        keys = torch.take_along_dim(k.unsqueeze(2), gather, dim=1)
        values = torch.take_along_dim(v.unsqueeze(2), gather, dim=1)

        logits = (q.unsqueeze(2) * keys).sum(-1) * self.scale

        fully_masked = None
        if key_mask is not None:
            mask = key_mask.to(logits.dtype)
            mask_window = torch.take_along_dim(mask.unsqueeze(2), indices.view(1, length, window), dim=1)
            logits = logits.masked_fill(mask_window.unsqueeze(-1) <= 0, float("-inf"))
            fully_masked = (mask_window.sum(dim=2, keepdim=True) == 0)
            logits = torch.where(fully_masked.unsqueeze(-1), torch.zeros_like(logits), logits)

        attn = torch.softmax(logits, dim=2).to(values.dtype)
        if fully_masked is not None:
            attn = torch.where(fully_masked.unsqueeze(-1), torch.zeros_like(attn), attn)
        attn = self.attn_drop(attn)

        return (attn.unsqueeze(-1) * values).sum(dim=2)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        *,
        shift_spatial: int = 0,
        shift_temporal: int = 0,
    ) -> torch.Tensor:
        """Apply one windowed spatial or temporal attention pass.

        Args:
            x: Token embeddings of shape ``(batch, num_channels * num_patches, dim)``.
            attn_mask: Optional ``(batch, num_tokens)`` mask, 1 for real and 0 for padded tokens.
            shift_spatial: Window shift applied on a spatial pass.
            shift_temporal: Window shift applied on a temporal pass.

        Returns:
            Tensor of shape ``(batch, num_tokens, dim)``.
        """
        batch, num_tokens, dim = x.shape
        channels = self.num_channels
        if num_tokens % channels != 0:
            raise ValueError("num_tokens must be divisible by num_channels")
        patches = num_tokens // channels

        qkv = self.qkv(x).reshape(batch, num_tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.normalize_qk:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)

        shape = (batch, channels, patches, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3).reshape(shape)
        k = k.permute(0, 2, 1, 3).reshape(shape)
        v = v.permute(0, 2, 1, 3).reshape(shape)
        mask = attn_mask.view(batch, channels, patches) if attn_mask is not None else None

        if self.spatial_pass:
            indices = self._window_indices(
                "spatial", channels, self.window_size_spatial, self.dilation_spatial, shift_spatial, x.device
            )
            flat = (batch * patches, channels, self.num_heads, self.head_dim)
            q_flat = q.permute(0, 2, 1, 3, 4).reshape(flat)
            k_flat = k.permute(0, 2, 1, 3, 4).reshape(flat)
            v_flat = v.permute(0, 2, 1, 3, 4).reshape(flat)
            mask_flat = mask.permute(0, 2, 1).reshape(batch * patches, channels) if mask is not None else None
            out = self._windowed_softmax_attention(q_flat, k_flat, v_flat, mask_flat, indices)
            out = out.reshape(batch, patches, channels, self.num_heads, self.head_dim).permute(0, 2, 1, 3, 4)
        else:
            indices = self._window_indices(
                "temporal", patches, self.window_size_temporal, self.dilation_temporal, shift_temporal, x.device
            )
            flat = (batch * channels, patches, self.num_heads, self.head_dim)
            q_flat = q.reshape(flat)
            k_flat = k.reshape(flat)
            v_flat = v.reshape(flat)
            mask_flat = mask.reshape(batch * channels, patches) if mask is not None else None
            out = self._windowed_softmax_attention(q_flat, k_flat, v_flat, mask_flat, indices)
            out = out.reshape(batch, channels, patches, self.num_heads, self.head_dim)

        out = out.reshape(batch, num_tokens, dim)
        out = self.proj_drop(self.proj(out))

        if attn_mask is not None:
            out = out * attn_mask.unsqueeze(-1).to(out.dtype)

        return out


class AlternatingAttention(nn.Module):
    """Full attention that alternates between the spatial and temporal axes.

    Even-indexed blocks attend over all channels at a fixed time index, odd-indexed
    blocks over all time indices within a channel. This is the unwindowed ablation
    of :class:`WindowedAlternatingAttention`.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        num_channels: int = 64,
        block_idx: int = 0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_channels = num_channels
        self.spatial_pass = (block_idx % 2 == 0)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply one full spatial or temporal attention pass.

        Args:
            x: Token embeddings of shape ``(batch, num_channels * num_patches, dim)``.
            attn_mask: Optional ``(batch, num_tokens)`` mask, 1 for real and 0 for padded tokens.

        Returns:
            Tensor of shape ``(batch, num_tokens, dim)``.
        """
        num_tokens = x.shape[1]
        patches = num_tokens // self.num_channels

        if self.spatial_pass:
            x = rearrange(x, "B (C T) D -> (B T) C D", C=self.num_channels)
            if attn_mask is not None:
                attn_mask = rearrange(attn_mask, "B (C T) -> (B T) C", C=self.num_channels)
            x = self._attend(x, attn_mask)
            return rearrange(x, "(B T) C D -> B (C T) D", T=patches)

        x = rearrange(x, "B (C T) D -> (B C) T D", C=self.num_channels)
        if attn_mask is not None:
            attn_mask = rearrange(attn_mask, "B (C T) -> (B C) T", C=self.num_channels)
        x = self._attend(x, attn_mask)
        return rearrange(x, "(B C) T D -> B (C T) D", C=self.num_channels)

    def _attend(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Scaled dot-product attention over the full second axis of ``x``."""
        batch, length, dim = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1).expand(batch, self.num_heads, length, length)
            attn = attn.masked_fill(attn_mask == 0, float("-inf"))

        attn = attn.softmax(dim=-1)
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask.sum(dim=-1, keepdim=True).eq(0), 0.0)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch, length, dim)
        return self.proj_drop(self.proj(x))


class FullAttention(nn.Module):
    """Standard multi-head self-attention over the complete token sequence."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Attend over all tokens.

        Args:
            x: Token embeddings of shape ``(batch, num_tokens, dim)``.
            attn_mask: Optional ``(batch, num_tokens)`` mask, 1 for real and 0 for padded tokens.

        Returns:
            Tensor of shape ``(batch, num_tokens, dim)``.
        """
        batch, num_tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, num_tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        if attn_mask is not None:
            expanded = attn_mask.unsqueeze(1).unsqueeze(1).expand(batch, self.num_heads, num_tokens, num_tokens)
            attn = attn.masked_fill(expanded == 0, float("-inf"))

        attn = attn.softmax(dim=-1)
        if attn_mask is not None:
            attn = attn.masked_fill(expanded.sum(dim=-1, keepdim=True).eq(0), 0.0)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch, num_tokens, dim)
        return self.proj_drop(self.proj(x))


class LayerScale(nn.Module):
    """Per-channel learnable rescaling of a residual branch."""

    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with a configurable attention mechanism.

    ``attention_type`` selects one of ``windowed-alternating`` (the CEReBrO
    default), ``alternating``, or ``full``. For the windowed variant, the
    dilation and shift schedules are indexed by spatial/temporal *pair* rather
    than by block, so a spatial block and the temporal block that follows it
    share the same schedule entry.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
        num_channels: int = 23,
        attention_type: str = "windowed-alternating",
        block_idx: int = 0,
        total_blocks: int = 12,
        spatial_first: bool = True,
        window_size_spatial: int = 7,
        window_size_temporal: int = 5,
        dilation_cycle_spatial: Iterable[int] = (1, 2, 4),
        dilation_cycle_temporal: Iterable[int] = (1, 2, 4),
        shift_cycle_spatial: Iterable[int] = (-1, 1, -2, 2),
        shift_cycle_temporal: Iterable[int] = (-1, 1, -2, 2),
        include_self: bool = True,
        normalize_qk: bool = False,
        use_axial_mode: bool = False,
    ) -> None:
        super().__init__()
        self.attention_type = attention_type
        self.block_idx = block_idx

        dilation_cycle_spatial = tuple(dilation_cycle_spatial)
        dilation_cycle_temporal = tuple(dilation_cycle_temporal)
        shift_cycle_spatial = tuple(shift_cycle_spatial)
        shift_cycle_temporal = tuple(shift_cycle_temporal)
        pair_idx = block_idx // 2

        self.shift_spatial = shift_cycle_spatial[pair_idx % len(shift_cycle_spatial)]
        self.shift_temporal = shift_cycle_temporal[pair_idx % len(shift_cycle_temporal)]

        self.norm1 = norm_layer(dim)

        if attention_type == "windowed-alternating":
            self.attn = WindowedAlternatingAttention(
                dim=dim,
                num_heads=num_heads,
                num_channels=num_channels,
                block_idx=block_idx,
                total_blocks=total_blocks,
                use_axial_mode=use_axial_mode,
                spatial_first=spatial_first,
                window_size_spatial=window_size_spatial,
                window_size_temporal=window_size_temporal,
                dilation_spatial=dilation_cycle_spatial[pair_idx % len(dilation_cycle_spatial)],
                dilation_temporal=dilation_cycle_temporal[pair_idx % len(dilation_cycle_temporal)],
                include_self=include_self,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                normalize_qk=normalize_qk,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )
        elif attention_type == "alternating":
            self.attn = AlternatingAttention(
                dim,
                num_heads=num_heads,
                num_channels=num_channels,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                norm_layer=norm_layer,
                block_idx=block_idx,
            )
        elif attention_type == "full":
            self.attn = FullAttention(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                norm_layer=norm_layer,
            )
        else:
            raise ValueError(
                f"Unknown attention_type '{attention_type}'; "
                "expected 'windowed-alternating', 'alternating', or 'full'"
            )

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run attention and the feed-forward network with residual connections."""
        if self.attention_type == "windowed-alternating":
            attended = self.attn(
                self.norm1(x),
                attn_mask,
                shift_spatial=self.shift_spatial,
                shift_temporal=self.shift_temporal,
            )
        else:
            attended = self.attn(self.norm1(x), attn_mask)

        x = x + self.drop_path1(self.ls1(attended))
        return x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
