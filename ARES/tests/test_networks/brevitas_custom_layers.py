# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Custom Brevitas Quantized Layers

Provides custom quantized operations not built into Brevitas:
- QuantAdd: Quantized element-wise addition (for ResNet skip connections)
- QuantConcatenate: Quantized channel concatenation (for DenseNet)
- IntegerSoftmax: Integer polynomial softmax for training (I-BERT style)
- QuantConv1dDepthwise: Depthwise 1D convolution for MAMBA
- QuantSiLU: SiLU activation with LUT export support
- QuantSSM: State Space Model core for MAMBA
- QuantPatchEmbed: Patch embedding for FEMBA (Conv2D + reshape to sequence)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from brevitas.nn import QuantIdentity, QuantLinear, QuantConv2d
from brevitas.quant import Int8ActPerTensorFloat


# Pre-computed LUT for integer softmax: exp(x) * 2^24 for x in [-128, 0]
# Index: x + 128 (so index 0 = exp(-128), index 128 = exp(0))
# This matches the C code in network_dma_pipeline.c.mako exactly
_I_SOFTMAX_LUT = torch.tensor([
             0,          0,          0,          0,          0,          0,          0,          0,
             0,          0,          0,          0,          0,          0,          0,          0,
             0,          0,          0,          0,          0,          0,          0,          0,
             0,          0,          0,          0,          0,          0,          0,          0,
             0,          0,          0,          0,          0,          0,          0,          0,
             0,          0,          0,          0,          0,          0,          0,          0,
             0,          0,          0,          0,          0,          0,          0,          0,
             0,          0,          0,          0,          0,          0,          0,          0,
             0,          0,          0,          0,          0,          0,          0,          0,
             0,          0,          0,          0,          0,          0,          0,          0,
             0,          0,          0,          0,          0,          0,          0,          0,
             0,          0,          0,          0,          0,          0,          0,          0,
             0,          0,          0,          0,          0,          0,          0,          0,
             0,          0,          0,          0,          0,          0,          0,          0,
             1,          5,         13,         37,        103,        280,        761,       2070,
          5628,      15298,      41586,     113043,     307285,     835288,    2270549,    6171992,
      16777216
], dtype=torch.int64)


class IntegerSoftmaxFunction(torch.autograd.Function):
    """
    LUT-based integer softmax for bit-exact matching with C code.

    Uses straight-through estimator for gradient flow:
    - Forward: LUT-based integer computation (matches GAP9 implementation)
    - Backward: Regular softmax gradients

    This allows training with the same implementation used during inference,
    enabling bit-exact matching between Python golden outputs and C execution.
    """

    @staticmethod
    def forward(ctx, x, quant_scale, n_levels):
        # Quantize input to INT8 range
        # The quant_scale maps attention scores to INT8 range
        x_max = x.max(dim=-1, keepdim=True)[0]
        x_shifted = x - x_max

        # Quantize to INT8 (shifted so max is 0)
        x_int = torch.round(x_shifted * quant_scale).clamp(-128, 0).long()

        # LUT lookup: index = x_int + 128 (so -128 maps to 0, 0 maps to 128)
        lut = _I_SOFTMAX_LUT.to(x.device)
        idx = (x_int + 128).clamp(0, 128)
        y = lut[idx]  # Exact exp() values in fixed point

        # Normalize to get probabilities (sum to n_levels-1, typically 255)
        y_sum = y.sum(dim=-1, keepdim=True).float()
        y_sum = y_sum.clamp(min=1.0)  # Prevent division by zero
        probs = (y.float() * (n_levels - 1)) / y_sum / (n_levels - 1)

        # Save for backward pass
        ctx.save_for_backward(probs)

        return probs

    @staticmethod
    def backward(ctx, grad_output):
        # Use standard softmax gradient (straight-through estimator)
        probs, = ctx.saved_tensors

        # Softmax gradient: grad_input = probs * (grad_output - (grad_output * probs).sum(-1, keepdim=True))
        grad_input = probs * (grad_output - (grad_output * probs).sum(dim=-1, keepdim=True))

        return grad_input, None, None


class IntegerSoftmax(nn.Module):
    """
    LUT-based integer softmax module for training.

    Uses a pre-computed lookup table with exact exp(x) * 2^24 values
    for x in [-128, 0]. This matches the C implementation in GAP9 exactly.

    Algorithm:
    1. Quantize input scores to INT8 (shifted so max = 0)
    2. Look up exp() values from LUT
    3. Normalize to get probabilities

    Args:
        quant_scale: Scale factor to map input to INT8 range (default: 16.0)
                    Higher values give more precision but risk overflow
        n_levels: Number of output levels (default: 256 for UINT8)
    """

    def __init__(self, quant_scale=16.0, n_levels=256):
        super().__init__()
        self.quant_scale = quant_scale
        self.n_levels = n_levels

    def forward(self, x):
        return IntegerSoftmaxFunction.apply(x, self.quant_scale, self.n_levels)

    def extra_repr(self):
        return f'quant_scale={self.quant_scale}, n_levels={self.n_levels}'


class QuantAdd(nn.Module):
    """
    Quantized element-wise addition.

    Used for ResNet-style skip connections where two tensors are added
    and the result is quantized to INT8.

    Args:
        bit_width: Bit width for quantization (default: 8)
        return_quant_tensor: Whether to return QuantTensor (default: True)
    """

    def __init__(self, bit_width=8, return_quant_tensor=True, **kwargs):
        super().__init__()
        self.quant = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=return_quant_tensor,
            **kwargs
        )

    def forward(self, x1, x2):
        """
        Add two tensors and quantize the result.

        Args:
            x1: First input tensor (QuantTensor or Tensor)
            x2: Second input tensor (QuantTensor or Tensor)

        Returns:
            Quantized sum (QuantTensor or Tensor based on return_quant_tensor)
        """
        # Extract values if QuantTensor
        if hasattr(x1, 'value'):
            x1 = x1.value
        if hasattr(x2, 'value'):
            x2 = x2.value

        # Add in FP32
        sum_fp32 = x1 + x2

        # Quantize the result
        return self.quant(sum_fp32)


class QuantConcatenate(nn.Module):
    """
    Quantized channel concatenation.

    Used for DenseNet-style dense connections where multiple tensors
    are concatenated along the channel dimension and quantized to INT8.

    Args:
        bit_width: Bit width for quantization (default: 8)
        return_quant_tensor: Whether to return QuantTensor (default: True)
        dim: Dimension along which to concatenate (default: 1 for channels)
    """

    def __init__(self, bit_width=8, return_quant_tensor=True, dim=1, **kwargs):
        super().__init__()
        self.dim = dim
        self.quant = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=return_quant_tensor,
            **kwargs
        )

    def forward(self, tensors):
        """
        Concatenate multiple tensors and quantize the result.

        Args:
            tensors: List of input tensors (QuantTensor or Tensor)

        Returns:
            Quantized concatenation (QuantTensor or Tensor based on return_quant_tensor)
        """
        # Extract values if QuantTensor
        values = []
        for t in tensors:
            if hasattr(t, 'value'):
                values.append(t.value)
            else:
                values.append(t)

        # Concatenate in FP32
        concat_fp32 = torch.cat(values, dim=self.dim)

        # Quantize the result
        return self.quant(concat_fp32)


class QuantMean(nn.Module):
    """
    Quantized mean pooling over a specified dimension.

    Used for sequence-to-vector pooling (e.g., mean pooling over sequence
    length in transformers to get a fixed-size representation).

    Args:
        dim: Dimension to reduce over (default: 1 for sequence dimension)
        keepdim: Whether to keep the reduced dimension (default: False)
        bit_width: Bit width for quantization (default: 8)
        return_quant_tensor: Whether to return QuantTensor (default: True)

    Example:
        Input: [B, seq_len, embed_dim] with dim=1
        Output: [B, embed_dim]
    """

    def __init__(self, dim=1, keepdim=False, bit_width=8, return_quant_tensor=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.quant = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=return_quant_tensor,
            **kwargs
        )

    def forward(self, x):
        """
        Apply mean pooling over the specified dimension and quantize.

        Args:
            x: Input tensor (QuantTensor or Tensor)

        Returns:
            Quantized mean (QuantTensor or Tensor based on return_quant_tensor)
        """
        # Extract value if QuantTensor
        if hasattr(x, 'value'):
            x = x.value

        # Mean pool in FP32
        mean_fp32 = x.mean(dim=self.dim, keepdim=self.keepdim)

        # Quantize the result
        return self.quant(mean_fp32)


class QuantPatchEmbed(nn.Module):
    """
    Quantized Patch Embedding for FEMBA.

    Converts 2D input (e.g., EMG spectrogram) into a sequence of embedded patches
    using a strided convolution, then reshapes to [B, seq_len, embed_dim] format.

    The convolution acts as a linear projection of each patch, and the reshape
    converts the 2D grid of patches into a 1D sequence suitable for MAMBA.

    Architecture:
        x: [B, in_chans, H, W]
            -> Conv2D (kernel=patch_size, stride=stride)
        -> [B, embed_dim, grid_h, grid_w]
            -> reshape to [B, grid_h * embed_dim, grid_w]
            -> permute to [B, grid_w, grid_h * embed_dim]
        -> [B, seq_len, d_model]

    Where:
        - seq_len = grid_w = (W - patch_size) // stride + 1
        - d_model = grid_h * embed_dim

    Args:
        inp_size: Input size (H, W) tuple
        patch_size: Patch size for both height and width
        stride: Stride for the convolution (typically equals patch_size for non-overlapping)
        in_chans: Number of input channels (default: 1 for EMG spectrograms)
        embed_dim: Embedding dimension per patch row (output channels of Conv2D)
        bit_width: Bit width for quantization (default: 8)
        return_quant_tensor: Whether to return QuantTensor (default: True)

    Example for EMG (inp_size=(10, 32), patch_size=2, stride=2, embed_dim=35):
        - Input: [B, 1, 10, 32]
        - After Conv2D: [B, 35, 5, 16]  (grid_h=5, grid_w=16)
        - After reshape: [B, 175, 16] -> permute -> [B, 16, 175]
        - Output: [B, 16, 175] where seq_len=16, d_model=175
    """

    def __init__(
        self,
        inp_size,
        patch_size,
        stride,
        in_chans=1,
        embed_dim=35,
        bit_width=8,
        return_quant_tensor=True,
    ):
        super().__init__()
        self.inp_size = inp_size if isinstance(inp_size, tuple) else (inp_size, inp_size)
        # Support both scalar and tuple patch_size/stride
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # Calculate output grid dimensions
        H, W = self.inp_size
        patch_h, patch_w = self.patch_size
        stride_h, stride_w = self.stride
        self.grid_h = (H - patch_h) // stride_h + 1
        self.grid_w = (W - patch_w) // stride_w + 1
        self.seq_len = self.grid_w  # Sequence length after permute
        self.d_model = self.grid_h * embed_dim  # Model dimension after reshape

        # Projection convolution: converts patches to embeddings
        self.proj = QuantConv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=self.patch_size,
            stride=self.stride,
            padding=0,
            bias=True,
            weight_bit_width=bit_width,
            return_quant_tensor=False  # We'll requantize after reshape
        )

        # Output quantization after reshape
        self.output_quant = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=return_quant_tensor
        )

    def forward(self, x):
        """
        Forward pass: project patches and reshape to sequence.

        Args:
            x: Input tensor [B, in_chans, H, W]

        Returns:
            Output tensor [B, seq_len, d_model]
            Where seq_len = grid_w, d_model = grid_h * embed_dim
        """
        # Extract value if QuantTensor
        if hasattr(x, 'value'):
            x = x.value

        B, C, H, W = x.shape

        # Apply patch projection: [B, in_chans, H, W] -> [B, embed_dim, grid_h, grid_w]
        x = self.proj(x)

        # Extract value if QuantTensor from proj
        if hasattr(x, 'value'):
            x = x.value

        # Reshape: [B, embed_dim, grid_h, grid_w] -> [B, embed_dim * grid_h, grid_w]
        x = x.reshape(B, self.embed_dim * self.grid_h, self.grid_w)

        # Permute: [B, embed_dim * grid_h, grid_w] -> [B, grid_w, embed_dim * grid_h]
        # This gives us [B, seq_len, d_model]
        x = x.permute(0, 2, 1)

        # Quantize output
        return self.output_quant(x)


class QuantSelfAttention(nn.Module):
    """
    Quantized self-attention block using QuantLinear projections.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        seq_len: Sequence length
        bit_width: Bit width for quantization
        return_quant_tensor: Whether to return QuantTensor
        pool_sequence: How to pool the sequence ("mean", "flat", or "none")
        use_integer_softmax: If True, use polynomial integer softmax during training.
                            This helps the model learn to be robust to integer
                            softmax approximation used during inference.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        seq_len: int = 1,
        bit_width: int = 8,
        return_quant_tensor: bool = True,
        pool_sequence: str = "mean",
        use_integer_softmax: bool = False,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.seq_len = seq_len
        self.pool_sequence = pool_sequence
        self.use_integer_softmax = use_integer_softmax

        # Integer softmax module (only used if use_integer_softmax=True)
        if use_integer_softmax:
            self.integer_softmax = IntegerSoftmax()

        # Import quantization configs
        from brevitas.quant import Int8WeightPerTensorFloat

        # Path A: INT8 projections with explicit output quantization
        # Projections return FP32, then we quantize with QuantIdentity to get scales
        self.q_proj = QuantLinear(
            embed_dim, embed_dim, bias=True,
            weight_bit_width=bit_width,
            return_quant_tensor=False
        )
        self.q_quant = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )

        self.k_proj = QuantLinear(
            embed_dim, embed_dim, bias=True,
            weight_bit_width=bit_width,
            return_quant_tensor=False
        )
        self.k_quant = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )

        self.v_proj = QuantLinear(
            embed_dim, embed_dim, bias=True,
            weight_bit_width=bit_width,
            return_quant_tensor=False
        )
        self.v_quant = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )

        self.out_proj = QuantLinear(
            embed_dim, embed_dim, bias=True,
            weight_bit_width=bit_width,
            return_quant_tensor=False
        )

        self.output_quant = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=return_quant_tensor
        )

    @property
    def softmax_scale(self) -> float:
        return 1.0 / (self.head_dim ** 0.5)

    def _extract_value(self, tensor):
        return tensor.value if hasattr(tensor, "value") else tensor

    def _ensure_sequence_shape(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            if self.seq_len is None:
                raise ValueError("seq_len must be provided for 2D attention inputs")
            B, features = x.shape
            if features != self.seq_len * self.embed_dim:
                raise ValueError("Input features must equal seq_len * embed_dim")
            x = x.view(B, self.seq_len, self.embed_dim)
        elif x.dim() == 3:
            if self.seq_len is None:
                self.seq_len = x.size(1)
        else:
            raise ValueError("Self-attention expects 2D or 3D input tensors")
        return x

    def _apply_quant_linear(self, module: QuantLinear, tokens: torch.Tensor) -> torch.Tensor:
        B, N, C = tokens.shape
        flat = tokens.reshape(B * N, C)
        out = module(flat)
        out = self._extract_value(out)
        return out.reshape(B, N, -1)

    def forward(self, x):
        tensor = self._extract_value(x)
        tensor = self._ensure_sequence_shape(tensor)
        B, N, _ = tensor.shape

        # Apply projections and quantize outputs (Path A)
        q = self._apply_quant_linear(self.q_proj, tensor)
        q = self.q_quant(q)  # Quantize Q projection output
        q = self._extract_value(q)  # Extract for FP32 computation

        k = self._apply_quant_linear(self.k_proj, tensor)
        k = self.k_quant(k)  # Quantize K projection output
        k = self._extract_value(k)  # Extract for FP32 computation

        v = self._apply_quant_linear(self.v_proj, tensor)
        v = self.v_quant(v)  # Quantize V projection output
        v = self._extract_value(v)  # Extract for FP32 computation

        def split_heads(t):
            return t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.softmax_scale

        # Use integer softmax during training if enabled
        if self.use_integer_softmax:
            attn = self.integer_softmax(scores)
        else:
            attn = torch.softmax(scores, dim=-1)

        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(B, N, self.embed_dim)

        out = self._apply_quant_linear(self.out_proj, context)
        if self.pool_sequence == "mean":
            out = out.mean(dim=1)
        elif self.pool_sequence == "flat":
            out = out.reshape(B, -1)

        return self.output_quant(out)


class QuantRoPESelfAttention(QuantSelfAttention):
    """
    Quantized self-attention with Rotary Position Embeddings (RoPE).

    This is structurally identical to QuantSelfAttention, but applies RoPE to Q and K
    after projection and before attention score computation.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        seq_len: int = 1,
        bit_width: int = 8,
        return_quant_tensor: bool = True,
        pool_sequence: str = "mean",
        use_integer_softmax: bool = False,
        rope_base: float = 10000.0,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            seq_len=seq_len,
            bit_width=bit_width,
            return_quant_tensor=return_quant_tensor,
            pool_sequence=pool_sequence,
            use_integer_softmax=use_integer_softmax,
        )
        self.use_rope = True
        self.rope_base = float(rope_base)
        inv_freq = 1.0 / (self.rope_base ** (torch.arange(0, self.head_dim, 2).float() / float(self.head_dim)))
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to a tensor in [B, H, N, D] layout.
        """
        if x.dim() != 4:
            raise ValueError(f"RoPE expects [B, H, N, D], got {tuple(x.shape)}")
        if (x.size(-1) % 2) != 0:
            raise ValueError(f"RoPE head_dim must be even, got {x.size(-1)}")

        N = x.size(-2)
        inv_freq = self.rope_inv_freq.to(device=x.device, dtype=x.dtype)
        positions = torch.arange(N, device=x.device, dtype=x.dtype)
        angles = torch.einsum("n,d->nd", positions, inv_freq)  # [N, D/2]
        cos = angles.cos()[None, None, :, :]  # [1,1,N,D/2]
        sin = angles.sin()[None, None, :, :]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd
        return out

    def forward(self, x):
        tensor = self._extract_value(x)
        tensor = self._ensure_sequence_shape(tensor)
        B, N, _ = tensor.shape

        # Apply projections and quantize outputs (Path A, like QuantSelfAttention)
        q = self._apply_quant_linear(self.q_proj, tensor)
        q = self.q_quant(q)
        q = self._extract_value(q)

        k = self._apply_quant_linear(self.k_proj, tensor)
        k = self.k_quant(k)
        k = self._extract_value(k)

        v = self._apply_quant_linear(self.v_proj, tensor)
        v = self.v_quant(v)
        v = self._extract_value(v)

        def split_heads(t):
            return t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        # RoPE on Q/K
        q = self._apply_rope(q)
        k = self._apply_rope(k)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.softmax_scale
        if self.use_integer_softmax:
            attn = self.integer_softmax(scores)
        else:
            attn = torch.softmax(scores, dim=-1)

        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(B, N, self.embed_dim)

        out = self._apply_quant_linear(self.out_proj, context)
        if self.pool_sequence == "mean":
            out = out.mean(dim=1)
        elif self.pool_sequence == "flat":
            out = out.reshape(B, -1)

        return self.output_quant(out)


class QuantCrossAttention(nn.Module):
    """
    Quantized cross-attention with learned queries.

    Q comes from a learned query embedding table, while K/V come from the input.
    This module is designed to be extracted as a single ARES layer type.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_queries: int,
        kv_len: int | None = None,
        bit_width: int = 8,
        return_quant_tensor: bool = True,
        use_integer_softmax: bool = False,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.num_queries = int(num_queries)
        self.kv_len = int(kv_len) if kv_len is not None else None
        self.use_integer_softmax = bool(use_integer_softmax)

        if use_integer_softmax:
            self.integer_softmax = IntegerSoftmax()

        # Learned queries (FP32), quantized by a dedicated QuantIdentity so the extractor can
        # capture a stable query scale and export the quantized table.
        self.query_embed = nn.Parameter(torch.randn(1, self.num_queries, self.embed_dim))
        self.query_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        self.q_proj = QuantLinear(
            self.embed_dim, self.embed_dim, bias=True, weight_bit_width=bit_width, return_quant_tensor=False
        )
        self.q_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        self.k_proj = QuantLinear(
            self.embed_dim, self.embed_dim, bias=True, weight_bit_width=bit_width, return_quant_tensor=False
        )
        self.k_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        self.v_proj = QuantLinear(
            self.embed_dim, self.embed_dim, bias=True, weight_bit_width=bit_width, return_quant_tensor=False
        )
        self.v_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        self.out_proj = QuantLinear(
            self.embed_dim, self.embed_dim, bias=True, weight_bit_width=bit_width, return_quant_tensor=False
        )
        self.output_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=return_quant_tensor)

    @property
    def softmax_scale(self) -> float:
        return 1.0 / (self.head_dim ** 0.5)

    def _extract_value(self, tensor):
        return tensor.value if hasattr(tensor, "value") else tensor

    def _apply_quant_linear(self, module: QuantLinear, tokens: torch.Tensor) -> torch.Tensor:
        B, N, C = tokens.shape
        flat = tokens.reshape(B * N, C)
        out = module(flat)
        out = self._extract_value(out)
        return out.reshape(B, N, -1)

    def forward(self, kv):
        x = self._extract_value(kv)
        if x.dim() != 3:
            raise ValueError(f"QuantCrossAttention expects [B, N, D], got {tuple(x.shape)}")
        B, N, D = x.shape
        if D != self.embed_dim:
            raise ValueError(f"embed_dim mismatch: expected {self.embed_dim}, got {D}")
        if self.kv_len is not None and N != self.kv_len:
            raise ValueError(f"kv_len mismatch: expected {self.kv_len}, got {N}")

        queries = self.query_embed.expand(B, -1, -1)
        queries = self.query_quant(queries)
        queries = self._extract_value(queries)

        q = self._apply_quant_linear(self.q_proj, queries)
        q = self.q_quant(q)
        q = self._extract_value(q)

        k = self._apply_quant_linear(self.k_proj, x)
        k = self.k_quant(k)
        k = self._extract_value(k)

        v = self._apply_quant_linear(self.v_proj, x)
        v = self.v_quant(v)
        v = self._extract_value(v)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, t.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        qh = split_heads(q)  # [B, H, Q, Dh]
        kh = split_heads(k)  # [B, H, N, Dh]
        vh = split_heads(v)  # [B, H, N, Dh]

        scores = torch.matmul(qh, kh.transpose(-2, -1)) * self.softmax_scale
        if self.use_integer_softmax:
            attn = self.integer_softmax(scores)
        else:
            attn = torch.softmax(scores, dim=-1)

        context = torch.matmul(attn, vh)  # [B, H, Q, Dh]
        context = context.transpose(1, 2).contiguous().view(B, self.num_queries, self.embed_dim)

        out = self._apply_quant_linear(self.out_proj, context)
        return self.output_quant(out)


class QuantMultiHeadAttention(QuantSelfAttention):
    """Multi-head self-attention convenience wrapper."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        seq_len: int,
        bit_width: int = 8,
        return_quant_tensor: bool = True,
        pool_sequence: str = "mean",
        use_integer_softmax: bool = False,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            seq_len=seq_len,
            bit_width=bit_width,
            return_quant_tensor=return_quant_tensor,
            pool_sequence=pool_sequence,
            use_integer_softmax=use_integer_softmax,
        )


class QuantAlternatingAttention(nn.Module):
    """
    Quantized Alternating Attention for Cerebro Transformer.

    Alternating pattern based on block_idx:
    - Even blocks: Channel attention (attend across channels for each time step)
      Reshape [B, C*T, D] -> [B*T, C, D], attend over C dimension
    - Odd blocks: Temporal attention (attend across time for each channel)
      Reshape [B, C*T, D] -> [B*C, T, D], attend over T dimension

    Uses combined QKV projection with 1/sqrt(head_dim) scaling applied to Q.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_channels: int,
        temporal_len: int,
        block_idx: int = 0,
        bit_width: int = 8,
        return_quant_tensor: bool = True,
        use_integer_softmax: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_channels = num_channels
        self.temporal_len = temporal_len
        self.block_idx = block_idx
        self.use_integer_softmax = use_integer_softmax
        self.scaling_factor = 1.0 / (self.head_dim ** 0.5)

        # Integer softmax for INT8 compatibility
        if use_integer_softmax:
            self.integer_softmax = IntegerSoftmax()

        # Combined QKV projection: [D] -> [3*D]
        self.qkv_proj = QuantLinear(
            embed_dim, 3 * embed_dim, bias=True,
            weight_bit_width=bit_width,
            return_quant_tensor=False
        )

        # Separate quantizers for Q, K, V after splitting
        # Q is scaled by 1/sqrt(head_dim) in FP32 then quantized
        self.q_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)
        self.k_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)
        self.v_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Output projection
        self.out_proj = QuantLinear(
            embed_dim, embed_dim, bias=True,
            weight_bit_width=bit_width,
            return_quant_tensor=False
        )

        self.output_quant = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=return_quant_tensor
        )

    @property
    def is_channel_attention(self) -> bool:
        """Even blocks do channel attention, odd blocks do temporal attention."""
        return self.block_idx % 2 == 0

    def _extract_value(self, tensor):
        return tensor.value if hasattr(tensor, "value") else tensor

    def forward(self, x):
        """
        Forward pass with alternating attention pattern.

        Args:
            x: Input tensor [B, seq_len, embed_dim] where seq_len = C * T

        Returns:
            Output tensor [B, seq_len, embed_dim]
        """
        x = self._extract_value(x)
        B, seq_total, D = x.shape

        assert seq_total == self.num_channels * self.temporal_len, \
            f"seq_len ({seq_total}) must equal num_channels * temporal_len ({self.num_channels * self.temporal_len})"

        # Reshape based on attention type
        # First view as [B, C, T, D]
        x_4d = x.view(B, self.num_channels, self.temporal_len, D)

        if self.is_channel_attention:
            # Channel attention: attend over C for each T
            # [B, C, T, D] -> [B, T, C, D] -> [B*T, C, D]
            x_attn = x_4d.transpose(1, 2).reshape(B * self.temporal_len, self.num_channels, D)
            local_batch = B * self.temporal_len
            local_seq = self.num_channels
        else:
            # Temporal attention: attend over T for each C
            # [B, C, T, D] -> [B*C, T, D]
            x_attn = x_4d.reshape(B * self.num_channels, self.temporal_len, D)
            local_batch = B * self.num_channels
            local_seq = self.temporal_len

        # QKV projection: [local_batch, local_seq, D] -> [local_batch, local_seq, 3*D]
        x_flat = x_attn.reshape(local_batch * local_seq, D)
        qkv = self.qkv_proj(x_flat)
        qkv = self._extract_value(qkv)
        qkv = qkv.reshape(local_batch, local_seq, 3, self.num_heads, self.head_dim)

        # Split Q, K, V and reshape to [local_batch, num_heads, local_seq, head_dim]
        q = qkv[:, :, 0].transpose(1, 2)  # [local_batch, num_heads, local_seq, head_dim]
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        # Apply 1/sqrt(head_dim) scaling to Q in FP32, then quantize
        q = q * self.scaling_factor
        q = self.q_quant(q)
        q = self._extract_value(q)

        k = self.k_quant(k)
        k = self._extract_value(k)

        v = self.v_quant(v)
        v = self._extract_value(v)

        # Attention scores: Q @ K^T
        # [local_batch, num_heads, local_seq, head_dim] @ [local_batch, num_heads, head_dim, local_seq]
        # -> [local_batch, num_heads, local_seq, local_seq]
        scores = torch.matmul(q, k.transpose(-2, -1))

        # Softmax
        if self.use_integer_softmax:
            attn = self.integer_softmax(scores)
        else:
            attn = torch.softmax(scores, dim=-1)

        # Context: attn @ V
        # [local_batch, num_heads, local_seq, local_seq] @ [local_batch, num_heads, local_seq, head_dim]
        # -> [local_batch, num_heads, local_seq, head_dim]
        context = torch.matmul(attn, v)

        # Reshape back: [local_batch, local_seq, embed_dim]
        context = context.transpose(1, 2).reshape(local_batch * local_seq, self.embed_dim)

        # Output projection
        out = self.out_proj(context)
        out = self._extract_value(out)
        out = out.reshape(local_batch, local_seq, self.embed_dim)

        # Reshape back to original [B, seq_total, D]
        if self.is_channel_attention:
            # [B*T, C, D] -> [B, T, C, D] -> [B, C, T, D] -> [B, C*T, D]
            out = out.reshape(B, self.temporal_len, self.num_channels, self.embed_dim)
            out = out.transpose(1, 2).reshape(B, seq_total, self.embed_dim)
        else:
            # [B*C, T, D] -> [B, C, T, D] -> [B, C*T, D]
            out = out.reshape(B, self.num_channels, self.temporal_len, self.embed_dim)
            out = out.reshape(B, seq_total, self.embed_dim)

        return self.output_quant(out)


# --- MAMBA-specific Quantized Layers ---

class QuantConv1dDepthwise(nn.Module):
    """
    Quantized depthwise 1D convolution for MAMBA.

    Each channel has its own filter (groups=C). Supports causal padding
    for autoregressive models.

    Args:
        in_channels: Number of input channels (= output channels for depthwise)
        kernel_size: Size of the convolution kernel (default: 4 for MAMBA)
        bias: Whether to include a bias term (default: True)
        causal: If True, use left-only padding for causal convolution
        bit_width: Bit width for quantization (default: 8)
        return_quant_tensor: Whether to return QuantTensor (default: True)
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 4,
        bias: bool = True,
        causal: bool = True,
        bit_width: int = 8,
        return_quant_tensor: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.causal = causal

        # Depthwise convolution: groups = in_channels
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            bias=bias,
            padding=0  # We handle padding manually for causal conv
        )

        # Weight quantization (will be extracted during export)
        self.weight_scale = nn.Parameter(torch.ones(1), requires_grad=False)

        # Output quantization
        self.output_quant = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=return_quant_tensor
        )

    def forward(self, x):
        """
        Forward pass with optional causal padding.

        Args:
            x: Input tensor [B, C, L] (batch, channels, length)

        Returns:
            Output tensor [B, C, L] with same shape (if causal)
        """
        # Extract value if QuantTensor
        if hasattr(x, 'value'):
            x = x.value

        # Apply causal padding (left-only)
        if self.causal:
            x = F.pad(x, (self.kernel_size - 1, 0))

        # Apply depthwise convolution
        out = self.conv(x)

        # Quantize output
        return self.output_quant(out)


class QuantSiLU(nn.Module):
    """
    Quantized SiLU (Swish) activation with LUT export support.

    SiLU(x) = x * sigmoid(x)

    For INT8 input, this uses a 256-entry lookup table for exact
    integer-only execution.

    Args:
        bit_width: Bit width for quantization (default: 8)
        return_quant_tensor: Whether to return QuantTensor (default: True)
    """

    def __init__(
        self,
        bit_width: int = 8,
        return_quant_tensor: bool = True,
    ):
        super().__init__()
        self.bit_width = bit_width

        # Output quantization
        self.output_quant = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=return_quant_tensor
        )

    def forward(self, x):
        """
        Apply SiLU activation.

        During training: standard SiLU in FP32
        During export: scale is captured for LUT generation

        Args:
            x: Input tensor (QuantTensor or Tensor)

        Returns:
            Output tensor with SiLU applied and quantized
        """
        # Extract value if QuantTensor
        if hasattr(x, 'value'):
            x = x.value

        # Apply SiLU (Swish) activation
        out = F.silu(x)

        # Quantize output
        return self.output_quant(out)


class QuantSSM(nn.Module):
    """
    Quantized State Space Model (SSM) core for MAMBA.

    Implements the selective state space recurrence:
        h[t] = dA * h[t-1] + dB' * x[t]
        y[t] = C[t] * h[t] + D * x[t]

    Where:
        - dA = exp(dt * A) (discretized state transition)
        - dB' = dt * B * s_x * phi1(dt * A) (discretized input matrix)
        - C is input-dependent (computed from x_proj)

    Args:
        d_inner: Inner dimension (number of channels M)
        d_state: State dimension (D)
        dt_rank: Rank of dt projection (default: d_inner // 16)
        bit_width: Bit width for quantization (default: 8)
        return_quant_tensor: Whether to return QuantTensor (default: True)
    """

    def __init__(
        self,
        d_inner: int,
        d_state: int = 16,
        dt_rank: int = None,
        bit_width: int = 8,
        return_quant_tensor: bool = True,
    ):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank if dt_rank is not None else max(1, d_inner // 16)

        # SSM parameters (learnable)
        # A_log: Log of A parameter (A = -exp(A_log), so A is negative)
        self.A_log = nn.Parameter(torch.randn(d_state, d_inner) * 0.1)

        # D: Skip connection coefficient
        self.D = nn.Parameter(torch.ones(d_inner))

        # x_proj: Projects x to dt, B, C
        # Output: [dt_rank, d_state, d_state] -> [dt_rank + 2*d_state]
        self.x_proj = QuantLinear(
            d_inner, self.dt_rank + 2 * d_state,
            bias=False,
            weight_bit_width=bit_width,
            return_quant_tensor=False
        )

        # dt_proj: Projects dt_input to full dt (d_inner dimensional)
        self.dt_proj = QuantLinear(
            self.dt_rank, d_inner,
            bias=True,  # Important: bias initializes dt range
            weight_bit_width=bit_width,
            return_quant_tensor=False
        )

        # Initialize dt_proj bias for appropriate dt range [0.001, 0.1]
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        # Initialize bias to produce dt values in target range
        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # Output quantization
        self.output_quant = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=return_quant_tensor
        )

    @property
    def A(self):
        """Compute A matrix from A_log (A is always negative)."""
        return -torch.exp(self.A_log)

    def forward(self, x, z=None):
        """
        Forward pass through SSM.

        Args:
            x: Input tensor [B, L, M] or [B, M] for single timestep
            z: Optional gate input [B, L, M] for SiLU gating

        Returns:
            y: Output tensor [B, L, M] or [B, M]
        """
        # Extract value if QuantTensor
        if hasattr(x, 'value'):
            x = x.value
        if z is not None and hasattr(z, 'value'):
            z = z.value

        # Handle single timestep input
        single_step = x.dim() == 2
        if single_step:
            x = x.unsqueeze(1)  # [B, 1, M]
            if z is not None:
                z = z.unsqueeze(1)

        B, L, M = x.shape
        D = self.d_state

        # Project x to get dt_input, B_ssm, C_ssm
        x_proj_out = self.x_proj(x.reshape(B * L, M))  # [B*L, dt_rank + 2*D]
        x_proj_out = x_proj_out.reshape(B, L, -1)

        # Split into dt_input, B, C
        dt_input = x_proj_out[:, :, :self.dt_rank]  # [B, L, dt_rank]
        B_ssm = x_proj_out[:, :, self.dt_rank:self.dt_rank + D]  # [B, L, D]
        C_ssm = x_proj_out[:, :, self.dt_rank + D:]  # [B, L, D]

        # Project dt_input to full dt and apply softplus
        dt = self.dt_proj(dt_input.reshape(B * L, self.dt_rank))  # [B*L, M]
        dt = F.softplus(dt).reshape(B, L, M)  # [B, L, M]

        # Compute A (negative exponential)
        A = self.A  # [D, M]

        # Initialize state
        h = torch.zeros(B, M, D, device=x.device, dtype=x.dtype)

        # Discretization and scan (sequential over time)
        y_list = []
        for t in range(L):
            dt_t = dt[:, t, :]  # [B, M]
            x_t = x[:, t, :]  # [B, M]
            B_t = B_ssm[:, t, :]  # [B, D]
            C_t = C_ssm[:, t, :]  # [B, D]

            # Discretize: dA = exp(dt * A), dB' = dt * B (simplified)
            # dA: [B, M] x [D, M] -> broadcast to [B, M, D]
            dA = torch.exp(dt_t.unsqueeze(-1) * A.T.unsqueeze(0))  # [B, M, D]

            # dB' = dt * B (simplified, without phi1 for training)
            dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)  # [B, M, D]

            # State update: h = dA * h + dB' * x
            h = dA * h + dB * x_t.unsqueeze(-1)  # [B, M, D]

            # Output: y = C * h + D * x
            y_t = torch.sum(h * C_t.unsqueeze(1), dim=-1)  # [B, M]
            y_t = y_t + self.D * x_t

            y_list.append(y_t)

        y = torch.stack(y_list, dim=1)  # [B, L, M]

        # Apply SiLU gate if z is provided
        if z is not None:
            gate = F.silu(z)
            y = y * gate

        # Remove time dimension if single step
        if single_step:
            y = y.squeeze(1)

        # Quantize output
        return self.output_quant(y)


class QuantMambaWrapper(nn.Module):
    """
    Bidirectional MAMBA wrapper for FEMBA-style models.

    Wraps two MAMBA blocks (forward and reverse) and combines their outputs.
    This enables bidirectional sequence processing.

    Architecture:
        x_in ────┬──────> mamba_fwd ──────────────┬──> scale_eq ──┐
                 │                                 │               │
                 └──> flip ──> mamba_rev ──> flip ─┘               ├──> add ──> out
                                                                   │
                                                                   └──────────┘

    The flip operations reverse the sequence dimension, allowing the reverse
    MAMBA block to process the sequence from end to start.

    Args:
        d_model: Model dimension (input/output)
        d_inner: Inner dimension (default: 2 * d_model)
        d_state: SSM state dimension (default: 16)
        conv_kernel: Conv1d kernel size (default: 4)
        bidirectional_strategy: How to combine outputs ("add" or "concat")
        bit_width: Bit width for quantization (default: 8)
        return_quant_tensor: Whether to return QuantTensor (default: True)
    """

    def __init__(
        self,
        d_model: int,
        d_inner: int = None,
        d_state: int = 16,
        conv_kernel: int = 4,
        bidirectional_strategy: str = "add",
        bit_width: int = 8,
        return_quant_tensor: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner if d_inner is not None else 2 * d_model
        self.d_state = d_state
        self.bidirectional_strategy = bidirectional_strategy

        # Forward MAMBA block
        self.mamba_fwd = QuantMambaBlock(
            d_model=d_model,
            d_inner=self.d_inner,
            d_state=d_state,
            conv_kernel=conv_kernel,
            bit_width=bit_width,
            return_quant_tensor=True
        )

        # Reverse MAMBA block (processes flipped sequence)
        self.mamba_rev = QuantMambaBlock(
            d_model=d_model,
            d_inner=self.d_inner,
            d_state=d_state,
            conv_kernel=conv_kernel,
            bit_width=bit_width,
            return_quant_tensor=True
        )

        # Scale equalizer for combining outputs (ensures same quantization scale)
        self.scale_equalizer = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )

        # Quantization layers for flip operations
        # These are needed because flip requires dequantization in Brevitas
        self.pre_flip_dequant = QuantIdentity(
            act_quant=None,  # Dequantize
            return_quant_tensor=False
        )
        self.post_flip_quant = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )

        # Output quantization for reverse path
        self.rev_pre_flip_dequant = QuantIdentity(
            act_quant=None,
            return_quant_tensor=False
        )
        self.rev_post_flip_quant = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )

        # Final output quantization
        self.output_quant = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=return_quant_tensor
        )

    def forward(self, x):
        """
        Bidirectional forward pass.

        Args:
            x: Input tensor [B, L, d_model]

        Returns:
            Output tensor [B, L, d_model] (combined forward and reverse)
        """
        # Extract value if QuantTensor
        if hasattr(x, 'value'):
            x_val = x.value
        else:
            x_val = x

        B, L, _ = x_val.shape

        # Forward path
        out_fwd = self.mamba_fwd(x)
        if hasattr(out_fwd, 'value'):
            out_fwd_val = out_fwd.value
        else:
            out_fwd_val = out_fwd

        # Reverse path: flip -> mamba_rev -> flip back
        # Dequantize for flip operation
        x_dequant = self.pre_flip_dequant(x)
        if hasattr(x_dequant, 'value'):
            x_dequant = x_dequant.value

        # Flip sequence dimension (dim=1 for [B, L, D])
        x_flipped = torch.flip(x_dequant, dims=[1])

        # Requantize after flip
        x_flipped = self.post_flip_quant(x_flipped)

        # Pass through reverse MAMBA block
        out_rev = self.mamba_rev(x_flipped)
        if hasattr(out_rev, 'value'):
            out_rev_val = out_rev.value
        else:
            out_rev_val = out_rev

        # Flip output back
        out_rev_dequant = self.rev_pre_flip_dequant(out_rev)
        if hasattr(out_rev_dequant, 'value'):
            out_rev_dequant = out_rev_dequant.value

        out_rev_flipped = torch.flip(out_rev_dequant, dims=[1])
        out_rev_flipped = self.rev_post_flip_quant(out_rev_flipped)
        if hasattr(out_rev_flipped, 'value'):
            out_rev_val = out_rev_flipped.value
        else:
            out_rev_val = out_rev_flipped

        # Combine outputs
        if self.bidirectional_strategy == "add":
            # Equalize scales before addition
            out_fwd_eq = self.scale_equalizer(out_fwd_val)
            out_rev_eq = self.scale_equalizer(out_rev_val)

            if hasattr(out_fwd_eq, 'value'):
                out_fwd_eq = out_fwd_eq.value
            if hasattr(out_rev_eq, 'value'):
                out_rev_eq = out_rev_eq.value

            combined = out_fwd_eq + out_rev_eq

        elif self.bidirectional_strategy == "concat":
            # Concatenate along feature dimension
            if hasattr(out_fwd_val, 'value'):
                out_fwd_val = out_fwd_val.value
            if hasattr(out_rev_val, 'value'):
                out_rev_val = out_rev_val.value
            combined = torch.cat([out_fwd_val, out_rev_val], dim=-1)

        else:
            raise ValueError(f"Unknown strategy: {self.bidirectional_strategy}")

        # Final quantization
        return self.output_quant(combined)


class QuantMambaBlock(nn.Module):
    """
    Complete quantized MAMBA block.

    Architecture:
        x_in -> in_proj -> [x, z] split
                            |
                            v
                      conv1d -> SiLU -> x_proj -> dt_proj -> SSM
                            |                                 |
                            +----------> SiLU gate <----------+
                                              |
                                              v
                                         out_proj -> x_out

    Args:
        d_model: Model dimension (input/output)
        d_inner: Inner dimension (default: 2 * d_model)
        d_state: SSM state dimension (default: 16)
        conv_kernel: Conv1d kernel size (default: 4)
        bit_width: Bit width for quantization (default: 8)
        return_quant_tensor: Whether to return QuantTensor (default: True)
    """

    def __init__(
        self,
        d_model: int,
        d_inner: int = None,
        d_state: int = 16,
        conv_kernel: int = 4,
        bit_width: int = 8,
        return_quant_tensor: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner if d_inner is not None else 2 * d_model
        self.d_state = d_state
        self.kernel_size = conv_kernel

        # Input projection: d_model -> 2 * d_inner (split into x and z)
        self.in_proj = QuantLinear(
            d_model, 2 * self.d_inner,
            bias=False,
            weight_bit_width=bit_width,
            return_quant_tensor=False
        )
        self.in_proj_quant = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )

        # Conv1d on x branch
        self.conv1d = QuantConv1dDepthwise(
            in_channels=self.d_inner,
            kernel_size=conv_kernel,
            causal=True,
            bit_width=bit_width,
            return_quant_tensor=True
        )

        # SiLU after conv1d
        self.silu = QuantSiLU(
            bit_width=bit_width,
            return_quant_tensor=True
        )

        # SSM core
        self.ssm = QuantSSM(
            d_inner=self.d_inner,
            d_state=d_state,
            bit_width=bit_width,
            return_quant_tensor=True
        )

        # Output projection: d_inner -> d_model
        self.out_proj = QuantLinear(
            self.d_inner, d_model,
            bias=False,
            weight_bit_width=bit_width,
            return_quant_tensor=False
        )
        self.output_quant = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=return_quant_tensor
        )

    def forward(self, x):
        """
        Forward pass through MAMBA block.

        Args:
            x: Input tensor [B, L, d_model]

        Returns:
            Output tensor [B, L, d_model]
        """
        # Extract value if QuantTensor
        if hasattr(x, 'value'):
            x = x.value

        B, L, _ = x.shape

        # Input projection and split
        xz = self.in_proj(x.reshape(B * L, self.d_model))
        xz = self.in_proj_quant(xz.reshape(B, L, -1))
        if hasattr(xz, 'value'):
            xz = xz.value

        x_branch, z_branch = xz.split([self.d_inner, self.d_inner], dim=-1)

        # x branch: conv1d -> SiLU -> SSM
        # Transpose for conv1d: [B, L, M] -> [B, M, L]
        x_branch = x_branch.transpose(1, 2)
        x_branch = self.conv1d(x_branch)
        if hasattr(x_branch, 'value'):
            x_branch = x_branch.value
        x_branch = x_branch.transpose(1, 2)  # Back to [B, L, M]

        x_branch = self.silu(x_branch)
        if hasattr(x_branch, 'value'):
            x_branch = x_branch.value

        # SSM with z gate
        y = self.ssm(x_branch, z_branch)
        if hasattr(y, 'value'):
            y = y.value

        # Output projection
        out = self.out_proj(y.reshape(B * L, self.d_inner))
        out = self.output_quant(out.reshape(B, L, self.d_model))

        return out




# --- LUNA Full Architecture Modules ---

class QuantPatchEmbedCNN(nn.Module):
    """
    Quantized CNN-based patch embedding for LUNA.

    Matches the original PatchEmbedNetwork: 3x Conv2D + GroupNorm + GELU.

    NOTE: This module expects 4D input [B, 1, num_tokens, patch_size] and outputs
    4D [B, out_channels, num_tokens, reduced_dim]. The caller must handle
    reshapes before/after for ARES code generation compatibility.

    Architecture:
        Input: (B, 1, 704, 40)
        -> Conv1 (1->16, k=(1,19), s=(1,10), p=(0,9)) + GN + GELU -> (B, 16, 704, 4)
        -> Conv2 (16->16, k=(1,3), s=(1,1), p=(0,1)) + GN + GELU -> (B, 16, 704, 4)
        -> Conv3 (16->16, k=(1,3), s=(1,1), p=(0,1)) + GN + GELU -> (B, 16, 704, 4)
        Output: (B, 16, 704, 4)

    Args:
        embed_dim: Embedding dimension (out_channels = embed_dim // 4)
        patch_size: Size of each patch in the time dimension
        bit_width: Bit width for quantization
        return_quant_tensor: Whether to return QuantTensor
    """

    def __init__(
        self,
        embed_dim: int = 64,
        patch_size: int = 40,
        bit_width: int = 8,
        return_quant_tensor: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # CNN parameters (matching original LUNA)
        self.out_channels = embed_dim // 4  # 16 for embed_dim=64
        self.groups = 4
        self.kernel_size = patch_size // 2  # 20 for patch_size=40

        # Conv1: Large kernel for initial feature extraction
        self.conv1 = QuantConv2d(
            in_channels=1,
            out_channels=self.out_channels,
            kernel_size=(1, self.kernel_size - 1),  # (1, 19)
            stride=(1, self.kernel_size // 2),       # (1, 10)
            padding=(0, self.kernel_size // 2 - 1),  # (0, 9)
            bias=False,
            weight_bit_width=bit_width,
            return_quant_tensor=False,
        )
        self.gn1 = nn.GroupNorm(self.groups, self.out_channels)
        self.gelu1 = nn.GELU()
        self.post_gelu1_quant = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True,
        )

        # Conv2: Refine features
        self.conv2 = QuantConv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1),
            bias=False,
            weight_bit_width=bit_width,
            return_quant_tensor=False,
        )
        self.gn2 = nn.GroupNorm(self.groups, self.out_channels)
        self.gelu2 = nn.GELU()
        self.post_gelu2_quant = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True,
        )

        # Conv3: Final refinement
        self.conv3 = QuantConv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1),
            bias=False,
            weight_bit_width=bit_width,
            return_quant_tensor=False,
        )
        self.gn3 = nn.GroupNorm(self.groups, self.out_channels)
        self.gelu3 = nn.GELU()
        self.output_quant = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=return_quant_tensor,
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (B, 1, num_tokens, patch_size) e.g., (1, 1, 704, 40)

        Returns:
            Output tensor (B, out_channels, num_tokens, reduced_dim) e.g., (1, 16, 704, 4)
        """
        if hasattr(x, 'value'):
            x = x.value

        # Conv1 + GN + GELU
        x = self.conv1(x)
        if hasattr(x, 'value'):
            x = x.value
        x = self.gn1(x)
        x = self.gelu1(x)
        x = self.post_gelu1_quant(x)
        if hasattr(x, 'value'):
            x = x.value

        # Conv2 + GN + GELU
        x = self.conv2(x)
        if hasattr(x, 'value'):
            x = x.value
        x = self.gn2(x)
        x = self.gelu2(x)
        x = self.post_gelu2_quant(x)
        if hasattr(x, 'value'):
            x = x.value

        # Conv3 + GN + GELU
        x = self.conv3(x)
        if hasattr(x, 'value'):
            x = x.value
        x = self.gn3(x)
        x = self.gelu3(x)

        return self.output_quant(x)


class QuantCrossAttentionWithSelfRefine(nn.Module):
    """
    Quantized cross-attention block with 3-layer self-attention refinement.

    Matches the original LUNA CrossAttentionBlock:
    1. Cross-attention: Q=learned queries, K/V=input tokens
    2. FFN with residual
    3. 3-layer self-attention encoder on the query outputs

    Args:
        num_queries: Number of learned queries
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        ff_dim: Feed-forward dimension (default: 4*embed_dim)
        bit_width: Bit width for quantization
        return_quant_tensor: Whether to return QuantTensor
        use_integer_softmax: Whether to use integer softmax
    """

    def __init__(
        self,
        num_queries: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int = None,
        kv_len: int = None,
        bit_width: int = 8,
        return_quant_tensor: bool = True,
        use_integer_softmax: bool = False,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_queries = int(num_queries)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.ff_dim = int(ff_dim) if ff_dim is not None else 4 * self.embed_dim
        self.kv_len = int(kv_len) if kv_len is not None else None
        self.use_integer_softmax = bool(use_integer_softmax)

        if use_integer_softmax:
            self.integer_softmax = IntegerSoftmax()
            self.self_attn_softmax = IntegerSoftmax()

        # Learned queries
        self.query_embed = nn.Parameter(torch.randn(1, self.num_queries, self.embed_dim))
        self.query_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Layer norms for cross-attention
        self.queries_norm = nn.LayerNorm(self.embed_dim)
        self.keys_norm = nn.LayerNorm(self.embed_dim)
        self.values_norm = nn.LayerNorm(self.embed_dim)
        self.post_qnorm_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)
        self.post_knorm_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)
        self.post_vnorm_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Cross-attention projections
        self.q_proj = QuantLinear(
            self.embed_dim, self.embed_dim, bias=True, weight_bit_width=bit_width, return_quant_tensor=False
        )
        self.q_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        self.k_proj = QuantLinear(
            self.embed_dim, self.embed_dim, bias=True, weight_bit_width=bit_width, return_quant_tensor=False
        )
        self.k_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        self.v_proj = QuantLinear(
            self.embed_dim, self.embed_dim, bias=True, weight_bit_width=bit_width, return_quant_tensor=False
        )
        self.v_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        self.cross_out_proj = QuantLinear(
            self.embed_dim, self.embed_dim, bias=True, weight_bit_width=bit_width, return_quant_tensor=False
        )
        self.cross_out_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # FFN after cross-attention
        self.ffn_fc1 = QuantLinear(
            self.embed_dim, self.ff_dim, bias=True, weight_bit_width=bit_width, return_quant_tensor=False
        )
        self.ffn_gelu = nn.GELU()
        self.post_ffn_gelu_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)
        self.ffn_fc2 = QuantLinear(
            self.ff_dim, self.embed_dim, bias=True, weight_bit_width=bit_width, return_quant_tensor=False
        )
        self.ffn_norm = nn.LayerNorm(self.embed_dim)
        self.post_ffn_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Add layer for residual
        self.ffn_add = QuantAdd(bit_width=bit_width, return_quant_tensor=True)

        # 3-layer self-attention refinement on queries
        self.self_attn_blocks = nn.ModuleList()
        for _ in range(3):
            block = nn.ModuleDict({
                'norm1': nn.LayerNorm(self.embed_dim),
                'post_norm1_quant': QuantIdentity(bit_width=bit_width, return_quant_tensor=True),
                'q_proj': QuantLinear(self.embed_dim, self.embed_dim, bias=True, weight_bit_width=bit_width, return_quant_tensor=False),
                'q_quant': QuantIdentity(bit_width=bit_width, return_quant_tensor=True),
                'k_proj': QuantLinear(self.embed_dim, self.embed_dim, bias=True, weight_bit_width=bit_width, return_quant_tensor=False),
                'k_quant': QuantIdentity(bit_width=bit_width, return_quant_tensor=True),
                'v_proj': QuantLinear(self.embed_dim, self.embed_dim, bias=True, weight_bit_width=bit_width, return_quant_tensor=False),
                'v_quant': QuantIdentity(bit_width=bit_width, return_quant_tensor=True),
                'out_proj': QuantLinear(self.embed_dim, self.embed_dim, bias=True, weight_bit_width=bit_width, return_quant_tensor=False),
                'out_quant': QuantIdentity(bit_width=bit_width, return_quant_tensor=True),
                'add1': QuantAdd(bit_width=bit_width, return_quant_tensor=True),
                'norm2': nn.LayerNorm(self.embed_dim),
                'post_norm2_quant': QuantIdentity(bit_width=bit_width, return_quant_tensor=True),
                'mlp_fc1': QuantLinear(self.embed_dim, self.ff_dim, bias=True, weight_bit_width=bit_width, return_quant_tensor=False),
                'mlp_gelu': nn.GELU(),
                'post_mlp_gelu_quant': QuantIdentity(bit_width=bit_width, return_quant_tensor=True),
                'mlp_fc2': QuantLinear(self.ff_dim, self.embed_dim, bias=True, weight_bit_width=bit_width, return_quant_tensor=False),
                'add2': QuantAdd(bit_width=bit_width, return_quant_tensor=True),
            })
            self.self_attn_blocks.append(block)

        self.output_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=return_quant_tensor)

    @property
    def softmax_scale(self) -> float:
        return 1.0 / (self.head_dim ** 0.5)

    def _extract_value(self, tensor):
        return tensor.value if hasattr(tensor, "value") else tensor

    def _apply_quant_linear(self, module: QuantLinear, tokens: torch.Tensor) -> torch.Tensor:
        B, N, C = tokens.shape
        flat = tokens.reshape(B * N, C)
        out = module(flat)
        out = self._extract_value(out)
        return out.reshape(B, N, -1)

    def forward(self, kv):
        """
        Forward pass.

        Args:
            kv: Key/value input tensor (B, N, D) e.g., (B, 22, 64)

        Returns:
            Output tensor (B, num_queries, D) e.g., (B, 4, 64)
        """
        x = self._extract_value(kv)
        if x.dim() != 3:
            raise ValueError(f"Expected [B, N, D], got {tuple(x.shape)}")
        B, N, D = x.shape

        # Expand learned queries
        queries = self.query_embed.expand(B, -1, -1)
        queries = self.query_quant(queries)
        queries = self._extract_value(queries)

        # Layer norms
        queries_normed = self.queries_norm(queries)
        queries_normed = self.post_qnorm_quant(queries_normed)
        queries_normed = self._extract_value(queries_normed)

        keys = self.keys_norm(x)
        keys = self.post_knorm_quant(keys)
        keys = self._extract_value(keys)

        values = self.values_norm(x)
        values = self.post_vnorm_quant(values)
        values = self._extract_value(values)

        # Cross-attention
        q = self._apply_quant_linear(self.q_proj, queries_normed)
        q = self.q_quant(q)
        q = self._extract_value(q)

        k = self._apply_quant_linear(self.k_proj, keys)
        k = self.k_quant(k)
        k = self._extract_value(k)

        v = self._apply_quant_linear(self.v_proj, values)
        v = self.v_quant(v)
        v = self._extract_value(v)

        def split_heads(t: torch.Tensor, seq_len: int) -> torch.Tensor:
            return t.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        qh = split_heads(q, self.num_queries)  # [B, H, Q, Dh]
        kh = split_heads(k, N)  # [B, H, N, Dh]
        vh = split_heads(v, N)  # [B, H, N, Dh]

        scores = torch.matmul(qh, kh.transpose(-2, -1)) * self.softmax_scale
        if self.use_integer_softmax:
            attn = self.integer_softmax(scores)
        else:
            attn = torch.softmax(scores, dim=-1)

        context = torch.matmul(attn, vh)  # [B, H, Q, Dh]
        context = context.transpose(1, 2).contiguous().view(B, self.num_queries, self.embed_dim)

        out = self._apply_quant_linear(self.cross_out_proj, context)
        out = self.cross_out_quant(out)
        out = self._extract_value(out)

        # FFN with residual
        residual = out
        ffn_out = self._apply_quant_linear(self.ffn_fc1, out)
        ffn_out = self.ffn_gelu(ffn_out)
        ffn_out = self.post_ffn_gelu_quant(ffn_out)
        ffn_out = self._extract_value(ffn_out)
        ffn_out = self._apply_quant_linear(self.ffn_fc2, ffn_out)
        out = self.ffn_add(ffn_out, residual)
        out = self._extract_value(out)

        # 3-layer self-attention refinement
        for block in self.self_attn_blocks:
            residual = out

            # LayerNorm + self-attention
            x_normed = block['norm1'](out)
            x_normed = block['post_norm1_quant'](x_normed)
            x_normed = self._extract_value(x_normed)

            # Self-attention projections
            sq = self._apply_quant_linear(block['q_proj'], x_normed)
            sq = block['q_quant'](sq)
            sq = self._extract_value(sq)

            sk = self._apply_quant_linear(block['k_proj'], x_normed)
            sk = block['k_quant'](sk)
            sk = self._extract_value(sk)

            sv = self._apply_quant_linear(block['v_proj'], x_normed)
            sv = block['v_quant'](sv)
            sv = self._extract_value(sv)

            sqh = split_heads(sq, self.num_queries)
            skh = split_heads(sk, self.num_queries)
            svh = split_heads(sv, self.num_queries)

            s_scores = torch.matmul(sqh, skh.transpose(-2, -1)) * self.softmax_scale
            if self.use_integer_softmax:
                s_attn = self.self_attn_softmax(s_scores)
            else:
                s_attn = torch.softmax(s_scores, dim=-1)

            s_context = torch.matmul(s_attn, svh)
            s_context = s_context.transpose(1, 2).contiguous().view(B, self.num_queries, self.embed_dim)

            s_out = self._apply_quant_linear(block['out_proj'], s_context)
            s_out = block['out_quant'](s_out)
            out = block['add1'](s_out, residual)
            out = self._extract_value(out)

            # MLP
            residual = out
            x_normed = block['norm2'](out)
            x_normed = block['post_norm2_quant'](x_normed)
            x_normed = self._extract_value(x_normed)
            mlp_out = self._apply_quant_linear(block['mlp_fc1'], x_normed)
            mlp_out = block['mlp_gelu'](mlp_out)
            mlp_out = block['post_mlp_gelu_quant'](mlp_out)
            mlp_out = self._extract_value(mlp_out)
            mlp_out = self._apply_quant_linear(block['mlp_fc2'], mlp_out)
            out = block['add2'](mlp_out, residual)
            out = self._extract_value(out)

        return self.output_quant(out)


class QuantClassificationHeadWithMLP(nn.Module):
    """
    Quantized classification head with learned aggregation query + MLP.

    Matches the original LUNA ClassificationHeadWithQueries:
    1. Learned aggregation query (cross-attention pooling)
    2. MLP: hidden_dim -> 4*hidden_dim -> num_classes

    Args:
        embed_dim: Embedding dimension per query
        num_queries: Number of queries (hidden_dim = embed_dim * num_queries)
        num_heads: Number of attention heads
        num_classes: Number of output classes
        bit_width: Bit width for quantization
        return_quant_tensor: Whether to return QuantTensor
        use_integer_softmax: Whether to use integer softmax
    """

    def __init__(
        self,
        embed_dim: int,
        num_queries: int,
        num_heads: int,
        num_classes: int,
        bit_width: int = 8,
        return_quant_tensor: bool = False,
        use_integer_softmax: bool = False,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_queries = int(num_queries)
        self.hidden_dim = self.embed_dim * self.num_queries  # 256 for embed=64, queries=4
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.num_classes = int(num_classes)
        self.use_integer_softmax = bool(use_integer_softmax)

        if use_integer_softmax:
            self.integer_softmax = IntegerSoftmax()

        # Learned aggregation query
        self.learned_agg = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        self.agg_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Cross-attention for pooling
        self.q_proj = QuantLinear(
            self.hidden_dim, self.hidden_dim, bias=True, weight_bit_width=bit_width, return_quant_tensor=False
        )
        self.q_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        self.k_proj = QuantLinear(
            self.hidden_dim, self.hidden_dim, bias=True, weight_bit_width=bit_width, return_quant_tensor=False
        )
        self.k_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        self.v_proj = QuantLinear(
            self.hidden_dim, self.hidden_dim, bias=True, weight_bit_width=bit_width, return_quant_tensor=False
        )
        self.v_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        self.attn_out_proj = QuantLinear(
            self.hidden_dim, self.hidden_dim, bias=True, weight_bit_width=bit_width, return_quant_tensor=False
        )
        self.attn_out_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # MLP classifier
        self.mlp_fc1 = QuantLinear(
            self.hidden_dim, 4 * self.hidden_dim, bias=True, weight_bit_width=bit_width, return_quant_tensor=False
        )
        self.mlp_gelu = nn.GELU()
        self.post_mlp_gelu_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)
        self.mlp_fc2 = QuantLinear(
            4 * self.hidden_dim, self.num_classes, bias=True, weight_bit_width=bit_width, return_quant_tensor=return_quant_tensor
        )

    @property
    def softmax_scale(self) -> float:
        return 1.0 / (self.head_dim ** 0.5)

    def _extract_value(self, tensor):
        return tensor.value if hasattr(tensor, "value") else tensor

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (B, seq_len, hidden_dim) e.g., (1, 32, 256)

        Returns:
            Output tensor (B, num_classes) e.g., (1, 2)
        """
        x = self._extract_value(x)
        B, N, D = x.shape

        # Expand learned aggregation query
        query = self.learned_agg.expand(B, -1, -1)  # (B, 1, hidden_dim)
        query = self.agg_quant(query)
        query = self._extract_value(query)

        # Project Q, K, V
        q = self.q_proj(query.view(B, self.hidden_dim))
        q = self.q_quant(q.view(B, 1, self.hidden_dim))
        q = self._extract_value(q)

        k = self.k_proj(x.reshape(B * N, self.hidden_dim))
        k = self.k_quant(k.view(B, N, self.hidden_dim))
        k = self._extract_value(k)

        v = self.v_proj(x.reshape(B * N, self.hidden_dim))
        v = self.v_quant(v.view(B, N, self.hidden_dim))
        v = self._extract_value(v)

        # Split heads
        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, Dh)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, Dh)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, Dh)

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.softmax_scale
        if self.use_integer_softmax:
            attn = self.integer_softmax(scores)
        else:
            attn = torch.softmax(scores, dim=-1)

        context = torch.matmul(attn, v)  # (B, H, 1, Dh)
        context = context.transpose(1, 2).contiguous().view(B, self.hidden_dim)  # (B, hidden_dim)

        # Output projection
        out = self.attn_out_proj(context)
        out = self.attn_out_quant(out)
        out = self._extract_value(out)

        # MLP classifier
        out = self.mlp_fc1(out)
        out = self.mlp_gelu(out)
        out = self.post_mlp_gelu_quant(out)
        out = self._extract_value(out)
        out = self.mlp_fc2(out)

        return out


def precompute_nerf_channel_embeddings(
    channel_locations: torch.Tensor,
    embed_dim: int,
    hidden_dim: int = None,
) -> torch.Tensor:
    """
    Pre-compute NeRF positional encoding for channel locations.

    This function computes the NeRF encoding offline and returns a tensor
    that can be stored as a weight (no sin/cos needed at runtime on GAP9).

    Args:
        channel_locations: (C, 3) tensor of 3D electrode coordinates
        embed_dim: Output embedding dimension
        hidden_dim: Hidden dimension for MLP projection (default: 2*embed_dim)

    Returns:
        (C, embed_dim) tensor of pre-computed channel embeddings
    """
    C, dim = channel_locations.shape
    if dim != 3:
        raise ValueError(f"Expected 3D coordinates, got {dim}D")

    hidden_dim = hidden_dim if hidden_dim is not None else 2 * embed_dim

    # Normalize coordinates to [0, 1]
    coords = channel_locations.clone()
    coord_min = coords.min(dim=0, keepdim=True)[0]
    coord_max = coords.max(dim=0, keepdim=True)[0]
    coords = (coords - coord_min) / (coord_max - coord_min + 1e-8)

    # NeRF encoding: sin/cos at multiple frequencies
    # Output size: 2 * dim * num_freqs (need to pad to embed_dim)
    num_freqs = embed_dim // (2 * dim)
    leftover = embed_dim - num_freqs * 2 * dim

    freq_bands = 2.0 ** torch.arange(num_freqs, dtype=coords.dtype, device=coords.device)
    scaled_coords = coords.unsqueeze(-1) * freq_bands.view(1, 1, -1)  # (C, dim, num_freqs)

    sin_enc = torch.sin(scaled_coords)  # (C, dim, num_freqs)
    cos_enc = torch.cos(scaled_coords)  # (C, dim, num_freqs)

    # Interleave sin/cos and flatten
    encoded = torch.stack([sin_enc, cos_enc], dim=-1)  # (C, dim, num_freqs, 2)
    encoded = encoded.permute(0, 2, 1, 3).reshape(C, num_freqs * dim * 2)  # (C, num_freqs*dim*2)

    # Pad if needed
    if leftover > 0:
        pad = torch.zeros(C, leftover, dtype=coords.dtype, device=coords.device)
        encoded = torch.cat([encoded, pad], dim=-1)

    # MLP projection: embed_dim -> hidden_dim -> embed_dim
    # This is done offline, so we use random initialized weights
    # In practice, you'd load trained weights from the original LUNA model
    with torch.no_grad():
        fc1 = nn.Linear(embed_dim, hidden_dim)
        fc2 = nn.Linear(hidden_dim, embed_dim)
        ln = nn.LayerNorm(hidden_dim)

        x = fc1(encoded)
        x = ln(x)
        x = F.gelu(x)
        x = fc2(x)

    return x


# Default 22-channel BCI electrode locations (standard 10-20 system approximation)
# These are normalized 3D coordinates for typical EEG electrode placements
DEFAULT_CHANNEL_LOCATIONS_22 = torch.tensor([
    # Frontal
    [-0.3, 0.9, 0.3],   # Fp1
    [0.3, 0.9, 0.3],    # Fp2
    [-0.5, 0.7, 0.4],   # F7
    [-0.2, 0.7, 0.5],   # F3
    [0.0, 0.7, 0.6],    # Fz
    [0.2, 0.7, 0.5],    # F4
    [0.5, 0.7, 0.4],    # F8
    # Central
    [-0.7, 0.3, 0.3],   # T7
    [-0.3, 0.3, 0.6],   # C3
    [0.0, 0.3, 0.7],    # Cz
    [0.3, 0.3, 0.6],    # C4
    [0.7, 0.3, 0.3],    # T8
    # Parietal
    [-0.5, -0.2, 0.5],  # P7
    [-0.2, -0.2, 0.6],  # P3
    [0.0, -0.2, 0.7],   # Pz
    [0.2, -0.2, 0.6],   # P4
    [0.5, -0.2, 0.5],   # P8
    # Occipital
    [-0.3, -0.6, 0.4],  # O1
    [0.0, -0.6, 0.5],   # Oz
    [0.3, -0.6, 0.4],   # O2
    # Extra
    [-0.6, 0.5, 0.3],   # FC5
    [0.6, 0.5, 0.3],    # FC6
], dtype=torch.float32)


if __name__ == "__main__":
    # Test IntegerSoftmax
    print("Testing IntegerSoftmax...")
    int_softmax = IntegerSoftmax()
    x = torch.randn(2, 4, 8)  # (batch, heads, seq_len)
    out_int = int_softmax(x)
    out_fp = torch.softmax(x, dim=-1)
    mae = (out_int - out_fp).abs().mean().item()
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out_int.shape}")
    print(f"  MAE vs FP32 softmax: {mae:.6f}")
    print(f"  Sum check (should be ~1.0): {out_int.sum(dim=-1).mean().item():.4f}")

    # Test gradient flow
    x_grad = torch.randn(2, 4, 8, requires_grad=True)
    out_grad = int_softmax(x_grad)
    loss = out_grad.sum()
    loss.backward()
    print(f"  Gradient flows: {x_grad.grad is not None}")
    print()

    # Test QuantAdd
    print("Testing QuantAdd...")
    add_layer = QuantAdd(bit_width=8)
    x1 = torch.randn(1, 16, 28, 28)
    x2 = torch.randn(1, 16, 28, 28)
    out = add_layer(x1, x2)
    print(f"  Input 1 shape: {x1.shape}")
    print(f"  Input 2 shape: {x2.shape}")
    print(f"  Output shape: {out.shape if not hasattr(out, 'value') else out.value.shape}")
    print(f"  Output type: {type(out)}")
    print()

    # Test QuantConcatenate
    print("Testing QuantConcatenate...")
    concat_layer = QuantConcatenate(bit_width=8)
    x1 = torch.randn(1, 8, 28, 28)
    x2 = torch.randn(1, 8, 28, 28)
    out = concat_layer([x1, x2])
    print(f"  Input 1 shape: {x1.shape}")
    print(f"  Input 2 shape: {x2.shape}")
    print(f"  Output shape: {out.shape if not hasattr(out, 'value') else out.value.shape}")
    print(f"  Output type: {type(out)}")
    print()

    # Test QuantSelfAttention with integer softmax
    print("Testing QuantSelfAttention with integer softmax...")
    attn_int = QuantSelfAttention(
        embed_dim=32, num_heads=2, seq_len=8,
        use_integer_softmax=True, pool_sequence="mean"
    )
    x = torch.randn(2, 8, 32)  # (batch, seq_len, embed_dim)
    out = attn_int(x)
    out_val = out.value if hasattr(out, 'value') else out
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out_val.shape}")
    print(f"  Uses integer softmax: {attn_int.use_integer_softmax}")
    print()

    # Test QuantConv1dDepthwise
    print("Testing QuantConv1dDepthwise...")
    conv1d = QuantConv1dDepthwise(
        in_channels=16, kernel_size=4, causal=True
    )
    x = torch.randn(2, 16, 32)  # (batch, channels, length)
    out = conv1d(x)
    out_val = out.value if hasattr(out, 'value') else out
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out_val.shape}")
    print(f"  Causal: {conv1d.causal}")
    assert out_val.shape == x.shape, "Causal conv should preserve length"
    print("  Length preserved: [OK]")
    print()

    # Test QuantSiLU
    print("Testing QuantSiLU...")
    silu = QuantSiLU()
    x = torch.randn(2, 16, 32)
    out = silu(x)
    out_val = out.value if hasattr(out, 'value') else out
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out_val.shape}")
    # Verify SiLU properties
    x_zero = torch.zeros(1, 1, 1)
    out_zero = silu(x_zero)
    out_zero_val = out_zero.value if hasattr(out_zero, 'value') else out_zero
    print(f"  SiLU(0) ≈ 0: {out_zero_val.abs().item() < 0.1}")
    print()

    # Test QuantSSM
    print("Testing QuantSSM...")
    ssm = QuantSSM(d_inner=16, d_state=4)
    x = torch.randn(2, 8, 16)  # (batch, seq_len, d_inner)
    z = torch.randn(2, 8, 16)  # Gate input
    out = ssm(x, z)
    out_val = out.value if hasattr(out, 'value') else out
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out_val.shape}")
    print(f"  d_inner: {ssm.d_inner}, d_state: {ssm.d_state}")
    print(f"  dt_rank: {ssm.dt_rank}")
    print()

    # Test QuantMambaBlock
    print("Testing QuantMambaBlock...")
    mamba = QuantMambaBlock(d_model=32, d_inner=64, d_state=8)
    x = torch.randn(2, 8, 32)  # (batch, seq_len, d_model)
    out = mamba(x)
    out_val = out.value if hasattr(out, 'value') else out
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out_val.shape}")
    print(f"  d_model: {mamba.d_model}, d_inner: {mamba.d_inner}")
    assert out_val.shape == x.shape, "MAMBA block should preserve shape"
    print("  Shape preserved: [OK]")
    print()

    # Test QuantMambaWrapper (Bidirectional)
    print("Testing QuantMambaWrapper (Bidirectional)...")
    bi_mamba = QuantMambaWrapper(
        d_model=32, d_inner=64, d_state=4,
        bidirectional_strategy="add"
    )
    x = torch.randn(2, 8, 32)  # (batch, seq_len, d_model)
    out = bi_mamba(x)
    out_val = out.value if hasattr(out, 'value') else out
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out_val.shape}")
    print(f"  d_model: {bi_mamba.d_model}, d_inner: {bi_mamba.d_inner}")
    print(f"  Strategy: {bi_mamba.bidirectional_strategy}")
    assert out_val.shape == x.shape, "Bidirectional MAMBA should preserve shape with 'add' strategy"
    print("  Shape preserved (add strategy): [OK]")

    # Verify forward and reverse are different
    with torch.no_grad():
        out_fwd = bi_mamba.mamba_fwd(x)
        out_fwd_val = out_fwd.value if hasattr(out_fwd, 'value') else out_fwd
        x_rev = torch.flip(x, dims=[1])
        out_rev = bi_mamba.mamba_rev(x_rev)
        out_rev_val = out_rev.value if hasattr(out_rev, 'value') else out_rev
        print(f"  Forward output norm: {out_fwd_val.norm().item():.4f}")
        print(f"  Reverse output norm: {out_rev_val.norm().item():.4f}")
        print(f"  Outputs different: {not torch.allclose(out_fwd_val, out_rev_val)}")
    print()

    # Test QuantPatchEmbed
    print("Testing QuantPatchEmbed...")
    # FEMBA-style config: 10x32 input, patch_size=2, stride=2, embed_dim=35
    patch_embed = QuantPatchEmbed(
        inp_size=(10, 32),
        patch_size=2,
        stride=2,
        in_chans=1,
        embed_dim=35
    )
    x = torch.randn(2, 1, 10, 32)  # (batch, channels, height, width)
    out = patch_embed(x)
    out_val = out.value if hasattr(out, 'value') else out
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out_val.shape}")
    print(f"  Expected: [B, seq_len={patch_embed.seq_len}, d_model={patch_embed.d_model}]")
    print(f"  Grid: {patch_embed.grid_h} x {patch_embed.grid_w}")
    assert out_val.shape == (2, patch_embed.seq_len, patch_embed.d_model), \
        f"Expected {(2, patch_embed.seq_len, patch_embed.d_model)}, got {out_val.shape}"
    print("  Shape correct: [OK]")
    print()

    print("[PASS] Custom layers test complete!")
