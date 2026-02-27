# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Search Space definitions for ARES auto-tuning.

Defines the configuration space to explore for each layer type during
auto-tuning. Configurations are constrained by hardware limits (L1 budget,
DMA alignment, etc.).

Usage:
    from codegen.optimization import SearchSpace

    space = SearchSpace()

    # Get candidates for a linear layer
    candidates = space.get_candidates(
        op_type='linear_int8',
        shape={'M': 400, 'N': 768, 'K': 192}
    )

    for config in candidates[:5]:
        print(config)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Iterator
import itertools


# Hardware constants
L1_BUDGET_BYTES = 100000  # ~100KB usable L1
L1_DOUBLE_BUFFER_FACTOR = 2  # Need 2x for double buffering
DMA_ALIGNMENT = 8  # 8-byte alignment for DMA
MIN_TILE_SIZE = 8  # Minimum practical tile size


@dataclass
class TuningConfig:
    """A single configuration to try during tuning."""
    tile_config: Dict[str, Any]
    kernel_config: Dict[str, Any]
    pipeline_config: Dict[str, Any]
    compile_flags: Dict[str, Any]
    description: str = ""

    def to_dict(self) -> Dict:
        return {
            'tile_config': self.tile_config,
            'kernel_config': self.kernel_config,
            'pipeline_config': self.pipeline_config,
            'compile_flags': self.compile_flags,
            'description': self.description,
        }

    def __repr__(self) -> str:
        return f"TuningConfig({self.description or self.tile_config})"


class SearchSpace:
    """Define and generate the search space for auto-tuning."""

    def __init__(self, l1_budget: int = L1_BUDGET_BYTES):
        self.l1_budget = l1_budget

    def get_candidates(self, op_type: str, shape: Dict[str, int],
                       max_candidates: int = 50) -> List[TuningConfig]:
        """
        Generate candidate configurations for a layer.

        Args:
            op_type: Operation type (e.g., 'linear_int8', 'conv2d_int8')
            shape: Layer shape parameters
            max_candidates: Maximum number of candidates to return

        Returns:
            List of TuningConfig candidates
        """
        method = getattr(self, f'_{op_type}_candidates', None)
        if method is None:
            return self._default_candidates(op_type, shape)

        candidates = list(method(shape))

        # Sort by expected quality (larger tiles generally better)
        candidates.sort(key=lambda c: -sum(c.tile_config.values()))

        return candidates[:max_candidates]

    def _linear_int8_candidates(self, shape: Dict[str, int]) -> Iterator[TuningConfig]:
        """Generate candidates for Linear INT8 layer.

        WIRING STATUS (which parameters actually affect generated code):
        ================================================================

        WIRED (affect kernel behavior):
        - tile_n: WIRED to LinearTileConfig.tile_out_features in _determine_linear_memory_tier()
                  Controls how output features are tiled across L1/L2
        - tile_k: WIRED to LinearTileConfig.tile_in_features (and k_tiling_enabled when tile_k < K)
                  Enables K-dimension tiling for large Linear layers
        - tile_m: WIRED to LinearTileConfig.tile_batch_tokens (and m_tiling_enabled when tile_m < M)
                  Enables M-dimension tiling for 3D linear layers (token/batch dimension)
        - l1_input_cache: WIRED to LINEAR_INT8_INPUT_L1_CACHE compile flag via EXTRA_CFLAGS
                          When True, input is DMA'd to L1 before compute
        - outf_unroll: WIRED to LINEAR_INT8_OUTF_UNROLL compile flag via EXTRA_CFLAGS
                       Output feature unroll factor (2 or 4) for SIMD accumulator loop

        All parameters are now fully wired.
        """
        M = shape.get('M', 1)  # batch_tokens
        N = shape.get('N', 256)  # out_features
        K = shape.get('K', 256)  # in_features

        # Tile size options
        # All parameters are now WIRED:
        # - tile_m enables M-dimension tiling (batch/token tiling)
        # - tile_n enables N-dimension tiling (output features)
        # - tile_k enables K-dimension tiling (input features) with INT32 accumulation
        tile_m_options = [m for m in [1, 4, 8, 16, 32, 64] if m <= M]
        tile_n_options = [n for n in [32, 64, 128, 256] if n <= N]
        tile_k_options = [k for k in [32, 64, 128, 256, 512] if k <= K]

        # Input caching options (WIRED to LINEAR_INT8_INPUT_L1_CACHE)
        input_cache_options = [True, False]

        # Output feature unroll options (WIRED to LINEAR_INT8_OUTF_UNROLL)
        outf_unroll_options = [2, 4]

        for tile_m, tile_n, tile_k in itertools.product(
            tile_m_options, tile_n_options, tile_k_options
        ):
            # Check L1 budget
            # Input tile: tile_m * tile_k
            # Weight tile: tile_k * tile_n
            # Output tile: tile_m * tile_n
            l1_required = (tile_m * tile_k + tile_k * tile_n + tile_m * tile_n)
            l1_required *= L1_DOUBLE_BUFFER_FACTOR

            if l1_required > self.l1_budget:
                continue

            for input_cache in input_cache_options:
                for outf_unroll in outf_unroll_options:
                    yield TuningConfig(
                        tile_config={
                            'tile_m': tile_m,
                            'tile_n': tile_n,
                            'tile_k': tile_k,
                            'l1_input_cache': input_cache,  # Wire to code generator
                        },
                        kernel_config={
                            'fast_qround': True,
                            'outf_unroll': outf_unroll,
                        },
                        pipeline_config={
                            'enable_overlap': True,
                            'prefetch_weights': True,
                        },
                        compile_flags={
                            'LINEAR_INT8_OUTF_UNROLL': outf_unroll,
                        },
                        description=f"tile_{tile_m}x{tile_n}x{tile_k}_cache{'_on' if input_cache else '_off'}_unroll{outf_unroll}"
                    )

    def _conv2d_int8_candidates(self, shape: Dict[str, int]) -> Iterator[TuningConfig]:
        """Generate candidates for Conv2D INT8 layer.

        WIRING STATUS (which parameters actually affect generated code):
        ================================================================

        WIRED (affect kernel behavior):
        - tile_h: WIRED to Conv2DTileConfig.tile_h via hint_tile_h in _determine_conv2d_memory_tier()
                  Controls output tile height for spatial tiling
        - tile_w: WIRED to Conv2DTileConfig.tile_w via hint_tile_w in _determine_conv2d_memory_tier()
                  Controls output tile width for 2D spatial tiling
        - outch_unroll: WIRED to CONV2D_IM2COL_OUTCH_UNROLL compile flag via EXTRA_CFLAGS
                        Controls output channel unroll factor in im2col kernel
        - px_unroll: WIRED to CONV1X1_PX_UNROLL compile flag via EXTRA_CFLAGS (1x1 convs only)
                     Controls pixel unroll factor (1=disabled, 2=two pixels per iteration)
        - use_gemm: WIRED to CONV2D_1X1_USE_GEMM compile flag (1x1 convs only)
                    Bypasses im2col and uses direct GEMM path for 1x1 convolutions

        All parameters are now fully wired.
        """
        in_h = shape.get('in_h', 28)
        in_w = shape.get('in_w', 28)
        in_ch = shape.get('in_channels', 64)
        out_ch = shape.get('out_channels', 64)
        kernel_h = shape.get('kernel_h', 3)
        kernel_w = shape.get('kernel_w', 3)

        # Special case for 1x1 convolution
        is_1x1 = (kernel_h == 1 and kernel_w == 1)

        # Tile size options for spatial dimensions
        tile_h_options = [h for h in [4, 7, 14, 28] if h <= in_h]
        tile_w_options = [w for w in [4, 7, 14, 28] if w <= in_w]

        # Output channel unroll options
        outch_unroll_options = [1, 2, 4, 8]

        # 1x1 specific: pixel unroll
        px_unroll_options = [1, 2, 4] if is_1x1 else [1]

        # 1x1 specific: direct GEMM bypass (avoids im2col)
        use_gemm_options = [True, False] if is_1x1 else [False]

        for tile_h, tile_w in itertools.product(tile_h_options, tile_w_options):
            # Calculate L1 requirements
            # Input tile: (tile_h + kernel_h - 1) * (tile_w + kernel_w - 1) * in_ch
            # Output tile: tile_h * tile_w * out_ch
            # Weights: kernel_h * kernel_w * in_ch * outch_tile
            input_tile = (tile_h + kernel_h - 1) * (tile_w + kernel_w - 1) * in_ch
            output_tile = tile_h * tile_w * out_ch

            base_l1 = (input_tile + output_tile) * L1_DOUBLE_BUFFER_FACTOR

            if base_l1 > self.l1_budget:
                continue

            for outch_unroll in outch_unroll_options:
                for px_unroll in px_unroll_options:
                    for use_gemm in use_gemm_options:
                        # Build compile flags
                        flags = {'CONV2D_IM2COL_OUTCH_UNROLL': outch_unroll}
                        if is_1x1:
                            # Wire px_unroll for 1x1 convolutions
                            flags['CONV1X1_PX_UNROLL'] = px_unroll
                            # Wire GEMM bypass for 1x1 convolutions
                            flags['CONV2D_1X1_USE_GEMM'] = 1 if use_gemm else 0

                        # Build description
                        desc = f"tile_{tile_h}x{tile_w}_unroll{outch_unroll}"
                        if is_1x1:
                            desc += f"_px{px_unroll}"
                            if use_gemm:
                                desc += "_gemm"

                        yield TuningConfig(
                            tile_config={
                                'tile_h': tile_h,
                                'tile_w': tile_w,
                                'l1_weight_caching': True,
                            },
                            kernel_config={
                                'outch_unroll': outch_unroll,
                                'px_unroll': px_unroll if is_1x1 else 1,
                                'use_gemm': use_gemm,
                            },
                            pipeline_config={
                                'enable_overlap': True,
                            },
                            compile_flags=flags,
                            description=desc
                        )

    def _mhsa_int8_candidates(self, shape: Dict[str, int]) -> Iterator[TuningConfig]:
        """Generate candidates for MHSA INT8 layer.

        WIRING STATUS (which parameters actually affect generated code):
        ================================================================

        WIRED (affect kernel behavior):
        - tile_q: WIRED to MHSATileConfig.tile_q via hint_tile_q in MHSA processing
                  Controls query sequence tiling for memory efficiency
        - fuse_qk_softmax_av: WIRED to MHSA_FUSE_QK_SOFTMAX_AV compile flag
                              Full fusion of QK matmul + softmax + AV matmul
        - fuse_softmax_av: WIRED to MHSA_FUSE_SOFTMAX_AV compile flag
                           Partial fusion of softmax + AV matmul only

        All parameters are now fully wired.
        """
        seq_len = shape.get('seq_len', 400)
        embed_dim = shape.get('embed_dim', 192)
        num_heads = shape.get('num_heads', 3)
        head_dim = shape.get('head_dim', embed_dim // num_heads)

        # tile_q options (must divide seq_len reasonably)
        tile_q_options = [q for q in [8, 12, 16, 24, 32, 48, 64] if q <= seq_len]

        # Fusion options
        # but is included for completeness in auto-tuning exploration
        fusion_options = [
            {'fuse_qk_softmax_av': True, 'fuse_softmax_av': False},
            {'fuse_qk_softmax_av': False, 'fuse_softmax_av': False},
            {'fuse_qk_softmax_av': False, 'fuse_softmax_av': True},  # Partial fusion (softmax+AV only)
        ]

        for tile_q in tile_q_options:
            # Check L1 budget
            # Q tile: tile_q * head_dim
            # K matrix: seq_len * head_dim
            # V matrix: seq_len * head_dim
            # Scores: tile_q * seq_len
            l1_per_head = (
                tile_q * head_dim +  # Q tile
                seq_len * head_dim +  # K
                seq_len * head_dim +  # V
                tile_q * seq_len  # scores
            )

            if l1_per_head * L1_DOUBLE_BUFFER_FACTOR > self.l1_budget:
                continue

            for fusion in fusion_options:
                # Determine fusion description
                if fusion['fuse_qk_softmax_av']:
                    fuse_desc = '_full'
                elif fusion['fuse_softmax_av']:
                    fuse_desc = '_softmax_av'
                else:
                    fuse_desc = '_none'

                yield TuningConfig(
                    tile_config={
                        'tile_q': tile_q,
                    },
                    kernel_config={
                        'qk_unroll': 4,
                        'v_transpose_in_l1': True,
                    },
                    pipeline_config={
                        'fuse_qk_softmax_av': fusion['fuse_qk_softmax_av'],
                        'fuse_softmax_av': fusion['fuse_softmax_av'],
                    },
                    compile_flags={
                        'MHSA_V_TRANSPOSE_REUSE_V_BUFFER': 1,
                        'MHSA_FUSE_QK_SOFTMAX_AV': 1 if fusion['fuse_qk_softmax_av'] else 0,
                        'MHSA_FUSE_SOFTMAX_AV': 1 if fusion['fuse_softmax_av'] else 0,
                    },
                    description=f"tile_q{tile_q}_fuse{fuse_desc}"
                )

    def _layernorm_int8_candidates(self, shape: Dict[str, int]) -> Iterator[TuningConfig]:
        """Generate candidates for LayerNorm INT8 layer."""
        num_tokens = shape.get('num_tokens', 400)
        embed_dim = shape.get('embed_dim', 192)

        # Tile over tokens
        tile_tokens_options = [t for t in [1, 4, 8, 16, 32] if t <= num_tokens]

        for tile_tokens in tile_tokens_options:
            yield TuningConfig(
                tile_config={
                    'tile_tokens': tile_tokens,
                },
                kernel_config={
                    'use_simd': True,
                },
                pipeline_config={
                    'enable_overlap': True,
                },
                compile_flags={},
                description=f"tile_tokens{tile_tokens}"
            )

    def _ssm_int8_candidates(self, shape: Dict[str, int]) -> Iterator[TuningConfig]:
        """Generate candidates for SSM INT8 layer.

        WIRING STATUS:
        - ph3_channel_tile: WIRED (maps to generate_c_code.py line 4968)
        - CONV1D_USE_SIMD: WIRED (compile flag, enables SIMD for Conv1D depthwise)

        Conv1D SIMD enables SumDotpSS optimization for 4-tap depthwise convolutions
        used in Mamba's input projection path. Default ON for kernel_size=4.
        """
        seq_len = shape.get('seq_len', 400)
        d_inner = shape.get('d_inner', 192)
        d_state = shape.get('d_state', 16)

        # SSM is mostly sequential over seq_len, but can tile over d_inner
        tile_d_options = [d for d in [32, 64, 96, 128, 192] if d <= d_inner]

        for tile_d in tile_d_options:
            yield TuningConfig(
                tile_config={
                    'ph3_channel_tile': tile_d,  # WIRED: maps to generate_c_code.py line 4968
                    'tile_d_inner': tile_d,       # Alias for documentation
                    'fully_l1_resident': tile_d * d_state * 4 < self.l1_budget // 2,
                },
                kernel_config={
                    'use_fixed_point': True,
                },
                pipeline_config={
                    'enable_overlap': True,
                },
                compile_flags={
                    # Conv1D SIMD for 4-tap depthwise (Mamba)
                    'CONV1D_USE_SIMD': 1,  # Always enable - no downside for kernel_size=4
                },
                description=f"tile_d{tile_d}"
            )

    def _maxpool_int8_candidates(self, shape: Dict[str, int]) -> Iterator[TuningConfig]:
        """Generate candidates for MaxPool INT8 layer.

        Pooling has limited tuning options - mainly tile sizes for spatial dimensions.
        Most pooling ops are already efficient; this provides a few tile size options.
        """
        H = shape.get('H', 28)
        W = shape.get('W', 28)
        C = shape.get('C', 32)

        # Try different spatial tile sizes
        tile_h_options = [h for h in [H, H // 2, 14, 7] if h > 0 and h <= H]
        tile_w_options = [w for w in [W, W // 2, 14, 7] if w > 0 and w <= W]

        # Remove duplicates and sort descending
        tile_h_options = sorted(set(tile_h_options), reverse=True)[:3]
        tile_w_options = sorted(set(tile_w_options), reverse=True)[:3]

        for tile_h in tile_h_options:
            for tile_w in tile_w_options:
                # Check L1 fits (input tile + output tile)
                pool_size = shape.get('kernel_size', 2)
                input_size = (tile_h * pool_size) * (tile_w * pool_size) * C
                output_size = tile_h * tile_w * C
                total = (input_size + output_size) * L1_DOUBLE_BUFFER_FACTOR

                if total <= self.l1_budget:
                    yield TuningConfig(
                        tile_config={'tile_h': tile_h, 'tile_w': tile_w},
                        kernel_config={},
                        pipeline_config={'enable_overlap': True},
                        compile_flags={},
                        description=f"tile_{tile_h}x{tile_w}"
                    )

    def _avgpool_int8_candidates(self, shape: Dict[str, int]) -> Iterator[TuningConfig]:
        """Generate candidates for AvgPool INT8 layer (same as maxpool)."""
        yield from self._maxpool_int8_candidates(shape)

    def _gelu_int8_candidates(self, shape: Dict[str, int]) -> Iterator[TuningConfig]:
        """Generate candidates for GELU INT8 layer.

        GELU is element-wise so tuning options are limited.
        Main options: tile size for processing and LUT vs polynomial approximation.
        """
        num_elements = shape.get('num_elements', 1024)

        # Tile size options for element-wise processing
        tile_options = [t for t in [256, 512, 1024, 2048, 4096] if t <= num_elements]
        if not tile_options:
            tile_options = [num_elements]

        # GELU implementation options
        impl_options = [
            {'use_lut': True, 'lut_bits': 8},   # LUT-based (faster, less accurate)
            {'use_lut': False},                  # Polynomial (slower, more accurate)
        ]

        for tile_size in tile_options:
            for impl in impl_options:
                yield TuningConfig(
                    tile_config={'tile_size': tile_size},
                    kernel_config=impl,
                    pipeline_config={'enable_overlap': True},
                    compile_flags={},
                    description=f"tile_{tile_size}_{'lut' if impl.get('use_lut') else 'poly'}"
                )

    def _add_int8_candidates(self, shape: Dict[str, int]) -> Iterator[TuningConfig]:
        """Generate candidates for Add INT8 layer.

        Element-wise addition with scale matching. Limited tuning options.
        """
        num_elements = shape.get('num_elements', 1024)

        # Tile size options
        tile_options = [t for t in [512, 1024, 2048, 4096] if t <= num_elements]
        if not tile_options:
            tile_options = [num_elements]

        for tile_size in tile_options:
            yield TuningConfig(
                tile_config={'tile_size': tile_size},
                kernel_config={},
                pipeline_config={'enable_overlap': True},
                compile_flags={},
                description=f"tile_{tile_size}"
            )

    def _concat_candidates(self, shape: Dict[str, int]) -> Iterator[TuningConfig]:
        """Generate candidates for Concat layer.

        Channel concatenation. Main option is whether to use DMA or memcpy.
        """
        # Concat has very limited tuning - mainly DMA strategy
        strategies = [
            {'use_dma': True, 'async': True},
            {'use_dma': True, 'async': False},
        ]

        for strategy in strategies:
            yield TuningConfig(
                tile_config={},
                kernel_config=strategy,
                pipeline_config={'enable_overlap': strategy.get('async', False)},
                compile_flags={},
                description=f"dma_{'async' if strategy.get('async') else 'sync'}"
            )

    def _embedding_candidates(self, shape: Dict[str, int]) -> Iterator[TuningConfig]:
        """Generate candidates for Embedding layer.

        Lookup table operation. Options: prefetch strategy, tile size.
        """
        vocab_size = shape.get('vocab_size', 1000)
        embed_dim = shape.get('embed_dim', 256)
        seq_len = shape.get('seq_len', 100)

        # Tile over sequence length
        tile_seq_options = [t for t in [16, 32, 64, 128] if t <= seq_len]
        if not tile_seq_options:
            tile_seq_options = [seq_len]

        # Prefetch options
        prefetch_options = [True, False]

        for tile_seq in tile_seq_options:
            for prefetch in prefetch_options:
                # Check L1 budget: tile_seq * embed_dim bytes
                l1_required = tile_seq * embed_dim * L1_DOUBLE_BUFFER_FACTOR
                if l1_required <= self.l1_budget:
                    yield TuningConfig(
                        tile_config={'tile_seq': tile_seq},
                        kernel_config={'prefetch_embeddings': prefetch},
                        pipeline_config={'enable_overlap': prefetch},
                        compile_flags={},
                        description=f"tile_{tile_seq}_{'prefetch' if prefetch else 'noprefetch'}"
                    )

    def _rfft_int8_candidates(self, shape: Dict[str, int]) -> Iterator[TuningConfig]:
        """Generate candidates for RFFT INT8 layer.

        Real FFT operation. Options: radix, twiddle factor precision.
        """
        fft_size = shape.get('fft_size', 256)

        # Radix options (must divide fft_size)
        radix_options = [r for r in [2, 4, 8] if fft_size % r == 0]

        # Twiddle precision options
        twiddle_options = [
            {'twiddle_bits': 16},  # Higher precision
            {'twiddle_bits': 8},   # Lower precision, faster
        ]

        for radix in radix_options:
            for twiddle in twiddle_options:
                yield TuningConfig(
                    tile_config={'radix': radix},
                    kernel_config=twiddle,
                    pipeline_config={'enable_overlap': True},
                    compile_flags={},
                    description=f"radix{radix}_tw{twiddle['twiddle_bits']}"
                )

    def _default_candidates(self, op_type: str,
                           shape: Dict[str, int]) -> List[TuningConfig]:
        """Default candidates for unknown op types."""
        return [
            TuningConfig(
                tile_config={},
                kernel_config={},
                pipeline_config={'enable_overlap': True},
                compile_flags={},
                description=f"default_{op_type}"
            )
        ]

    def estimate_l1_usage(self, op_type: str, config: TuningConfig,
                          shape: Dict[str, int]) -> int:
        """
        Estimate L1 memory usage for a configuration.

        Args:
            op_type: Operation type
            config: Configuration to estimate
            shape: Layer shape

        Returns:
            Estimated L1 usage in bytes
        """
        tc = config.tile_config

        if op_type == 'linear_int8':
            tile_m = tc.get('tile_m', 1)
            tile_n = tc.get('tile_n', 64)
            tile_k = tc.get('tile_k', 64)
            return (tile_m * tile_k + tile_k * tile_n + tile_m * tile_n) * 2

        elif op_type == 'conv2d_int8':
            tile_h = tc.get('tile_h', 7)
            tile_w = tc.get('tile_w', 7)
            in_ch = shape.get('in_channels', 64)
            out_ch = shape.get('out_channels', 64)
            kernel_h = shape.get('kernel_h', 3)
            kernel_w = shape.get('kernel_w', 3)
            input_tile = (tile_h + kernel_h - 1) * (tile_w + kernel_w - 1) * in_ch
            output_tile = tile_h * tile_w * out_ch
            return (input_tile + output_tile) * 2

        elif op_type == 'mhsa_int8':
            tile_q = tc.get('tile_q', 16)
            seq_len = shape.get('seq_len', 400)
            head_dim = shape.get('head_dim', 64)
            return (tile_q * head_dim + 2 * seq_len * head_dim + tile_q * seq_len) * 2

        return 0


def get_search_space(op_type: str, shape: Dict[str, int],
                     max_candidates: int = 50) -> List[TuningConfig]:
    """
    Convenience function to get search space candidates.

    Args:
        op_type: Operation type
        shape: Layer shape parameters
        max_candidates: Maximum number of candidates

    Returns:
        List of TuningConfig candidates
    """
    space = SearchSpace()
    return space.get_candidates(op_type, shape, max_candidates)
