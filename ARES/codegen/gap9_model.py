# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
GAP9 hardware model and tiling heuristics.

This module provides:
- Conservative memory budgets for GAP9 (L1/L2/L3) used by codegen planning
- Tiling calculators (Conv2D/Linear/etc.) that decide whether an op runs:
  - fully in L2,
  - with inner-loop L1 tiling, and/or
  - with outer-loop L3→L2 streaming ("slabs") for very large weights/activations.

The goal is to keep decisions predictable and safe: it's better to fall back to
an L2 execution path than to generate an L1-tiled plan that will fail at runtime.
"""

import math

try:
    from .tiling.strategies import compute_tile_plan_with_strategy
except ImportError:
    from tiling.strategies import compute_tile_plan_with_strategy

class GAP9HardwareModel:
    """
    GAP9 memory budget constants used by code generation.

    Notes on L1 sizing:
        GAP9's TCDM (L1) is 128KB total, but not all of that is usable for
        tiling buffers. Stack, runtime bookkeeping, and scratch allocations
        (especially on the orchestrator core) reduce the safe buffer budget.

        We use a conservative 104KB estimate so tile calculators don't produce
        plans that barely fit on paper but fail at runtime.
    """

    # Hardware totals as seen by allocators/linker scripts.
    L1_TOTAL_BYTES = (128 * 1024) - 4
    L2_TOTAL_BYTES = int(1.5 * 1024 * 1024)

    # Memory Constraints
    L1_SIZE_BYTES = 104 * 1024      # Conservative usable L1 budget for tiling buffers
    L2_SIZE_BYTES = 1500 * 1024     # ~1.5 MB L2 (conservative)

    # Reserved Memory
    L1_RESERVED = 4 * 1024          # Stack/Shared vars
    L2_RESERVED = 100 * 1024        # Code/Stack/Heap

    # L2 Tiling Budget: Conservative estimate for tiling decisions
    # This accounts for runtime overhead beyond L2_RESERVED:
    #   - L2 Arena for activation buffers: ~200-300 KB
    #   - Golden comparison buffers: ~50 KB
    #   - Weight/parameter staging: ~50 KB
    #   - Runtime stack and heap margin: ~100 KB
    # Total overhead: ~400-500 KB, so we use 1100 KB as safe tiling budget
    # (Hardware L2 = 1500 KB, minus 400 KB overhead = ~1100 KB for tiling decisions)
    L2_TILING_BUDGET = 1100 * 1024  # Used consistently across Conv2D, Linear, MHSA, Mamba

    # L3 Staging Threshold: Weights larger than this are pre-loaded from L3
    # Used for layers that don't have dynamic L3 tiling (e.g., MHSA projections)
    # 32KB chosen as balance between L2 pressure and L3 latency overhead
    L3_STAGE_THRESHOLD = 32 * 1024  # 32 KB

    # L3 fallback sizing policy for oversized activation buffers.
    L2_ACTIVATION_RESERVED = 550 * 1024
    L3_FALLBACK_SINGLE_BUFFER_THRESHOLD = 700 * 1024

    @classmethod
    def get_l1_total_bytes(cls):
        return cls.L1_TOTAL_BYTES

    @classmethod
    def get_l2_total_bytes(cls):
        return cls.L2_TOTAL_BYTES

    @classmethod
    def get_l1_budget(cls):
        return cls.L1_SIZE_BYTES - cls.L1_RESERVED

    @classmethod
    def get_l2_budget(cls):
        return cls.L2_SIZE_BYTES - cls.L2_RESERVED

    @classmethod
    def get_l2_tiling_budget(cls):
        """Get conservative L2 budget for tiling decisions (accounts for runtime overhead)."""
        return cls.L2_TILING_BUDGET

    @classmethod
    def get_l3_stage_threshold(cls):
        """Get threshold for L3 staging (weights >= this size are pre-loaded from L3)."""
        return cls.L3_STAGE_THRESHOLD

    @classmethod
    def get_l2_activation_reserved_bytes(cls):
        """Reserved L2 bytes used when deciding activation L3 fallback."""
        return cls.L2_ACTIVATION_RESERVED

    @classmethod
    def get_l3_fallback_single_buffer_threshold_bytes(cls):
        """Single-buffer threshold for forcing L3 fallback."""
        return cls.L3_FALLBACK_SINGLE_BUFFER_THRESHOLD


# Weight Residency Types (constants for consistency)
WEIGHT_RESIDENCY_L2 = 'L2'                    # Small weights, fit in L2
WEIGHT_RESIDENCY_L3_STAGED = 'L3_STAGED'      # Pre-load from L3 before execution
WEIGHT_RESIDENCY_L3_TILED = 'L3_TILED'        # Stream during execution (dynamic tiling)
WEIGHT_RESIDENCY_MAMBA_SCRATCH = 'MAMBA_SCRATCH'  # Shared Mamba scratch buffer


def determine_weight_residency(
    weight_size_bytes: int,
    layer_type: str,
    memory_tier: str = None,
    uses_mamba_scratch: bool = False
) -> str:
    """
    Unified function for determining weight residency across all layer types.

    This consolidates the weight residency logic that was previously scattered
    across different layer type handlers in build_execution_plan().

    Args:
        weight_size_bytes: Size of weights in bytes (INT8 = 1 byte per element)
        layer_type: Type of layer ('conv2d', 'linear', 'mhsa_projection', 'mamba', etc.)
        memory_tier: Memory tier from tile calculation ('L1_TILED', 'L2_FULL', 'L3_TILED')
        uses_mamba_scratch: Whether this layer uses shared Mamba scratch buffer

    Returns:
        Weight residency string: 'L2', 'L3_STAGED', 'L3_TILED', or 'MAMBA_SCRATCH'

    Decision Logic:
        1. Mamba scratch takes priority (shared buffer for all Mamba params)
        2. If tile calculation determined L3_TILED, use that (streaming weights)
        3. For MHSA projections, use threshold-based staging (>=32KB -> L3_STAGED)
        4. Default: L2 resident (weights fit in L2)
    """
    # Priority 1: Mamba scratch (shared buffer for Mamba block parameters)
    if uses_mamba_scratch:
        return WEIGHT_RESIDENCY_MAMBA_SCRATCH

    # Priority 2: Dynamic L3 tiling (determined by tile calculation)
    # This is used for large Conv2D/Linear layers that stream weights during execution
    if memory_tier == 'L3_TILED':
        return WEIGHT_RESIDENCY_L3_TILED

    # Priority 3: Threshold-based L3 staging for layers without dynamic tiling
    # MHSA projections and other small-ish layers that don't use tile-based L3
    if layer_type in ('mhsa_projection', 'mhsa'):
        if weight_size_bytes >= GAP9HardwareModel.L3_STAGE_THRESHOLD:
            return WEIGHT_RESIDENCY_L3_STAGED

    # Default: L2 resident (small weights that fit in L2)
    return WEIGHT_RESIDENCY_L2

class TileConfig:
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

class Conv2DTileConfig(TileConfig):
    def __init__(self):
        # L1 Tiling (Inner)
        self.tile_h = 0
        self.tile_w = 0
        self.tile_h_with_halo = 0
        self.tile_w_with_halo = 0
        self.num_tiles = 0
        self.num_tiles_h = 0
        self.num_tiles_w = 0
        self.l1_input_bytes = 0
        self.l1_output_bytes = 0
        self.out_tile_h = 0
        self.out_tile_w = 0

        # L1 Weight Caching (step )
        self.weight_tiling_enabled = False  # True if weights cached in L1
        self.tile_out_ch = 0                # Output channels per weight tile
        self.num_out_ch_tiles = 1           # Number of output channel tiles
        self.l1_weight_bytes = 0            # Weight tile size in bytes

        # Triple-buffer weight pipeline (step )
        # Uses 3 weight buffers to eliminate blocking wait on first weight load
        # Only enabled when num_out_ch_tiles >= 3 (enough tiles to fill pipeline)
        self.triple_buffer_weights = False

        # L3 Tiling (Outer)
        self.l3_tiling_enabled = False
        self.l3_tile_h = 0
        self.l3_tile_h_halo = 0
        self.num_l3_tiles = 1
        self.l2_buffer_size = 0

class LinearTileConfig(TileConfig):
    def __init__(self):
        self.tile_out_features = 0
        self.num_tiles = 0
        self.l1_input_bytes = 0
        self.l1_output_bytes = 0
        self.l1_weight_bytes = 0
        self.input_features = 0

        # K-dimension tiling (input feature tiling)
        self.tile_in_features = 0      # Input features per K-tile (0 = no K-tiling)
        self.num_k_tiles = 1           # Number of input feature tiles
        self.k_tiling_enabled = False  # Flag: True if K-tiling is active

        # M-dimension tiling (batch/token tiling for 3D linear)
        self.tile_batch_tokens = 0     # Tokens per M-tile (0 = no M-tiling)
        self.num_m_tiles = 1           # Number of token tiles
        self.m_tiling_enabled = False  # Flag: True if M-tiling is active
        self.batch_tokens = 1          # Total batch tokens (for reference)

        # Input L1 caching (auto-tunable)
        # When True, DMA input to L1 before compute for better locality
        # When False, compute directly from L2 input pointer
        self.l1_input_cache = False

        # L3 Tiling (Outer)
        self.l3_tiling_enabled = False
        self.l3_tile_out_features = 0 # Number of output features in one L2 Slab
        self.num_l3_tiles = 1
        self.l2_buffer_size = 0

class MHSATileConfig(TileConfig):
    def __init__(self):
        self.tile_q = 0
        self.num_tiles = 0
        self.persistent_bytes = 0
        self.tile_bytes = 0
        
        # L3 Tiling (Outer)
        self.l3_tiling_enabled = False
        self.l3_seq_len = 0  # Sequence length in one L2 Slab
        self.num_l3_tiles = 1
        self.l2_buffer_size = 0

# Pooling tile configurations
class MaxPoolTileConfig(TileConfig):
    def __init__(self):
        # L1 Tiling (Inner)
        self.tile_h = 0              # Input tile height
        self.tile_w = 0              # Input tile width
        self.out_tile_h = 0          # Output tile height
        self.out_tile_w = 0          # Output tile width
        self.num_tiles = 0           # Total number of tiles
        self.num_tiles_h = 0         # Number of tiles in height dimension
        self.num_tiles_w = 0         # Number of tiles in width dimension
        self.l1_input_bytes = 0      # Input buffer size per tile
        self.l1_output_bytes = 0     # Output buffer size per tile
        self.l1_buffer_bytes = 0     # Total L1 buffer (for consistency)

        # L3 Tiling (Outer) - for very large activations
        self.l3_tiling_enabled = False
        self.l3_tile_h = 0           # Output height per L2 slab
        self.l3_tile_h_in = 0        # Input height per L2 slab (with halo)
        self.num_l3_tiles = 1        # Number of L3 slabs
        self.l2_buffer_size = 0      # L2 buffer size for double-buffered slabs


class AvgPoolTileConfig(TileConfig):
    def __init__(self):
        # L1 Tiling (Inner)
        self.tile_h = 0
        self.tile_w = 0
        self.out_tile_h = 0
        self.out_tile_w = 0
        self.num_tiles = 0
        self.num_tiles_h = 0
        self.num_tiles_w = 0
        self.l1_input_bytes = 0
        self.l1_output_bytes = 0
        self.l1_buffer_bytes = 0

        # L3 Tiling (Outer)
        self.l3_tiling_enabled = False
        self.l3_tile_h = 0
        self.l3_tile_h_in = 0
        self.num_l3_tiles = 1
        self.l2_buffer_size = 0


class GlobalAvgPoolTileConfig(TileConfig):
    def __init__(self):
        self.tile_h = 0              # Input tile height
        self.tile_w = 0              # Input tile width
        self.in_h = 0                # Full input height
        self.in_w = 0                # Full input width
        self.num_tiles = 0
        self.num_tiles_h = 0
        self.num_tiles_w = 0
        self.l1_input_bytes = 0
        self.l1_output_bytes = 0
        self.l1_partial_sum_bytes = 0  # INT32 accumulator space

def calculate_conv2d_tile_size(
    in_h, in_w, in_channels, out_channels,
    kernel_size=None, stride=None, padding=None,
    kernel_h=None, kernel_w=None, stride_h=None, stride_w=None, pad_h=None, pad_w=None,
    l1_budget=None, l2_budget=None,
    hint_tile_h=None,  # Optional: preferred output tile height (for auto-tuning)
    hint_tile_w=None   # Optional: preferred output tile width (for auto-tuning)
):
    """
    Compute a Conv2D tiling configuration for GAP9.

    The returned `Conv2DTileConfig` describes:
    - Whether outer-loop L3 tiling ("slabs") is needed to fit L2
    - Inner-loop L1 tiling sizes (tile_h/tile_w + halo) to fit TCDM
    - Byte budgets for the required L1 input/output buffers

    Args:
        in_h, in_w, in_channels: Input tensor shape (NHWC-ish without batch)
        out_channels: Output channels
        kernel_size/stride/padding or explicit kernel_h/kernel_w/...: Conv parameters
        l1_budget: Optional override for usable L1 bytes
        l2_budget: Optional override for usable L2 bytes
        hint_tile_h: Optional preferred output tile height (for auto-tuning)
        hint_tile_w: Optional preferred output tile width (for auto-tuning)

    Returns:
        Conv2DTileConfig on success, or None if no feasible plan fits budgets.
    """
    # Normalize args
    kh = kernel_h if kernel_h is not None else kernel_size
    kw = kernel_w if kernel_w is not None else kernel_size
    sh = stride_h if stride_h is not None else stride
    sw = stride_w if stride_w is not None else stride
    ph = pad_h if pad_h is not None else padding
    pw = pad_w if pad_w is not None else padding
    
    if l1_budget is None: l1_budget = GAP9HardwareModel.get_l1_budget()
    if l2_budget is None: l2_budget = GAP9HardwareModel.get_l2_budget()

    config = Conv2DTileConfig()
    
    total_input_bytes = in_h * in_w * in_channels
    total_output_bytes = (in_h // sh) * (in_w // sw) * out_channels
    total_l2_req = total_input_bytes + total_output_bytes

    if total_l2_req <= l2_budget:
        config.l3_tiling_enabled = False
        config.l3_tile_h = in_h
        config.l3_tile_h_halo = in_h + 2*ph
        config.num_l3_tiles = 1
        config.l2_buffer_size = total_l2_req
        target_h = in_h
    else:
        config.l3_tiling_enabled = True
        num_slabs = 2
        while True:
            slab_h = math.ceil(in_h / num_slabs)
            slab_h_halo = slab_h * sh + kh - 1
            if slab_h_halo > in_h: slab_h_halo = in_h
            slab_input_bytes = slab_h_halo * in_w * in_channels
            slab_output_bytes = slab_h * (in_w // sw) * out_channels
            req_l2 = 2 * (slab_input_bytes + slab_output_bytes)
            if req_l2 <= l2_budget:
                config.l3_tile_h = slab_h
                config.l3_tile_h_halo = slab_h_halo
                config.num_l3_tiles = num_slabs
                config.l2_buffer_size = req_l2
                target_h = slab_h_halo
                break
            num_slabs += 1
            if num_slabs > in_h: return None

    # L1 Tiling Logic (with optional W-dimension tiling support)
    out_slab_h = (config.l3_tile_h if config.l3_tiling_enabled else math.floor((in_h + 2*ph - kh)/sh + 1))
    out_slab_w = math.floor((in_w + 2*pw - kw)/sw + 1)
    found = False

    # If hint_tile_h is provided, try it first (for auto-tuning)
    tile_h_candidates = list(range(out_slab_h, 0, -1))
    if hint_tile_h is not None and 0 < hint_tile_h <= out_slab_h:
        # Move hint to front of candidate list
        if hint_tile_h in tile_h_candidates:
            tile_h_candidates.remove(hint_tile_h)
        tile_h_candidates.insert(0, hint_tile_h)

    # If hint_tile_w is provided, use it; otherwise default to full width (no W-tiling)
    tile_w_candidates = [out_slab_w]  # Default: full width
    if hint_tile_w is not None and 0 < hint_tile_w <= out_slab_w:
        # Try hint first, then fall back to full width
        tile_w_candidates = [hint_tile_w, out_slab_w] if hint_tile_w != out_slab_w else [out_slab_w]

    for tile_h_out in tile_h_candidates:
        tile_h_in = (tile_h_out - 1) * sh + kh
        for tile_w_out in tile_w_candidates:
            tile_w_in = (tile_w_out - 1) * sw + kw
            size_in = tile_h_in * tile_w_in * in_channels
            size_out = tile_h_out * tile_w_out * out_channels
            total_l1 = 2 * (size_in + size_out)

            if total_l1 <= l1_budget:
                config.tile_h = tile_h_out
                config.tile_w = tile_w_out
                config.tile_h_with_halo = tile_h_in
                config.tile_w_with_halo = tile_w_in
                config.out_tile_h = tile_h_out
                config.out_tile_w = tile_w_out
                config.num_tiles_h = math.ceil(out_slab_h / tile_h_out)
                config.num_tiles_w = math.ceil(out_slab_w / tile_w_out)
                config.num_tiles = config.num_tiles_h * config.num_tiles_w
                config.l1_input_bytes = size_in
                config.l1_output_bytes = size_out
                found = True
                break
        if found:
            break

    if not found: return None # Simplified fallback
    return config


def calculate_conv2d_tile_size_with_weights(
    in_h, in_w, in_channels, out_channels,
    kernel_size=None, stride=None, padding=None,
    kernel_h=None, kernel_w=None, stride_h=None, stride_w=None, pad_h=None, pad_w=None,
    l1_budget=None, l2_budget=None,
    min_out_ch_tile=8,  # Minimum output channels per weight tile (for efficiency)
    hint_tile_h=None,  # Optional: preferred output tile height (for auto-tuning)
    hint_tile_w=None,  # Optional: preferred output tile width (for auto-tuning)
    groups=1           # Number of groups (1=standard conv, in_channels=depthwise)
):
    """
    Calculate Conv2D tile configuration WITH L1 weight caching.

    This extends the basic tiling to also cache weights in L1, reducing
    the 10x L2 latency to 1x L1 latency for weight reads.

    L1 Layout (double-buffered):
        2 * (input_tile + output_tile + weight_tile)

    Weight Tiling Strategy:
        - Weights are tiled along the output channel dimension
        - Full spatial kernel (kH x kW) and all input channels stay together
        - Each weight tile: [tile_out_ch, in_ch, kH, kW] for standard conv
        - For depthwise: [tile_out_ch, 1, kH, kW] (each output ch uses 1 input ch)

    Returns:
        Conv2DTileConfig with weight_tiling_enabled=True if successful,
        or falls back to standard tiling (weight_tiling_enabled=False) if weights don't fit.
    """
    # Normalize args
    kh = kernel_h if kernel_h is not None else kernel_size
    kw = kernel_w if kernel_w is not None else kernel_size
    sh = stride_h if stride_h is not None else stride
    sw = stride_w if stride_w is not None else stride
    ph = pad_h if pad_h is not None else padding
    pw = pad_w if pad_w is not None else padding

    if l1_budget is None: l1_budget = GAP9HardwareModel.get_l1_budget()
    if l2_budget is None: l2_budget = GAP9HardwareModel.get_l2_budget()

    config = Conv2DTileConfig()

    # Calculate output dimensions
    out_h = (in_h + 2*ph - kh) // sh + 1
    out_w = (in_w + 2*pw - kw) // sw + 1

    # Detect depthwise convolution (groups == in_channels == out_channels)
    is_depthwise = (groups > 1) and (groups == in_channels)

    # Total weight size for this layer
    # Depthwise: each output channel has 1 input channel (kh * kw weights per output)
    # Standard: each output channel has in_channels input channels
    if is_depthwise:
        total_weight_bytes = out_channels * kh * kw
    else:
        total_weight_bytes = out_channels * in_channels * kh * kw

    # ---
    # Step 1: L3/L2 Tiling (Same as before - handle large activations)
    # ---
    total_input_bytes = in_h * in_w * in_channels
    total_output_bytes = out_h * out_w * out_channels
    total_l2_req = total_input_bytes + total_output_bytes

    if total_l2_req <= l2_budget:
        config.l3_tiling_enabled = False
        config.l3_tile_h = in_h
        config.l3_tile_h_halo = in_h + 2*ph
        config.num_l3_tiles = 1
        config.l2_buffer_size = total_l2_req
        target_h = in_h
        target_out_h = out_h
    else:
        config.l3_tiling_enabled = True
        num_slabs = 2
        while True:
            slab_h = math.ceil(in_h / num_slabs)
            slab_h_halo = slab_h * sh + kh - 1
            if slab_h_halo > in_h: slab_h_halo = in_h
            slab_input_bytes = slab_h_halo * in_w * in_channels
            slab_output_bytes = slab_h * (in_w // sw) * out_channels
            req_l2 = 2 * (slab_input_bytes + slab_output_bytes)
            if req_l2 <= l2_budget:
                config.l3_tile_h = slab_h
                config.l3_tile_h_halo = slab_h_halo
                config.num_l3_tiles = num_slabs
                config.l2_buffer_size = req_l2
                target_h = slab_h_halo
                target_out_h = slab_h
                break
            num_slabs += 1
            if num_slabs > in_h:
                return None  # Can't fit even with L3 tiling

    # ---
    # Step 2: L1 Tiling WITH Weight Caching
    # ---
    # Try to find a tile size that fits input + output + weights in L1

    out_slab_h = target_out_h
    out_slab_w = out_w

    # Build candidate lists, prioritizing hints if provided
    tile_h_candidates = list(range(out_slab_h, 0, -1))
    if hint_tile_h is not None and 0 < hint_tile_h <= out_slab_h:
        if hint_tile_h in tile_h_candidates:
            tile_h_candidates.remove(hint_tile_h)
        tile_h_candidates.insert(0, hint_tile_h)

    # W-dimension candidate list (default to full width)
    tile_w_candidates = [out_slab_w]
    if hint_tile_w is not None and 0 < hint_tile_w <= out_slab_w:
        tile_w_candidates = [hint_tile_w, out_slab_w] if hint_tile_w != out_slab_w else [out_slab_w]

    # Try tile configurations, prioritizing hints
    for tile_h_out in tile_h_candidates:
        tile_h_in = (tile_h_out - 1) * sh + kh
        for tile_w_out in tile_w_candidates:
            tile_w_in = (tile_w_out - 1) * sw + kw

            size_in = tile_h_in * tile_w_in * in_channels
            size_out = tile_h_out * tile_w_out * out_channels

            # Available L1 per buffer after input/output (for weights)
            # Layout: 2 * (input + output + weights)
            available_for_weights = (l1_budget // 2) - size_in - size_out

            if available_for_weights <= 0:
                continue  # Input + output too big, try smaller tile

            # Weight tile size per output channel:
            # - Standard conv: in_ch * kH * kW (each output channel connects to all inputs)
            # - Depthwise: kH * kW (each output channel connects to 1 input channel)
            #
            # For Conv2D im2col+SIMD, if this is not a multiple of 4 then consecutive
            # output-channel rows become misaligned for `v4s` loads. The runtime can
            # optionally pad each output-channel row to a 4-byte stride in L1 (see
            # CONV2D_WEIGHT_ROW_STRIDE_PAD), so size L1 weight tiles accordingly.
            if is_depthwise:
                weight_per_out_ch = kh * kw
            else:
                weight_per_out_ch = in_channels * kh * kw
            weight_per_out_ch_padded = (weight_per_out_ch + 3) & ~3

            # How many output channels can we fit in available weight budget?
            max_out_ch = available_for_weights // weight_per_out_ch_padded

            if max_out_ch >= out_channels:
                # All weights fit in L1! Single weight tile.
                num_tiles_h = math.ceil(out_slab_h / tile_h_out)
                num_tiles_w = math.ceil(out_slab_w / tile_w_out)
                config.weight_tiling_enabled = True
                config.tile_out_ch = out_channels
                config.num_out_ch_tiles = 1
                config.l1_weight_bytes = out_channels * weight_per_out_ch_padded

                config.tile_h = tile_h_out
                config.tile_w = tile_w_out
                config.tile_h_with_halo = tile_h_in
                config.tile_w_with_halo = tile_w_in
                config.out_tile_h = tile_h_out
                config.out_tile_w = tile_w_out
                config.num_tiles_h = num_tiles_h
                config.num_tiles_w = num_tiles_w
                config.num_tiles = num_tiles_h * num_tiles_w
                config.l1_input_bytes = size_in
                config.l1_output_bytes = size_out
                return config

            elif max_out_ch >= min_out_ch_tile:
                # Partial weight tiling - tile output channels
                num_tiles_h = math.ceil(out_slab_h / tile_h_out)
                num_tiles_w = math.ceil(out_slab_w / tile_w_out)

                # Check if triple-buffer weight pipeline is beneficial
                # Triple-buffer layout: 2*(input+output) + 3*weights
                # This eliminates blocking wait on first weight load
                num_out_ch_tiles_double = math.ceil(out_channels / max_out_ch)
                use_triple_buffer = False

                if num_out_ch_tiles_double >= 3:
                    # Try triple-buffer: calculate available space for 3 weight buffers
                    # Layout: 2*(input+output) + 3*weights <= l1_budget
                    available_for_triple_weights = l1_budget - 2 * (size_in + size_out)
                    if available_for_triple_weights > 0:
                        max_out_ch_triple = available_for_triple_weights // (3 * weight_per_out_ch_padded)
                        if max_out_ch_triple >= min_out_ch_tile:
                            num_out_ch_tiles_triple = math.ceil(out_channels / max_out_ch_triple)
                            # Only use triple-buffer if we still have >= 3 tiles
                            if num_out_ch_tiles_triple >= 3:
                                use_triple_buffer = True
                                max_out_ch = max_out_ch_triple

                config.weight_tiling_enabled = True
                config.tile_out_ch = max_out_ch
                config.num_out_ch_tiles = math.ceil(out_channels / max_out_ch)
                config.l1_weight_bytes = max_out_ch * weight_per_out_ch_padded
                config.triple_buffer_weights = use_triple_buffer

                config.tile_h = tile_h_out
                config.tile_w = tile_w_out
                config.tile_h_with_halo = tile_h_in
                config.tile_w_with_halo = tile_w_in
                config.out_tile_h = tile_h_out
                config.out_tile_w = tile_w_out
                config.num_tiles_h = num_tiles_h
                config.num_tiles_w = num_tiles_w
                config.num_tiles = num_tiles_h * num_tiles_w
                config.l1_input_bytes = size_in
                config.l1_output_bytes = size_out
                return config

    # ---
    # Step 3: Fallback to standard tiling (no L1 weight caching)
    # ---
    # If we can't fit weights in L1, use the existing non-weight-cached approach
    print(f"  WARNING: Conv2D weights too large for L1 caching ({total_weight_bytes} bytes), using L2 fallback")
    config.weight_tiling_enabled = False
    config.tile_out_ch = 0
    config.num_out_ch_tiles = 1
    config.l1_weight_bytes = 0

    # Standard L1 tiling (input + output only)
    # Build candidate lists, prioritizing hints if provided
    tile_h_candidates_fallback = list(range(out_slab_h, 0, -1))
    if hint_tile_h is not None and 0 < hint_tile_h <= out_slab_h:
        if hint_tile_h in tile_h_candidates_fallback:
            tile_h_candidates_fallback.remove(hint_tile_h)
        tile_h_candidates_fallback.insert(0, hint_tile_h)

    tile_w_candidates_fallback = [out_slab_w]
    if hint_tile_w is not None and 0 < hint_tile_w <= out_slab_w:
        tile_w_candidates_fallback = [hint_tile_w, out_slab_w] if hint_tile_w != out_slab_w else [out_slab_w]

    for tile_h_out in tile_h_candidates_fallback:
        tile_h_in = (tile_h_out - 1) * sh + kh
        for tile_w_out in tile_w_candidates_fallback:
            tile_w_in = (tile_w_out - 1) * sw + kw

            size_in = tile_h_in * tile_w_in * in_channels
            size_out = tile_h_out * tile_w_out * out_channels
            total_l1 = 2 * (size_in + size_out)

            if total_l1 <= l1_budget:
                config.tile_h = tile_h_out
                config.tile_w = tile_w_out
                config.tile_h_with_halo = tile_h_in
                config.tile_w_with_halo = tile_w_in
                config.out_tile_h = tile_h_out
                config.out_tile_w = tile_w_out
                config.num_tiles_h = math.ceil(out_slab_h / tile_h_out)
                config.num_tiles_w = math.ceil(out_slab_w / tile_w_out)
                config.num_tiles = config.num_tiles_h * config.num_tiles_w
                config.l1_input_bytes = size_in
                config.l1_output_bytes = size_out
                return config

    return None  # Can't fit at all


def calculate_linear_tile_size(input_features, output_features, batch_size=1, l1_budget=None, l2_budget=None, output_element_size=1):
    """
    Compute a Linear tiling configuration for GAP9.

    This chooses between:
    - L2-full execution (weights + activations fit in L2)
    - L3 weight streaming (split output features into slabs) when weights are too large
    - L1 inner-loop tiling strategies:
      - batch_size == 1: input stays in L1; weights/output are tiled in L1
      - batch_size > 1 (token sequences): weight-only L1 tiling to maximize reuse

    Args:
        input_features: K dimension
        output_features: N dimension
        batch_size: M dimension (tokens); `1` for classic 2D linear
        l1_budget: Optional override for usable L1 bytes
        l2_budget: Optional override for usable L2 bytes
        output_element_size: Output element bytes (1 for int8, 4 for fp32)

    Returns:
        LinearTileConfig on success, or None if no feasible plan fits budgets.
    """
    if l1_budget is None: l1_budget = GAP9HardwareModel.get_l1_budget()
    if l2_budget is None: l2_budget = GAP9HardwareModel.get_l2_budget()
    
    config = LinearTileConfig()
    config.input_features = input_features

    # 1. Check L2 Budget (Weights + Input + Output)
    # For Linear, weights are usually the bottleneck.
    weight_size = input_features * output_features
    input_size = input_features * batch_size
    output_size = output_features * batch_size * output_element_size # Account for element size
    total_req = weight_size + input_size + output_size

    if total_req <= l2_budget:
        # Fits in L2
        config.l3_tiling_enabled = False
        config.l3_tile_out_features = output_features
        config.num_l3_tiles = 1
        config.l2_buffer_size = total_req
        target_out_features = output_features
    else:
        # Needs L3 Tiling: Split Output Features (Weight Streaming)
        # This assumes Input fits in L2. If Input > L2, we need sequence tiling (future).
        if input_size > l2_budget / 2:
            return None # Input too big for Weight Streaming strategy
        
        config.l3_tiling_enabled = True
        num_slabs = 2
        while True:
            slab_out = math.ceil(output_features / num_slabs)
            # Size = Input (Full) + Slab_Weights (Double Buff) + Slab_Output (Double Buff)
            # We double buffer the slabs for streaming
            slab_weight_size = slab_out * input_features
            slab_output_size = slab_out * batch_size * output_element_size # Account for element size
            
            # L2 Req = Input + 2 * (Slab_Weights + Slab_Output)
            req_l2 = input_size + 2 * (slab_weight_size + slab_output_size)
            
            if req_l2 <= l2_budget:
                config.l3_tile_out_features = slab_out
                config.num_l3_tiles = num_slabs
                config.l2_buffer_size = req_l2
                target_out_features = slab_out
                break
            num_slabs += 1
            if num_slabs > output_features: return None

    # 2. L1 Tiling (Inner Loop)
    #
    # Two strategies:
    # - 2D Linear (batch_size == 1): Load input once, double-buffer weights + output tiles in L1.
    # - 3D Linear (batch_size > 1, INT8 output): Keep input/output in L2 and tile only weights in L1.
    #   This maximizes weight reuse across tokens (transformer MLP), avoids allocating huge L1 output tiles.
    #
    if batch_size > 1 and output_element_size == 1:
        # Weight-only tiling: double-buffer weight tiles
        max_features = (l1_budget // 2) // input_features
        max_features &= ~3  # SIMD-friendly (groups of 4)
        if max_features < 4:
            return None

        tile_out = min(target_out_features, max_features)

        # Prefer large aligned tiles to reduce DMA overhead
        for align in (64, 32, 16, 4):
            aligned = (tile_out // align) * align
            if aligned >= 4:
                tile_out = aligned
                break

        if tile_out < 1:
            return None

        config.tile_out_features = int(tile_out)
        config.num_tiles = math.ceil(target_out_features / tile_out)
        config.l1_input_bytes = 0
        config.l1_output_bytes = 0
        config.l1_weight_bytes = config.tile_out_features * input_features
        return config

    # 2D Linear / FP32 output: input+output+weights tiling in L1
    if input_size > l1_budget:
        return None  # Input too big for L1

    remaining = l1_budget - input_size
    bytes_per_feature = input_features + (batch_size * output_element_size)
    max_features = remaining // (2 * bytes_per_feature)  # Double buffer

    if max_features < 1:
        return None

    if max_features >= target_out_features:
        config.tile_out_features = target_out_features
        config.num_tiles = 1
    else:
        config.tile_out_features = int(max_features)
        config.num_tiles = math.ceil(target_out_features / max_features)

    config.l1_input_bytes = input_size
    config.l1_output_bytes = config.tile_out_features * batch_size * output_element_size
    config.l1_weight_bytes = config.tile_out_features * input_features
    
    return config

def calculate_mhsa_tile_size(seq_len, head_dim, num_heads, l1_budget=None, l2_budget=None,
                              hint_tile_q=None  # Optional: preferred query tile size (for auto-tuning)
                              ):
    if l1_budget is None: l1_budget = GAP9HardwareModel.get_l1_budget()
    if l2_budget is None: l2_budget = GAP9HardwareModel.get_l2_budget()

    # Codegen allocates at least ~110KB L1 when MHSA is present (see network.c template).
    # Use that budget here so MHSA can pick larger tiles (fewer iterations) safely.
    l1_budget = max(l1_budget, 110000)
    
    config = MHSATileConfig()
    embed_dim = num_heads * head_dim
    
    # 1. Check L2 Budget (Q, K, V, Output)
    # All buffers are [Seq_Len, Embed_Dim]
    # Total = 4 * (Seq_Len * Embed_Dim)
    total_size = 4 * seq_len * embed_dim
    
    if total_size <= l2_budget:
        config.l3_tiling_enabled = False
        config.l3_seq_len = seq_len
        config.num_l3_tiles = 1
        config.l2_buffer_size = total_size
        target_seq_len = seq_len
    else:
        # L3 Tiling: Split Sequence Length
        config.l3_tiling_enabled = True
        num_slabs = 2
        while True:
            slab_seq = math.ceil(seq_len / num_slabs)
            # Size = 2 * (4 buffers * slab_seq * embed_dim) (Double Buffered Slabs)
            req_l2 = 2 * (4 * slab_seq * embed_dim)
            
            if req_l2 <= l2_budget:
                config.l3_seq_len = slab_seq
                config.num_l3_tiles = num_slabs
                config.l2_buffer_size = req_l2
                target_seq_len = slab_seq
                break
            num_slabs += 1
            if num_slabs > seq_len: return None

    # 2. L1 Tiling (Inner Loop)
    # Persistent K,V (Full Head) + Tiled Q, Output
    # Persistent = 2 * target_seq_len * head_dim
    persistent_bytes = 2 * target_seq_len * head_dim
    
    if persistent_bytes > l1_budget / 2: 
        # If K/V don't fit, we need a different L1 strategy (e.g. tile K/V too).
        # For now, fail.
        return None
        
    remaining = l1_budget - persistent_bytes
    
    # Tile Q: Need Q_tile, Scores_tile, M_tile
    #
    # IMPORTANT: The runtime MHSA L1 kernel double-buffers only the Q and M tiles,
    # while the score buffers (INT32 + INT8 + UINT8 for iSoftmax) are single-buffered.
    #
    # Integer softmax path memory model (bytes):
    #   persistent K,V: 2 * seq_len * head_dim
    #   Q tiles (ping/pong): 2 * tile_q * head_dim
    #   M tiles (ping/pong): 2 * tile_q * head_dim
    #
    # Score buffers (single-buffered):
    #   - INT32 scores: tile_q * seq_len * 4
    #   - UINT8 attention weights: tile_q * seq_len * 1
    #   - Optional INT8 intermediate scores are disabled by default
    #     (`MHSA_TILED_QK_STORE_INT8_SCORES=0`), so we do not budget for them here.
    #
    # Total required:
    #   persistent_bytes + tile_q * (5*seq_len + 4*head_dim)
    #
    # This matches the runtime check in `mhsa_tiled_l1_inner_loop()`.
    bytes_per_row = 5 * target_seq_len + 4 * head_dim
    max_rows = remaining // bytes_per_row
    
    if max_rows < 1: return None
    max_rows = int(min(max_rows, target_seq_len))
    
    if max_rows >= target_seq_len:
        config.tile_q = target_seq_len
        config.num_tiles = 1
    else:
        # If hint_tile_q is provided and valid, use it directly
        if hint_tile_q is not None and 0 < hint_tile_q <= max_rows:
            config.tile_q = hint_tile_q
            config.num_tiles = math.ceil(target_seq_len / hint_tile_q)
        else:
            # Choose a tile_q that balances:
            # - fewer tiles (less per-tile overhead)
            # - better multi-core load balance (avoid a 4th row-iteration on some cores)
            #
            # Simple heuristic proxy for runtime cost:
            #   num_tiles * (ceil(tile_q / NUM_CORES) + 1)
            # The `+1` approximates fixed per-tile overhead (fork/args/DMA setup).
            NUM_CORES = 8
            best_tile_q = 1
            best_score = None
            for tile_q in range(1, max_rows + 1):
                num_tiles = math.ceil(target_seq_len / tile_q)
                iters = math.ceil(tile_q / NUM_CORES)
                score = num_tiles * (iters + 1)
                if best_score is None or score < best_score or (score == best_score and tile_q > best_tile_q):
                    best_score = score
                    best_tile_q = tile_q
            config.tile_q = best_tile_q
            config.num_tiles = math.ceil(target_seq_len / best_tile_q)
        
    config.persistent_bytes = persistent_bytes
    # tile_bytes is used for rough L1 sizing in codegen. Keep it representing the
    # per-tile Q+M footprint (single-buffer each); Q and M are double-buffered at runtime.
    config.tile_bytes = config.tile_q * (2 * head_dim)
    
    return config

class SSMTileConfig(TileConfig):
    def __init__(self):
        self.tile_m = 0
        self.num_tiles = 0
        self.l1_state_bytes = 0
        self.l1_io_bytes = 0
        
        # L3 Tiling (Outer)
        self.l3_tiling_enabled = False
        self.l3_m = 0
        self.num_l3_tiles = 1
        self.l2_buffer_size = 0

def calculate_ssm_tile_size(seq_len, d_inner, d_state, l1_budget=None, l2_budget=None):
    """
    Calculate Tiling for SSM (Mamba) Layer.
    
    Tiling Strategy:
    - L3 Tiling: Split d_inner (Channels). State 'h' and params 'A' partition cleanly.
    - L1 Tiling: Split d_inner (Channels).
    - Time (seq_len) is NOT tiled (streaming scan).
    
    Args:
        seq_len (L): Sequence length
        d_inner (M): Inner dimension (channels)
        d_state (N): State dimension
    """
    if l1_budget is None: l1_budget = GAP9HardwareModel.get_l1_budget()
    if l2_budget is None: l2_budget = GAP9HardwareModel.get_l2_budget()
    
    config = SSMTileConfig()
    
    # 1. Check L2 Budget
    # Inputs: x [L, M], z [L, M], dt [L, M] (actually dt is internal usually, but let's assume input)
    # Weights: A [M, N], D [M], dt_w [M], dt_b [M]
    # B, C are usually computed on the fly or passed in. Let's assume passed in [L, N].
    # Total L2 = Inputs + Outputs + Weights
    # Major buffers: x, z, y (IO), A (Weights)
    
    io_bytes = 3 * (seq_len * d_inner) # x(1) + z(1) + y(1)
    weight_bytes = d_inner * d_state * 4 # A_f32
    # B, C are [L, N].
    bc_bytes = seq_len * d_state * (4 + 2) # B_f32 + C_q15
    
    total_l2 = io_bytes + weight_bytes + bc_bytes
    
    if total_l2 <= l2_budget:
        config.l3_tiling_enabled = False
        config.l3_m = d_inner
        config.num_l3_tiles = 1
        config.l2_buffer_size = total_l2
        target_m = d_inner
    else:
        # L3 Tiling: Split M (Channels)
        # B and C are shared (broadcast), so they must fit in L2 fully or be streamed.
        # Assuming B, C fit.
        config.l3_tiling_enabled = True
        num_slabs = 2
        while True:
            slab_m = math.ceil(d_inner / num_slabs)
            req_l2 = 2 * (3 * seq_len * slab_m + slab_m * d_state * 4) + bc_bytes
            if req_l2 <= l2_budget:
                config.l3_m = slab_m
                config.num_l3_tiles = num_slabs
                config.l2_buffer_size = req_l2
                target_m = slab_m
                break
            num_slabs += 1
            if num_slabs > d_inner: return None

    # 2. L1 Tiling
    # We tile M. 
    # L and N are kept full.
    # Memory per tile (Double Buffered):
    # State 'h' [Tile_M, N] (int32) -> Persistent within tile? No, it's scanned.
    # h is updated t=0..L. We only need h[Tile_M, N] current state.
    # Wait, if we tile M, we iterate L inside the tile.
    # So we need memory for h [Tile_M, N].
    
    # IO Buffers (Double Buffered):
    # x_tile [L, Tile_M] (int8)
    # z_tile [L, Tile_M] (int8)
    # y_tile [L, Tile_M] (int8)
    # dt_acc [L, Tile_M] (int32) -- Intermediate, huge!
    # 
    # If L is large, we CANNOT store full [L, Tile_M] intermediate dt_acc in L1.
    # The SSM kernel in 'ssm_anna_L2.c' computes dt_acc, then discretizes, then scans.
    # It seems to require full L buffers.
    # IF L is too big, we must Chunk L.
    # For now, assume L fits or we reduce Tile_M drastically.
    
    # Let's iterate Tile_M.
    
    # Fixed overhead: B, C, LUTs
    # B [L, N] (float), C [L, N] (int16)
    bc_size = seq_len * d_state * (4 + 2)
    luts_size = 2048 # Approx
    
    remaining = l1_budget - bc_size - luts_size
    if remaining <= 0: return None
    
    # Per Tile costs (Double Buffered -> * 2)
    # x(1) + z(1) + y(1) + dt_acc(4) + dt_f(4) + dA(2) + dB(2) + y_acc(4)
    # This is A LOT of intermediates per M.
    # Bytes per (L*1) column:
    # 1+1+1 + 4+4 + 2+2 + 4 = 19 bytes per element?
    # Plus state h [1, N] * 4 bytes.
    
    # Approx bytes per channel column:
    # col_size = L * 20 + N * 4
    
    bytes_per_m = seq_len * 24 + d_state * 4
    
    max_m = remaining // (2 * bytes_per_m)
    
    if max_m < 1: return None
    
    if max_m >= target_m:
        config.tile_m = target_m
        config.num_tiles = 1
    else:
        config.tile_m = int(max_m)
        config.num_tiles = math.ceil(target_m / max_m)
        
    return config

def _calculate_pool_tile_size_impl(
    in_h, in_w, channels,
    kernel_size, stride, padding,
    l1_budget, l2_budget,
    config
):
    """Shared implementation for MaxPool and AvgPool tile calculation.

    Both pooling operations have identical tiling logic - only the kernel
    computation differs at runtime. This function computes the tile configuration.

    Args:
        in_h, in_w, channels: Input dimensions
        kernel_size, stride, padding: Pooling parameters
        l1_budget, l2_budget: Memory budgets
        config: Either MaxPoolTileConfig or AvgPoolTileConfig instance

    Returns:
        The populated config, or None if tiling is impossible
    """
    # Calculate output dimensions
    out_h = (in_h + 2 * padding - kernel_size) // stride + 1
    out_w = (in_w + 2 * padding - kernel_size) // stride + 1

    # Calculate total L2 requirement
    total_input_bytes = in_h * in_w * channels
    total_output_bytes = out_h * out_w * channels
    total_l2_req = total_input_bytes + total_output_bytes

    # step : Determine L3 tiling (outer loop)
    if total_l2_req <= l2_budget:
        config.l3_tiling_enabled = False
        config.l3_tile_h = out_h
        config.l3_tile_h_in = in_h
        config.num_l3_tiles = 1
        config.l2_buffer_size = total_l2_req
        target_out_h = out_h
    else:
        config.l3_tiling_enabled = True
        num_slabs = 2
        while True:
            slab_out_h = math.ceil(out_h / num_slabs)
            slab_in_h = (slab_out_h - 1) * stride + kernel_size
            if slab_in_h > in_h:
                slab_in_h = in_h

            slab_input_bytes = slab_in_h * in_w * channels
            slab_output_bytes = slab_out_h * out_w * channels
            req_l2 = 2 * (slab_input_bytes + slab_output_bytes)

            if req_l2 <= l2_budget:
                config.l3_tile_h = slab_out_h
                config.l3_tile_h_in = slab_in_h
                config.num_l3_tiles = num_slabs
                config.l2_buffer_size = req_l2
                target_out_h = slab_out_h
                break
            num_slabs += 1
            if num_slabs > out_h:
                return None

    # step : L1 tiling within each L2 slab
    found = False
    for out_tile_h in range(target_out_h, 0, -1):
        for out_tile_w in range(out_w, 0, -1):
            in_tile_h = (out_tile_h - 1) * stride + kernel_size
            in_tile_w = (out_tile_w - 1) * stride + kernel_size

            if in_tile_h > config.l3_tile_h_in:
                in_tile_h = config.l3_tile_h_in
            if in_tile_w > in_w + 2 * padding:
                in_tile_w = in_w + 2 * padding

            input_bytes = in_tile_h * in_tile_w * channels
            output_bytes = out_tile_h * out_tile_w * channels
            total_l1 = 2 * (input_bytes + output_bytes)

            if total_l1 <= l1_budget:
                config.tile_h = in_tile_h
                config.tile_w = in_tile_w
                config.out_tile_h = out_tile_h
                config.out_tile_w = out_tile_w
                config.num_tiles_h = math.ceil(target_out_h / out_tile_h)
                config.num_tiles_w = math.ceil(out_w / out_tile_w)
                config.num_tiles = config.num_tiles_h * config.num_tiles_w
                config.l1_input_bytes = input_bytes
                config.l1_output_bytes = output_bytes
                config.l1_buffer_bytes = total_l1
                found = True
                break
        if found:
            break

    if not found:
        return None

    return config


def calculate_maxpool_tile_size(
    in_h, in_w, channels,
    kernel_size=2, stride=2, padding=0,
    l1_budget=None, l2_budget=None
):
    """Calculate L1/L3 tile configuration for MaxPool operation.

    MaxPool reduces spatial dimensions: out_h = (in_h + 2*pad - k) / stride + 1
    No weights, just input/output double-buffering.

    L3 Tiling: If activations exceed L2 budget, split into height slabs.
    L1 Tiling: Within each L2 slab, tile for L1 double-buffering.
    """
    if l1_budget is None:
        l1_budget = GAP9HardwareModel.get_l1_budget()
    if l2_budget is None:
        l2_budget = GAP9HardwareModel.get_l2_tiling_budget()

    config = MaxPoolTileConfig()
    return _calculate_pool_tile_size_impl(
        in_h, in_w, channels,
        kernel_size, stride, padding,
        l1_budget, l2_budget,
        config
    )


def calculate_avgpool_tile_size(
    in_h, in_w, channels,
    kernel_size=2, stride=2, padding=0,
    l1_budget=None, l2_budget=None
):
    """Calculate L1/L3 tile configuration for AvgPool operation.

    Same tiling logic as MaxPool - only the kernel computation differs.

    L3 Tiling: If activations exceed L2 budget, split into height slabs.
    L1 Tiling: Within each L2 slab, tile for L1 double-buffering.
    """
    if l1_budget is None:
        l1_budget = GAP9HardwareModel.get_l1_budget()
    if l2_budget is None:
        l2_budget = GAP9HardwareModel.get_l2_tiling_budget()

    config = AvgPoolTileConfig()
    return _calculate_pool_tile_size_impl(
        in_h, in_w, channels,
        kernel_size, stride, padding,
        l1_budget, l2_budget,
        config
    )


def calculate_globalavgpool_tile_size(
    in_h, in_w, channels, batch=1,
    l1_budget=None
):
    """Calculate L1 tile configuration for GlobalAvgPool operation.

    GlobalAvgPool reduces [B, C, H, W] -> [B, C, 1, 1].
    We tile over spatial dimensions, accumulating partial sums in L1.
    Output is tiny (just channels), so we focus on input tiling.
    """
    if l1_budget is None:
        l1_budget = GAP9HardwareModel.get_l1_budget()

    config = GlobalAvgPoolTileConfig()

    # Output is always [batch, channels] = batch * channels bytes
    output_bytes = batch * channels

    # Need space for partial sums (INT32 for accumulation)
    partial_sum_bytes = batch * channels * 4  # 4 bytes per INT32

    # Try to find largest input tile that fits
    for tile_h in range(in_h, 0, -1):
        for tile_w in range(in_w, 0, -1):
            input_bytes = tile_h * tile_w * channels * batch

            # Double-buffering for input + partial sums (shared) + output
            total_l1 = 2 * input_bytes + partial_sum_bytes + output_bytes

            if total_l1 <= l1_budget:
                config.tile_h = tile_h
                config.tile_w = tile_w
                config.in_h = in_h
                config.in_w = in_w
                config.num_tiles_h = math.ceil(in_h / tile_h)
                config.num_tiles_w = math.ceil(in_w / tile_w)
                config.num_tiles = config.num_tiles_h * config.num_tiles_w
                config.l1_input_bytes = input_bytes
                config.l1_output_bytes = output_bytes
                config.l1_partial_sum_bytes = partial_sum_bytes
                return config

    return None


# --- Element-wise Operation Tiling (ReLU, GELU, Requantize, etc.) ---

class ElementwiseTileConfig(TileConfig):
    """Tile configuration for in-place element-wise operations (ReLU, GELU, Requantize).

    These operations process data element-by-element with 1:1 input:output ratio.
    For in-place ops, we use double-buffering with the same buffer for input/output.
    """
    def __init__(self):
        self.tile_size = 0           # Elements per tile
        self.num_tiles = 0           # Total number of tiles
        self.l1_buffer_bytes = 0     # L1 buffer size (2 x tile_size for double-buffering)


class AddTileConfig(TileConfig):
    """Tile configuration for binary element-wise operations (Add).

    These operations need two input buffers and one output buffer.
    Double-buffering: 2 x (input1 + input2 + output) = 6 x tile_size
    """
    def __init__(self):
        self.tile_size = 0           # Elements per tile
        self.num_tiles = 0           # Total number of tiles
        self.l1_buffer_bytes = 0     # L1 buffer size


class ConcatTileConfig(TileConfig):
    """Tile configuration for Concat operation.

    Concat copies multiple input tensors into a single output tensor along channel dim.
    We tile along the spatial dimensions (HxW), processing all channels for each tile.
    """
    def __init__(self):
        self.tile_size = 0           # Spatial elements per tile (HxW chunk)
        self.num_tiles = 0           # Total number of tiles
        self.l1_buffer_bytes = 0     # L1 buffer size


def calculate_elementwise_tile_size(
    num_elements: int,
    l1_budget: int = None,
    in_place: bool = True,
    hint_tile_size: int = None
) -> ElementwiseTileConfig:
    """Calculate L1 tile configuration for element-wise operations.

    For in-place operations (ReLU, GELU), we need:
        2 x tile_size (double-buffering, same buffer for in/out)

    For separate input/output (Requantize), we need:
        2 x (input_tile + output_tile) = 4 x tile_size

    Args:
        num_elements: Total number of elements to process
        l1_budget: L1 memory budget in bytes (default: hardware model)
        in_place: If True, input/output share same buffer
        hint_tile_size: Optional hint for tile size (from auto-tuner)

    Returns:
        ElementwiseTileConfig with tile parameters, or None if cannot fit
    """
    if l1_budget is None:
        l1_budget = GAP9HardwareModel.get_l1_budget()

    config = ElementwiseTileConfig()

    # Calculate maximum tile size based on double-buffering requirements
    if in_place:
        # In-place: 2 buffers x tile_size bytes
        max_tile_size = l1_budget // 2
    else:
        # Separate: 2 x (input + output) = 4 x tile_size bytes
        max_tile_size = l1_budget // 4

    # Use hint if provided and valid
    if hint_tile_size and hint_tile_size <= max_tile_size and hint_tile_size <= num_elements:
        tile_size = hint_tile_size
    else:
        # Clamp to actual number of elements
        tile_size = min(num_elements, max_tile_size)

    # Ensure we can fit at least 1 element (should always be true)
    if tile_size < 1:
        return None

    # Calculate number of tiles
    num_tiles = math.ceil(num_elements / tile_size)

    config.tile_size = tile_size
    config.num_tiles = num_tiles
    if in_place:
        config.l1_buffer_bytes = 2 * tile_size
    else:
        config.l1_buffer_bytes = 4 * tile_size

    return config


def calculate_add_tile_size(
    num_elements: int,
    l1_budget: int = None
) -> AddTileConfig:
    """Calculate L1 tile configuration for Add operation.

    Add needs two input buffers and one output buffer (or in-place to one input).
    For double-buffering with separate output:
        2 x (input1 + input2 + output) = 6 x tile_size bytes

    For in-place (output overwrites input1):
        2 x (input1 + input2) = 4 x tile_size bytes

    We use in-place mode since Add typically overwrites one of its inputs.

    Args:
        num_elements: Total number of elements to process
        l1_budget: L1 memory budget in bytes

    Returns:
        AddTileConfig with tile parameters, or None if cannot fit
    """
    if l1_budget is None:
        l1_budget = GAP9HardwareModel.get_l1_budget()

    config = AddTileConfig()

    # In-place Add: 2 x (input1 + input2) = 4 x tile_size
    # Using 4x because we need to double-buffer both inputs
    max_tile_size = l1_budget // 4

    tile_size = min(num_elements, max_tile_size)

    if tile_size < 1:
        return None

    num_tiles = math.ceil(num_elements / tile_size)

    config.tile_size = tile_size
    config.num_tiles = num_tiles
    config.l1_buffer_bytes = 4 * tile_size  # 2 x (input1 + input2)

    return config


def calculate_concat_tile_size(
    num_inputs: int,
    total_channels: int,
    spatial_size: int,  # H x W
    l1_budget: int = None
) -> ConcatTileConfig:
    """Calculate L1 tile configuration for Concat operation.

    Concat processes multiple input tensors, concatenating along channel dimension.
    We tile along spatial dimensions, processing all channels for each spatial tile.

    For double-buffering:
        2 x (sum of input tiles + output tile) per spatial chunk
        = 2 x (total_channels x spatial_tile_size)

    Args:
        num_inputs: Number of input tensors
        total_channels: Sum of all input channels (= output channels)
        spatial_size: Total spatial elements (H x W)
        l1_budget: L1 memory budget in bytes

    Returns:
        ConcatTileConfig with tile parameters, or None if cannot fit
    """
    if l1_budget is None:
        l1_budget = GAP9HardwareModel.get_l1_budget()

    config = ConcatTileConfig()

    # We need to load spatial tiles from all inputs and write to output
    # For double-buffering: 2 x (all_channels x spatial_tile)
    # This simplifies to: 2 x total_channels x spatial_tile_size bytes
    max_spatial_tile = l1_budget // (2 * total_channels)

    spatial_tile_size = min(spatial_size, max_spatial_tile)

    if spatial_tile_size < 1:
        return None

    num_tiles = math.ceil(spatial_size / spatial_tile_size)

    config.tile_size = spatial_tile_size
    config.num_tiles = num_tiles
    config.l1_buffer_bytes = 2 * total_channels * spatial_tile_size

    return config


class LayerNormTileConfig(TileConfig):
    """Tile configuration for LayerNorm operation.

    LayerNorm normalizes across the last dimension (normalized_dim).
    We tile by tokens (each token = normalized_dim elements).
    Weights (gamma, beta) are FP32 and shared across all tokens.

    L1 requirements:
        - Data (double-buffered): 4 x tokens_per_batch x normalized_dim bytes
        - Weights (loaded once): 2 x normalized_dim x 4 bytes (gamma + beta, FP32)
    """
    def __init__(self):
        self.tokens_per_batch = 0    # Number of tokens per tile
        self.normalized_dim = 0      # Elements per token (embed_dim)
        self.num_tiles = 0           # Total number of token batches
        self.l1_data_bytes = 0       # L1 buffer for input/output data (double-buffered)
        self.l1_weight_bytes = 0     # L1 buffer for weights (gamma + beta, loaded once)
        self.l1_buffer_bytes = 0     # Total L1 bytes


def calculate_layernorm_tile_size(
    num_tokens: int,
    normalized_dim: int,
    l1_budget: int = None,
    hint_tile_tokens: int = None
) -> LayerNormTileConfig:
    """Calculate L1 tile configuration for LayerNorm operation.

    LayerNorm normalizes each token (vector of normalized_dim elements) independently.
    We tile by tokens since we cannot split within a token's normalization.

    L1 requirements:
        - Data buffers (double-buffered): 4 x tokens_per_batch x normalized_dim
          (2x for ping-pong, 2x for input+output)
        - Weight buffers (not double-buffered): 2 x normalized_dim x 4
          (gamma and beta, both FP32)

    Args:
        num_tokens: Number of tokens to normalize
        normalized_dim: Dimension of each token (embed_dim)
        l1_budget: L1 memory budget in bytes
        hint_tile_tokens: Optional hint for tokens per batch (from auto-tuner)

    Returns:
        LayerNormTileConfig with tile parameters, or None if cannot fit
    """
    if l1_budget is None:
        l1_budget = GAP9HardwareModel.get_l1_budget()

    config = LayerNormTileConfig()
    config.normalized_dim = normalized_dim

    # Weight buffer: gamma (FP32) + beta (FP32)
    weight_bytes = 2 * normalized_dim * 4  # 2 x normalized_dim x sizeof(float)

    # Check if weights alone exceed L1
    if weight_bytes >= l1_budget:
        return None

    # Remaining budget for data (input + output, double-buffered)
    data_budget = l1_budget - weight_bytes

    # Data per token: input (INT8) + output (INT8) = 2 x normalized_dim
    bytes_per_token = 2 * normalized_dim

    # Double-buffering: 2 x bytes_per_token per token
    max_tokens_per_batch = data_budget // (2 * bytes_per_token)

    if max_tokens_per_batch < 1:
        return None

    # Use hint if provided and valid
    if hint_tile_tokens and hint_tile_tokens <= max_tokens_per_batch and hint_tile_tokens <= num_tokens:
        tokens_per_batch = hint_tile_tokens
    else:
        tokens_per_batch = min(num_tokens, max_tokens_per_batch)
    num_tiles = math.ceil(num_tokens / tokens_per_batch)

    config.tokens_per_batch = tokens_per_batch
    config.num_tiles = num_tiles
    config.l1_data_bytes = 2 * bytes_per_token * tokens_per_batch  # Double-buffered
    config.l1_weight_bytes = weight_bytes
    config.l1_buffer_bytes = config.l1_data_bytes + config.l1_weight_bytes

    return config


class Transpose2dTileConfig(TileConfig):
    """Tile configuration for Transpose_2d operation.

    Transpose_2d swaps dimensions: [B, D1, D2] → [B, D2, D1]
    We tile along D2 (input's last dim) which becomes contiguous in output.

    L1 requirements:
        - Input tile: D1 x tile_d2 bytes
        - Output tile: tile_d2 x D1 bytes (same size, different layout)
        - Double-buffered: 4 x D1 x tile_d2 bytes
    """
    def __init__(self):
        self.tile_d2 = 0             # Tile size along D2 (input's last dim)
        self.dim1 = 0                # Full D1 dimension (not tiled)
        self.dim2 = 0                # Full D2 dimension (tiled)
        self.num_tiles = 0           # Number of tiles
        self.l1_buffer_bytes = 0     # Total L1 bytes


def calculate_transpose2d_tile_size(
    dim1: int,
    dim2: int,
    l1_budget: int = None
) -> Transpose2dTileConfig:
    """Calculate L1 tile configuration for Transpose_2d operation.

    Transpose swaps [B, D1, D2] → [B, D2, D1].
    We tile along D2 (input) which becomes the first spatial dimension in output.

    L1 requirements:
        - For each tile: input [D1, tile_d2] + output [tile_d2, D1]
        - Double-buffered: 2 x (D1 x tile_d2 + tile_d2 x D1) = 4 x D1 x tile_d2

    Args:
        dim1: First dimension (not tiled, kept whole)
        dim2: Second dimension (tiled)
        l1_budget: L1 memory budget in bytes

    Returns:
        Transpose2dTileConfig with tile parameters, or None if cannot fit
    """
    if l1_budget is None:
        l1_budget = GAP9HardwareModel.get_l1_budget()

    config = Transpose2dTileConfig()
    config.dim1 = dim1
    config.dim2 = dim2

    # Each tile: input + output, both are D1 x tile_d2 bytes (INT8)
    # Double-buffered: 2 x 2 x D1 x tile_d2 = 4 x D1 x tile_d2
    bytes_per_tile_element = 4 * dim1  # 4x for double-buffering input+output

    # Maximum tile_d2 that fits in L1
    max_tile_d2 = l1_budget // bytes_per_tile_element

    if max_tile_d2 < 1:
        return None

    tile_d2 = min(dim2, max_tile_d2)
    num_tiles = math.ceil(dim2 / tile_d2)

    config.tile_d2 = tile_d2
    config.num_tiles = num_tiles
    config.l1_buffer_bytes = 4 * dim1 * tile_d2

    return config


# --- NE16 Depthwise 3x3 Spatial Tiling ---

class NE16DepthwiseTileConfig(TileConfig):
    """Tile configuration for NE16 depthwise 3x3 convolution with spatial tiling.

    NE16 depthwise produces INT32 output, requiring SW requantization.
    We tile along the height dimension to fit large activations in L1.

    L1 layout (double-buffered):
        [input_A (U8)][output_A (S32)][input_B (U8)][output_B (S32)][weights][bias]

    Weights and bias are loaded once and shared across all tiles.
    """
    def __init__(self):
        self.spatial_tiling_enabled = False  # True if tiling needed
        self.tile_h_out = 0                  # Output height per tile
        self.tile_h_in = 0                   # Input height with halo (tile_h_out + 2)
        self.num_tiles = 1                   # Number of spatial tiles
        self.l1_input_tile_bytes = 0         # Input tile size (U8)
        self.l1_output_tile_bytes = 0        # Output tile size (S32)
        self.l1_weight_bytes = 0             # Packed weight size
        self.l1_bias_bytes = 0               # Bias size (INT32)
        self.l1_total_bytes = 0              # Total L1 usage


def calculate_ne16_depthwise_tile_size(
    in_h: int,
    in_w: int,
    channels: int,
    pad_h: int = 1,
    pad_w: int = 1,
    l1_budget: int = None
) -> NE16DepthwiseTileConfig:
    """Calculate spatial tiling for NE16 depthwise 3x3 convolution.

    NE16 depthwise 3x3 operates on the full padded input width but can be
    tiled along height. For each tile:
      - Input tile: (tile_h_in) x (in_w + 2*pad_w) x channels (U8)
      - Output tile: tile_h_out x in_w x channels (INT32 = 4 bytes)

    Halo requirement for 3x3 kernel:
      - tile_h_in = tile_h_out + 2 (kernel extends 1 row above and below)

    Memory model (double-buffered):
      - 2 x (input_tile + output_tile) + weights + bias

    Args:
        in_h: Original input height (before padding)
        in_w: Original input width (before padding)
        channels: Number of channels (same for input/output in depthwise)
        pad_h: Vertical padding (default 1 for same-padding)
        pad_w: Horizontal padding (default 1 for same-padding)
        l1_budget: L1 memory budget in bytes

    Returns:
        NE16DepthwiseTileConfig with tiling parameters
    """
    if l1_budget is None:
        l1_budget = GAP9HardwareModel.get_l1_budget()

    config = NE16DepthwiseTileConfig()

    # Padded dimensions (NE16 expects pre-padded input)
    padded_w = in_w + 2 * pad_w
    out_h = in_h  # With pad=1, output spatial = input spatial for 3x3 stride=1
    out_w = in_w

    # Weight and bias sizes (fixed, not tiled)
    # Depthwise packed weights: ceil(channels/16) * 144 bytes
    nb_k = (channels + 15) // 16
    weight_bytes = nb_k * 8 * 3 * 3 * 2  # nb_k * 144
    bias_bytes = channels * 4  # INT32 corrected bias

    config.l1_weight_bytes = weight_bytes
    config.l1_bias_bytes = bias_bytes

    # Reserved space for weights + bias
    fixed_overhead = weight_bytes + bias_bytes

    # Available budget for double-buffered input/output tiles
    available_for_tiles = l1_budget - fixed_overhead

    if available_for_tiles <= 0:
        # Weights alone exceed L1 budget - shouldn't happen for reasonable configs
        return config

    # Calculate tile sizes for full activation (no tiling case)
    full_input_u8_size = (in_h + 2 * pad_h) * padded_w * channels
    full_output_s32_size = out_h * out_w * channels * 4

    # Check if full activation fits without tiling
    full_double_buffer = 2 * (full_input_u8_size + full_output_s32_size)
    if full_double_buffer + fixed_overhead <= l1_budget:
        config.spatial_tiling_enabled = False
        config.tile_h_out = out_h
        config.tile_h_in = in_h + 2 * pad_h
        config.num_tiles = 1
        config.l1_input_tile_bytes = full_input_u8_size
        config.l1_output_tile_bytes = full_output_s32_size
        config.l1_total_bytes = full_double_buffer + fixed_overhead
        return config

    # Need spatial tiling - find largest tile_h_out that fits
    # tile_h_in = tile_h_out + 2 (halo for 3x3 kernel)
    # Input tile size = tile_h_in x padded_w x channels (U8)
    # Output tile size = tile_h_out x out_w x channels x 4 (S32)
    #
    # Double-buffered: 2 x (input + output)

    config.spatial_tiling_enabled = True

    # Binary search for largest fitting tile_h_out
    for tile_h_out in range(out_h, 0, -1):
        tile_h_in = tile_h_out + 2  # Halo for 3x3

        input_tile_bytes = tile_h_in * padded_w * channels
        output_tile_bytes = tile_h_out * out_w * channels * 4

        double_buffer_bytes = 2 * (input_tile_bytes + output_tile_bytes)
        total_l1_needed = double_buffer_bytes + fixed_overhead

        if total_l1_needed <= l1_budget:
            config.tile_h_out = tile_h_out
            config.tile_h_in = tile_h_in
            config.num_tiles = math.ceil(out_h / tile_h_out)
            config.l1_input_tile_bytes = input_tile_bytes
            config.l1_output_tile_bytes = output_tile_bytes
            config.l1_total_bytes = total_l1_needed
            return config

    # If we get here, even a single output row doesn't fit
    # This shouldn't happen for reasonable channel counts
    return config


# --- Tiling Strategy Adapter ---


def _require_op_keys(op_spec, required_keys):
    missing = [key for key in required_keys if key not in op_spec]
    if missing:
        raise ValueError(f"Missing required op_spec keys: {missing}")


def _compute_tile_plan_parity_default(op_spec, memory_constraints=None):
    """Compute tile plan using the existing default calculators."""
    if not isinstance(op_spec, dict):
        raise ValueError("op_spec must be a dict")
    if "op_type" not in op_spec:
        raise ValueError("op_spec must include 'op_type'")

    op_type = op_spec["op_type"]
    l1_budget = None
    l2_budget = None
    if isinstance(memory_constraints, dict):
        l1_budget = memory_constraints.get("l1_budget")
        l2_budget = memory_constraints.get("l2_budget")

    config = None

    if op_type == "conv2d":
        _require_op_keys(op_spec, [
            "in_h", "in_w", "in_channels", "out_channels",
            "kernel_h", "kernel_w", "stride_h", "stride_w",
            "pad_h", "pad_w",
        ])
        config = calculate_conv2d_tile_size_with_weights(
            op_spec["in_h"], op_spec["in_w"],
            op_spec["in_channels"], op_spec["out_channels"],
            op_spec["kernel_h"], op_spec["kernel_w"],
            op_spec["stride_h"], op_spec["stride_w"],
            op_spec["pad_h"], op_spec["pad_w"],
            l1_budget=l1_budget,
            l2_budget=l2_budget,
            hint_tile_h=op_spec.get("hint_tile_h"),
            hint_tile_w=op_spec.get("hint_tile_w"),
        )

    elif op_type == "linear":
        _require_op_keys(op_spec, ["input_features", "output_features"])
        config = calculate_linear_tile_size(
            op_spec["input_features"],
            op_spec["output_features"],
            batch_size=op_spec.get("batch_size", 1),
            l1_budget=l1_budget,
            l2_budget=l2_budget,
            output_element_size=op_spec.get("output_element_size", 1),
        )

    elif op_type == "mhsa":
        _require_op_keys(op_spec, ["seq_len", "head_dim", "num_heads"])
        config = calculate_mhsa_tile_size(
            op_spec["seq_len"],
            op_spec["head_dim"],
            op_spec["num_heads"],
            l1_budget=l1_budget,
            l2_budget=l2_budget,
            hint_tile_q=op_spec.get("hint_tile_q"),
        )

    elif op_type == "maxpool":
        _require_op_keys(op_spec, ["in_h", "in_w", "channels"])
        config = calculate_maxpool_tile_size(
            op_spec["in_h"], op_spec["in_w"], op_spec["channels"],
            kernel_size=op_spec.get("kernel_size", 2),
            stride=op_spec.get("stride", 2),
            padding=op_spec.get("padding", 0),
            l1_budget=l1_budget,
            l2_budget=l2_budget,
        )

    elif op_type == "avgpool":
        _require_op_keys(op_spec, ["in_h", "in_w", "channels"])
        config = calculate_avgpool_tile_size(
            op_spec["in_h"], op_spec["in_w"], op_spec["channels"],
            kernel_size=op_spec.get("kernel_size", 2),
            stride=op_spec.get("stride", 2),
            padding=op_spec.get("padding", 0),
            l1_budget=l1_budget,
            l2_budget=l2_budget,
        )

    elif op_type == "globalavgpool":
        _require_op_keys(op_spec, ["in_h", "in_w", "channels"])
        config = calculate_globalavgpool_tile_size(
            op_spec["in_h"], op_spec["in_w"], op_spec["channels"],
            batch=op_spec.get("batch", 1),
            l1_budget=l1_budget,
        )

    else:
        raise ValueError(f"Unsupported op_type for compute_tile_plan: {op_type}")

    return config


def compute_tile_plan(op_spec, memory_constraints=None, strategy=None):
    """
    Normalized tiling strategy entrypoint.

    Behavior:
    - Route strategy selection through `codegen.tiling.strategies`.
    - Compute plans with existing default calculators.
    - Record selected/rejected strategy metadata without changing tile output.
    """
    return compute_tile_plan_with_strategy(
        op_spec=op_spec,
        memory_constraints=memory_constraints,
        strategy=strategy,
        parity_compute_fn=_compute_tile_plan_parity_default,
    )
