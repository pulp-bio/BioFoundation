# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
C Code Generator for embedded deployment

Generates production-ready C code from verified INT8 network using Mako templates.
Now includes MemoryPlanner for Arena-based L2 allocation.
Usage:
    python codegen/generate_c_code.py

Outputs:
    generated/
    ├── inc/       # Headers
    ├── src/       # Implementation
    ├── bin/       # Binary data files
    └── Makefile   # Build configuration
"""

import json
import os
import math
import re
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
from mako.template import Template
from mako.lookup import TemplateLookup
import warnings

# NE16 accelerator support
try:
    from .ne16_packing import (
        ne16_pack_linear_weights,
        ne16_pack_conv1x1_weights,
        ne16_pack_conv3x3_weights,
        ne16_pack_conv3x3_depthwise_weights,
        ne16_pack_conv3x3_depthwise_weights_with_requant,
        is_ne16_eligible_linear,
        is_ne16_eligible_conv2d,
        get_ne16_packed_weight_size,
        compute_ne16_requant_params,
    )
    NE16_PACKING_AVAILABLE = True
except ImportError:
    NE16_PACKING_AVAILABLE = False


@dataclass
class LayerBuildContext:
    """
    Mutable state passed through layer spec building.

    This context holds all the tracking state that layer handlers need to read
    and update during the _build_layer_specs() loop. Using a context object
    allows layer handlers to be extracted as separate methods.
    """
    # Current tensor state (updated by each layer)
    current_shape: List[int] = field(default_factory=list)
    current_scale: float = 1.0
    current_buffer: str = ""

    # Output tracking (layer_name -> value)
    layer_output_buffer: Dict[str, str] = field(default_factory=dict)
    layer_output_scale: Dict[str, float] = field(default_factory=dict)

    # Input tracking for parallel branches (e.g., SwiGLU W1/W3)
    layer_input_buffer: Dict[str, str] = field(default_factory=dict)
    layer_input_scale: Dict[str, float] = field(default_factory=dict)

    # Buffer management
    buffer_map: Dict[str, dict] = field(default_factory=dict)
    buffer_scale: Dict[str, float] = field(default_factory=dict)
    activation_buffers: List[dict] = field(default_factory=list)

    # Results
    specs: List[dict] = field(default_factory=list)
    param_layers: List[str] = field(default_factory=list)

    # Transformer block tracking
    block_input_shape: Dict[int, List[int]] = field(default_factory=dict)
    block_input_scale: Dict[int, float] = field(default_factory=dict)
    block_input_buffer: Dict[int, str] = field(default_factory=dict)

# Import L1 tiling support
from .gap9_model import (
    calculate_conv2d_tile_size,
    calculate_conv2d_tile_size_with_weights,  # L1 weight caching
    Conv2DTileConfig,
    calculate_linear_tile_size,
    LinearTileConfig,
    calculate_maxpool_tile_size,
    MaxPoolTileConfig,
    calculate_avgpool_tile_size,
    AvgPoolTileConfig,
    calculate_globalavgpool_tile_size,
    GlobalAvgPoolTileConfig,
    calculate_mhsa_tile_size,
    MHSATileConfig,
    # Element-wise operation tiling
    calculate_elementwise_tile_size,
    ElementwiseTileConfig,
    calculate_add_tile_size,
    AddTileConfig,
    calculate_concat_tile_size,
    ConcatTileConfig,
    # LayerNorm tiling
    calculate_layernorm_tile_size,
    LayerNormTileConfig,
    # Transpose_2d tiling
    calculate_transpose2d_tile_size,
    Transpose2dTileConfig,
    # NE16 depthwise spatial tiling
    calculate_ne16_depthwise_tile_size,
    NE16DepthwiseTileConfig,
    # Weight residency labels
    WEIGHT_RESIDENCY_L2,
    WEIGHT_RESIDENCY_L3_STAGED,
    WEIGHT_RESIDENCY_L3_TILED,
    WEIGHT_RESIDENCY_MAMBA_SCRATCH,
)

from .constants import SCALE_EPSILON
from .targets import TargetBase, available_targets, create_target
from .layers import handle_embedding as handle_embedding_v2
from .layers import handle_patchembed as handle_patchembed_v2
from .layers import handle_positionalembedding as handle_positionalembedding_v2
from .layers import handle_gelu as handle_gelu_v2
from .layers import handle_groupnorm as handle_groupnorm_v2
from .layers import handle_layernorm as handle_layernorm_v2
from .layers import handle_rmsnorm as handle_rmsnorm_v2
from .layers import handle_squeeze as handle_squeeze_v2
from .layers import handle_flatten as handle_flatten_v2
from .layers import handle_reshape as handle_reshape_v2
from .layers import handle_permute as handle_permute_v2
from .layers import handle_quant_identity as handle_quant_identity_v2
from .layers import handle_quant_relu as handle_quant_relu_v2
from .layers import handle_silu as handle_silu_v2
from .layers import handle_rfft as handle_rfft_v2
from .layers import handle_maxpool2d as handle_maxpool2d_v2
from .layers import handle_avgpool2d as handle_avgpool2d_v2
from .layers import handle_globalavgpool as handle_globalavgpool_v2
from .layers import handle_adaptive_avgpool1d as handle_adaptive_avgpool1d_v2
from .layers import handle_zeropad2d as handle_zeropad2d_v2
from .layers import handle_add as handle_add_v2
from .layers import handle_concatenate as handle_concatenate_v2
from .layers import handle_mean as handle_mean_v2
from .layers import handle_quantconv2d as handle_quantconv2d_v2
from .layers import handle_conv1d_depthwise as handle_conv1d_depthwise_v2
from .layers import handle_quantlinear as handle_quantlinear_v2
from .layers import handle_alternating_attention as handle_alternating_attention_v2
from .layers import handle_ssm as handle_ssm_v2
from .layers import handle_mambablock as handle_mambablock_v2
from .layers import handle_mambawrapper as handle_mambawrapper_v2

# Import optimization knowledge base (auto-tuning integration)
try:
    from .optimization import KnowledgeBase, extract_shape_from_layer
    KB_AVAILABLE = True
except ImportError:
    KB_AVAILABLE = False
    KnowledgeBase = None
    extract_shape_from_layer = None

# Import checkpoint export/replay helpers
try:
    from .checkpoints import (
        CHECKPOINT_STAGE_POST_FUSION,
        CHECKPOINT_STAGE_POST_MEMORY_PLAN,
        CHECKPOINT_STAGE_POST_TILING,
        CHECKPOINT_STAGE_PRE_FUSION,
        CheckpointManager,
    )
    CHECKPOINTS_AVAILABLE = True
except ImportError:
    CHECKPOINTS_AVAILABLE = False
    CheckpointManager = None
    CHECKPOINT_STAGE_PRE_FUSION = "pre_fusion"
    CHECKPOINT_STAGE_POST_FUSION = "post_fusion"
    CHECKPOINT_STAGE_POST_TILING = "post_tiling"
    CHECKPOINT_STAGE_POST_MEMORY_PLAN = "post_memory_plan"

# Import extracted fusion module
try:
    from .optimization.fusion import (
        build_default_registry as build_fusion_registry_v2,
        detect_fusion_opportunities as detect_fusion_opportunities_v2,
        transform_specs_for_fusion as transform_specs_for_fusion_v2,
        validate_default_registry as validate_fusion_registry_v2,
        validate_fusions as validate_fusions_v2,
        write_fusion_report as write_fusion_report_v2,
    )
    FUSION_MODULE_AVAILABLE = True
except ImportError:
    FUSION_MODULE_AVAILABLE = False
    build_fusion_registry_v2 = None
    detect_fusion_opportunities_v2 = None
    transform_specs_for_fusion_v2 = None
    validate_fusion_registry_v2 = None
    validate_fusions_v2 = None
    write_fusion_report_v2 = None


class MemoryPlanner:
    """Compatibility adapter around policy-based planner implementations."""

    def __init__(
        self,
        specs,
        activation_buffers,
        shared_pool,
        policy: str = "arena_first_fit",
        allow_experimental: bool = False,
    ):
        self.specs = specs
        self.activation_buffers = activation_buffers
        self.shared_pool = shared_pool
        self.policy = policy
        self.allow_experimental = allow_experimental

        self.backend_name = None
        self.result = None
        self.lifetimes = {}
        self.offsets = {}
        self.total_size = 0
        self.unresolved_conflicts = []

    def analyze(self):
        """Run selected planner policy and expose planner result fields."""
        from .memory.planners import create_planner

        planner_impl = create_planner(self.policy, allow_experimental=self.allow_experimental)
        self.backend_name = planner_impl.policy_name
        self.result = planner_impl.plan(
            specs=self.specs,
            activation_buffers=self.activation_buffers,
            shared_pool=self.shared_pool,
        )

        self.lifetimes = self.result.lifetimes
        self.offsets = self.result.offsets
        self.total_size = self.result.total_size
        self.unresolved_conflicts = self.result.unresolved_conflicts


class CCodeGenerator:
    """
    Generate production-ready C code from INT8 network.

    This class orchestrates the full code generation pipeline:

    1. **Binary Generation**: Converts numpy weights/biases to .bin files
       for HyperFlash storage (L3 → L2 → L1 streaming).

    2. **Memory Planning**: Uses MemoryPlanner to assign static L2 arena
       offsets, eliminating dynamic allocation for activations.

    3. **Tiling Strategy**: Determines per-layer execution strategy:
       - L1 Tiled: Fits in L1 with double buffering (fastest)
       - L2 Fallback: Compute directly on L2 (medium)
       - L3 Staged: Stream weights from L3 (for large models)

    4. **Template Rendering**: Uses Mako templates to generate:
       - src/main.c: Entry point and cluster dispatch
       - src/network.c: Layer-by-layer execution logic
       - inc/*.h: Headers and data declarations
       - Makefile: Build configuration

    The generated code uses a data-driven executor that iterates over
    a `network_layers[]` array of LayerSpec structs, dispatching to
    the appropriate kernel based on layer type.

    Attributes:
        network_info: Extracted metadata from BrevitasExtractor
        layer_info: Per-layer configuration (scales, shapes, etc.)
        target: active target object used for memory and capability decisions
        planner: MemoryPlanner for L2 arena allocation
    """

    def __init__(
        self,
        network_info_path="golden_outputs/network_info.json",
        weights_dir="golden_outputs/weights",
        test_case_dir="golden_outputs/test_cases/test_case_3",
        output_dir="generated",
        target: Optional["TargetBase"] = None,
        target_name: Optional[str] = None,
        enable_l1_tiling=True,  # Enable L1 DMA tiling for Conv2D operations
        l1_budget_bytes=None,
        enable_fusion=True,  # Enable cross-layer fusion optimizations
        board_mode=False,  # Board-ready code: minimal prints, no golden checks, just cycles
        disable_l1_weight_caching=False,  # Disable L1 weight caching for baseline benchmarks
        layer_config_overrides=None,  # Per-layer tile config overrides (dict or path to JSON)
        int8_classifier_output=False,  # Output INT8 from classifier head (vs FP32 logits)
        use_hwc_layout="auto",  # "auto", True, or False - HWC layout for conv2d/pool/pad
        use_ne16="auto",  # Enable NE16: True, False, or "auto" (auto-detect beneficial layers)
        ne16_min_macs=1024,  # Minimum MACs for NE16 eligibility
        ne16_auto_threshold=100000  # Auto mode: only use NE16 for layers with >= this many MACs
    ):
        """
        Initialize code generator.

        Args:
            network_info_path: Path to network_info.json
            weights_dir: Directory containing weight/bias .npy files
            test_case_dir: Directory containing test case data
            output_dir: Output directory for generated C code
            target: Target abstraction object (overrides target_name if provided)
            target_name: Target name from registry (default: "gap9")
            enable_l1_tiling: Enable L1 memory tiling (Conv2D only)
            l1_budget_bytes: Available L1 memory (bytes). If None, uses target default.
        """
        with open(network_info_path) as f:
            self.network_info = json.load(f)
        self.layer_info = {k: v for k, v in self.network_info.items() if not k.startswith('__')}
        self.layer_names = list(self.layer_info.keys())
        self.layer_order = self.network_info.get('__layer_order__', self.layer_names)
        self.weights_dir = Path(weights_dir)
        self.test_case_dir = Path(test_case_dir)
        if not self.test_case_dir.exists():
            fallback = self.test_case_dir.parent / "test_case_1"
            if fallback.exists():
                self.test_case_dir = fallback
        self.output_dir = Path(output_dir)
        self.template_dir = Path(__file__).parent / "templates"
        if target is not None and target_name is not None:
            raise ValueError("Specify either 'target' or 'target_name', not both.")
        if target is not None:
            self.target = target
        else:
            self.target = create_target(target_name or "gap9")
        self.target.validate_required_capabilities()

        # Configuration
        self.enable_l1_tiling = enable_l1_tiling
        default_l1_budget = self.target.l1_budget_bytes
        self.l1_budget_bytes = l1_budget_bytes or default_l1_budget
        self.enable_fusion = enable_fusion
        self.board_mode = board_mode  # Board-ready: minimal prints, no golden, just cycles
        self.disable_l1_weight_caching = disable_l1_weight_caching  # Disable L1 weight caching for baseline benchmarks
        self.int8_classifier_output = int8_classifier_output  # Output INT8 from classifier (reduces memory traffic)
        # HWC layout: can be True, False, or "auto" (auto-detect based on kernel patterns)
        self.use_hwc_layout = use_hwc_layout if isinstance(use_hwc_layout, bool) else False
        # NE16 accelerator configuration
        # use_ne16 can be True, False, or "auto"
        # - True: enable NE16 for all eligible layers (>= ne16_min_macs)
        # - False: disable NE16 completely
        # - "auto": enable NE16 only for large layers (>= ne16_auto_threshold MACs)
        self.use_ne16 = use_ne16
        self.ne16_min_macs = ne16_min_macs
        self.ne16_auto_threshold = ne16_auto_threshold
        if use_ne16 and use_ne16 != "auto" and not NE16_PACKING_AVAILABLE:
            warnings.warn("NE16 requested but ne16_packing module not available. Falling back to SW kernels.")
            self.use_ne16 = False
        if use_ne16 == "auto" and not NE16_PACKING_AVAILABLE:
            self.use_ne16 = False  # Can't auto-enable without packing support
        if self.use_ne16 == "auto":
            print(f"  [NE16] Auto-detection enabled (threshold: {self.ne16_auto_threshold:,} MACs)")

        # Checkpoint export (disabled by default)
        self.checkpoint_output_dir = os.getenv("ARES_CHECKPOINT_DIR", "").strip()
        self.checkpoint_tag = os.getenv("ARES_CHECKPOINT_TAG", "").strip()
        self.checkpoint_manager = None
        self._checkpoint_written_stages = set()

        # Data structures
        self.binary_files = []
        self.weight_entries = {}
        self.bias_entries = {}
        self.intermediate_entries = []
        self.intermediate_layer_entries = {}
        self.input_entry = None
        self.golden_entry = None
        self.ssm_entries = []  # SSM layer parameter entries
        self.mamba_block_entries = []  # MambaBlock parameter entries
        self.alt_attn_ne16_entries = []  # Alternating attention NE16 parameter entries
        self.mamba_slab_sizes = {}  # Max sizes for shared Mamba weight slab (L3 streaming)
        # L3 streamed golden validation
        self.max_golden_size = 0  # Largest intermediate golden output
        self.golden_chunk_size = 0  # Staging buffer size (capped for chunked comparison)
        # NE16 accelerator data structures
        self.ne16_weight_entries = {}  # layer_name -> packed weight binary entry
        self.ne16_bias_entries = {}    # layer_name -> corrected bias binary entry
        self.ne16_scale_entries = {}   # layer_name -> HW outquant scale (qbias) entry
        self.ne16_scale_shift_entries = {}  # layer_name -> HW outquant scale_shift (qnorm) entry
        self.ne16_eligible_layers = set()  # Set of layer names using NE16
        self.total_golden_size = 0  # Total size of all intermediate goldens
        self.use_streamed_golden = False  # Auto-enabled when total exceeds threshold
        self.ssm_dt_scale_q = {}  # I-Mamba step 6: dt_acc to Q16.16 scale factors
        self.ssm_dt_scale_shift = {}  # I-Mamba step 6: dt_acc to Q16.16 shift amounts
        self.ssm_bc_scale_factor = {}  # I-Mamba step 8: B/C to Q15 scale factors
        self.ssm_output_scale_q = {}  # I-Mamba step 9: Output conversion scale factors

        # Metadata
        self.input_shape = None
        self.input_numel = None
        # Detect input quantization layer(s) - may be 'input_quant' or branch-specific like 'eeg_input_quant'
        self.input_quant_layers = self._detect_input_quant_layers()
        self.input_scale = self._get_primary_input_scale()
        self.final_linear_name = self._determine_final_linear()
        self.layer_specs = []
        self.activation_buffers = []
        self.param_layers = []
        self.num_classes = None
        self.total_macs = 0  # Total multiply-accumulate operations for the network
        self._metadata_ready = False
        self.layer_symbol_counts = {}
        self.buffer_symbol_counts = {}
        self.binary_symbol_counts = {}

        # Memory Management
        self.block_activation_buffers = {} 
        self.shared_activation_pool = []    
        self.block_buffer_role_map = {}     
        self.l2_arena_size = 0  # Total size of L2 arena
        self.planner = None     # MemoryPlanner instance
        self.planner_policy = "arena_first_fit"
        self.planner_debug_dump_path = None

        # Reporting
        self.l1_tiled_layers = []
        self.l2_fallback_layers = []
        self.fused_layers = []
        self.memory_hierarchy = {}
        self.buffer_memory_annotations = {}
        self.memory_level_report = {}
        self.memory_level_report_path = None
        self.fusion_report_path = None
        self._memory_levels_ready = False

        # Optimization Knowledge Base integration
        self.knowledge_base = None
        self.optimization_decisions = []  # Track KB lookups and decisions
        if KB_AVAILABLE:
            try:
                self.knowledge_base = KnowledgeBase()
                print(f"  [KB] Loaded knowledge base with {len(self.knowledge_base)} entries")
            except Exception as e:
                print(f"  [KB] Warning: Could not load knowledge base: {e}")

        # Layer config overrides (for auto-tuning)
        self.layer_config_overrides = {}
        if layer_config_overrides:
            if isinstance(layer_config_overrides, dict):
                self.layer_config_overrides = layer_config_overrides
            elif isinstance(layer_config_overrides, (str, Path)):
                override_path = Path(layer_config_overrides)
                if override_path.exists():
                    with open(override_path) as f:
                        self.layer_config_overrides = json.load(f)
                    print(f"  [TUNE] Loaded {len(self.layer_config_overrides)} layer config overrides")

        if self.layer_config_overrides:
            for layer_name, config in self.layer_config_overrides.items():
                print(f"    - {layer_name}: {config}")

        # Auto-detect HWC layout if use_hwc_layout is "auto"
        if use_hwc_layout == "auto":
            self.use_hwc_layout = self._should_use_hwc_layout()
            if self.use_hwc_layout:
                print(f"  [AUTO] Selected HWC layout (detected 1D-style conv kernels)")
            else:
                print(f"  [AUTO] Selected CHW layout (standard 2D convolutions)")

        self._setup_checkpointing()

    def _get_l2_tiling_budget_bytes(self) -> int:
        """Return target-specific L2 tiling budget (required for tile planners)."""
        budget = self.target.l2_tiling_budget_bytes
        if budget is None:
            raise ValueError(f"Target '{self.target.name}' must define l2_tiling_bytes.")
        return budget

    def _get_l1_total_bytes_for_allocation(self) -> int:
        """Return allocator-visible L1 total used by generated network.c."""
        total = self.target.memory.l1_total_bytes
        if total is not None:
            return total
        raise ValueError(
            f"Target '{self.target.name}' missing memory.l1_total_bytes. "
            f"Add l1_total_bytes to the target's MemoryCapabilities."
        )

    def _get_l2_total_bytes_for_fallback(self) -> int:
        """Return total L2 bytes used by activation fallback heuristics."""
        total = self.target.memory.l2_total_bytes
        if total is not None:
            return total
        fallback = self.target.l2_budget_bytes
        if fallback is not None:
            return fallback
        raise ValueError(
            f"Target '{self.target.name}' missing memory.l2_total_bytes and l2_budget_bytes. "
            f"Add l2_total_bytes to the target's MemoryCapabilities."
        )

    def _get_l2_activation_reserved_bytes_for_fallback(self) -> int:
        """Return reserved bytes for activation fallback decisions."""
        reserved = self.target.memory.l2_activation_reserved_bytes
        if reserved is not None:
            return reserved
        l2_total = self._get_l2_total_bytes_for_fallback()
        l2_tiling = self._get_l2_tiling_budget_bytes()
        if l2_total > l2_tiling:
            return l2_total - l2_tiling
        return 0

    def _get_l3_fallback_single_buffer_threshold_bytes(self) -> int:
        """Return threshold for moving one oversized activation buffer to L3."""
        threshold = self.target.memory.l3_fallback_single_buffer_threshold_bytes
        if threshold is not None:
            return threshold
        raise ValueError(
            f"Target '{self.target.name}' missing memory.l3_fallback_single_buffer_threshold_bytes. "
            f"Add l3_fallback_single_buffer_threshold_bytes to the target's MemoryCapabilities."
        )

    def _determine_weight_residency(
        self,
        weight_size_bytes: int,
        layer_type: str,
        memory_tier: Optional[str] = None,
        uses_mamba_scratch: bool = False,
    ) -> str:
        """Route weight residency decisions through the active target object."""
        return self.target.determine_weight_residency(
            weight_size_bytes=weight_size_bytes,
            layer_type=layer_type,
            memory_tier=memory_tier,
            uses_mamba_scratch=uses_mamba_scratch,
        )

    def _should_use_hwc_layout(self) -> bool:
        """
        Auto-detect whether HWC layout would be beneficial for this network.

        HWC is beneficial when:
        - Network uses predominantly 1xK or Kx1 kernels (1D-style convolutions)
        - Small input channel counts (≤32) where SIMD over channels is limited
        - Network has NE16-eligible 3x3 convolutions (NE16 3x3 outputs HWC)

        HWC is safe when:
        - Network uses GlobalAvgPool before Linear (reduces to 1x1 spatial, layout irrelevant)

        HWC is NOT used when:
        - Network has Flatten followed by Linear with spatial dimensions >1x1
          (weights trained with NCHW order from flattened spatial tensor)

        Returns:
            True if HWC layout is recommended, False for CHW
        """
        layer_names = list(self.layer_info.keys())

        # Check if network has GlobalAvgPool/AdaptiveAvgPool before Linear
        # This makes HWC safe because 1x1 spatial means layout doesn't matter
        has_global_pool_before_linear = False
        for i, layer_name in enumerate(layer_names):
            layer_type = self.layer_info[layer_name].get('type', '')
            if 'GlobalAvgPool' in layer_type or 'AdaptiveAvgPool' in layer_type:
                # Check if any subsequent layer is Linear
                for j in range(i + 1, len(layer_names)):
                    next_type = self.layer_info[layer_names[j]].get('type', '')
                    if 'Linear' in next_type:
                        has_global_pool_before_linear = True
                        break
                if has_global_pool_before_linear:
                    break

        # Check if network has Flatten -> Linear pattern (incompatible with HWC)
        # Linear layer weights are trained with NCHW-ordered inputs from Flatten
        has_flatten_linear = False
        for i, layer_name in enumerate(layer_names):
            layer_type = self.layer_info[layer_name].get('type', '')
            if 'Flatten' in layer_type and i + 1 < len(layer_names):
                next_type = self.layer_info[layer_names[i + 1]].get('type', '')
                if 'Linear' in next_type:
                    has_flatten_linear = True
                    break

        # Check for spatial-to-linear pattern (conv/pool output directly to linear without GlobalAvgPool)
        has_spatial_linear = self._has_spatial_to_linear_pattern() and not has_global_pool_before_linear

        if has_flatten_linear or has_spatial_linear:
            print(f"  [HWC] Disabled: Flatten->Linear or spatial->Linear pattern detected")
            return False

        # Collect conv layer info
        conv_layers = []
        ne16_3x3_candidates = 0
        for layer_name, layer_info in self.layer_info.items():
            layer_type = layer_info.get('type', '')
            if 'Conv2d' in layer_type:
                kernel = layer_info.get('kernel_size', [1, 1])
                if isinstance(kernel, int):
                    kernel = [kernel, kernel]
                in_ch = layer_info.get('in_channels', 1)
                out_ch = layer_info.get('out_channels', 1)
                stride = layer_info.get('stride', [1, 1])
                if isinstance(stride, int):
                    stride = [stride, stride]

                conv_layers.append({
                    'name': layer_name,
                    'kernel_h': kernel[0],
                    'kernel_w': kernel[1],
                    'in_ch': in_ch
                })

                # Count NE16 3x3 candidates (3x3 kernel, stride=1, sufficient MACs)
                if kernel[0] == 3 and kernel[1] == 3 and stride[0] == 1 and stride[1] == 1:
                    output_shape = layer_info.get('output_shape', [1, out_ch, 7, 7])
                    out_h = output_shape[2] if len(output_shape) > 2 else 7
                    out_w = output_shape[3] if len(output_shape) > 3 else 7
                    total_macs = in_ch * out_ch * 9 * out_h * out_w
                    if total_macs >= 100000:  # NE16 auto threshold
                        ne16_3x3_candidates += 1

        if not conv_layers:
            return False  # No conv layers, default to CHW

        # Count 1D-style kernels (1xK or Kx1)
        one_d_count = sum(1 for c in conv_layers if c['kernel_h'] == 1 or c['kernel_w'] == 1)
        two_d_count = len(conv_layers) - one_d_count

        # Count small channel convs (where HWC helps most)
        small_ch_count = sum(1 for c in conv_layers if c['in_ch'] <= 32)

        # Use HWC if:
        # 1. Majority (>50%) of conv layers are 1D-style, OR
        # 2. All conv layers have small channels (≤32) and at least some are 1D-style
        _ = ne16_3x3_candidates
        if one_d_count > two_d_count:
            return True
        if small_ch_count == len(conv_layers) and one_d_count > 0:
            return True

        return False

    def _has_spatial_to_linear_pattern(self) -> bool:
        """Check if network has spatial layers feeding into Linear (requires CHW layout).

        This detects networks where:
        - There are spatial layers (conv, pool) that produce 2D+ output
        - These feed into a Linear layer (either directly or via Flatten)
        - The Linear weights were trained with NCHW-ordered flattened inputs

        In such cases, NE16 3x3 (which outputs HWC) would corrupt the data layout.
        """
        has_spatial_layer = False
        has_linear_layer = False

        for layer_name, layer_data in self.layer_info.items():
            layer_type = layer_data.get('type', '')
            # Spatial layers that produce HW dimensions
            if any(s in layer_type for s in ['Conv2d', 'Pool', 'MaxPool', 'AvgPool']):
                has_spatial_layer = True
            # Linear layers that consume flattened spatial output
            if 'Linear' in layer_type:
                # Check if input is multi-dimensional (flattened from spatial)
                in_features = layer_data.get('in_features', 0)
                # If in_features > 1024 and we have spatial layers, likely taking flattened spatial input
                if in_features > 64 and has_spatial_layer:
                    has_linear_layer = True

        return has_spatial_layer and has_linear_layer

    def _should_use_ne16_for_layer(self, layer_type: str, in_features: int, out_features: int,
                                    kernel_size: tuple = None, spatial_size: tuple = None,
                                    groups: int = 1) -> bool:
        """
        Determine if NE16 should be used for a specific layer.

        Args:
            layer_type: 'QuantLinear' or 'QuantConv2d'
            in_features: Input features (or channels for conv)
            out_features: Output features (or channels for conv)
            kernel_size: Tuple (kh, kw) for conv layers, None for linear
            spatial_size: Tuple (h, w) output spatial dimensions for conv layers
            groups: Number of groups (1=normal conv, >1=grouped/depthwise)

        Returns:
            True if NE16 should be used for this layer
        """
        if not NE16_PACKING_AVAILABLE:
            return False

        # Explicit disable
        if self.use_ne16 is False:
            return False

        if layer_type == 'QuantLinear':
            if not self.target.supports_ne16_linear():
                return False
        elif layer_type == 'QuantConv2d':
            if kernel_size is None:
                return False
            if not self.target.supports_ne16_conv2d_kernel(
                kernel_size=kernel_size,
                stride=(1, 1),  # Callers guard NE16 packing paths to stride=1.
                groups=groups,
                in_channels=in_features,
                use_hwc_layout=self.use_hwc_layout,
                memory_tier=None,
            ):
                return False
        else:
            return False

        # Calculate MACs for this layer
        # Depthwise: channels * H * W * kernel (no cross-channel accumulation)
        # Regular: in_ch * out_ch * kernel * spatial
        is_depthwise = (groups > 1) and (groups == in_features)
        if is_depthwise:
            total_macs = in_features  # channels
            if kernel_size:
                kh, kw = kernel_size
                total_macs *= kh * kw
            if spatial_size:
                sh, sw = spatial_size
                total_macs *= sh * sw
        else:
            total_macs = in_features * out_features
            if kernel_size:
                kh, kw = kernel_size
                total_macs *= kh * kw
            if spatial_size:
                sh, sw = spatial_size
                total_macs *= sh * sw

        # Minimum MACs threshold for NE16 depthwise (overhead dominates for small layers)
        NE16_DEPTHWISE_MIN_MACS = 200000  # 200K MACs

        # Helper to check if NE16 depthwise can be executed (with spatial tiling if needed)
        def depthwise_fits_l1(channels, spatial_h, spatial_w, pad_h=1, pad_w=1):
            """Check if NE16 depthwise can fit in L1 (with tiling if needed).

            For large layers requiring many tiles, SW depthwise is more efficient
            because the tile loading (S8->U8 conversion) and storing (INT32->INT8 requant)
            overhead negates the NE16 advantage. Only use NE16 when tiling is minimal.
            """
            # Use the tiling calculator to check feasibility
            config = calculate_ne16_depthwise_tile_size(
                spatial_h, spatial_w, channels, pad_h, pad_w
            )
            if config is None or config.tile_h_out <= 0:
                return False

            # If spatial tiling is not needed (single tile), NE16 is efficient
            if not config.spatial_tiling_enabled:
                return True

            # For tiled execution, only use NE16 if num_tiles <= 4
            # Beyond that, SW depthwise is more efficient due to tile overhead
            MAX_EFFICIENT_TILES = 4
            if config.num_tiles > MAX_EFFICIENT_TILES:
                return False

            return True

        # Explicit enable: use standard eligibility check
        if self.use_ne16 is True:
            if layer_type == 'QuantLinear':
                return is_ne16_eligible_linear(in_features, out_features,
                                               self.ne16_min_macs, allow_tiling=True)
            elif layer_type == 'QuantConv2d':
                # Depthwise 3x3: check MAC threshold AND L1 fit (no tiling support)
                if is_depthwise and kernel_size == (3, 3):
                    if total_macs < NE16_DEPTHWISE_MIN_MACS:
                        return False
                    sh, sw = spatial_size if spatial_size else (7, 7)
                    return depthwise_fits_l1(in_features, sh, sw)
                # Regular conv: 1x1 or 3x3
                if kernel_size in [(1, 1), (3, 3)]:
                    sh, sw = spatial_size if spatial_size else (7, 7)
                    return is_ne16_eligible_conv2d(in_features, out_features, kernel_size,
                                                   sh, sw, self.ne16_min_macs)
            return False

        # Auto mode: only enable for layers with >= ne16_auto_threshold MACs
        if self.use_ne16 == "auto":
            if total_macs < self.ne16_auto_threshold:
                return False  # Too small, SW kernels are likely faster

            # Layer meets threshold - check eligibility
            if layer_type == 'QuantLinear':
                return is_ne16_eligible_linear(in_features, out_features,
                                               self.ne16_min_macs, allow_tiling=True)
            elif layer_type == 'QuantConv2d':
                # Depthwise 3x3: check MAC threshold AND L1 fit (no tiling support)
                if is_depthwise and kernel_size == (3, 3):
                    if total_macs < NE16_DEPTHWISE_MIN_MACS:
                        return False
                    sh, sw = spatial_size if spatial_size else (7, 7)
                    return depthwise_fits_l1(in_features, sh, sw)
                # Regular conv: 1x1 or 3x3
                if kernel_size in [(1, 1), (3, 3)]:
                    sh, sw = spatial_size if spatial_size else (7, 7)
                    return is_ne16_eligible_conv2d(in_features, out_features, kernel_size,
                                                   sh, sw, self.ne16_min_macs)
            return False

        return False

    def _normalize_layer_name(self, layer_name: str) -> str:
        """
        Normalize layer name for consistent lookup across profile parser and code generator.

        Handles variations like 'layer.0.conv' vs 'layer_0_conv'.
        """
        # Keep original Python-style names (with dots) as the canonical form
        # since that's what network_info.json uses
        return layer_name.strip()

    def _get_layer_override(self, layer_name: str) -> Optional[dict]:
        """
        Get config override for a layer if one exists.

        Returns:
            dict with tile_config, kernel_config, etc. or None
        """
        if not self.layer_config_overrides:
            return None
        norm_name = self._normalize_layer_name(layer_name)
        # Try exact match first
        if norm_name in self.layer_config_overrides:
            return self.layer_config_overrides[norm_name]
        # Try matching with underscores converted to dots (for profile parser compatibility)
        alt_name = norm_name.replace('_', '.')
        return self.layer_config_overrides.get(alt_name)

    def _query_kb_for_layer(self, layer_name: str, op_type: str, shape: dict, auto_apply: bool = True) -> dict:
        """
        Query the knowledge base for optimization configuration.

        Args:
            layer_name: Name of the layer
            op_type: Operation type (e.g., "linear_int8", "conv2d_int8")
            shape: Dictionary of layer dimensions
            auto_apply: If True, automatically add KB config to layer_config_overrides

        Returns:
            dict with keys:
                - 'found': bool - whether a KB entry was found
                - 'entry': OptimizationEntry or None
                - 'source': str - "knowledge_base" or "heuristic"
                - 'confidence': float
        """
        result = {
            'found': False,
            'entry': None,
            'source': 'heuristic',
            'confidence': 0.5,
        }

        if self.knowledge_base is None:
            return result

        entry = self.knowledge_base.lookup(op_type, shape, min_confidence=0.5)
        if entry:
            result['found'] = True
            result['entry'] = entry
            result['source'] = 'knowledge_base'
            result['confidence'] = entry.confidence

            # Track this decision
            self.optimization_decisions.append({
                'layer_name': layer_name,
                'op_type': op_type,
                'shape': shape,
                'kb_entry': entry.description or entry.source,
                'confidence': entry.confidence,
                'expected_macs_per_cycle': entry.performance.macs_per_cycle,
            })

            # Auto-apply KB config if not already overridden by user
            if auto_apply and layer_name not in self.layer_config_overrides:
                kb_override = self._kb_entry_to_override(entry)
                if kb_override:
                    self.layer_config_overrides[layer_name] = kb_override
                    print(f"  [KB] Auto-applying config for {layer_name}: {kb_override.get('tile_config', {})}")

        return result

    def _kb_entry_to_override(self, entry) -> dict:
        """Convert a KB entry to a layer config override dict."""
        override = {}

        # Extract tile config
        if hasattr(entry, 'tile_config') and entry.tile_config and entry.tile_config.config:
            override['tile_config'] = dict(entry.tile_config.config)

        # Extract kernel config
        if hasattr(entry, 'kernel_config') and entry.kernel_config and entry.kernel_config.config:
            override['kernel_config'] = dict(entry.kernel_config.config)

        # Extract pipeline config (PipelineConfig uses to_dict() rather than .config)
        if hasattr(entry, 'pipeline_config') and entry.pipeline_config:
            pipeline_dict = entry.pipeline_config.to_dict()
            if pipeline_dict:
                override['pipeline_config'] = pipeline_dict

        # Extract compile flags
        if hasattr(entry, 'compile_flags') and entry.compile_flags and entry.compile_flags.flags:
            override['compile_flags'] = dict(entry.compile_flags.flags)

        return override if override else None

    def _prepare_kb_config(self, layer_name: str, op_type: str, shape: dict) -> bool:
        """
        Query KB and prepare layer_config_overrides BEFORE calling tiling functions.

        This method MUST be called before any _determine_*_memory_tier() or
        _get_layer_override() calls to ensure KB configs are applied.

        Args:
            layer_name: Name of the layer
            op_type: Operation type (e.g., "linear_int8", "mhsa_int8")
            shape: Dictionary of layer dimensions

        Returns:
            True if a KB config was found and applied, False otherwise.
        """
        result = self._query_kb_for_layer(layer_name, op_type, shape, auto_apply=True)
        if result['found']:
            conf = result.get('confidence', 0)
            print(f"  [KB] Using knowledge base config for {layer_name} ({conf:.0%} confidence)")
            return True
        return False

    def generate_optimization_report(self):
        """Generate a report showing optimization decisions made during code generation."""
        report_path = self.output_dir / "optimization_report.md"

        lines = [
            "# Optimization Report",
            "",
            f"Network: {self.network_info.get('__model_name__', 'Unknown')}",
            f"Generated by ARES Knowledge Base v{self.knowledge_base.version if self.knowledge_base else 'N/A'}",
            "",
        ]

        # Summary
        kb_count = sum(1 for d in self.optimization_decisions if d.get('kb_entry'))
        heuristic_count = len(self.layer_specs) - kb_count

        lines.append("## Summary")
        lines.append("")
        lines.append(f"- Total layers: {len(self.layer_specs)}")
        lines.append(f"- Total MACs: {self.total_macs:,}")
        lines.append(f"- KB-optimized layers: {kb_count}")
        lines.append(f"- Heuristic layers: {heuristic_count}")
        lines.append(f"- L1-tiled layers: {len(self.l1_tiled_layers)}")
        lines.append(f"- L2-fallback layers: {len(self.l2_fallback_layers)}")
        lines.append("")

        # Detailed decisions
        if self.optimization_decisions:
            lines.append("## Knowledge Base Matches")
            lines.append("")
            lines.append("| Layer | Op Type | KB Entry | Confidence | Expected MACs/cycle |")
            lines.append("|-------|---------|----------|------------|---------------------|")

            for d in self.optimization_decisions:
                macs = d.get('expected_macs_per_cycle')
                macs_str = f"{macs:.1f}" if macs else "N/A"
                lines.append(
                    f"| {d['layer_name']} | {d['op_type']} | {d.get('kb_entry', 'N/A')} | "
                    f"{d['confidence']:.0%} | {macs_str} |"
                )
            lines.append("")

        # Per-layer MACs breakdown
        layers_with_macs = [(s['name'], s.get('op', ''), s.get('macs', 0)) for s in self.layer_specs if s.get('macs', 0) > 0]
        if layers_with_macs:
            lines.append("## Layer MACs Breakdown")
            lines.append("")
            lines.append("| Layer | Op Type | MACs | % of Total |")
            lines.append("|-------|---------|------|------------|")
            for name, op, macs in sorted(layers_with_macs, key=lambda x: -x[2]):
                pct = 100.0 * macs / self.total_macs if self.total_macs > 0 else 0
                lines.append(f"| {name} | {op} | {macs:,} | {pct:.1f}% |")
            lines.append("")

        # Layers without KB matches
        kb_layers = {d['layer_name'] for d in self.optimization_decisions}
        non_kb_layers = [s['name'] for s in self.layer_specs if s['name'] not in kb_layers]

        if non_kb_layers:
            lines.append("## Layers Using Heuristics")
            lines.append("")
            lines.append("These layers did not match any KB entries and use default heuristics:")
            lines.append("")
            for name in non_kb_layers[:20]:  # Limit to first 20
                lines.append(f"- {name}")
            if len(non_kb_layers) > 20:
                lines.append(f"- ... and {len(non_kb_layers) - 20} more")
            lines.append("")

        # Memory Arena and Buffer Liveness
        if self.planner and self.planner.lifetimes:
            lines.append("## Memory Arena")
            lines.append("")

            # Calculate memory efficiency metrics
            naive_size = sum(life['size'] for life in self.planner.lifetimes.values())
            optimized_size = self.l2_arena_size
            savings = naive_size - optimized_size
            savings_pct = (savings / naive_size * 100) if naive_size > 0 else 0

            # Count buffer reuse (buffers sharing address ranges)
            reuse_count = 0
            for name, life in self.planner.lifetimes.items():
                offset = self.planner.offsets.get(name, 0)
                # Check if this buffer's address range overlaps with any earlier buffer
                for other_name, other_life in self.planner.lifetimes.items():
                    if other_name == name:
                        continue
                    other_offset = self.planner.offsets.get(other_name, 0)
                    # Space overlap check
                    if not (offset + life['size'] <= other_offset or offset >= other_offset + other_life['size']):
                        # Time non-overlap check (they must not overlap in time for valid reuse)
                        if life['end'] < other_life['start'] or life['start'] > other_life['end']:
                            reuse_count += 1
                            break  # Count each buffer only once

            lines.append(f"- Naive allocation (no reuse): {naive_size:,} bytes ({naive_size / 1024:.1f} KB)")
            lines.append(f"- Optimized allocation: {optimized_size:,} bytes ({optimized_size / 1024:.1f} KB)")
            lines.append(f"- **Memory saved: {savings:,} bytes ({savings_pct:.1f}%)**")
            lines.append(f"- Number of buffers: {len(self.planner.lifetimes)}")
            lines.append(f"- Buffers reusing space: {reuse_count}")
            lines.append("")

            # Buffer liveness table
            lines.append("### Buffer Liveness")
            lines.append("")
            lines.append("| Buffer | Size (bytes) | Start Layer | End Layer | Arena Offset |")
            lines.append("|--------|--------------|-------------|-----------|--------------|")

            # Sort by start time, then by size (largest first)
            sorted_buffers = sorted(
                self.planner.lifetimes.items(),
                key=lambda x: (x[1]['start'], -x[1]['size'])
            )

            for name, lifetime in sorted_buffers[:30]:  # Limit to first 30
                offset = self.planner.offsets.get(name, 'N/A')
                offset_str = f"{offset:,}" if isinstance(offset, int) else offset
                lines.append(
                    f"| {name[:40]} | {lifetime['size']:,} | {lifetime['start']} | {lifetime['end']} | {offset_str} |"
                )

            if len(sorted_buffers) > 30:
                lines.append(f"| ... | ... | ... | ... | ... |")
                lines.append(f"| ({len(sorted_buffers) - 30} more buffers) | | | | |")
            lines.append("")

        # Write report
        with open(report_path, 'w') as f:
            f.write('\n'.join(lines))

        print(f"  [KB] Optimization report: {report_path}")

    def _is_feature_flag_enabled(self, flag_name: str, default: bool = False) -> bool:
        """Return True when an environment feature flag is enabled."""
        default_value = "1" if default else "0"
        value = os.getenv(flag_name, default_value).strip().lower()
        return value in ("1", "true", "yes", "on")

    def _setup_checkpointing(self) -> None:
        """Initialize checkpoint manager if ARES_CHECKPOINT_DIR is configured."""
        if not self.checkpoint_output_dir:
            return

        if not CHECKPOINTS_AVAILABLE:
            warnings.warn(
                "Checkpoint export requested via ARES_CHECKPOINT_DIR but checkpoint module is unavailable.",
                RuntimeWarning,
            )
            return

        base_metadata = {
            "test_case_dir": str(self.test_case_dir),
            "output_dir": str(self.output_dir),
            "fusion_enabled": bool(self.enable_fusion),
        }
        if self.checkpoint_tag:
            base_metadata["tag"] = self.checkpoint_tag

        self.checkpoint_manager = CheckpointManager(
            self.checkpoint_output_dir,
            base_metadata=base_metadata,
        )
        print(f"  [Checkpoint] Export enabled: {self.checkpoint_output_dir}")

    def _checkpointing_enabled(self) -> bool:
        return self.checkpoint_manager is not None

    def _snapshot_codegen_state(self, specs_override=None, include_memory=False):
        """
        Capture serializable codegen state for checkpoint export.

        The snapshot intentionally includes only planning/debug structures and no
        runtime output artifacts.
        """
        specs = specs_override if specs_override is not None else self.layer_specs
        state = {
            "layer_specs": specs,
            "layer_count": len(specs),
            "activation_buffers": self.activation_buffers,
            "shared_activation_pool": self.shared_activation_pool,
            "param_layers": self.param_layers,
            "fused_layers": self.fused_layers,
            "l1_tiled_layers": self.l1_tiled_layers,
            "l2_fallback_layers": self.l2_fallback_layers,
        }

        if include_memory:
            state["l2_arena_size"] = self.l2_arena_size
            state["planner_policy"] = self.planner_policy
            if self.planner is not None:
                state["planner_offsets"] = getattr(self.planner, "offsets", {})
                state["planner_lifetimes"] = getattr(self.planner, "lifetimes", {})
                state["planner_unresolved_conflicts"] = getattr(
                    self.planner, "unresolved_conflicts", []
                )

        return state

    def _write_phase_checkpoint(self, stage: str, specs_override=None, include_memory=False) -> None:
        """Write one stage checkpoint if export is enabled."""
        if not self._checkpointing_enabled():
            return
        if stage in self._checkpoint_written_stages:
            return

        metadata = {
            "pipeline_v2_enabled": True,
            "fused_layers_count": len(self.fused_layers),
        }
        if self.checkpoint_tag:
            metadata["tag"] = self.checkpoint_tag

        try:
            state = self._snapshot_codegen_state(
                specs_override=specs_override,
                include_memory=include_memory,
            )
            path = self.checkpoint_manager.write_stage(
                stage=stage,
                state=state,
                metadata=metadata,
            )
            self._checkpoint_written_stages.add(stage)
            print(f"  [Checkpoint] {stage} -> {path}")
        except Exception as exc:
            warnings.warn(
                f"Failed to write checkpoint for stage '{stage}': {exc}",
                RuntimeWarning,
            )

    def _run_pipeline_v2(self):
        """Run the default code generation pipeline."""
        from .pipeline.pipeline import run_default_pipeline

        context = run_default_pipeline(self)
        if context.stage_order:
            print("  [PipelineV2] Stage timing summary:")
            for stage_name in context.stage_order:
                elapsed_s = context.stage_timings.get(stage_name, 0.0)
                print(f"    - {stage_name}: {elapsed_s:.3f}s")
        return context

    def generate_all(self):
        """Generate complete C project."""
        print("="*80)
        print(f"C Code Generation for {self.target.display_name}")
        print("="*80)
        print()

        print("Pipeline mode: default")
        print()

        # Create output structure
        self.create_structure()

        # Generate binaries first (needed for sizes/checksums)
        self.generate_binaries()

        self._run_pipeline_v2()

        # Generate optimization report (always, for MACs tracking)
        self.generate_optimization_report()

        print()
        print("="*80)
        print("[PASS] C code generation complete!")
        print("="*80)
        print()
        print(f"Output directory: {self.output_dir}")
        print()
        print("Next steps:")
        print("  1. cd generated/")
        print("  2. make clean all")
        print(f"  3. make run (on {self.target.display_name} board/gvsoc)")

    def create_structure(self):
        """Create output directory structure."""
        print("Creating directory structure...")
        (self.output_dir / "inc").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "inc" / "ops").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "src").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "src" / "net").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "src" / "ops").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "bin").mkdir(parents=True, exist_ok=True)
        self._prune_stale_runtime_duplicates()
        print(f"[OK] Created {self.output_dir}/")

    def _prune_stale_runtime_duplicates(self):
        """Remove generated runtime duplicates that can drift from codegen/runtime.

        `codegen/runtime/{src,inc}` is the single source of truth for shared code.
        Generated projects should not carry local copies of these files; otherwise they
        can be compiled/used accidentally (e.g., by `$(wildcard src/*.c)`), causing ODR
        conflicts or signature mismatches.
        """
        stale = [
            # Shared runtime sources (now compiled from `$(RUNTIME_DIR)/src`)
            "src/mem.c",
            "src/network_kernels.c",
            "src/network_dma_pipeline.c",
            "src/network_l3_prefetch.c",
            "src/network_buffer_mgmt.c",
            # Shared runtime headers (now included from `$(RUNTIME_DIR)/inc`)
            "inc/mem.h",
            "inc/network_kernels.h",
            "inc/network_dma_pipeline.h",
            "inc/network_l3_prefetch.h",
            "inc/network_buffer_mgmt.h",
            "inc/tile_buffer_manager.h",
            "inc/l3_prefetch.h",
            "inc/layer_descriptors.h",
            # Legacy split artifacts (compiled src/net/*.c is now authoritative)
            "src/net/layer_descriptors.inc",
            "src/net/core.inc",
            "src/net/cluster_entry.inc",
            "src/net/fc_entry.inc",
        ]

        for relpath in stale:
            path = self.output_dir / relpath
            if path.is_file():
                path.unlink()

    def generate_binaries(self):
        """Convert NumPy arrays to binary files with checksums."""
        print()
        print("Generating binary files...")
        self.binary_symbol_counts = {}

        bin_dir = self.output_dir / "bin"

        # Helper to save binary and record manifest
        def save_binary(filename, data, label):
            path = bin_dir / filename
            data.tofile(path)
            size = path.stat().st_size
            checksum = self.compute_checksum(path)

            entry = {
                'filename': filename,
                'path': f'bin/{filename}',
                'size': size,
                'checksum': checksum,
                'label': label,
                'index': len(self.binary_files),
            }
            entry['c_symbol'] = self._unique_binary_symbol(Path(filename).stem)
            self.binary_files.append(entry)

            print(f"  [OK] {filename:40s} {size:6d} bytes  ck=0x{checksum:08X}")
            return entry

        # Load test input (may be multiple inputs for multi-input models).
        # Fall back to input0_fp32.npy for generators that use numbered naming.
        _input_path = self.test_case_dir / "input_fp32.npy"
        if not _input_path.exists():
            _input_path = self.test_case_dir / "input0_fp32.npy"
        input_fp32 = np.load(_input_path)
        self.input_shape = list(input_fp32.shape)
        # Quantize input using detected input scale
        input_scale = self.input_scale
        input_int8 = np.clip(np.round(input_fp32 / input_scale), -128, 127).astype(np.int8)

        # Convert input from NCHW to NHWC if HWC layout is enabled
        # This avoids the need for runtime CHW->HWC conversion
        if self.use_hwc_layout and input_int8.ndim == 4:
            # [N, C, H, W] -> [N, H, W, C]
            input_int8 = np.transpose(input_int8, (0, 2, 3, 1)).copy()
            print(f"  [HWC] Converted input from NCHW to NHWC: {self.input_shape} -> {list(input_int8.shape)}")

        self.input_entry = save_binary("input_test1.bin", input_int8, "Test input (INT8)")

        # For multi-input models, also save additional inputs and track their entries
        self.additional_binary_input_entries = []  # Track {quant_layer, entry} for template
        for i, quant_layer in enumerate(self.input_quant_layers[1:], start=1):
            additional_input_path = self.test_case_dir / f"input_{i}_fp32.npy"
            if additional_input_path.exists():
                add_fp32 = np.load(additional_input_path)
                add_scale = self.network_info.get(quant_layer, {}).get('scale', 1.0)
                add_int8 = np.clip(np.round(add_fp32 / add_scale), -128, 127).astype(np.int8)

                # Convert additional input from NCHW to NHWC if HWC layout is enabled
                if self.use_hwc_layout and add_int8.ndim == 4:
                    add_int8 = np.transpose(add_int8, (0, 2, 3, 1)).copy()
                    print(f"  [HWC] Converted input_{i} from NCHW to NHWC")

                entry = save_binary(f"input_{i}_test1.bin", add_int8, f"Test input {i} (INT8)")
                self.additional_binary_input_entries.append({
                    'quant_layer': quant_layer,
                    'entry': entry,
                    'index': entry['index'],  # Index in binary_files array
                    'numel': add_int8.size,
                    'scale': add_scale,
                })

        # Calculate buffer size for input_quant based on its output shape (after any permute/reshape)
        # This is needed because the input may be permuted before reaching input_quant
        primary_input_quant = self.input_quant_layers[0] if self.input_quant_layers else 'input_quant'
        input_quant_shape = self.network_info.get(primary_input_quant, {}).get('output_shape', list(input_int8.shape))
        self.input_numel = int(np.prod(input_quant_shape[1:]))  # Exclude batch dimension

        # Weights and biases for each layer
        for layer_name in self.layer_names:
            layer_data = self.layer_info[layer_name]
            layer_type = layer_data['type']

            if layer_type in ['QuantConv2d', 'QuantLinear']:
                # Weight (INT8)
                weight_path = self.weights_dir / f"{layer_name}_weight_int8.npy"
                if weight_path.exists():
                    weight_int8 = np.load(weight_path)
                    # For Conv2D with HWC layout: reorder weights from [out_ch, in_ch, kh, kw]
                    # to [out_ch, kh, kw, in_ch] so channels are contiguous per kernel position
                    if self.use_hwc_layout and layer_type == 'QuantConv2d' and weight_int8.ndim == 4:
                        weight_int8 = np.transpose(weight_int8, (0, 2, 3, 1)).copy()
                    entry = save_binary(f"{layer_name}_weight.bin", weight_int8,
                                        f"{layer_name} weight (INT8)")
                    self.weight_entries[layer_name] = entry

                # Bias (INT32 for conv/linear intermediate, FP32 for final unless int8_classifier_output)
                bias_path = self.weights_dir / f"{layer_name}_bias_fp32.npy"
                if bias_path.exists():
                    bias_fp32 = np.load(bias_path)
                    # Use INT32 bias for: all Conv2D, all non-final Linear, OR final Linear when int8_classifier_output
                    is_final_linear = (layer_type == 'QuantLinear' and layer_name == self.final_linear_name)
                    convert_to_int32 = (
                        layer_type == 'QuantConv2d' or
                        (layer_type == 'QuantLinear' and layer_name != self.final_linear_name) or
                        (is_final_linear and self.int8_classifier_output)
                    )

                    if convert_to_int32:
                        scale_x = layer_data.get('scale_input', 1.0)
                        # For projection shortcuts, ensure we use the block input scale (already harmonized in network_info)
                        scale_w = layer_data['scale_weight']
                        scale_bias = scale_x * scale_w
                        bias_data = np.round(bias_fp32 / scale_bias).astype(np.int32)
                        label = f"{layer_name} bias (INT32)"
                    else:
                        bias_data = bias_fp32.astype(np.float32, copy=False)
                        label = f"{layer_name} bias (FP32)"

                    entry = save_binary(f"{layer_name}_bias.bin", bias_data, label)
                    self.bias_entries[layer_name] = entry

                # NE16 weight packing for eligible Linear layers
                # Auto-detection: use_ne16="auto" enables NE16 for large layers (>= ne16_auto_threshold MACs)
                if layer_type == 'QuantLinear' and layer_name in self.weight_entries:
                    in_features = layer_data.get('in_features', 0)
                    out_features = layer_data.get('out_features', 0)
                    if self._should_use_ne16_for_layer(layer_type, in_features, out_features):
                        # Reload original weight and compute INT32 bias for NE16 correction
                        weight_int8 = np.load(self.weights_dir / f"{layer_name}_weight_int8.npy")
                        bias_fp32 = np.load(bias_path) if bias_path.exists() else np.zeros(out_features, dtype=np.float32)
                        scale_x = layer_data.get('scale_input', 1.0)
                        scale_w = layer_data['scale_weight']
                        scale_bias = scale_x * scale_w
                        bias_int32 = np.round(bias_fp32 / scale_bias).astype(np.int32)

                        # Pack weights and correct bias for NE16
                        packed_weights, bias_corrected = ne16_pack_linear_weights(weight_int8, bias_int32)

                        # Save packed weights
                        packed_entry = save_binary(f"{layer_name}_ne16_packed.bin", packed_weights,
                                                   f"{layer_name} NE16 packed weights")
                        self.ne16_weight_entries[layer_name] = packed_entry

                        # Save corrected bias
                        bias_corr_entry = save_binary(f"{layer_name}_ne16_bias_corr.bin", bias_corrected,
                                                       f"{layer_name} NE16 corrected bias (INT32)")
                        self.ne16_bias_entries[layer_name] = bias_corr_entry

                        # Compute HW outquant scale parameters (reference-compatible)
                        scale_output = layer_data.get('scale_output', 1.0)
                        if scale_output is not None and scale_output > 0:
                            hw_scale, hw_scale_shift = compute_ne16_requant_params(
                                scale_x, scale_w, scale_output, out_features
                            )
                            # Save HW outquant scale arrays
                            scale_entry = save_binary(f"{layer_name}_ne16_hw_scale.bin", hw_scale,
                                                      f"{layer_name} NE16 HW outquant scale (uint8)")
                            self.ne16_scale_entries[layer_name] = scale_entry
                            scale_shift_entry = save_binary(f"{layer_name}_ne16_hw_scale_shift.bin", hw_scale_shift,
                                                            f"{layer_name} NE16 HW outquant scale_shift (uint8)")
                            self.ne16_scale_shift_entries[layer_name] = scale_shift_entry

                        self.ne16_eligible_layers.add(layer_name)
                        print(f"    → NE16 packed: {layer_name} ({in_features}→{out_features})")

                # NE16 weight packing for eligible Conv2D 1x1 layers
                # Auto-detection: use_ne16="auto" enables NE16 for large layers (>= ne16_auto_threshold MACs)
                if layer_type == 'QuantConv2d' and layer_name in self.weight_entries:
                    kernel_size = layer_data.get('kernel_size', [1, 1])
                    if isinstance(kernel_size, int):
                        kernel_size = [kernel_size, kernel_size]
                    kernel_h, kernel_w = kernel_size[0], kernel_size[1] if len(kernel_size) > 1 else kernel_size[0]

                    if kernel_h == 1 and kernel_w == 1:
                        in_channels = layer_data.get('in_channels', 0)
                        out_channels = layer_data.get('out_channels', 0)
                        groups = layer_data.get('groups', 1)
                        # Get output spatial dimensions for MACs calculation
                        output_shape = layer_data.get('output_shape', [1, out_channels, 7, 7])
                        out_h = output_shape[2] if len(output_shape) > 2 else 7
                        out_w = output_shape[3] if len(output_shape) > 3 else 7

                        # NE16 1x1 only supports stride=1
                        stride = layer_data.get('stride', 1)
                        stride_h = stride[0] if isinstance(stride, (list, tuple)) else stride
                        stride_w = stride[1] if isinstance(stride, (list, tuple)) and len(stride) > 1 else stride_h

                        if stride_h == 1 and stride_w == 1 and self._should_use_ne16_for_layer(layer_type, in_channels, out_channels,
                                                          kernel_size=(kernel_h, kernel_w),
                                                          spatial_size=(out_h, out_w),
                                                          groups=groups):
                            # Reload original weight and compute INT32 bias for NE16 correction
                            weight_int8 = np.load(self.weights_dir / f"{layer_name}_weight_int8.npy")
                            bias_fp32 = np.load(bias_path) if bias_path.exists() else np.zeros(out_channels, dtype=np.float32)
                            scale_x = layer_data.get('scale_input', 1.0)
                            scale_w = layer_data['scale_weight']
                            scale_bias = scale_x * scale_w
                            bias_int32 = np.round(bias_fp32 / scale_bias).astype(np.int32)

                            # Pack weights and correct bias for NE16 (1x1 conv uses same packing as linear)
                            packed_weights, bias_corrected = ne16_pack_conv1x1_weights(weight_int8, bias_int32)

                            # Save packed weights
                            packed_entry = save_binary(f"{layer_name}_ne16_packed.bin", packed_weights,
                                                       f"{layer_name} NE16 packed weights")
                            self.ne16_weight_entries[layer_name] = packed_entry

                            # Save corrected bias
                            bias_corr_entry = save_binary(f"{layer_name}_ne16_bias_corr.bin", bias_corrected,
                                                           f"{layer_name} NE16 corrected bias (INT32)")
                            self.ne16_bias_entries[layer_name] = bias_corr_entry

                            # Compute HW outquant scale parameters (reference-compatible)
                            scale_output = layer_data.get('scale_output', 1.0)
                            if scale_output is not None and scale_output > 0:
                                hw_scale, hw_scale_shift = compute_ne16_requant_params(
                                    scale_x, scale_w, scale_output, out_channels
                                )
                                # Save HW outquant scale arrays
                                scale_entry = save_binary(f"{layer_name}_ne16_hw_scale.bin", hw_scale,
                                                          f"{layer_name} NE16 HW outquant scale (uint8)")
                                self.ne16_scale_entries[layer_name] = scale_entry
                                scale_shift_entry = save_binary(f"{layer_name}_ne16_hw_scale_shift.bin", hw_scale_shift,
                                                                f"{layer_name} NE16 HW outquant scale_shift (uint8)")
                                self.ne16_scale_shift_entries[layer_name] = scale_shift_entry

                            self.ne16_eligible_layers.add(layer_name)
                            print(f"    → NE16 packed (conv1x1): {layer_name} ({in_channels}→{out_channels})")

                    # NE16 3x3 convolution support
                    elif kernel_h == 3 and kernel_w == 3:
                        in_channels = layer_data.get('in_channels', 0)
                        out_channels = layer_data.get('out_channels', 0)
                        groups = layer_data.get('groups', 1)
                        # NE16 3x3 only supports stride=1
                        stride = layer_data.get('stride', 1)
                        stride_h = stride[0] if isinstance(stride, (list, tuple)) else stride
                        stride_w = stride[1] if isinstance(stride, (list, tuple)) and len(stride) > 1 else stride_h
                        # Get output spatial dimensions for MACs calculation
                        output_shape = layer_data.get('output_shape', [1, out_channels, 7, 7])
                        out_h = output_shape[2] if len(output_shape) > 2 else 7
                        out_w = output_shape[3] if len(output_shape) > 3 else 7

                        if stride_h == 1 and stride_w == 1 and self._should_use_ne16_for_layer(layer_type, in_channels, out_channels,
                                                          kernel_size=(kernel_h, kernel_w),
                                                          spatial_size=(out_h, out_w),
                                                          groups=groups):
                            # Reload original weight and compute INT32 bias for NE16 correction
                            weight_int8 = np.load(self.weights_dir / f"{layer_name}_weight_int8.npy")
                            bias_fp32 = np.load(bias_path) if bias_path.exists() else np.zeros(out_channels, dtype=np.float32)
                            scale_x = layer_data.get('scale_input', 1.0)
                            scale_w = layer_data['scale_weight']
                            scale_bias = scale_x * scale_w
                            bias_int32 = np.round(bias_fp32 / scale_bias).astype(np.int32)

                            # Check if depthwise (groups == in_channels)
                            is_depthwise = (groups > 1) and (groups == in_channels)
                            if is_depthwise:
                                # Pack weights for NE16 depthwise 3x3 with HW requant params
                                scale_input = layer_data.get('scale_input', 1.0)
                                scale_weight = layer_data.get('scale_weight', 1.0)
                                scale_output = layer_data.get('scale_output', 1.0)

                                packed_weights, bias_corrected, hw_scale, hw_scale_shift = \
                                    ne16_pack_conv3x3_depthwise_weights_with_requant(
                                        weight_int8, bias_int32,
                                        scale_input, scale_weight, scale_output
                                    )
                                pack_desc = "NE16 packed depthwise 3x3 weights"
                                log_msg = f"    → NE16 packed (conv3x3_dw): {layer_name} ({in_channels} channels, HW requant)"

                                # Save HW requant scale arrays
                                hw_scale_entry = save_binary(f"{layer_name}_ne16_hw_scale.bin", hw_scale,
                                                             f"{layer_name} NE16 HW requant scale (UINT8)")
                                hw_scale_shift_entry = save_binary(f"{layer_name}_ne16_hw_scale_shift.bin", hw_scale_shift,
                                                                   f"{layer_name} NE16 HW requant shift (UINT8)")

                                # Track HW requant entries for this layer
                                if not hasattr(self, 'ne16_hw_scale_entries'):
                                    self.ne16_hw_scale_entries = {}
                                if not hasattr(self, 'ne16_hw_scale_shift_entries'):
                                    self.ne16_hw_scale_shift_entries = {}
                                self.ne16_hw_scale_entries[layer_name] = hw_scale_entry
                                self.ne16_hw_scale_shift_entries[layer_name] = hw_scale_shift_entry
                            else:
                                # Pack weights for regular NE16 3x3
                                packed_weights, bias_corrected = ne16_pack_conv3x3_weights(weight_int8, bias_int32)
                                pack_desc = "NE16 packed 3x3 weights"
                                log_msg = f"    → NE16 packed (conv3x3): {layer_name} ({in_channels}→{out_channels})"

                            # Save packed weights
                            packed_entry = save_binary(f"{layer_name}_ne16_packed.bin", packed_weights,
                                                       f"{layer_name} {pack_desc}")
                            self.ne16_weight_entries[layer_name] = packed_entry

                            # Save corrected bias
                            bias_corr_entry = save_binary(f"{layer_name}_ne16_bias_corr.bin", bias_corrected,
                                                           f"{layer_name} NE16 corrected bias (INT32)")
                            self.ne16_bias_entries[layer_name] = bias_corr_entry

                            self.ne16_eligible_layers.add(layer_name)
                            print(log_msg)

            elif layer_type == 'PatchEmbed':
                # PatchEmbed: Conv2D projection weights and biases
                weight_path = self.weights_dir / f"{layer_name}_proj_weight_int8.npy"
                if weight_path.exists():
                    weight_int8 = np.load(weight_path)
                    entry = save_binary(f"{layer_name}_weight.bin", weight_int8,
                                        f"{layer_name} proj weight (INT8)")
                    self.weight_entries[layer_name] = entry

                bias_path = self.weights_dir / f"{layer_name}_proj_bias_fp32.npy"
                if bias_path.exists():
                    bias_fp32 = np.load(bias_path)
                    # Convert bias to INT32 (like Conv2D)
                    scale_x = layer_data.get('scale_input', 1.0)
                    scale_w = layer_data.get('proj_weight_scale', 1.0)
                    scale_bias = scale_x * scale_w
                    bias_int32 = np.round(bias_fp32 / scale_bias).astype(np.int32)
                    entry = save_binary(f"{layer_name}_bias.bin", bias_int32,
                                        f"{layer_name} proj bias (INT32)")
                    self.bias_entries[layer_name] = entry

            elif layer_type == 'PositionalEmbedding':
                # PositionalEmbedding: INT8 learnable positional embedding
                pos_embed_path = self.weights_dir / f"{layer_name}_int8.npy"
                if pos_embed_path.exists():
                    pos_embed_int8 = np.load(pos_embed_path)
                    entry = save_binary(f"{layer_name}.bin", pos_embed_int8,
                                        f"{layer_name} (INT8)")
                    self.weight_entries[layer_name] = entry

            elif layer_type == 'Embedding':
                # Embedding: INT8 embedding table + captured INT32 indices
                weight_path = self.weights_dir / f"{layer_name}_weight_int8.npy"
                if weight_path.exists():
                    weight_int8 = np.load(weight_path).astype(np.int8, copy=False)
                    entry = save_binary(f"{layer_name}_weight.bin", weight_int8,
                                        f"{layer_name} embedding weight (INT8)")
                    self.weight_entries[layer_name] = entry

                indices_path = self.weights_dir / f"{layer_name}_indices_int32.npy"
                if indices_path.exists():
                    indices_int32 = np.load(indices_path).astype(np.int32, copy=False)
                    entry = save_binary(f"{layer_name}_indices.bin", indices_int32,
                                        f"{layer_name} embedding indices (INT32)")
                    # Treat indices as a "bias" blob so we can reuse the existing (weight,bias) loader path.
                    self.bias_entries[layer_name] = entry

            elif layer_type == 'MultiheadSelfAttention':
                # MHSA projections: save weights and quantize biases to INT32 (like conv/linear)
                scale_x = layer_data.get('scale_input', 1.0)
                for prefix in ('q', 'k', 'v', 'out'):
                    weight_key = f"{prefix}_weight_int8"
                    if weight_key in layer_data:
                        weight_int8 = np.array(layer_data[weight_key], dtype=np.int8)
                        entry = save_binary(f"{layer_name}_{prefix}_weight.bin", weight_int8,
                                            f"{layer_name} {prefix.upper()} weight (INT8)")
                        self.weight_entries[f"{layer_name}::{prefix}"] = entry
                    bias_key = f"{prefix}_bias_fp32"
                    if bias_key in layer_data and layer_data[bias_key] is not None:
                        bias_fp32 = np.array(layer_data[bias_key], dtype=np.float32)
                        scale_w = layer_data.get(f"{prefix}_scale_weight", 1.0)
                        # Output projection bias uses context scale (scale_v), not input scale
                        if prefix == 'out':
                            # Context has scale_v (from V projection output)
                            scale_v = layer_data.get('v_scale_output')
                            if scale_v is None:
                                # Fallback: compute scale_v from input * v_weight_scale
                                scale_v_weight = layer_data.get('v_scale_weight', 1.0)
                                scale_v = scale_x * scale_v_weight
                            scale_bias = scale_v * scale_w
                        else:
                            scale_bias = scale_x * scale_w
                        bias_int32 = np.round(bias_fp32 / scale_bias).astype(np.int32)
                        entry = save_binary(f"{layer_name}_{prefix}_bias.bin", bias_int32,
                                            f"{layer_name} {prefix.upper()} bias (INT32)")
                        self.bias_entries[f"{layer_name}::{prefix}"] = entry
                if layer_data.get('use_rope', False):
                    cos_path = self.weights_dir / f"{layer_name}_rope_cos_q15.npy"
                    sin_path = self.weights_dir / f"{layer_name}_rope_sin_q15.npy"
                    if not cos_path.exists() or not sin_path.exists():
                        raise FileNotFoundError(
                            f"RoPE MHSA {layer_name}: missing {cos_path.name} / {sin_path.name} in weights dir"
                        )
                    cos_q15 = np.load(cos_path).astype(np.int16, copy=False)
                    sin_q15 = np.load(sin_path).astype(np.int16, copy=False)
                    entry_cos = save_binary(
                        f"{layer_name}_rope_cos_q15.bin",
                        cos_q15,
                        f"{layer_name} RoPE cos table (Q15 INT16)",
                    )
                    entry_sin = save_binary(
                        f"{layer_name}_rope_sin_q15.bin",
                        sin_q15,
                        f"{layer_name} RoPE sin table (Q15 INT16)",
                    )
                    self.weight_entries[f"{layer_name}::rope_cos"] = entry_cos
                    self.weight_entries[f"{layer_name}::rope_sin"] = entry_sin

            elif layer_type == 'CrossAttention':
                # CrossAttention: learned query table + Q/K/V/Out projections (INT8) with INT32 biases.
                scale_x = layer_data.get('scale_input', 1.0)  # KV input scale
                scale_query = layer_data.get('query_scale', 1.0)  # query embedding input scale

                # Query embedding table (INT8)
                if 'query_embed_int8' in layer_data:
                    query_int8 = np.array(layer_data['query_embed_int8'], dtype=np.int8)
                    entry = save_binary(
                        f"{layer_name}_query_embed.bin",
                        query_int8,
                        f"{layer_name} query_embed (INT8)",
                    )
                    self.weight_entries[f"{layer_name}::query_embed"] = entry

                for prefix in ('q', 'k', 'v', 'out'):
                    weight_key = f"{prefix}_weight_int8"
                    if weight_key in layer_data:
                        weight_int8 = np.array(layer_data[weight_key], dtype=np.int8)
                        entry = save_binary(
                            f"{layer_name}_{prefix}_weight.bin",
                            weight_int8,
                            f"{layer_name} {prefix.upper()} weight (INT8)",
                        )
                        self.weight_entries[f"{layer_name}::{prefix}"] = entry

                    bias_key = f"{prefix}_bias_fp32"
                    if bias_key in layer_data and layer_data[bias_key] is not None:
                        bias_fp32 = np.array(layer_data[bias_key], dtype=np.float32)
                        scale_w = layer_data.get(f"{prefix}_scale_weight", 1.0)

                        if prefix == 'q':
                            scale_bias = scale_query * scale_w
                        elif prefix == 'out':
                            # Context has scale_v (from V projection output)
                            scale_v = layer_data.get('v_scale_output')
                            if scale_v is None:
                                scale_v_weight = layer_data.get('v_scale_weight', 1.0)
                                scale_v = scale_x * scale_v_weight
                            scale_bias = scale_v * scale_w
                        else:
                            scale_bias = scale_x * scale_w

                        bias_int32 = np.round(bias_fp32 / scale_bias).astype(np.int32)
                        entry = save_binary(
                            f"{layer_name}_{prefix}_bias.bin",
                            bias_int32,
                            f"{layer_name} {prefix.upper()} bias (INT32)",
                        )
                        self.bias_entries[f"{layer_name}::{prefix}"] = entry

            elif layer_type == 'CrossAttentionWithSelfRefine':
                # Composite block: cross-attention + FFN + 3x self-attention refinement.
                # Generates binaries for all sub-weights (norms, projections, FFN, SA blocks).
                embed_dim = int(layer_data.get('embed_dim', 0))
                num_queries = int(layer_data.get('num_queries', 4))
                ff_dim = int(layer_data.get('ff_dim', 256))
                num_sa_blocks = int(layer_data.get('num_self_attn_blocks', 3))

                # Helper to save FP32 norm weights
                def save_norm_weight(prefix):
                    for suffix in ('weight', 'bias'):
                        key = f"{prefix}_{suffix}"
                        if key in layer_data and layer_data[key] is not None:
                            data = np.array(layer_data[key], dtype=np.float32)
                            entry = save_binary(
                                f"{layer_name}_{prefix}_{suffix}.bin", data,
                                f"{layer_name} {prefix} {suffix} (FP32)",
                            )
                            self.weight_entries[f"{layer_name}::{prefix}_{suffix}"] = entry

                # Helper to save INT8 projection weight + INT32 bias
                def save_projection(prefix, scale_input_for_bias):
                    w_key = f"{prefix}_weight_int8"
                    if w_key in layer_data:
                        w = np.array(layer_data[w_key], dtype=np.int8)
                        entry = save_binary(
                            f"{layer_name}_{prefix}_weight.bin", w,
                            f"{layer_name} {prefix} weight (INT8)",
                        )
                        self.weight_entries[f"{layer_name}::{prefix}"] = entry
                    b_key = f"{prefix}_bias_fp32"
                    if b_key in layer_data and layer_data[b_key] is not None:
                        b_fp32 = np.array(layer_data[b_key], dtype=np.float32)
                        s_w = layer_data.get(f"{prefix}_scale_weight", 1.0)
                        b_int32 = np.round(b_fp32 / (scale_input_for_bias * s_w)).astype(np.int32)
                        entry = save_binary(
                            f"{layer_name}_{prefix}_bias.bin", b_int32,
                            f"{layer_name} {prefix} bias (INT32)",
                        )
                        self.bias_entries[f"{layer_name}::{prefix}"] = entry

                # Query embedding (INT8)
                if 'query_embed_int8' in layer_data:
                    q_embed = np.array(layer_data['query_embed_int8'], dtype=np.int8)
                    entry = save_binary(
                        f"{layer_name}_query_embed.bin", q_embed,
                        f"{layer_name} query_embed (INT8)",
                    )
                    self.weight_entries[f"{layer_name}::query_embed"] = entry

                # Stage 1: LayerNorm weights
                save_norm_weight('queries_norm')
                save_norm_weight('keys_norm')
                save_norm_weight('values_norm')

                # Stage 2: Cross-attention projections
                queries_norm_scale = float(layer_data.get('queries_norm_scale_output', layer_data.get('query_scale', 1.0)))
                keys_norm_scale = float(layer_data.get('keys_norm_scale_output', layer_data.get('scale_input', 1.0)))
                values_norm_scale = float(layer_data.get('values_norm_scale_output', layer_data.get('scale_input', 1.0)))
                v_scale_output = float(layer_data.get('v_scale_output', layer_data.get('scale_input', 1.0)))
                out_scale_output = float(layer_data.get('out_scale_output', layer_data.get('scale_input', 1.0)))

                save_projection('q', queries_norm_scale)
                save_projection('k', keys_norm_scale)
                save_projection('v', values_norm_scale)
                save_projection('out', v_scale_output)

                # Stage 4: FFN
                save_projection('ffn_fc1', out_scale_output)
                ffn_gelu_scale = float(layer_data.get('ffn_gelu_scale', layer_data.get('scale_input', 1.0)))
                save_projection('ffn_fc2', ffn_gelu_scale)

                # Stage 5: Self-attention refinement blocks
                for sa_idx in range(num_sa_blocks):
                    pfx = f"sa{sa_idx}"
                    save_norm_weight(f"{pfx}_norm1")
                    save_norm_weight(f"{pfx}_norm2")
                    sa_norm1_scale = float(layer_data.get(f'{pfx}_norm1_scale_output', layer_data.get('scale_input', 1.0)))
                    sa_v_scale = float(layer_data.get(f'{pfx}_v_scale_output', layer_data.get('scale_input', 1.0)))
                    sa_out_scale = float(layer_data.get(f'{pfx}_out_scale_output', layer_data.get('scale_input', 1.0)))
                    sa_norm2_scale = float(layer_data.get(f'{pfx}_norm2_scale_output', layer_data.get('scale_input', 1.0)))
                    sa_gelu_scale = float(layer_data.get(f'{pfx}_mlp_gelu_scale', layer_data.get('scale_input', 1.0)))

                    save_projection(f"{pfx}_q", sa_norm1_scale)
                    save_projection(f"{pfx}_k", sa_norm1_scale)
                    save_projection(f"{pfx}_v", sa_norm1_scale)
                    save_projection(f"{pfx}_out", sa_v_scale)
                    save_projection(f"{pfx}_mlp_fc1", sa_norm2_scale)
                    save_projection(f"{pfx}_mlp_fc2", sa_gelu_scale)

            elif layer_type == 'ClassificationHeadWithMLP':
                # Composite block: cross-attention pooling + MLP classifier.
                hidden_dim = int(layer_data.get('hidden_dim', layer_data.get('embed_dim', 256)))

                # Learned aggregation query (INT8)
                if 'learned_agg_int8' in layer_data:
                    agg = np.array(layer_data['learned_agg_int8'], dtype=np.int8)
                    entry = save_binary(
                        f"{layer_name}_learned_agg.bin", agg,
                        f"{layer_name} learned_agg (INT8)",
                    )
                    self.weight_entries[f"{layer_name}::learned_agg"] = entry

                # Cross-attention projections
                agg_scale = float(layer_data.get('agg_scale', 1.0))
                scale_input = float(layer_data.get('scale_input', 1.0))
                v_scale_out = float(layer_data.get('v_scale_output', scale_input))

                for prefix in ('q', 'k', 'v', 'out'):
                    w_key = f"{prefix}_weight_int8"
                    if w_key in layer_data:
                        w = np.array(layer_data[w_key], dtype=np.int8)
                        entry = save_binary(
                            f"{layer_name}_{prefix}_weight.bin", w,
                            f"{layer_name} {prefix.upper()} weight (INT8)",
                        )
                        self.weight_entries[f"{layer_name}::{prefix}"] = entry
                    b_key = f"{prefix}_bias_fp32"
                    if b_key in layer_data and layer_data[b_key] is not None:
                        b_fp32 = np.array(layer_data[b_key], dtype=np.float32)
                        s_w = layer_data.get(f"{prefix}_scale_weight", 1.0)
                        if prefix == 'q':
                            s_bias = agg_scale * s_w
                        elif prefix == 'out':
                            s_bias = v_scale_out * s_w
                        else:
                            s_bias = scale_input * s_w
                        b_int32 = np.round(b_fp32 / s_bias).astype(np.int32)
                        entry = save_binary(
                            f"{layer_name}_{prefix}_bias.bin", b_int32,
                            f"{layer_name} {prefix.upper()} bias (INT32)",
                        )
                        self.bias_entries[f"{layer_name}::{prefix}"] = entry

                # MLP fc1/fc2
                out_scale_output = float(layer_data.get('out_scale_output', scale_input))
                for prefix in ('mlp_fc1', 'mlp_fc2'):
                    w_key = f"{prefix}_weight_int8"
                    if w_key in layer_data:
                        w = np.array(layer_data[w_key], dtype=np.int8)
                        entry = save_binary(
                            f"{layer_name}_{prefix}_weight.bin", w,
                            f"{layer_name} {prefix} weight (INT8)",
                        )
                        self.weight_entries[f"{layer_name}::{prefix}"] = entry
                    b_key = f"{prefix}_bias_fp32"
                    if b_key in layer_data and layer_data[b_key] is not None:
                        b_fp32 = np.array(layer_data[b_key], dtype=np.float32)
                        s_w = layer_data.get(f"{prefix}_scale_weight", 1.0)
                        if prefix == 'mlp_fc1':
                            s_bias = out_scale_output * s_w
                        else:
                            gelu_scale = float(layer_data.get('mlp_gelu_scale', scale_input))
                            s_bias = gelu_scale * s_w
                        b_int32 = np.round(b_fp32 / s_bias).astype(np.int32)
                        entry = save_binary(
                            f"{layer_name}_{prefix}_bias.bin", b_int32,
                            f"{layer_name} {prefix} bias (INT32)",
                        )
                        self.bias_entries[f"{layer_name}::{prefix}"] = entry

            elif layer_type == 'AlternatingAttention':
                # AlternatingAttention: combined QKV projection + output projection
                scale_x = layer_data.get('scale_input', 1.0)
                embed_dim = layer_data.get('embed_dim', 0)

                # QKV projection weights (INT8)
                qkv_weight_int8 = None
                qkv_bias_int32 = None
                if 'qkv_weight_int8' in layer_data:
                    qkv_weight_int8 = np.array(layer_data['qkv_weight_int8'], dtype=np.int8)
                    entry = save_binary(f"{layer_name}_qkv_weight.bin", qkv_weight_int8,
                                        f"{layer_name} QKV weight (INT8)")
                    self.weight_entries[f"{layer_name}::qkv"] = entry

                # QKV projection bias (INT32)
                if 'qkv_bias_fp32' in layer_data and layer_data['qkv_bias_fp32'] is not None:
                    bias_fp32 = np.array(layer_data['qkv_bias_fp32'], dtype=np.float32)
                    scale_w = layer_data.get('qkv_scale_weight', 1.0)
                    scale_bias = scale_x * scale_w
                    qkv_bias_int32 = np.round(bias_fp32 / scale_bias).astype(np.int32)
                    entry = save_binary(f"{layer_name}_qkv_bias.bin", qkv_bias_int32,
                                        f"{layer_name} QKV bias (INT32)")
                    self.bias_entries[f"{layer_name}::qkv"] = entry

                # NE16 packing for QKV projection
                if self.use_ne16 and qkv_weight_int8 is not None and NE16_PACKING_AVAILABLE:
                    qkv_in = embed_dim
                    qkv_out = 3 * embed_dim
                    if qkv_bias_int32 is None:
                        qkv_bias_int32 = np.zeros(qkv_out, dtype=np.int32)
                    packed_weights, bias_corrected = ne16_pack_linear_weights(qkv_weight_int8, qkv_bias_int32)
                    packed_entry = save_binary(f"{layer_name}_qkv_ne16_packed.bin", packed_weights,
                                               f"{layer_name} QKV NE16 packed weights")
                    self.ne16_weight_entries[f"{layer_name}::qkv"] = packed_entry
                    bias_corr_entry = save_binary(f"{layer_name}_qkv_ne16_bias_corr.bin", bias_corrected,
                                                   f"{layer_name} QKV NE16 corrected bias (INT32)")
                    self.ne16_bias_entries[f"{layer_name}::qkv"] = bias_corr_entry
                    # HW outquant scale
                    scale_w = layer_data.get('qkv_scale_weight', 1.0)
                    scale_output = layer_data.get('qkv_scale_output', 1.0)
                    if scale_output > 0:
                        hw_scale, hw_scale_shift = compute_ne16_requant_params(scale_x, scale_w, scale_output, qkv_out)
                        scale_entry = save_binary(f"{layer_name}_qkv_ne16_hw_scale.bin", hw_scale,
                                                  f"{layer_name} QKV NE16 HW scale")
                        self.ne16_scale_entries[f"{layer_name}::qkv"] = scale_entry
                        scale_shift_entry = save_binary(f"{layer_name}_qkv_ne16_hw_scale_shift.bin", hw_scale_shift,
                                                        f"{layer_name} QKV NE16 HW scale_shift")
                        self.ne16_scale_shift_entries[f"{layer_name}::qkv"] = scale_shift_entry
                    self.ne16_eligible_layers.add(f"{layer_name}::qkv")
                    print(f"    → NE16 packed: {layer_name}::qkv ({qkv_in}→{qkv_out})")

                # Output projection weights (INT8)
                out_weight_int8 = None
                out_bias_int32 = None
                if 'out_weight_int8' in layer_data:
                    out_weight_int8 = np.array(layer_data['out_weight_int8'], dtype=np.int8)
                    entry = save_binary(f"{layer_name}_out_weight.bin", out_weight_int8,
                                        f"{layer_name} OUT weight (INT8)")
                    self.weight_entries[f"{layer_name}::out"] = entry

                # Output projection bias (INT32, uses V scale for input)
                if 'out_bias_fp32' in layer_data and layer_data['out_bias_fp32'] is not None:
                    bias_fp32 = np.array(layer_data['out_bias_fp32'], dtype=np.float32)
                    scale_w = layer_data.get('out_scale_weight', 1.0)
                    # Output projection input comes from context (at V scale)
                    scale_v = layer_data.get('v_scale_output', 1.0)
                    scale_bias = scale_v * scale_w
                    out_bias_int32 = np.round(bias_fp32 / scale_bias).astype(np.int32)
                    entry = save_binary(f"{layer_name}_out_bias.bin", out_bias_int32,
                                        f"{layer_name} OUT bias (INT32)")
                    self.bias_entries[f"{layer_name}::out"] = entry

                # NE16 packing for output projection
                if self.use_ne16 and out_weight_int8 is not None and NE16_PACKING_AVAILABLE:
                    out_in = embed_dim
                    out_out = embed_dim
                    if out_bias_int32 is None:
                        out_bias_int32 = np.zeros(out_out, dtype=np.int32)
                    packed_weights, bias_corrected = ne16_pack_linear_weights(out_weight_int8, out_bias_int32)
                    packed_entry = save_binary(f"{layer_name}_out_ne16_packed.bin", packed_weights,
                                               f"{layer_name} OUT NE16 packed weights")
                    self.ne16_weight_entries[f"{layer_name}::out"] = packed_entry
                    bias_corr_entry = save_binary(f"{layer_name}_out_ne16_bias_corr.bin", bias_corrected,
                                                   f"{layer_name} OUT NE16 corrected bias (INT32)")
                    self.ne16_bias_entries[f"{layer_name}::out"] = bias_corr_entry
                    # HW outquant scale
                    scale_w = layer_data.get('out_scale_weight', 1.0)
                    scale_output = layer_data.get('scale_output', 1.0)
                    if scale_output > 0:
                        hw_scale, hw_scale_shift = compute_ne16_requant_params(scale_v, scale_w, scale_output, out_out)
                        scale_entry = save_binary(f"{layer_name}_out_ne16_hw_scale.bin", hw_scale,
                                                  f"{layer_name} OUT NE16 HW scale")
                        self.ne16_scale_entries[f"{layer_name}::out"] = scale_entry
                        scale_shift_entry = save_binary(f"{layer_name}_out_ne16_hw_scale_shift.bin", hw_scale_shift,
                                                        f"{layer_name} OUT NE16 HW scale_shift")
                        self.ne16_scale_shift_entries[f"{layer_name}::out"] = scale_shift_entry
                    self.ne16_eligible_layers.add(f"{layer_name}::out")
                    print(f"    → NE16 packed: {layer_name}::out ({out_in}→{out_out})")

            elif layer_type == 'LayerNorm':
                # LayerNorm FP32 weight (gamma) and bias (beta)
                weight_path = self.weights_dir / f"{layer_name}_weight_fp32.npy"
                if weight_path.exists():
                    weight_fp32 = np.load(weight_path)
                    entry = save_binary(f"{layer_name}_weight.bin", weight_fp32,
                                        f"{layer_name} weight (FP32)")
                    self.weight_entries[layer_name] = entry

                bias_path = self.weights_dir / f"{layer_name}_bias_fp32.npy"
                if bias_path.exists():
                    bias_fp32 = np.load(bias_path)
                    entry = save_binary(f"{layer_name}_bias.bin", bias_fp32,
                                        f"{layer_name} bias (FP32)")
                    self.bias_entries[layer_name] = entry

            elif layer_type == 'RMSNorm':
                # RMSNorm FP32 weight (gamma only, no bias)
                weight_path = self.weights_dir / f"{layer_name}_weight_fp32.npy"
                if weight_path.exists():
                    weight_fp32 = np.load(weight_path)
                    entry = save_binary(f"{layer_name}_weight.bin", weight_fp32,
                                        f"{layer_name} weight (FP32)")
                    self.weight_entries[layer_name] = entry

            elif layer_type == 'GroupNorm':
                # GroupNorm FP32 weight (gamma) and bias (beta), per-channel
                weight_path = self.weights_dir / f"{layer_name}_weight_fp32.npy"
                if weight_path.exists():
                    weight_fp32 = np.load(weight_path)
                    entry = save_binary(f"{layer_name}_weight.bin", weight_fp32,
                                        f"{layer_name} weight (FP32)")
                    self.weight_entries[layer_name] = entry

                bias_path = self.weights_dir / f"{layer_name}_bias_fp32.npy"
                if bias_path.exists():
                    bias_fp32 = np.load(bias_path)
                    entry = save_binary(f"{layer_name}_bias.bin", bias_fp32,
                                        f"{layer_name} bias (FP32)")
                    self.bias_entries[layer_name] = entry

            elif layer_type == 'Conv1dDepthwise':
                # Conv1D Depthwise weight (INT8)
                weight_path = self.weights_dir / f"{layer_name}_weight_int8.npy"
                if weight_path.exists():
                    weight_int8 = np.load(weight_path)
                    entry = save_binary(f"{layer_name}_weight.bin", weight_int8,
                                        f"{layer_name} weight (INT8)")
                    self.weight_entries[layer_name] = entry

                # Conv1D Depthwise bias (INT32, quantized from FP32)
                bias_path = self.weights_dir / f"{layer_name}_bias_fp32.npy"
                if bias_path.exists():
                    bias_fp32 = np.load(bias_path)
                    # Quantize bias to INT32 using scale_x * scale_w
                    scale_x = layer_data.get('scale_input', 1.0)
                    scale_w = layer_data.get('scale_weight', 1.0)
                    scale_bias = scale_x * scale_w
                    bias_int32 = np.round(bias_fp32 / scale_bias).astype(np.int32)
                    entry = save_binary(f"{layer_name}_bias.bin", bias_int32,
                                        f"{layer_name} bias (INT32)")
                    self.bias_entries[layer_name] = entry

            elif layer_type == 'SiLU':
                # Generate and save SiLU LUT
                scale_in = layer_data.get('scale_input', 0.1)
                scale_out = layer_data.get('scale_output', 0.1)
                lut = self._generate_silu_lut(scale_in, scale_out)
                entry = save_binary(f"{layer_name}_lut.bin", lut,
                                    f"{layer_name} SiLU LUT (INT8)")
                self.weight_entries[f"{layer_name}::lut"] = entry

            elif layer_type == 'SSM':
                # SSM parameters for complete SSM layer

                # x_proj_weight (INT8)
                x_proj_path = self.weights_dir / f"{layer_name}_x_proj_weight_int8.npy"
                if x_proj_path.exists():
                    x_proj_weight = np.load(x_proj_path)
                    entry = save_binary(f"{layer_name}_x_proj_weight.bin", x_proj_weight,
                                        f"{layer_name} x_proj weight (INT8)")
                    self.weight_entries[f"{layer_name}::x_proj_weight"] = entry

                # dt_proj_weight (INT8 - kept as INT8 for proper quantized matmul)
                dt_weight_path = self.weights_dir / f"{layer_name}_dt_proj_weight_int8.npy"
                if dt_weight_path.exists():
                    dt_weight = np.load(dt_weight_path).astype(np.int8)
                    entry = save_binary(f"{layer_name}_dt_proj_weight.bin", dt_weight,
                                        f"{layer_name} dt_proj weight (INT8)")
                    self.weight_entries[f"{layer_name}::dt_proj_weight"] = entry

                # dt_proj_bias (FP32)
                dt_bias_path = self.weights_dir / f"{layer_name}_dt_proj_bias_fp32.npy"
                if dt_bias_path.exists():
                    dt_bias = np.load(dt_bias_path).astype(np.float32)
                    entry = save_binary(f"{layer_name}_dt_proj_bias.bin", dt_bias,
                                        f"{layer_name} dt_proj bias (FP32)")
                    self.weight_entries[f"{layer_name}::dt_proj_bias"] = entry

                    # I-Mamba step 6: Q16.16 fixed-point bias (always fits INT32)
                    # Q16.16 format: 16 integer bits, 16 fractional bits
                    # bias_q16_16 = round(bias_fp32 * 65536)
                    # Bias range [-7, +3] -> [-458752, +196608] (fits INT32 easily)
                    scale_x = layer_data.get('scale_input', 0.007812)  # Default ~1/128
                    scale_x_proj = layer_data.get('x_proj_scale_weight', 0.001)
                    scale_dt_proj = layer_data.get('dt_proj_scale_weight', 0.01)
                    scale_output = layer_data.get('scale_output', scale_x)  # Output scale

                    # Q16.16 bias (always works)
                    dt_bias_q16_16 = np.round(dt_bias * 65536.0).astype(np.int32)
                    entry_q16 = save_binary(f"{layer_name}_dt_proj_bias_q16_16.bin", dt_bias_q16_16,
                                            f"{layer_name} dt_proj bias (Q16.16 INT32, I-Mamba step 6)")
                    self.weight_entries[f"{layer_name}::dt_proj_bias_q16_16"] = entry_q16

                    # Compute dt_acc to Q16.16 conversion scale factor
                    # combined_scale (full precision) = (scale_x * scale_x_proj) * scale_dt_proj
                    # To convert dt_acc to Q16.16: dt_acc_q16_16 = dt_acc * combined_scale * 65536
                    # Use fixed-point: dt_acc_q16_16 = (dt_acc * DT_SCALE_Q) >> DT_SCALE_SHIFT
                    combined_scale_full = (scale_x * scale_x_proj) * scale_dt_proj
                    DT_SCALE_SHIFT = 24  # 24-bit shift for good precision
                    dt_scale_q = int(round(combined_scale_full * 65536.0 * (1 << DT_SCALE_SHIFT)))

                    # Store scale factor for kernel (will be added to layer spec)
                    self.ssm_dt_scale_q[layer_name] = dt_scale_q
                    self.ssm_dt_scale_shift[layer_name] = DT_SCALE_SHIFT

                    # I-Mamba step 8: Precompute B/C to Q15 scale factor
                    # bc_scale_factor = x_proj_scale * 32768 * 2^BC_SHIFT
                    # where x_proj_scale = scale_x * scale_x_proj
                    BC_SHIFT = 16
                    bc_scale_factor = int(round(scale_x * scale_x_proj * 32768.0 * (1 << BC_SHIFT)))
                    self.ssm_bc_scale_factor[layer_name] = bc_scale_factor

                    # I-Mamba step 9: Precompute output conversion scale factor
                    # output = y_acc * scale_x / (32768 * scale_output)
                    # output_scale_q = scale_x / (32768 * scale_output) * 2^OUTPUT_SHIFT
                    OUTPUT_SHIFT = 24
                    output_scale_q = int(round(scale_x / (32768.0 * scale_output) * (1 << OUTPUT_SHIFT)))
                    self.ssm_output_scale_q[layer_name] = output_scale_q

                    print(f"  step : Q16.16 bias range [{dt_bias_q16_16.min()}, {dt_bias_q16_16.max()}], "
                          f"dt_scale_q={dt_scale_q}, shift={DT_SCALE_SHIFT}")
                    print(f"  step : bc_scale_factor={bc_scale_factor}")
                    print(f"  step : output_scale_q={output_scale_q}, shift={OUTPUT_SHIFT}")

                # A parameter: Pre-compute A = -exp(A_log) for I-Mamba step 2b
                # This eliminates the runtime -exp() computation in the SSM kernel
                a_log_path = self.weights_dir / f"{layer_name}_A_log_fp32.npy"
                if a_log_path.exists():
                    a_log = np.load(a_log_path).astype(np.float32)
                    A_param = -np.exp(a_log)  # I-Mamba: Pre-compute A = -exp(A_log)
                    entry = save_binary(f"{layer_name}_A.bin", A_param,
                                        f"{layer_name} A (FP32, pre-computed)")
                    self.weight_entries[f"{layer_name}::A"] = entry
                    # I-Mamba step 4: Q15 version of A for full dyadic SSM
                    # A is negative (decay factor), typically in range [-1, 0)
                    # Clamp to Q15 range to avoid overflow
                    a_q15 = np.clip(np.round(A_param * 32768), -32768, 32767).astype(np.int16)
                    entry_q15 = save_binary(f"{layer_name}_A_q15.bin", a_q15,
                                            f"{layer_name} A (Q15)")
                    self.weight_entries[f"{layer_name}::A_q15"] = entry_q15

                # D (FP32 for current path, Q15 for I-Mamba)
                d_path = self.weights_dir / f"{layer_name}_D_fp32.npy"
                if d_path.exists():
                    d_param = np.load(d_path).astype(np.float32)
                    # FP32 copy retained for consumers that expect float parameters.
                    entry = save_binary(f"{layer_name}_D.bin", d_param,
                                        f"{layer_name} D (FP32)")
                    self.weight_entries[f"{layer_name}::D"] = entry
                    # I-Mamba step 2c: Q15 version for dyadic arithmetic
                    d_q15 = np.clip(np.round(d_param * 32768), -32768, 32767).astype(np.int16)
                    entry_q15 = save_binary(f"{layer_name}_D_q15.bin", d_q15,
                                            f"{layer_name} D (Q15)")
                    self.weight_entries[f"{layer_name}::D_q15"] = entry_q15

                # I-Mamba step 4: Generate Softplus and Exp LUTs for standalone SSM
                # Generate Q8.8 Softplus LUT for dt (INT16)
                softplus_scale_in = 0.1  # Fixed scale for I-Mamba
                softplus_lut = self._generate_softplus_lut_q8_8(softplus_scale_in)
                entry = save_binary(f"{layer_name}_softplus_lut.bin", softplus_lut,
                                    f"{layer_name} Softplus LUT (Q8.8 INT16)")
                self.weight_entries[f"{layer_name}::softplus_lut"] = entry

                # Generate Q15 Exp LUT for SSM discretization (dA = exp(dt * A))
                exp_scale_in = 0.1  # Fixed scale for I-Mamba
                exp_lut = self._generate_exp_neg_lut_q15(exp_scale_in)
                entry = save_binary(f"{layer_name}_exp_lut.bin", exp_lut,
                                    f"{layer_name} Exp LUT (Q15 INT16)")
                self.weight_entries[f"{layer_name}::exp_lut"] = entry

                # Build SSM entry for template - tracking parameter indices and sizes
                ssm_entry = {
                    'c_name': self.sanitize_c_name(layer_name),
                    'layer_name': layer_name,
                    'x_proj_weight_index': self.weight_entries.get(f"{layer_name}::x_proj_weight", {}).get('index'),
                    'x_proj_weight_elements': np.load(x_proj_path).size if x_proj_path.exists() else 0,
                    'dt_proj_weight_index': self.weight_entries.get(f"{layer_name}::dt_proj_weight", {}).get('index'),
                    'dt_proj_weight_elements': np.load(dt_weight_path).size if dt_weight_path.exists() else 0,
                    'dt_proj_bias_index': self.weight_entries.get(f"{layer_name}::dt_proj_bias", {}).get('index'),
                    'dt_proj_bias_elements': np.load(dt_bias_path).size if dt_bias_path.exists() else 0,
                    'A_index': self.weight_entries.get(f"{layer_name}::A", {}).get('index'),
                    'A_elements': np.load(a_log_path).size if a_log_path.exists() else 0,
                    # I-Mamba step 4: Q15 A for full dyadic SSM
                    'A_q15_index': self.weight_entries.get(f"{layer_name}::A_q15", {}).get('index'),
                    'D_index': self.weight_entries.get(f"{layer_name}::D", {}).get('index'),
                    'D_elements': np.load(d_path).size if d_path.exists() else 0,
                    # I-Mamba step 2c: Q15 D for dyadic arithmetic
                    'D_q15_index': self.weight_entries.get(f"{layer_name}::D_q15", {}).get('index'),
                    # I-Mamba step 4: LUT indices for full integer SSM
                    'softplus_lut_index': self.weight_entries.get(f"{layer_name}::softplus_lut", {}).get('index'),
                    'exp_lut_index': self.weight_entries.get(f"{layer_name}::exp_lut", {}).get('index'),
                    # I-Mamba step 6: Q16.16 bias and scale factors for full integer dt_proj
                    'dt_proj_bias_q16_16_index': self.weight_entries.get(f"{layer_name}::dt_proj_bias_q16_16", {}).get('index'),
                    'dt_scale_q': self.ssm_dt_scale_q.get(layer_name, 0),
                    'dt_scale_shift': self.ssm_dt_scale_shift.get(layer_name, 24),
                }
                self.ssm_entries.append(ssm_entry)

            elif layer_type == 'MambaBlock':
                # Full MambaBlock has multiple weight components
                scale_x = layer_data.get('scale_input', 1.0)

                # in_proj weight (INT8)
                weight_path = self.weights_dir / f"{layer_name}_in_proj_weight_int8.npy"
                if weight_path.exists():
                    weight = np.load(weight_path)
                    entry = save_binary(f"{layer_name}_in_proj_weight.bin", weight,
                                        f"{layer_name} in_proj weight (INT8)")
                    self.weight_entries[f"{layer_name}::in_proj_weight"] = entry

                # conv1d weight (INT8)
                weight_path = self.weights_dir / f"{layer_name}_conv1d_weight_int8.npy"
                if weight_path.exists():
                    weight = np.load(weight_path)
                    entry = save_binary(f"{layer_name}_conv1d_weight.bin", weight,
                                        f"{layer_name} conv1d weight (INT8)")
                    self.weight_entries[f"{layer_name}::conv1d_weight"] = entry

                # conv1d bias (INT32, quantized from FP32)
                bias_path = self.weights_dir / f"{layer_name}_conv1d_bias_fp32.npy"
                if bias_path.exists():
                    bias_fp32 = np.load(bias_path)
                    # Scale = in_proj_scale_output * conv1d_scale_weight
                    in_proj_scale_out = layer_data.get('in_proj_scale_output', scale_x)
                    conv1d_scale_w = layer_data.get('conv1d_scale_weight', 1.0)
                    scale_bias = in_proj_scale_out * conv1d_scale_w
                    bias_int32 = np.round(bias_fp32 / scale_bias).astype(np.int32)
                    entry = save_binary(f"{layer_name}_conv1d_bias.bin", bias_int32,
                                        f"{layer_name} conv1d bias (INT32)")
                    self.weight_entries[f"{layer_name}::conv1d_bias"] = entry

                # SSM sub-module weights (have ssm_ prefix in extracted files)
                # x_proj weight (INT8)
                weight_path = self.weights_dir / f"{layer_name}_ssm_x_proj_weight_int8.npy"
                if weight_path.exists():
                    weight = np.load(weight_path)
                    entry = save_binary(f"{layer_name}_x_proj_weight.bin", weight,
                                        f"{layer_name} x_proj weight (INT8)")
                    self.weight_entries[f"{layer_name}::x_proj_weight"] = entry

                # dt_proj weight (INT8 - kept as INT8 for proper quantized matmul)
                weight_path = self.weights_dir / f"{layer_name}_ssm_dt_proj_weight_int8.npy"
                if weight_path.exists():
                    weight = np.load(weight_path).astype(np.int8)
                    entry = save_binary(f"{layer_name}_dt_proj_weight.bin", weight,
                                        f"{layer_name} dt_proj weight (INT8)")
                    self.weight_entries[f"{layer_name}::dt_proj_weight"] = entry

                # dt_proj bias (FP32)
                bias_path = self.weights_dir / f"{layer_name}_ssm_dt_proj_bias_fp32.npy"
                if bias_path.exists():
                    bias = np.load(bias_path).astype(np.float32)
                    entry = save_binary(f"{layer_name}_dt_proj_bias.bin", bias,
                                        f"{layer_name} dt_proj bias (FP32)")
                    self.weight_entries[f"{layer_name}::dt_proj_bias"] = entry

                    # I-Mamba step 6: Q16.16 fixed-point bias (always fits INT32)
                    # For MambaBlock: scale_x = in_proj_scale_output, scale_x_proj = ssm_x_proj_scale_weight
                    scale_x = layer_data.get('in_proj_scale_output', layer_data.get('scale_input', 0.007812))
                    scale_x_proj = layer_data.get('ssm_x_proj_scale_weight', 0.001)
                    scale_dt_proj = layer_data.get('ssm_dt_proj_scale_weight', 0.01)
                    scale_output = layer_data.get('ssm_scale_output', scale_x)  # I-Mamba step 9

                    # Q16.16 bias (always works)
                    dt_bias_q16_16 = np.round(bias * 65536.0).astype(np.int32)
                    entry_q16 = save_binary(f"{layer_name}_dt_proj_bias_q16_16.bin", dt_bias_q16_16,
                                            f"{layer_name} dt_proj bias (Q16.16 INT32, I-Mamba step 6)")
                    self.weight_entries[f"{layer_name}::dt_proj_bias_q16_16"] = entry_q16

                    # Compute dt_acc to Q16.16 conversion scale factor
                    combined_scale_full = (scale_x * scale_x_proj) * scale_dt_proj
                    DT_SCALE_SHIFT = 24
                    dt_scale_q = int(round(combined_scale_full * 65536.0 * (1 << DT_SCALE_SHIFT)))

                    # Store scale factor for kernel
                    self.ssm_dt_scale_q[layer_name] = dt_scale_q
                    self.ssm_dt_scale_shift[layer_name] = DT_SCALE_SHIFT

                    # I-Mamba step 8: Precompute B/C to Q15 scale factor
                    BC_SHIFT = 16
                    bc_scale_factor = int(round(scale_x * scale_x_proj * 32768.0 * (1 << BC_SHIFT)))
                    self.ssm_bc_scale_factor[layer_name] = bc_scale_factor

                    # I-Mamba step 9: Precompute output conversion scale factor
                    OUTPUT_SHIFT = 24
                    output_scale_q = int(round(scale_x / (32768.0 * scale_output) * (1 << OUTPUT_SHIFT)))
                    self.ssm_output_scale_q[layer_name] = output_scale_q

                    print(f"  step : Q16.16 bias range [{dt_bias_q16_16.min()}, {dt_bias_q16_16.max()}], "
                          f"dt_scale_q={dt_scale_q}, shift={DT_SCALE_SHIFT}")
                    print(f"  step : bc_scale_factor={bc_scale_factor}")
                    print(f"  step : output_scale_q={output_scale_q}, shift={OUTPUT_SHIFT}")

                # A parameter: Pre-compute A = -exp(A_log) for I-Mamba step 2b
                a_log_path = self.weights_dir / f"{layer_name}_ssm_A_log_fp32.npy"
                if a_log_path.exists():
                    a_log = np.load(a_log_path).astype(np.float32)
                    A_param = -np.exp(a_log)  # I-Mamba: Pre-compute A = -exp(A_log)
                    entry = save_binary(f"{layer_name}_A.bin", A_param,
                                        f"{layer_name} A (FP32, pre-computed)")
                    self.weight_entries[f"{layer_name}::A"] = entry
                    # I-Mamba step 4: Q15 version of A for full dyadic SSM
                    # A is negative (decay factor), typically in range [-1, 0)
                    a_q15 = np.clip(np.round(A_param * 32768), -32768, 32767).astype(np.int16)
                    entry_q15 = save_binary(f"{layer_name}_A_q15.bin", a_q15,
                                            f"{layer_name} A (Q15)")
                    self.weight_entries[f"{layer_name}::A_q15"] = entry_q15

                # D (FP32 for current path, Q15 for I-Mamba)
                d_path = self.weights_dir / f"{layer_name}_ssm_D_fp32.npy"
                if d_path.exists():
                    d_param = np.load(d_path).astype(np.float32)
                    # FP32 copy retained for consumers that expect float parameters.
                    entry = save_binary(f"{layer_name}_D.bin", d_param,
                                        f"{layer_name} D (FP32)")
                    self.weight_entries[f"{layer_name}::D"] = entry
                    # I-Mamba step 2c: Q15 version for dyadic arithmetic
                    d_q15 = np.clip(np.round(d_param * 32768), -32768, 32767).astype(np.int16)
                    entry_q15 = save_binary(f"{layer_name}_D_q15.bin", d_q15,
                                            f"{layer_name} D (Q15)")
                    self.weight_entries[f"{layer_name}::D_q15"] = entry_q15

                # out_proj weight (INT8)
                weight_path = self.weights_dir / f"{layer_name}_out_proj_weight_int8.npy"
                if weight_path.exists():
                    weight = np.load(weight_path)
                    entry = save_binary(f"{layer_name}_out_proj_weight.bin", weight,
                                        f"{layer_name} out_proj weight (INT8)")
                    self.weight_entries[f"{layer_name}::out_proj_weight"] = entry

                # Generate SiLU LUT for this MambaBlock (INT8, for activation after conv1d)
                scale_in = layer_data.get('scale_input', 0.1)
                scale_out = layer_data.get('scale_output', 0.1)
                lut = self._generate_silu_lut(scale_in, scale_out)
                entry = save_binary(f"{layer_name}_silu_lut.bin", lut,
                                    f"{layer_name} SiLU LUT (INT8)")
                self.weight_entries[f"{layer_name}::silu_lut"] = entry

                # Generate Q13 SiLU Gate LUT for gating (INT16, for ssm_out * silu(z) >> 13)
                z_scale = layer_data.get('in_proj_scale_output', scale_in)
                gate_lut_q13 = self._generate_silu_gate_lut_q13(z_scale)
                entry = save_binary(f"{layer_name}_silu_gate_lut_q13.bin", gate_lut_q13,
                                    f"{layer_name} SiLU Gate LUT (Q13 INT16)")
                self.weight_entries[f"{layer_name}::silu_gate_lut_q13"] = entry

                # Generate Q8.8 Softplus LUT for SSM dt (INT16, for integer softplus)
                # Input scale: dt_proj output typically in range [-10, 10], use 0.1 to cover [-12.8, 12.7]
                softplus_scale_in = 0.1  # Fixed scale for I-Mamba step 1
                softplus_lut = self._generate_softplus_lut_q8_8(softplus_scale_in)
                entry = save_binary(f"{layer_name}_softplus_lut.bin", softplus_lut,
                                    f"{layer_name} Softplus LUT (Q8.8 INT16)")
                self.weight_entries[f"{layer_name}::softplus_lut"] = entry

                # Generate Q15 Exp LUT for SSM discretization (dA = exp(dt * A))
                # Input: dt * A where dt > 0, A < 0, so input is negative
                # Range: typically -10 to 0, use scale 0.1 to cover [-12.8, 12.7]
                exp_scale_in = 0.1  # Fixed scale for I-Mamba step 2
                exp_lut = self._generate_exp_neg_lut_q15(exp_scale_in)
                entry = save_binary(f"{layer_name}_exp_lut.bin", exp_lut,
                                    f"{layer_name} Exp LUT (Q15 INT16)")
                self.weight_entries[f"{layer_name}::exp_lut"] = entry

            elif layer_type == 'MambaWrapper':
                # Bidirectional MambaWrapper: generate binaries for both fwd and rev directions
                scale_x = layer_data.get('scale_input', 1.0)

                for direction in ['fwd', 'rev']:
                    src_prefix = f"{direction}_"  # Key prefix in layer_data
                    entry_prefix = f"{layer_name}_{direction}"  # Key prefix for weight_entries

                    # in_proj weight (INT8 or packed 2-bit)
                    weight_key = f'{src_prefix}in_proj_weight_int8'
                    packed_key = f'{src_prefix}in_proj_weight_packed_2bit'
                    bit_width_key = f'{src_prefix}in_proj_weight_bit_width'
                    bit_width = layer_data.get(bit_width_key, 8)

                    if packed_key in layer_data and bit_width == 2:
                        # Save packed 2-bit weights (4 weights per byte)
                        weight = np.array(layer_data[packed_key]).astype(np.uint8)
                        entry = save_binary(f"{layer_name}_{direction}_in_proj_weight.bin", weight,
                                            f"{layer_name} {direction} in_proj weight (packed 2-bit)")
                        self.weight_entries[f"{entry_prefix}::in_proj_weight"] = entry
                        self.weight_entries[f"{entry_prefix}::in_proj_weight_bit_width"] = 2
                    elif weight_key in layer_data:
                        weight = np.array(layer_data[weight_key]).astype(np.int8)
                        entry = save_binary(f"{layer_name}_{direction}_in_proj_weight.bin", weight,
                                            f"{layer_name} {direction} in_proj weight (INT8)")
                        self.weight_entries[f"{entry_prefix}::in_proj_weight"] = entry
                        self.weight_entries[f"{entry_prefix}::in_proj_weight_bit_width"] = 8

                    # conv1d weight (INT8)
                    weight_key = f'{src_prefix}conv1d_weight_int8'
                    if weight_key in layer_data:
                        weight = np.array(layer_data[weight_key]).astype(np.int8)
                        entry = save_binary(f"{layer_name}_{direction}_conv1d_weight.bin", weight,
                                            f"{layer_name} {direction} conv1d weight (INT8)")
                        self.weight_entries[f"{entry_prefix}::conv1d_weight"] = entry

                    # conv1d bias (INT32, quantized from FP32)
                    bias_key = f'{src_prefix}conv1d_bias_fp32'
                    if bias_key in layer_data and layer_data[bias_key] is not None:
                        bias_fp32 = np.array(layer_data[bias_key])
                        in_proj_scale_out = layer_data.get(f'{src_prefix}in_proj_scale_output', scale_x)
                        conv1d_scale_w = layer_data.get(f'{src_prefix}conv1d_scale_weight', 1.0)
                        scale_bias = in_proj_scale_out * conv1d_scale_w
                        bias_int32 = np.round(bias_fp32 / scale_bias).astype(np.int32)
                        entry = save_binary(f"{layer_name}_{direction}_conv1d_bias.bin", bias_int32,
                                            f"{layer_name} {direction} conv1d bias (INT32)")
                        self.weight_entries[f"{entry_prefix}::conv1d_bias"] = entry

                    # SSM x_proj weight (INT8 or packed 2-bit)
                    weight_key = f'{src_prefix}ssm_x_proj_weight_int8'
                    packed_key = f'{src_prefix}ssm_x_proj_weight_packed_2bit'
                    bit_width_key = f'{src_prefix}ssm_x_proj_weight_bit_width'
                    bit_width = layer_data.get(bit_width_key, 8)

                    if packed_key in layer_data and bit_width == 2:
                        weight = np.array(layer_data[packed_key]).astype(np.uint8)
                        entry = save_binary(f"{layer_name}_{direction}_x_proj_weight.bin", weight,
                                            f"{layer_name} {direction} x_proj weight (packed 2-bit)")
                        self.weight_entries[f"{entry_prefix}::x_proj_weight"] = entry
                        self.weight_entries[f"{entry_prefix}::x_proj_weight_bit_width"] = 2
                    elif weight_key in layer_data:
                        weight = np.array(layer_data[weight_key]).astype(np.int8)
                        entry = save_binary(f"{layer_name}_{direction}_x_proj_weight.bin", weight,
                                            f"{layer_name} {direction} x_proj weight (INT8)")
                        self.weight_entries[f"{entry_prefix}::x_proj_weight"] = entry
                        self.weight_entries[f"{entry_prefix}::x_proj_weight_bit_width"] = 8

                    # SSM dt_proj weight (INT8 or packed 2-bit)
                    weight_key = f'{src_prefix}ssm_dt_proj_weight_int8'
                    packed_key = f'{src_prefix}ssm_dt_proj_weight_packed_2bit'
                    bit_width_key = f'{src_prefix}ssm_dt_proj_weight_bit_width'
                    bit_width = layer_data.get(bit_width_key, 8)

                    if packed_key in layer_data and bit_width == 2:
                        weight = np.array(layer_data[packed_key]).astype(np.uint8)
                        entry = save_binary(f"{layer_name}_{direction}_dt_proj_weight.bin", weight,
                                            f"{layer_name} {direction} dt_proj weight (packed 2-bit)")
                        self.weight_entries[f"{entry_prefix}::dt_proj_weight"] = entry
                        self.weight_entries[f"{entry_prefix}::dt_proj_weight_bit_width"] = 2
                    elif weight_key in layer_data:
                        weight = np.array(layer_data[weight_key]).astype(np.int8)
                        entry = save_binary(f"{layer_name}_{direction}_dt_proj_weight.bin", weight,
                                            f"{layer_name} {direction} dt_proj weight (INT8)")
                        self.weight_entries[f"{entry_prefix}::dt_proj_weight"] = entry
                        self.weight_entries[f"{entry_prefix}::dt_proj_weight_bit_width"] = 8

                    # dt_proj bias (FP32 and Q16.16)
                    bias_key = f'{src_prefix}ssm_dt_proj_bias_fp32'
                    if bias_key in layer_data and layer_data[bias_key] is not None:
                        bias = np.array(layer_data[bias_key]).astype(np.float32)
                        entry = save_binary(f"{layer_name}_{direction}_dt_proj_bias.bin", bias,
                                            f"{layer_name} {direction} dt_proj bias (FP32)")
                        self.weight_entries[f"{entry_prefix}::dt_proj_bias"] = entry

                        # Q16.16 bias
                        dt_bias_q16_16 = np.round(bias * 65536.0).astype(np.int32)
                        entry_q16 = save_binary(f"{layer_name}_{direction}_dt_proj_bias_q16_16.bin", dt_bias_q16_16,
                                                f"{layer_name} {direction} dt_proj bias (Q16.16 INT32)")
                        self.weight_entries[f"{entry_prefix}::dt_proj_bias_q16_16"] = entry_q16

                        # Compute scale factors
                        dir_scale_x = layer_data.get(f'{src_prefix}in_proj_scale_output', scale_x)
                        scale_x_proj = layer_data.get(f'{src_prefix}ssm_x_proj_scale_weight', 0.001)
                        scale_dt_proj = layer_data.get(f'{src_prefix}ssm_dt_proj_scale_weight', 0.01)
                        scale_output = layer_data.get(f'{src_prefix}ssm_scale_output', dir_scale_x)

                        combined_scale_full = (dir_scale_x * scale_x_proj) * scale_dt_proj
                        DT_SCALE_SHIFT = 24
                        dt_scale_q = int(round(combined_scale_full * 65536.0 * (1 << DT_SCALE_SHIFT)))

                        # Store for both fwd and rev using entry_prefix
                        self.ssm_dt_scale_q[entry_prefix] = dt_scale_q
                        self.ssm_dt_scale_shift[entry_prefix] = DT_SCALE_SHIFT

                        BC_SHIFT = 16
                        bc_scale_factor = int(round(dir_scale_x * scale_x_proj * 32768.0 * (1 << BC_SHIFT)))
                        self.ssm_bc_scale_factor[entry_prefix] = bc_scale_factor

                        OUTPUT_SHIFT = 24
                        output_scale_q = int(round(dir_scale_x / (32768.0 * scale_output) * (1 << OUTPUT_SHIFT)))
                        self.ssm_output_scale_q[entry_prefix] = output_scale_q

                    # A parameter (Q15)
                    a_log_key = f'{src_prefix}ssm_A_log_fp32'
                    if a_log_key in layer_data:
                        a_log = np.array(layer_data[a_log_key]).astype(np.float32)
                        A_param = -np.exp(a_log)
                        entry = save_binary(f"{layer_name}_{direction}_A.bin", A_param,
                                            f"{layer_name} {direction} A (FP32)")
                        self.weight_entries[f"{entry_prefix}::A"] = entry

                        a_q15 = np.clip(np.round(A_param * 32768), -32768, 32767).astype(np.int16)
                        entry_q15 = save_binary(f"{layer_name}_{direction}_A_q15.bin", a_q15,
                                                f"{layer_name} {direction} A (Q15)")
                        self.weight_entries[f"{entry_prefix}::A_q15"] = entry_q15

                    # D parameter (Q15)
                    d_key = f'{src_prefix}ssm_D_fp32'
                    if d_key in layer_data:
                        d_param = np.array(layer_data[d_key]).astype(np.float32)
                        entry = save_binary(f"{layer_name}_{direction}_D.bin", d_param,
                                            f"{layer_name} {direction} D (FP32)")
                        self.weight_entries[f"{entry_prefix}::D"] = entry

                        d_q15 = np.clip(np.round(d_param * 32768), -32768, 32767).astype(np.int16)
                        entry_q15 = save_binary(f"{layer_name}_{direction}_D_q15.bin", d_q15,
                                                f"{layer_name} {direction} D (Q15)")
                        self.weight_entries[f"{entry_prefix}::D_q15"] = entry_q15

                    # out_proj weight (INT8 or packed 2-bit)
                    weight_key = f'{src_prefix}out_proj_weight_int8'
                    packed_key = f'{src_prefix}out_proj_weight_packed_2bit'
                    bit_width_key = f'{src_prefix}out_proj_weight_bit_width'
                    bit_width = layer_data.get(bit_width_key, 8)

                    if packed_key in layer_data and bit_width == 2:
                        weight = np.array(layer_data[packed_key]).astype(np.uint8)
                        entry = save_binary(f"{layer_name}_{direction}_out_proj_weight.bin", weight,
                                            f"{layer_name} {direction} out_proj weight (packed 2-bit)")
                        self.weight_entries[f"{entry_prefix}::out_proj_weight"] = entry
                        self.weight_entries[f"{entry_prefix}::out_proj_weight_bit_width"] = 2
                    elif weight_key in layer_data:
                        weight = np.array(layer_data[weight_key]).astype(np.int8)
                        entry = save_binary(f"{layer_name}_{direction}_out_proj_weight.bin", weight,
                                            f"{layer_name} {direction} out_proj weight (INT8)")
                        self.weight_entries[f"{entry_prefix}::out_proj_weight"] = entry
                        self.weight_entries[f"{entry_prefix}::out_proj_weight_bit_width"] = 8

                    # LUTs for this direction
                    dir_scale_in = layer_data.get(f'{src_prefix}scale_input', layer_data.get('scale_input', 0.1))
                    dir_scale_out = layer_data.get(f'{src_prefix}scale_output', layer_data.get('scale_output', 0.1))

                    # SiLU LUT (INT8)
                    lut = self._generate_silu_lut(dir_scale_in, dir_scale_out)
                    entry = save_binary(f"{layer_name}_{direction}_silu_lut.bin", lut,
                                        f"{layer_name} {direction} SiLU LUT (INT8)")
                    self.weight_entries[f"{entry_prefix}::silu_lut"] = entry

                    # SiLU Gate LUT (Q13)
                    z_scale = layer_data.get(f'{src_prefix}in_proj_scale_output', dir_scale_in)
                    gate_lut_q13 = self._generate_silu_gate_lut_q13(z_scale)
                    entry = save_binary(f"{layer_name}_{direction}_silu_gate_lut_q13.bin", gate_lut_q13,
                                        f"{layer_name} {direction} SiLU Gate LUT (Q13 INT16)")
                    self.weight_entries[f"{entry_prefix}::silu_gate_lut_q13"] = entry

                    # Softplus LUT (Q8.8)
                    softplus_lut = self._generate_softplus_lut_q8_8(0.1)
                    entry = save_binary(f"{layer_name}_{direction}_softplus_lut.bin", softplus_lut,
                                        f"{layer_name} {direction} Softplus LUT (Q8.8 INT16)")
                    self.weight_entries[f"{entry_prefix}::softplus_lut"] = entry

                    # Exp LUT (Q15)
                    exp_lut = self._generate_exp_neg_lut_q15(0.1)
                    entry = save_binary(f"{layer_name}_{direction}_exp_lut.bin", exp_lut,
                                        f"{layer_name} {direction} Exp LUT (Q15 INT16)")
                    self.weight_entries[f"{entry_prefix}::exp_lut"] = entry


        # Golden output (FP32)
        output_fp32 = np.load(self.test_case_dir / "output_fp32.npy")
        self.golden_entry = save_binary("golden_output.bin", output_fp32, "Golden output (FP32)")

        # Intermediate golden outputs (INT8) for debugging
        intermediate_dir = self.test_case_dir / "intermediate_int8"
        if intermediate_dir.exists():
            npy_files = sorted(intermediate_dir.glob("*.npy"))

            # Skip intermediate goldens for large models to save L2 memory
            # Large transformer models (e.g., TinyMyo with 8 blocks x 8 layers = 64+ intermediate outputs)
            # can consume ~20 MB in L3 when loaded, leaving insufficient L2 for activations
            # Allow deeper CNNs (e.g., ResNet-18 ~76 layers) to keep validation;
            # still guard extremely large transformer stacks.
            MAX_INTERMEDIATE_GOLDENS = 120  # Threshold: >120 layers = skip intermediate validation

            if len(npy_files) > MAX_INTERMEDIATE_GOLDENS:
                print(f"  [WARN]  Skipping {len(npy_files)} intermediate golden outputs (model too large)")
                print(f"      Large models need L2 memory for activations - intermediate validation disabled")
                # Keep intermediate_entries empty to prevent FC from loading golden files
            else:
                # Iterate in topological order (layer_order) to match network.c expectations
                # Previous alphabetical sort caused mismatch between handles[] index and layer_specs iteration
                for layer_name in self.layer_order:
                    npy_filename = f"{layer_name}_int8.npy"
                    npy_path = intermediate_dir / npy_filename

                    if npy_path.exists():
                        data_int8 = np.load(npy_path).astype(np.int8, copy=False)

                        # Convert 4D NCHW tensors to HWC layout if use_hwc_layout is enabled
                        # PyTorch outputs are NCHW, but GAP9 C code uses HWC for efficient access
                        if self.use_hwc_layout and data_int8.ndim == 4:
                            # NCHW (N, C, H, W) -> NHWC (N, H, W, C)
                            data_int8 = np.transpose(data_int8, (0, 2, 3, 1)).copy()

                        bin_name = f"{layer_name}_golden.bin"
                        label = f"{layer_name} output (INT8)"
                        entry = save_binary(bin_name, data_int8, label)
                        entry['layer_name'] = layer_name
                        entry['slot'] = len(self.intermediate_entries)
                        self.intermediate_entries.append(entry)
                        self.intermediate_layer_entries[layer_name] = entry

        print(f"  Total: {len(self.binary_files)} binary files")

    def compute_checksum(self, filepath):
        """Compute simple checksum (sum of bytes, matching example)."""
        with open(filepath, 'rb') as f:
            data = f.read()
        return sum(data) & 0xFFFFFFFF

    def detect_l3_activation_fallback(self):
        """
        Detect when activation buffers exceed L2 capacity and mark for L3 fallback.

        Large transformer models can exceed target L2 when combined with other
        activation buffers. This method:
        1. Calculates total L2 requirement for activation buffers
        2. Identifies oversized buffers that should use L3 instead
        3. Marks them with 'use_l3_fallback' flag for template generation
        """
        print()
        print("Checking activation buffer sizes for L3 fallback...")

        # Target-specific L2 policy for activation fallback.
        l2_total_bytes = self._get_l2_total_bytes_for_fallback()
        l2_reserved_bytes = self._get_l2_activation_reserved_bytes_for_fallback()
        l2_available_for_activations = max(0, l2_total_bytes - l2_reserved_bytes)

        # Single buffer threshold: if one buffer exceeds this, use L3
        # Only use L3 fallback when absolutely necessary.
        single_buffer_l3_threshold = self._get_l3_fallback_single_buffer_threshold_bytes()

        # Calculate total activation buffer size
        total_activation_bytes = 0
        buffer_list = []

        # Regular activation buffers
        for buf in self.activation_buffers:
            size_bytes = buf['numel'] * self._sizeof(buf['ctype'])
            total_activation_bytes += size_bytes
            buffer_list.append({
                'buf': buf,
                'size_bytes': size_bytes,
                'is_pool': False,
            })

        # Shared activation pool buffers
        for pool_buf in self.shared_activation_pool:
            size_bytes = pool_buf['numel'] * self._sizeof(pool_buf['ctype'])
            total_activation_bytes += size_bytes
            buffer_list.append({
                'buf': pool_buf,
                'size_bytes': size_bytes,
                'is_pool': True,
            })

        print(f"  Total activation buffers: {len(buffer_list)}")
        print(f"  Total size: {total_activation_bytes:,} bytes ({total_activation_bytes / 1024 / 1024:.2f} MB)")
        print(
            f"  L2 available: {l2_available_for_activations:,} bytes "
            f"({l2_available_for_activations / 1024 / 1024:.2f} MB)"
        )

        # Check if we need L3 fallback
        if total_activation_bytes <= l2_available_for_activations:
            print(f"  [OK] All activation buffers fit in L2 (no fallback needed)")
            return

        print(
            f"  [WARN]  Activation buffers exceed L2 capacity by "
            f"{(total_activation_bytes - l2_available_for_activations) / 1024:.1f} KB"
        )
        print(f"  Applying L3 fallback strategy...")

        # Sort buffers by size (largest first)
        buffer_list.sort(key=lambda x: x['size_bytes'], reverse=True)

        # Mark oversized buffers for L3 fallback
        l3_fallback_count = 0
        l3_fallback_bytes = 0

        for item in buffer_list:
            buf = item['buf']
            size_bytes = item['size_bytes']

            # Strategy: ONLY move buffers that exceed the single buffer threshold
            # The secondary "total exceeds capacity" check is disabled until streaming is implemented
            needs_l3 = False

            # Skip L3 fallback for buffers marked as l2_required (e.g., L3-tiled weight slabs)
            # These buffers MUST stay in L2 because they receive L3→L2 streaming
            if buf.get('l2_required', False):
                buf['use_l3_fallback'] = False
                print(f"    → {buf['name']}: {size_bytes / 1024:.1f} KB → L2 (l2_required=True, streaming destination)")
                continue

            if size_bytes >= single_buffer_l3_threshold:
                needs_l3 = True
                reason = (
                    "exceeds single buffer threshold "
                    f"({size_bytes / 1024:.1f} KB >= {single_buffer_l3_threshold / 1024:.1f} KB)"
                )
            # DISABLED: Secondary check would move more buffers but L3 streaming isn't implemented
            # elif total_activation_bytes > l2_available_for_activations:
            #     needs_l3 = True
            #     reason = f"total exceeds L2 capacity (moving largest buffer: {size_bytes / 1024:.1f} KB)"

            if needs_l3:
                buf['use_l3_fallback'] = True
                l3_fallback_count += 1
                l3_fallback_bytes += size_bytes
                total_activation_bytes -= size_bytes  # Remove from total to see if we fit now
                print(f"    → {buf['name']}: {size_bytes / 1024:.1f} KB → L3 ({reason})")
            else:
                buf['use_l3_fallback'] = False

        if l3_fallback_count > 0:
            print(f"  [OK] Marked {l3_fallback_count} buffer(s) for L3 fallback ({l3_fallback_bytes / 1024:.1f} KB total)")
            print(f"  Remaining L2 usage: {(total_activation_bytes) / 1024:.1f} KB")
        else:
            print(f"  [OK] No L3 fallback needed after analysis")

    def _sizeof(self, ctype):
        """Return size in bytes for C type."""
        if ctype in ('int8_t', 'uint8_t'):
            return 1
        if ctype in ('int16_t', 'uint16_t'):
            return 2
        if ctype in ('int32_t', 'uint32_t', 'float', 'fp32'):
            return 4
        raise ValueError(f"Unknown C type: {ctype}")

    def _run_memory_level_annotation(self):
        """Run memory-level annotation/report generation."""
        from .memory.annotation_pass import annotate_generator_memory_levels

        report = annotate_generator_memory_levels(self)
        self._memory_levels_ready = True
        print(f"  [OK] Memory-level report: {self.memory_level_report_path}")
        return report

    def _prepare_memory_levels(self):
        """Ensure L3 fallback flags and memory-level annotations are up to date."""
        if self._memory_levels_ready:
            return
        self.detect_l3_activation_fallback()
        self._run_memory_level_annotation()

    def _resolve_planner_policy(self):
        """Resolve planner policy from environment."""
        policy = os.getenv("ARES_PLANNER_POLICY", "arena_first_fit").strip().lower()
        if not policy:
            policy = "arena_first_fit"
        return policy

    def _emit_planner_debug_dump(self):
        """Emit planner debug artifact for diagnostics and replay checks."""
        if self.planner is None:
            return

        debug_path = self.output_dir / "planner_debug_dump.json"
        payload = {
            "policy": getattr(self.planner, "policy", self.planner_policy),
            "backend": getattr(self.planner, "backend_name", None),
            "peak_arena_bytes": self.l2_arena_size,
            "lifetimes": getattr(self.planner, "lifetimes", {}),
            "offsets": getattr(self.planner, "offsets", {}),
            "unresolved_conflicts": getattr(self.planner, "unresolved_conflicts", []),
        }
        with open(debug_path, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")

        self.planner_debug_dump_path = str(debug_path)
        print(f"  [OK] Planner debug dump: {self.planner_debug_dump_path}")

    def _run_memory_planner(self):
        """Execute the memory planning step."""
        allow_experimental = self._is_feature_flag_enabled(
            "ARES_ENABLE_EXPERIMENTAL_PLANNERS", default=False
        )
        requested_policy = self._resolve_planner_policy()
        print(f"\nRunning Memory Planner (policy={requested_policy})...")

        # Preserve default behavior via arena_first_fit policy.
        self.planner_policy = requested_policy
        self.planner = MemoryPlanner(
            self.layer_specs,
            self.activation_buffers,
            self.shared_activation_pool,
            policy=requested_policy,
            allow_experimental=allow_experimental,
        )

        try:
            self.planner.analyze()
        except ValueError as exc:
            if requested_policy != "arena_first_fit":
                print(f"  [Planner] Warning: {exc}")
                print("  [Planner] Falling back to 'arena_first_fit'.")
                self.planner_policy = "arena_first_fit"
                self.planner = MemoryPlanner(
                    self.layer_specs,
                    self.activation_buffers,
                    self.shared_activation_pool,
                    policy=self.planner_policy,
                    allow_experimental=allow_experimental,
                )
                self.planner.analyze()
            else:
                raise

        self.l2_arena_size = self.planner.total_size

        # Apply offsets to buffer objects so the template can use them
        mapped_count = 0
        for name, offset in self.planner.offsets.items():
            # Update regular buffers
            for buf in self.activation_buffers:
                if buf['name'] == name:
                    buf['offset'] = offset
                    mapped_count += 1
            # Update pools
            for buf in self.shared_activation_pool:
                if buf['name'] == name:
                    buf['offset'] = offset
                    mapped_count += 1

        print(f"  [OK] Planned L2 Arena: {self.l2_arena_size} bytes for {mapped_count} buffers")
        print(f"  [OK] Reduction: {mapped_count} mallocs -> 1 malloc")
        self._emit_planner_debug_dump()

        # Export memory-planned checkpoint for replay
        self._write_phase_checkpoint(
            CHECKPOINT_STAGE_POST_MEMORY_PLAN,
            include_memory=True,
        )

    def _emit_headers_from_current_state(self):
        """Render header templates from current generator state."""
        self.render_template("network_data.h.mako", "inc/network_data.h", binary_files=self.binary_files)
        self.render_template("network.h.mako", "inc/network.h", network_info=self.network_info)

    def generate_headers(self):
        print("\nGenerating headers...")
        self._ensure_codegen_metadata()
        self._prepare_memory_levels()
        # Run Memory Planner after memory-level annotations are ready
        self._run_memory_planner()

        self._emit_headers_from_current_state()
        print("  [OK] Generated headers")
    
    def _generate_layer_array(self):
        """Generate the static C array of LayerSpecs."""
        c_code = []
        c_code.append("static const LayerSpec network_layers[] = {")
        
        for spec in self.layer_specs:
            op_type = spec['op']
            c_code.append(f"    // Layer: {spec['name']}")
            c_code.append("    {")
            
            # Map Python op string to C Enum
            if op_type == 'conv2d':
                c_code.append(f"        .type = OP_CONV2D,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.conv2d = {{")
                # Map Conv2D params directly to the struct fields
                c_code.append(f"            .layer_name = \"{spec['name']}\",")
                # Pointers will be set at runtime in network.c based on Arena/Offsets
                # We initialize dimensions and constants here
                c_code.append(f"            .in_h = {spec['in_h']}, .in_w = {spec['in_w']}, .in_ch = {spec['in_ch']},")
                c_code.append(f"            .out_h = {spec['out_h']}, .out_w = {spec['out_w']}, .out_ch = {spec['out_ch']},")
                c_code.append(f"            .kernel_h = {spec.get('kernel_h', spec.get('kernel',1))}, .kernel_w = {spec.get('kernel_w', spec.get('kernel',1))},")
                c_code.append(f"            .stride_h = {spec.get('stride_h', spec.get('stride',1))}, .stride_w = {spec.get('stride_w', spec.get('stride',1))},")
                c_code.append(f"            .pad_h = {spec.get('pad_h', spec.get('padding',0))}, .pad_w = {spec.get('pad_w', spec.get('padding',0))},")
                c_code.append(f"            .groups = {spec.get('groups', 1)},")  # Depthwise: groups=in_ch
                # Tiling info
                if 'tile_config' in spec:
                    tc = spec['tile_config']
                    c_code.append(f"            .tile_h = {tc['tile_h']}, .tile_w = {tc['tile_w']},")
                    c_code.append(f"            .tile_h_halo = {tc['tile_h_with_halo']}, .tile_w_halo = {tc['tile_w_with_halo']},")
                    c_code.append(f"            .num_tiles = {tc['num_tiles']}, .num_tiles_h = {tc['num_tiles_h']}, .num_tiles_w = {tc['num_tiles_w']},")
                    c_code.append(f"            .l1_input_size = {tc['l1_input_bytes']}, .l1_output_size = {tc['l1_output_bytes']},")
                    c_code.append(f"            .out_tile_h = {tc['out_tile_h']}, .out_tile_w = {tc['out_tile_w']},")
                    c_code.append(f"            .l3_tiling_enabled = {1 if tc.get('l3_tiling_enabled') else 0},")
                    if tc.get('l3_tiling_enabled'):
                        c_code.append(f"            .l3_tile_h = {tc['l3_tile_h']}, .l3_tile_h_halo = {tc['l3_tile_h_halo']},")
                        c_code.append(f"            .num_l3_tiles = {tc['num_l3_tiles']},")
                    # L1 weight caching fields
                    c_code.append(f"            .weight_tiling_enabled = {1 if tc.get('weight_tiling_enabled') else 0},")
                    c_code.append(f"            .tile_out_ch = {tc.get('tile_out_ch', 0)},")
                    c_code.append(f"            .num_out_ch_tiles = {tc.get('num_out_ch_tiles', 1)},")
                    c_code.append(f"            .l1_weight_size = {tc.get('l1_weight_bytes', 0)},")
                    # Triple-buffer weight pipeline (eliminates blocking wait on first weight load)
                    c_code.append(f"            .triple_buffer_weights = {1 if tc.get('triple_buffer_weights') else 0},")
                else:
                     c_code.append(f"            .num_tiles = 0, // L2 fallback")
                
                # Quantization
                c_code.append(f"            .scale_input = {spec['scale_input']}f, .scale_weight = {spec['scale_weight']}f, .scale_output = {spec['scale_output']}f,")
                c_code.append(f"            .fusion_relu = {1 if spec.get('fusion_relu') else 0},")
                c_code.append(f"            .fusion_quant = {1 if spec.get('fusion_quant') else 0},")
                qsi = spec.get('quant_scale_in', 0.0)
                qso = spec.get('quant_scale_out', 0.0)
                # Format floats properly: 0.0 -> "0.0f", not "0f"
                qsi_str = f"{qsi:.15g}" if qsi != 0 else "0.0"
                qso_str = f"{qso:.15g}" if qso != 0 else "0.0"
                c_code.append(f"            .quant_scale_in = {qsi_str}f, .quant_scale_out = {qso_str}f,")
                # Conv+MaxPool fusion
                c_code.append(f"            .fusion_maxpool = {1 if spec.get('fusion_maxpool') else 0},")
                if spec.get('fusion_maxpool'):
                    c_code.append(f"            .pool_kernel_h = {spec['pool_kernel_h']}, .pool_kernel_w = {spec['pool_kernel_w']},")
                    c_code.append(f"            .pool_stride_h = {spec['pool_stride_h']}, .pool_stride_w = {spec['pool_stride_w']},")
                    c_code.append(f"            .pool_out_h = {spec['pool_out_h']}, .pool_out_w = {spec['pool_out_w']},")
                    c_code.append(f"            .fused_output_buffer_l2 = NULL,  // Set at runtime")
                layout_str = "LAYOUT_HWC" if self.use_hwc_layout else "LAYOUT_CHW"
                c_code.append(f"            .layout = {layout_str}")
                c_code.append(f"        }}")

            elif op_type == 'linear_int8':
                c_code.append(f"        .type = OP_LINEAR_INT8,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.linear_int8 = {{")
                c_code.append(f"            .layer_name = \"{spec['name']}\",")
                c_code.append(f"            .in_features = {spec['in_features']}, .out_features = {spec['out_features']},")
                c_code.append(f"            .batch_tokens = {spec.get('batch_tokens', 1)},")
                if 'tile_config' in spec:
                    tc = spec['tile_config']
                    c_code.append(f"            .tile_out_features = {tc['tile_out_features']}, .num_tiles = {tc['num_tiles']},")

                    # K-dimension tiling
                    tile_in = tc.get('tile_in_features', spec['in_features'])
                    num_k_tiles = tc.get('num_k_tiles', 1)
                    k_tiling_enabled = 1 if tc.get('k_tiling_enabled', False) else 0
                    c_code.append(f"            .tile_in_features = {tile_in}, .num_k_tiles = {num_k_tiles}, .k_tiling_enabled = {k_tiling_enabled},")

                    # M-dimension tiling (batch/token tiling)
                    tile_batch = tc.get('tile_batch_tokens', spec.get('batch_tokens', 1))
                    num_m_tiles = tc.get('num_m_tiles', 1)
                    m_tiling_enabled = 1 if tc.get('m_tiling_enabled', False) else 0
                    c_code.append(f"            .tile_batch_tokens = {tile_batch}, .num_m_tiles = {num_m_tiles}, .m_tiling_enabled = {m_tiling_enabled},")

                    c_code.append(f"            .l1_input_size = {tc['l1_input_bytes']}, .l1_output_size = {tc['l1_output_bytes']}, .l1_weight_size = {tc['l1_weight_bytes']},")
                    c_code.append(f"            .l3_tiling_enabled = {1 if tc.get('l3_tiling_enabled') else 0},")
                    if tc.get('l3_tiling_enabled'):
                        c_code.append(f"            .l3_tile_out_features = {tc['l3_tile_out_features']},")
                        c_code.append(f"            .num_l3_tiles = {tc['num_l3_tiles']},")
                else:
                    c_code.append(f"            .num_tiles = 0,")
                    c_code.append(f"            .tile_in_features = {spec['in_features']}, .num_k_tiles = 1, .k_tiling_enabled = 0,")
                    c_code.append(f"            .tile_batch_tokens = {spec.get('batch_tokens', 1)}, .num_m_tiles = 1, .m_tiling_enabled = 0,")
                c_code.append(f"            .scale_input = {spec['scale_input']}f, .scale_weight = {spec['scale_weight']}f, .scale_output = {spec['scale_output']}f,")
                c_code.append(f"            .fusion_relu = {1 if spec.get('fusion_relu') else 0},")
                c_code.append(f"            .fusion_quant = {1 if spec.get('fusion_quant') else 0},")
                # ReLU output scale for intermediate rescale
                relu_out_scale = spec.get('relu_output_scale', 0.0)
                relu_out_scale_str = f"{relu_out_scale:.15g}" if relu_out_scale != 0 else "0.0"
                c_code.append(f"            .relu_output_scale = {relu_out_scale_str}f,")
                qsi = spec.get('quant_scale_in', 0.0)
                qso = spec.get('quant_scale_out', 0.0)
                # Format floats properly: 0.0 -> "0.0f", not "0f"
                qsi_str = f"{qsi:.15g}" if qsi != 0 else "0.0"
                qso_str = f"{qso:.15g}" if qso != 0 else "0.0"
                c_code.append(f"            .quant_scale_in = {qsi_str}f, .quant_scale_out = {qso_str}f")
                c_code.append(f"        }}")

            elif op_type == 'linear_fp32':
                c_code.append(f"        .type = OP_LINEAR_FP32,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.linear_fp32 = {{")
                c_code.append(f"            .layer_name = \"{spec['name']}\",")
                c_code.append(f"            .in_features = {spec['in_features']}, .out_features = {spec['out_features']},")
                if 'tile_config' in spec:
                    tc = spec['tile_config']
                    c_code.append(f"            .tile_out_features = {tc['tile_out_features']}, .num_tiles = {tc['num_tiles']},")
                    c_code.append(f"            .l1_input_size = {tc['l1_input_bytes']}, .l1_output_size = {tc['l1_output_bytes']}, .l1_weight_size = {tc['l1_weight_bytes']},")
                    c_code.append(f"            .l3_tiling_enabled = {1 if tc.get('l3_tiling_enabled') else 0},")
                    if tc.get('l3_tiling_enabled'):
                        c_code.append(f"            .l3_tile_out_features = {tc['l3_tile_out_features']},")
                        c_code.append(f"            .num_l3_tiles = {tc['num_l3_tiles']},")
                else:
                    c_code.append(f"            .num_tiles = 0,")
                c_code.append(f"            .scale_input = {spec['scale_input']}f, .scale_weight = {spec['scale_weight']}f")
                c_code.append(f"        }}")

            elif op_type == 'linear_ne16':
                c_code.append(f"        .type = OP_LINEAR_NE16,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.linear_ne16 = {{")
                c_code.append(f"            .batch = 1,")
                c_code.append(f"            .num_tokens = {spec.get('batch_tokens', 1)},")
                c_code.append(f"            .in_features = {spec['in_features']}, .out_features = {spec['out_features']},")
                c_code.append(f"            .out_stride = {spec['out_features']},")
                c_code.append(f"            .scale_input = {spec['scale_input']}f, .scale_weight = {spec['scale_weight']}f, .scale_output = {spec['scale_output']}f,")
                c_code.append(f"            .tile_tokens = {spec.get('ne16_tile_tokens', spec.get('batch_tokens', 1))},")
                # NE16 packed weights and corrected bias will be set at runtime
                c_code.append(f"            .weights_packed = NULL,  // Set at runtime from binary index {spec.get('ne16_packed_weight_index', -1)}")
                c_code.append(f"            .bias_corrected = NULL,  // Set at runtime from binary index {spec.get('ne16_bias_corr_index', -1)}")
                # HW outquant parameters (set at runtime if available)
                use_hw_requant = spec.get('ne16_use_hw_requant', False)
                hw_scale_idx = spec.get('ne16_hw_scale_index', -1)
                hw_scale_shift_idx = spec.get('ne16_hw_scale_shift_index', -1)
                c_code.append(f"            .hw_scale = NULL,  // Set at runtime from binary index {hw_scale_idx}")
                c_code.append(f"            .hw_scale_shift = NULL,  // Set at runtime from binary index {hw_scale_shift_idx}")
                c_code.append(f"            .use_hw_requant = {1 if use_hw_requant else 0},")
                # Calculate scratch buffer sizes (1 byte for HW requant, 4 bytes for SW)
                tile_tokens = spec.get('ne16_tile_tokens', spec.get('batch_tokens', 1))
                in_features = spec['in_features']
                out_features = spec['out_features']
                output_bytes_per_elem = 1 if use_hw_requant else 4
                c_code.append(f"            .scratch_input_size = {tile_tokens * in_features},")
                c_code.append(f"            .scratch_output_size = {tile_tokens * out_features * output_bytes_per_elem}")
                c_code.append(f"        }}")

            elif op_type == 'conv2d_1x1_ne16':
                c_code.append(f"        .type = OP_CONV2D_1X1_NE16,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.conv2d_ne16 = {{")
                c_code.append(f"            .batch = 1,")
                c_code.append(f"            .in_h = {spec['in_h']}, .in_w = {spec['in_w']}, .in_channels = {spec['in_ch']},")
                c_code.append(f"            .out_channels = {spec['out_ch']},")
                c_code.append(f"            .scale_input = {spec['scale_input']}f, .scale_weight = {spec['scale_weight']}f, .scale_output = {spec['scale_output']}f,")
                # NE16 packed weights and corrected bias will be set at runtime
                c_code.append(f"            .weights_packed = NULL,  // Set at runtime from binary index {spec.get('ne16_packed_weight_index', -1)}")
                c_code.append(f"            .bias_corrected = NULL   // Set at runtime from binary index {spec.get('ne16_bias_corr_index', -1)}")
                c_code.append(f"        }}")

            elif op_type == 'conv2d_3x3_ne16':
                c_code.append(f"        .type = OP_CONV2D_3X3_NE16,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.conv2d_ne16 = {{")
                c_code.append(f"            .batch = 1,")
                c_code.append(f"            .in_h = {spec['in_h']}, .in_w = {spec['in_w']}, .in_channels = {spec['in_ch']},")
                c_code.append(f"            .out_channels = {spec['out_ch']},")
                c_code.append(f"            .kernel_h = 3, .kernel_w = 3,")
                c_code.append(f"            .stride_h = {spec['stride_h']}, .stride_w = {spec['stride_w']},")
                c_code.append(f"            .pad_h = {spec['pad_h']}, .pad_w = {spec['pad_w']},")
                c_code.append(f"            .scale_input = {spec['scale_input']}f, .scale_weight = {spec['scale_weight']}f, .scale_output = {spec['scale_output']}f,")
                # NE16 packed weights and corrected bias will be set at runtime
                c_code.append(f"            .weights_packed = NULL,  // Set at runtime from binary index {spec.get('ne16_packed_weight_index', -1)}")
                c_code.append(f"            .bias_corrected = NULL   // Set at runtime from binary index {spec.get('ne16_bias_corr_index', -1)}")
                c_code.append(f"        }}")

            elif op_type == 'conv2d_3x3_dw_ne16':
                # Depthwise 3x3 NE16: in_channels == out_channels
                c_code.append(f"        .type = OP_CONV2D_3X3_DW_NE16,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.conv2d_ne16 = {{")
                c_code.append(f"            .batch = 1,")
                c_code.append(f"            .in_h = {spec['in_h']}, .in_w = {spec['in_w']}, .in_channels = {spec['in_ch']},")
                c_code.append(f"            .out_channels = {spec['out_ch']},")  # Same as in_ch for depthwise
                c_code.append(f"            .kernel_h = 3, .kernel_w = 3,")
                c_code.append(f"            .stride_h = {spec['stride_h']}, .stride_w = {spec['stride_w']},")
                c_code.append(f"            .pad_h = {spec['pad_h']}, .pad_w = {spec['pad_w']},")
                c_code.append(f"            .scale_input = {spec['scale_input']}f, .scale_weight = {spec['scale_weight']}f, .scale_output = {spec['scale_output']}f,")
                # NE16 packed weights and corrected bias will be set at runtime
                c_code.append(f"            .weights_packed = NULL,  // Set at runtime from binary index {spec.get('ne16_packed_weight_index', -1)}")
                c_code.append(f"            .bias_corrected = NULL,  // Set at runtime from binary index {spec.get('ne16_bias_corr_index', -1)}")
                # HW requantization parameters (depthwise only)
                c_code.append(f"            .hw_scale = NULL,  // Set at runtime from binary index {spec.get('ne16_hw_scale_index', -1)}")
                c_code.append(f"            .hw_scale_shift = NULL,  // Set at runtime from binary index {spec.get('ne16_hw_scale_shift_index', -1)}")
                c_code.append(f"            .use_hw_requant = 1,  // Enable HW requantization")
                # NE16 depthwise spatial tiling parameters
                spatial_tiling = spec.get('ne16_dw_spatial_tiling', 0)
                num_tiles = spec.get('ne16_dw_num_tiles', 1)
                tile_h_out = spec.get('ne16_dw_tile_h_out', spec['in_h'])
                tile_h_in = spec.get('ne16_dw_tile_h_in', spec['in_h'] + 2 * spec['pad_h'])
                c_code.append(f"            .ne16_dw_spatial_tiling = {spatial_tiling},")
                c_code.append(f"            .ne16_dw_num_tiles = {num_tiles},")
                c_code.append(f"            .ne16_dw_tile_h_out = {tile_h_out},")
                c_code.append(f"            .ne16_dw_tile_h_in = {tile_h_in}")
                c_code.append(f"        }}")

            elif op_type == 'relu':
                c_code.append(f"        .type = OP_RELU,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.relu = {{")
                c_code.append(f"            .buffer = NULL, // Set at runtime")
                c_code.append(f"            .size = {spec['numel']},")
                c_code.append(f"            .scale_in = {spec['scale_in']}f, .scale_out = {spec['scale_out']}f")
                c_code.append(f"        }}")

            elif op_type == 'requantize':
                c_code.append(f"        .type = OP_REQUANTIZE,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.requantize = {{")
                c_code.append(f"            .buffer = NULL, // Set at runtime")
                c_code.append(f"            .size = {spec['numel']},")
                c_code.append(f"            .scale_in = {spec['scale_in']}f,")
                c_code.append(f"            .scale_out = {spec['scale_out']}f")
                c_code.append(f"        }}")

            elif op_type == 'maxpool':
                c_code.append(f"        .type = OP_MAXPOOL,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.maxpool = {{")
                c_code.append(f"            .in_h = {spec['in_h']}, .in_w = {spec['in_w']}, .channels = {spec['channels']},")
                c_code.append(f"            .out_h = {spec['out_h']}, .out_w = {spec['out_w']},")
                c_code.append(f"            .kernel_h = {spec['kernel_h']}, .kernel_w = {spec['kernel_w']},")
                c_code.append(f"            .stride_h = {spec['stride_h']}, .stride_w = {spec['stride_w']},")
                c_code.append(f"            .pad_h = {spec['pad_h']}, .pad_w = {spec['pad_w']},")
                if 'tile_config' in spec:
                    tc = spec['tile_config']
                    c_code.append(f"            .tile_h = {tc['tile_h']}, .tile_w = {tc['tile_w']},")
                    c_code.append(f"            .num_tiles = {tc['num_tiles']}, .num_tiles_h = {tc['num_tiles_h']}, .num_tiles_w = {tc['num_tiles_w']},")
                    c_code.append(f"            .l1_input_size = {tc['l1_input_bytes']}, .l1_output_size = {tc['l1_output_bytes']},")
                    c_code.append(f"            .out_tile_h = {tc['out_tile_h']}, .out_tile_w = {tc['out_tile_w']},")
                    c_code.append(f"            .l3_tiling_enabled = {1 if tc.get('l3_tiling_enabled') else 0},")
                    if tc.get('l3_tiling_enabled'):
                        c_code.append(f"            .l3_tile_h = {tc['l3_tile_h']}, .l3_tile_h_halo = {tc['l3_tile_h_halo']},")
                        c_code.append(f"            .num_l3_tiles = {tc['num_l3_tiles']},")
                else:
                    c_code.append(f"            .num_tiles = 0,")
                c_code.append(f"            .fusion_quant = {1 if spec.get('fusion_quant') else 0},")
                qsi = spec.get('quant_scale_in', 0.0)
                qso = spec.get('quant_scale_out', 0.0)
                # Format floats properly: 0.0 -> "0.0f", not "0f"
                qsi_str = f"{qsi:.15g}" if qsi != 0 else "0.0"
                qso_str = f"{qso:.15g}" if qso != 0 else "0.0"
                c_code.append(f"            .quant_scale_in = {qsi_str}f, .quant_scale_out = {qso_str}f,")
                layout_str = "LAYOUT_HWC" if self.use_hwc_layout else "LAYOUT_CHW"
                c_code.append(f"            .layout = {layout_str}")
                c_code.append(f"        }}")

            elif op_type == 'avgpool':
                c_code.append(f"        .type = OP_AVGPOOL,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.avgpool = {{")
                c_code.append(f"            .in_h = {spec['in_h']}, .in_w = {spec['in_w']}, .channels = {spec['channels']},")
                c_code.append(f"            .out_h = {spec['out_h']}, .out_w = {spec['out_w']},")
                c_code.append(f"            .kernel_h = {spec['kernel_h']}, .kernel_w = {spec['kernel_w']},")
                c_code.append(f"            .stride_h = {spec['stride_h']}, .stride_w = {spec['stride_w']},")
                c_code.append(f"            .scale_in = {spec['scale_input']}f, .scale_out = {spec['scale_output']}f,")
                if 'tile_config' in spec:
                    tc = spec['tile_config']
                    c_code.append(f"            .tile_h = {tc['tile_h']}, .tile_w = {tc['tile_w']},")
                    c_code.append(f"            .num_tiles = {tc['num_tiles']}, .num_tiles_h = {tc['num_tiles_h']}, .num_tiles_w = {tc['num_tiles_w']},")
                    c_code.append(f"            .l1_input_size = {tc['l1_input_bytes']}, .l1_output_size = {tc['l1_output_bytes']},")
                    c_code.append(f"            .out_tile_h = {tc['out_tile_h']}, .out_tile_w = {tc['out_tile_w']},")
                    c_code.append(f"            .l3_tiling_enabled = {1 if tc.get('l3_tiling_enabled') else 0},")
                    if tc.get('l3_tiling_enabled'):
                        c_code.append(f"            .l3_tile_h = {tc['l3_tile_h']}, .l3_tile_h_halo = {tc['l3_tile_h_halo']},")
                        c_code.append(f"            .num_l3_tiles = {tc['num_l3_tiles']},")
                else:
                    c_code.append(f"            .num_tiles = 0,")
                c_code.append(f"            .fusion_quant = {1 if spec.get('fusion_quant') else 0},")
                qsi = spec.get('quant_scale_in', 0.0)
                qso = spec.get('quant_scale_out', 0.0)
                # Format floats properly: 0.0 -> "0.0f", not "0f"
                qsi_str = f"{qsi:.15g}" if qsi != 0 else "0.0"
                qso_str = f"{qso:.15g}" if qso != 0 else "0.0"
                c_code.append(f"            .quant_scale_in = {qsi_str}f, .quant_scale_out = {qso_str}f,")
                layout_str = "LAYOUT_HWC" if self.use_hwc_layout else "LAYOUT_CHW"
                c_code.append(f"            .layout = {layout_str}")
                c_code.append(f"        }}")

            elif op_type == 'global_avgpool':
                c_code.append(f"        .type = OP_GLOBAL_AVGPOOL,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.global_avgpool = {{")
                c_code.append(f"            .batch = {spec.get('batch', 1)}, .channels = {spec['channels']},")
                c_code.append(f"            .h = {spec['height']}, .w = {spec['width']},")
                c_code.append(f"            .scale_in = {spec['scale_input']}f, .scale_out = {spec['scale_output']}f,")
                layout_str = "LAYOUT_HWC" if self.use_hwc_layout else "LAYOUT_CHW"
                c_code.append(f"            .layout = {layout_str}")
                c_code.append(f"        }}")

            elif op_type == 'add':
                c_code.append(f"        .type = OP_ADD,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.add = {{")
                c_code.append(f"            .size = {spec['size']},")
                c_code.append(f"            .scale_a = {spec['scale_x1']}f, .scale_b = {spec['scale_x2']}f, .scale_out = {spec['scale_output']}f")
                c_code.append(f"        }}")

            elif op_type == 'concat':
                c_code.append(f"        .type = OP_CONCAT,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.concat = {{")
                c_code.append(f"            .num_inputs = {spec['num_inputs']},")
                c_code.append(f"            .height = {spec['height']}, .width = {spec['width']},")
                c_code.append(f"            .scale_output = {spec['scale_output']}f")
                c_code.append(f"            // channels_per_input and input_scales arrays set at runtime")
                c_code.append(f"        }}")

            elif op_type == 'mhsa':
                c_code.append(f"        .type = OP_MHSA,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.mhsa = {{")
                c_code.append(f"            .layer_name = \"{spec['name']}\",")
                c_code.append(f"            .seq_len = {spec['seq_len']}, .num_heads = {spec['num_heads']},")
                c_code.append(f"            .head_dim = {spec['head_dim']}, .embed_dim = {spec['embed_dim']},")
                c_code.append(f"            .pool_mode = {spec['pool_mode']},")
                c_code.append(f"            .use_fp32_projections = {1 if spec.get('use_fp32_projections') else 0},")
                c_code.append(f"            .scale_input = {spec['scale_input']}f,")
                c_code.append(f"            .scale_q_weight = {spec['q_scale_weight']}f, .scale_k_weight = {spec['k_scale_weight']}f,")
                c_code.append(f"            .scale_v_weight = {spec['v_scale_weight']}f, .scale_out_weight = {spec['out_scale_weight']}f,")
                tc = spec.get('tile_config')
                if tc:
                    if 'tile_q' not in tc:
                        print(f"[WARN] MHSA tile_config missing fields for {spec['name']}: {tc}")
                    c_code.append(f"            .tile_q = {tc['tile_q']}, .num_tiles = {tc['num_tiles']},")
                    c_code.append(f"            .persistent_bytes = {tc['persistent_bytes']}, .tile_bytes = {tc['tile_bytes']},")
                    c_code.append(f"            .l3_tiling_enabled = {1 if tc.get('l3_tiling_enabled') else 0},")
                    if tc.get('l3_tiling_enabled'):
                        c_code.append(f"            .l3_seq_len = {tc['l3_seq_len']},")
                        c_code.append(f"            .num_l3_tiles = {tc['num_l3_tiles']},")
                else:
                    c_code.append(f"            .num_tiles = 0,")
                c_code.append(f"            .scale_q = {spec['scale_q']}f, .scale_k = {spec['scale_k']}f,")
                c_code.append(f"            .scale_v = {spec['scale_v']}f, .scale_output = {spec['scale_output']}f,")
                c_code.append(f"            .softmax_scale = {spec['softmax_scale']}f,")
                c_code.append(f"            .softmax_lut = i_softmax_lut,  // Enable bit-exact i-Softmax")
                # L1 weight caching fields
                c_code.append(f"            .l1_weight_caching_enabled = {1 if spec.get('l1_weight_caching_enabled') else 0},")
                c_code.append(f"            .l1_proj_weight_size = {spec.get('l1_proj_weight_size', 0)}")
                c_code.append(f"        }}")

            elif op_type == 'cross_attention':
                c_code.append(f"        .type = OP_CROSS_ATTENTION,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.cross_attention = {{")
                c_code.append(f"            .batch = {spec['batch']},")
                c_code.append(f"            .kv_len = {spec['kv_len']},")
                c_code.append(f"            .num_queries = {spec['num_queries']},")
                c_code.append(f"            .embed_dim = {spec['embed_dim']},")
                c_code.append(f"            .num_heads = {spec['num_heads']},")
                c_code.append(f"            .head_dim = {spec['head_dim']},")
                c_code.append(f"            .scale_kv_in = {spec['scale_input']}f,")
                c_code.append(f"            .scale_query_in = {spec['query_scale']}f,")
                c_code.append(f"            .scale_q_weight = {spec['q_scale_weight']}f, .scale_k_weight = {spec['k_scale_weight']}f,")
                c_code.append(f"            .scale_v_weight = {spec['v_scale_weight']}f, .scale_out_weight = {spec['out_scale_weight']}f,")
                c_code.append(f"            .scale_q = {spec['scale_q']}f, .scale_k = {spec['scale_k']}f, .scale_v = {spec['scale_v']}f,")
                c_code.append(f"            .scale_output = {spec['scale_output']}f,")
                c_code.append(f"            .softmax_scale = {spec['softmax_scale']}f")
                c_code.append(f"        }}")

            elif op_type == 'cross_attn_self_refine':
                c_code.append(f"        .type = OP_CROSS_ATTN_SELF_REFINE,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.cross_attn_self_refine = {{")
                c_code.append(f"            .batch = {spec['batch']},")
                c_code.append(f"            .kv_len = {spec['kv_len']},")
                c_code.append(f"            .num_queries = {spec['num_queries']},")
                c_code.append(f"            .embed_dim = {spec['embed_dim']},")
                c_code.append(f"            .num_heads = {spec['num_heads']},")
                c_code.append(f"            .head_dim = {spec['head_dim']},")
                c_code.append(f"            .ff_dim = {spec['ff_dim']},")
                c_code.append(f"            .num_sa_blocks = {spec['num_sa_blocks']},")
                c_code.append(f"            .softmax_scale = {spec['softmax_scale']}f,")
                c_code.append(f"            .scale_input = {spec['scale_input']}f,")
                c_code.append(f"            .scale_output = {spec['scale_output']}f")
                c_code.append(f"        }}")

            elif op_type == 'classification_head_mlp':
                c_code.append(f"        .type = OP_CLASSIFICATION_HEAD_MLP,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.classification_head_mlp = {{")
                c_code.append(f"            .batch = {spec['batch']},")
                c_code.append(f"            .seq_len = {spec['seq_len']},")
                c_code.append(f"            .hidden_dim = {spec['hidden_dim']},")
                c_code.append(f"            .num_heads = {spec['num_heads']},")
                c_code.append(f"            .head_dim = {spec['head_dim']},")
                c_code.append(f"            .mlp_hidden_dim = {spec['mlp_hidden_dim']},")
                c_code.append(f"            .num_classes = {spec['num_classes']},")
                c_code.append(f"            .softmax_scale = {spec['softmax_scale']}f,")
                c_code.append(f"            .scale_input = {spec['scale_input']}f,")
                c_code.append(f"            .scale_output = {spec['scale_output']}f")
                c_code.append(f"        }}")

            elif op_type == 'layernorm':
                c_code.append(f"        .type = OP_LAYERNORM,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.layernorm = {{")
                c_code.append(f"            .num_tokens = {spec['num_tokens']}, .embed_dim = {spec['embed_dim']},")
                c_code.append(f"            .scale_in = {spec['scale_input']}f, .scale_out = {spec['scale_output']}f")
                c_code.append(f"        }}")

            elif op_type == 'rmsnorm':
                c_code.append(f"        .type = OP_RMSNORM,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.rmsnorm = {{")
                c_code.append(f"            .num_vectors = {spec['num_tokens']}, .normalized_dim = {spec['normalized_dim']},")
                c_code.append(f"            .scale_in = {spec['scale_input']}f, .scale_out = {spec['scale_output']}f,")
                c_code.append(f"            .eps = {spec['eps']}f")
                c_code.append(f"        }}")

            elif op_type == 'gelu':
                c_code.append(f"        .type = OP_GELU,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.gelu = {{")
                c_code.append(f"            .num_elements = {spec['num_elements']},")
                c_code.append(f"            .scale_in = {spec['scale_input']}f, .scale_out = {spec['scale_output']}f")
                c_code.append(f"        }}")

            elif op_type == 'mean':
                c_code.append(f"        .type = OP_MEAN,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.mean = {{")
                c_code.append(f"            .batch = {spec['batch']}, .seq_len = {spec['seq_len']}, .features = {spec['features']},")
                c_code.append(f"            .dim = {spec['dim']}, .keepdim = {spec['keepdim']},")
                c_code.append(f"            .scale_in = {spec['scale_input']}f, .scale_out = {spec['scale_output']}f")
                c_code.append(f"        }}")

            elif op_type == 'alternating_attention':
                c_code.append(f"        .type = OP_ALTERNATING_ATTENTION,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.alternating_attention = {{")
                c_code.append(f"            .batch = {spec['batch']}, .seq_len = {spec['seq_len']},")
                c_code.append(f"            .embed_dim = {spec['embed_dim']}, .num_heads = {spec['num_heads']},")
                c_code.append(f"            .head_dim = {spec['head_dim']},")
                c_code.append(f"            .num_channels = {spec['num_channels']}, .temporal_len = {spec['temporal_len']},")
                c_code.append(f"            .block_idx = {spec['block_idx']},")
                c_code.append(f"            .scaling_factor = {spec['scaling_factor']}f,")
                c_code.append(f"            .scale_in = {spec['scale_input']}f,")
                c_code.append(f"            .scale_qkv_weight = {spec['qkv_scale_weight']}f,")
                c_code.append(f"            .scale_qkv_out = {spec['qkv_scale_output']}f,")
                c_code.append(f"            .scale_q = {spec['scale_q']}f, .scale_k = {spec['scale_k']}f, .scale_v = {spec['scale_v']}f,")
                c_code.append(f"            .scale_out_weight = {spec['out_scale_weight']}f,")
                c_code.append(f"            .scale_out = {spec['scale_output']}f,")
                # NE16 support for attention projections
                use_ne16_qkv = spec.get('use_ne16_qkv', False)
                use_ne16_out = spec.get('use_ne16_out', False)
                c_code.append(f"            .use_ne16_qkv = {1 if use_ne16_qkv else 0},")
                c_code.append(f"            .qkv_ne16_packed = NULL,  // Set at runtime from binary idx {spec.get('qkv_ne16_packed_index', -1)}")
                c_code.append(f"            .qkv_ne16_bias = NULL,")
                c_code.append(f"            .qkv_ne16_scale = NULL,")
                c_code.append(f"            .qkv_ne16_scale_shift = NULL,")
                c_code.append(f"            .use_ne16_out = {1 if use_ne16_out else 0},")
                c_code.append(f"            .out_ne16_packed = NULL,  // Set at runtime from binary idx {spec.get('out_ne16_packed_index', -1)}")
                c_code.append(f"            .out_ne16_bias = NULL,")
                c_code.append(f"            .out_ne16_scale = NULL,")
                c_code.append(f"            .out_ne16_scale_shift = NULL")
                c_code.append(f"        }}")

            elif op_type == 'flatten':
                c_code.append(f"        .type = OP_FLATTEN,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.flatten = {{")
                c_code.append(f"            // No parameters needed (aliasing operation)")
                c_code.append(f"        }}")

            elif op_type == 'squeeze':
                c_code.append(f"        .type = OP_SQUEEZE,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.squeeze = {{")
                c_code.append(f"            // No parameters needed (dimension removal, no data movement)")
                c_code.append(f"        }}")

            elif op_type == 'reshape':
                c_code.append(f"        .type = OP_RESHAPE,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.reshape = {{")
                c_code.append(f"            // Shape stored in spec but not typically needed at runtime")
                c_code.append(f"        }}")

            elif op_type == 'transpose_2d':
                c_code.append(f"        .type = OP_TRANSPOSE_2D,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.transpose_2d = {{")
                c_code.append(f"            .batch = {spec.get('batch_size', spec.get('batch', 1))}, .dim0 = {spec.get('dim1', 1)}, .dim1 = {spec.get('dim2', 1)},")
                c_code.append(f"            .scale = {spec.get('scale', spec.get('scale_output', 1.0))}f")
                c_code.append(f"        }}")

            elif op_type == 'chw_to_hwc':
                c_code.append(f"        .type = OP_CHW_TO_HWC,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.layout_convert = {{")
                c_code.append(f"            .channels = {spec['channels']},")
                c_code.append(f"            .height = {spec['height']},")
                c_code.append(f"            .width = {spec['width']}")
                c_code.append(f"        }}")

            elif op_type == 'hwc_to_chw':
                c_code.append(f"        .type = OP_HWC_TO_CHW,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.layout_convert = {{")
                c_code.append(f"            .channels = {spec['channels']},")
                c_code.append(f"            .height = {spec['height']},")
                c_code.append(f"            .width = {spec['width']}")
                c_code.append(f"        }}")

            elif op_type == 'zeropad2d':
                c_code.append(f"        .type = OP_ZEROPAD2D,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.zeropad2d = {{")
                c_code.append(f"            .in_h = {spec['in_h']}, .in_w = {spec['in_w']}, .channels = {spec['channels']},")
                c_code.append(f"            .out_h = {spec['out_h']}, .out_w = {spec['out_w']},")
                c_code.append(f"            .pad_left = {spec['pad_left']}, .pad_right = {spec['pad_right']},")
                c_code.append(f"            .pad_top = {spec['pad_top']}, .pad_bottom = {spec['pad_bottom']},")
                layout_str = "LAYOUT_HWC" if self.use_hwc_layout else "LAYOUT_CHW"
                c_code.append(f"            .layout = {layout_str}")
                c_code.append(f"        }}")

            elif op_type == 'embedding':
                c_code.append(f"        .type = OP_EMBEDDING,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.embedding = {{")
                c_code.append(f"            .vocab_size = {spec['vocab_size']},")
                c_code.append(f"            .embed_dim = {spec['embed_dim']},")
                c_code.append(f"            .num_indices = {spec['num_indices']},")
                c_code.append(f"            .scale_out = {spec.get('scale_out', spec.get('scale_output', 1.0))}f")
                c_code.append(f"        }}")

            elif op_type == 'groupnorm':
                c_code.append(f"        .type = OP_GROUPNORM,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.groupnorm = {{")
                c_code.append(f"            .batch = {spec['batch']},")
                c_code.append(f"            .channels = {spec['channels']},")
                c_code.append(f"            .spatial_size = {spec['spatial_size']},")
                c_code.append(f"            .num_groups = {spec['num_groups']},")
                c_code.append(
                    f"            .scale_in = {spec.get('scale_input', spec.get('scale_in', 1.0))}f,"
                    f" .scale_out = {spec.get('scale_output', spec.get('scale_out', 1.0))}f"
                )
                c_code.append(f"        }}")

            elif op_type == 'rfft':
                c_code.append(f"        .type = OP_RFFT,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.rfft = {{")
                patch_size = int(spec.get('patch_size', spec.get('n_fft', 0)))
                num_bins = int(spec.get('num_bins', (patch_size // 2 + 1) if patch_size else 0))
                c_code.append(f"            .num_patches = {spec['num_patches']},")
                c_code.append(f"            .patch_size = {patch_size},")
                c_code.append(f"            .num_bins = {num_bins},")
                c_code.append(
                    f"            .scale_in = {spec.get('scale_input', spec.get('scale_in', 1.0))}f,"
                    f" .scale_out = {spec.get('scale_output', spec.get('scale_out', 1.0))}f"
                )
                c_code.append(f"        }}")

            elif op_type == 'adaptive_avgpool1d':
                # Map AdaptiveAvgPool1d(output_size=1) to OP_GLOBAL_AVGPOOL for LayerSpec metadata.
                # Runtime execution still uses the dedicated `adaptive_avgpool1d` code path in `network.c.mako`.
                if spec.get('output_size', 1) != 1:
                    c_code.append(f"        .type = OP_UNKNOWN,")
                    c_code.append(f"        .name = \"{spec['name']}\",")
                    c_code.append(
                        f"        // Op adaptive_avgpool1d(output_size={spec.get('output_size')}) not yet mapped in generator (only output_size=1 supported)"
                    )
                else:
                    c_code.append(f"        .type = OP_GLOBAL_AVGPOOL,")
                    c_code.append(f"        .name = \"{spec['name']}\",")
                    c_code.append(f"        .params.global_avgpool = {{")
                    c_code.append(
                        f"            .batch = {spec.get('batch', 1)}, .channels = {spec['channels']}, .h = 1, .w = {spec['input_len']},"
                    )
                    c_code.append(f"            .scale_in = {spec['scale']}f, .scale_out = {spec['scale']}f")
                    c_code.append(f"        }}")

            elif op_type == 'conv1d_depthwise':
                c_code.append(f"        .type = OP_CONV1D_DEPTHWISE,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.conv1d_depthwise = {{")
                c_code.append(f"            .channels = {spec['channels']}, .length = {spec['length']},")
                c_code.append(f"            .kernel_size = {spec['kernel_size']}, .causal = {spec.get('causal', 0)},")
                c_code.append(
                    f"            .scale_in = {spec.get('scale_input', 1.0)}f, .scale_w = {spec.get('scale_weight', 1.0)}f,"
                    f" .scale_out = {spec.get('scale_output', 1.0)}f"
                )
                c_code.append(f"        }}")

            elif op_type == 'silu':
                c_code.append(f"        .type = OP_SILU,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.silu = {{")
                c_code.append(f"            .num_elements = {spec['num_elements']},")
                c_code.append(f"            .scale_in = {spec.get('scale_in', 1.0)}f, .scale_out = {spec.get('scale_out', 1.0)}f,")
                c_code.append(f"            .lut = NULL")
                c_code.append(f"        }}")

            elif op_type == 'ssm':
                c_code.append(f"        .type = OP_SSM,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.ssm = {{")
                c_code.append(f"            .batch = {spec['batch']}, .seq_len = {spec['seq_len']}, .d_inner = {spec['d_inner']}, .d_state = {spec['d_state']}, .dt_rank = {spec['dt_rank']},")
                c_code.append(f"            .scale_x = {spec.get('scale_x', spec.get('scale_in', 1.0))}f, .scale_output = {spec.get('scale_output', spec.get('scale_out', 1.0))}f")
                c_code.append(f"        }}")

            elif op_type == 'mamba_block':
                c_code.append(f"        .type = OP_MAMBA_BLOCK,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.mamba_block = {{")
                c_code.append(f"            .batch = {spec['batch']}, .seq_len = {spec['seq_len']},")
                c_code.append(f"            .d_model = {spec['d_model']}, .d_inner = {spec['d_inner']}, .d_state = {spec['d_state']},")
                c_code.append(f"            .dt_rank = {spec['dt_rank']}, .kernel_size = {spec['kernel_size']},")
                c_code.append(f"            .scale_in = {spec.get('scale_in', spec.get('scale_x', 1.0))}f, .scale_out = {spec.get('scale_out', spec.get('scale_output', 1.0))}f")
                c_code.append(f"        }}")

            elif op_type == 'mamba_wrapper':
                c_code.append(f"        .type = OP_MAMBA_WRAPPER,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.mamba_wrapper = {{")
                c_code.append(f"            .batch = {spec['batch']}, .seq_len = {spec['seq_len']},")
                c_code.append(f"            .d_model = {spec['d_model']}, .d_inner = {spec['d_inner']}, .d_state = {spec['d_state']},")
                c_code.append(f"            .dt_rank = {spec['dt_rank']}, .kernel_size = {spec['kernel_size']},")
                c_code.append(f"            .scale_in = {spec.get('scale_in', 1.0)}f, .scale_out = {spec.get('scale_out', 1.0)}f")
                c_code.append(f"        }}")


            elif op_type == 'patch_embed':
                c_code.append(f"        .type = OP_PATCH_EMBED,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.patch_embed = {{")
                c_code.append(f"            .batch = {spec['batch']}, .in_chans = {spec['in_chans']},")
                c_code.append(f"            .inp_h = {spec['inp_h']}, .inp_w = {spec['inp_w']},")
                c_code.append(f"            .patch_h = {spec['patch_h']}, .patch_w = {spec['patch_w']},")
                c_code.append(f"            .stride_h = {spec['stride_h']}, .stride_w = {spec['stride_w']},")
                c_code.append(f"            .embed_dim = {spec['embed_dim']},")
                c_code.append(f"            .grid_h = {spec['grid_h']}, .grid_w = {spec['grid_w']},")
                c_code.append(f"            .seq_len = {spec['seq_len']}, .d_model = {spec['d_model']},")
                c_code.append(f"            .scale_in = {spec['scale_in']}f, .scale_weight = {spec['scale_weight']}f,")
                c_code.append(f"            .scale_out = {spec['scale_out']}f")
                c_code.append(f"        }}")

            elif op_type == 'pos_embed':
                c_code.append(f"        .type = OP_POS_EMBED,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        .params.pos_embed = {{")
                c_code.append(f"            .batch = {spec['batch']},")
                c_code.append(f"            .seq_len = {spec['seq_len']}, .d_model = {spec['d_model']},")
                c_code.append(f"            .scale_pos = {spec['scale_pos']}f,")
                c_code.append(f"            .scale_input = {spec['scale_input']}f,")
                c_code.append(f"            .scale_out = {spec['scale_out']}f")
                c_code.append(f"        }}")

            else:
                c_code.append(f"        .type = OP_UNKNOWN,")
                c_code.append(f"        .name = \"{spec['name']}\",")
                c_code.append(f"        // Op {op_type} not yet mapped in generator")
            
            c_code.append("    },")

        c_code.append("};")
        return "\n".join(c_code)

    def generate_sources(self):
        """Generate all source files."""
        print()
        print("Generating sources...")
        self._ensure_codegen_metadata()
        # Planner already run in generate_headers, but safe to run if independent
        if self.l2_arena_size == 0: 
            self._prepare_memory_levels()
            self._run_memory_planner()

        # Generate array code FIRST, before rendering network.c
        layer_array_code = self._generate_layer_array()

        ops_present = {spec.get('op') for spec in self.layer_specs if spec.get('op')}

        self.render_template("main.c.mako", "src/main.c",
                            binary_files=self.binary_files,
                            network_info=self.network_info)

        # Render a monolithic network.c then split it into separately compiled modules.
        network_c = self.render_template(
            "network.c.mako",
            None,
            network_info=self.network_info,
            layer_array_code=layer_array_code,
        )
        self._write_phase47_network_sources(network_c, ops_present)

        # shared runtime (codegen/runtime/src/ops/) and compiled from there via the
        # Makefile. No longer generating local copies to avoid conflicts.

        print("  [OK] Generated source files")
    
    

    def generate_makefile(self):
        """Generate Makefile."""
        print()
        print("Generating Makefile...")

        self.render_template("Makefile.mako", "Makefile")

        print("  [OK] Generated Makefile")

    # codegen/runtime/{src,inc} directly via the Makefile. The _prune_stale_runtime_duplicates()
    # method removes any leftover copies to prevent duplicate symbol errors.

    def _write_text(self, output_name, content):
        output_path = self.output_dir / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(content)

    def _write_phase47_network_sources(self, network_c, ops_present):
        try:
            sources, headers = self._split_network_c_to_phase47(network_c, ops_present)
        except Exception as e:
            self._write_text("src/network.c", network_c)
            print(f"  [WARN]  Network split skipped (wrote monolithic network.c): {e}")
            return

        for relpath, content in headers.items():
            self._write_text(relpath, content)
        for relpath, content in sources.items():
            self._write_text(relpath, content)

    def _make_file_header(self, filename: str, description: str) -> str:
        """Generate a descriptive file header banner."""
        network_name = self._infer_network_name()
        return f"""/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 *
 * {filename} - {description}
 * Auto-generated by ARES for network: {network_name}
 */

"""

    def _split_network_c_to_phase47(self, network_c, ops_present):
        lines = network_c.splitlines(keepends=True)

        def find_idx(predicate, label):
            for idx, line in enumerate(lines):
                if predicate(line):
                    return idx
            raise ValueError(f"Could not find {label} in rendered network.c")

        # LayerSpec array
        layer_start = find_idx(
            lambda l: l.lstrip().startswith("const LayerSpec network_layers[]")
                      or l.lstrip().startswith("static const LayerSpec network_layers[]"),
            "LayerSpec array (network_layers)",
        )
        layer_end = None
        for idx in range(layer_start, len(lines)):
            if lines[idx].strip() == "};":
                layer_end = idx
                break
        if layer_end is None:
            raise ValueError("Could not find end of network_layers array (expected a line with just `};`)")

        # Extract network_cl_args_t typedef
        struct_end = find_idx(lambda l: l.strip() == "} network_cl_args_t;", "network_cl_args_t typedef end")
        struct_start = None
        for idx in range(struct_end, -1, -1):
            if lines[idx].lstrip().startswith("typedef struct"):
                struct_start = idx
                break
        if struct_start is None:
            raise ValueError("Could not find start of network_cl_args_t typedef (expected `typedef struct`)")

        # Cluster entry function
        cluster_start = find_idx(
            lambda l: l.lstrip().startswith("static void network_cl_entry(")
                      or l.lstrip().startswith("void network_cl_entry("),
            "cluster entry function (network_cl_entry)",
        )

        # FC entry function (and marker comment)
        fc_comment_idx = None
        for idx, line in enumerate(lines):
            if line.strip().startswith("// --- FC Entry Point ---"):
                fc_comment_idx = idx
                break

        if fc_comment_idx is not None:
            cluster_end = fc_comment_idx
            fc_start = fc_comment_idx + 1
        else:
            fc_start = find_idx(
                lambda l: l.lstrip().startswith("int network_run_test_from_l3("),
                "FC entry function (network_run_test_from_l3)",
            )
            cluster_end = fc_start

        # Find the FC function signature line(s)
        fc_sig_start = find_idx(
            lambda l: l.lstrip().startswith("int network_run_test_from_l3("),
            "FC function signature (network_run_test_from_l3)",
        )
        fc_sig_end = None
        for idx in range(fc_sig_start, len(lines)):
            if ")" in lines[idx]:
                fc_sig_end = idx
                break
        if fc_sig_end is None:
            raise ValueError("Could not find end of FC function signature (missing `)`)")

        preamble_lines = lines[:layer_start]
        include_lines = []
        for line in preamble_lines:
            stripped = line.lstrip()
            if not stripped.startswith("#include "):
                continue
            # Older generated layouts used `#include "ops/op_*.c"` to keep one TU.
            # Current layout compiles ops separately, so never include op .c files via headers.
            if stripped.startswith('#include "ops/op_') or stripped.startswith('#include "../ops/op_'):
                continue
            include_lines.append(line)
        define_lines = [l for l in preamble_lines if l.lstrip().startswith("#define ")]

        # Internal header
        internal_h = []
        internal_h.append("#pragma once\n")
        internal_h.append(self._make_file_header("network_internal.h", "Shared Declarations and Types"))
        internal_h.extend(include_lines)
        if any(op in ops_present for op in ('maxpool', 'avgpool', 'global_avgpool')):
            internal_h.append('#include "ops/op_pool.h"\n')
        internal_h.append("\n")
        internal_h.extend(define_lines)
        internal_h.append("\n")
        internal_h.append("extern pi_cl_dma_copy_t g_load_dma_descs[MAX_DMA_DESCRIPTORS];\n")
        internal_h.append("extern pi_cl_dma_copy_t g_store_dma_descs[MAX_DMA_DESCRIPTORS];\n\n")
        internal_h.extend(lines[struct_start:struct_end + 1])
        internal_h.append("\n\n")
        internal_h.append("extern network_cl_args_t g_network_cl_args;\n")
        internal_h.append("extern const LayerSpec network_layers[];\n\n")
        internal_h.append("void network_cl_entry(void *arg);\n")

        fc_sig = "".join(lines[fc_sig_start:fc_sig_end + 1]).rstrip()
        fc_impl_proto = fc_sig.replace("int network_run_test_from_l3(", "int network_run_test_from_l3_impl(", 1) + ";\n"
        internal_h.append(fc_impl_proto)

        headers = {"inc/network_internal.h": "".join(internal_h)}

        # Helper: build call arg list for wrapper
        def extract_param_names(sig_text):
            open_idx = sig_text.find("(")
            close_idx = sig_text.rfind(")")
            if open_idx == -1 or close_idx == -1 or close_idx <= open_idx:
                raise ValueError("Unable to parse FC signature parameters")
            params_text = sig_text[open_idx + 1:close_idx].strip()
            if not params_text:
                return []
            parts = [p.strip() for p in params_text.split(",")]
            names = []
            for p in parts:
                last = p.split()[-1]
                last = last.strip()
                while last.startswith("*"):
                    last = last[1:]
                if last.endswith("[]"):
                    last = last[:-2]
                names.append(last)
            return names

        call_args = ", ".join(extract_param_names(fc_sig))

        # src/network.c wrapper (public API)
        wrapper = []
        wrapper.append(self._make_file_header("network.c", "Public API Entry Point"))
        wrapper.append('#include "network_internal.h"\n\n')
        wrapper.append(fc_sig + "\n")
        wrapper.append("{\n")
        wrapper.append(f"    return network_run_test_from_l3_impl({call_args});\n")
        wrapper.append("}\n")

        # src/net/network_layers.c
        layer_block = "".join(lines[layer_start:layer_end + 1])
        layer_block = layer_block.replace("static const LayerSpec network_layers[]", "const LayerSpec network_layers[]", 1)
        net_layers = []
        net_layers.append(self._make_file_header("network_layers.c", "Layer Specifications and Parameters"))
        net_layers.append('#include "network_internal.h"\n\n')
        net_layers.append(layer_block)
        net_layers.append("\n")

        # src/net/network_cluster.c: everything between layer_end and FC comment, minus the args typedef/global and op-includes
        cluster_lines = list(lines[layer_end + 1:cluster_end])

        # Strip the struct typedef (now in network_internal.h)
        for idx in range(struct_start - (layer_end + 1), struct_end - (layer_end + 1) + 1):
            if 0 <= idx < len(cluster_lines):
                cluster_lines[idx] = ""

        # Strip the static `g_network_cl_args` definition line (now in network_globals.c)
        for i, line in enumerate(cluster_lines):
            if "g_network_cl_args" in line and line.lstrip().startswith("static network_cl_args_t"):
                cluster_lines[i] = ""

        # Strip op include fragments from older include-based layouts
        cluster_lines = [
            "" if l.lstrip().startswith('#include "ops/op_') or l.lstrip().startswith('#include "../ops/op_') else l
            for l in cluster_lines
        ]

        # Find section markers to split into helpers, workers, entry
        cluster_text = "".join(cluster_lines)
        end_sec4_idx = cluster_text.find("/* End Helper Functions */")
        end_sec5_idx = cluster_text.find("/* End Worker Functions */")

        if end_sec4_idx != -1 and end_sec5_idx != -1:
            # Split into three files based on section markers
            helpers_body = cluster_text[:end_sec4_idx].strip()
            workers_body = cluster_text[end_sec4_idx + len("/* End Helper Functions */"):end_sec5_idx].strip()
            entry_body = cluster_text[end_sec5_idx + len("/* End Worker Functions */"):].strip()

            # Make cluster entry externally visible
            entry_body = entry_body.replace("static void network_cl_entry(void *arg)", "void network_cl_entry(void *arg)", 1)

            # network_helpers.c: Golden comparison, L1 fusion kernels
            net_helpers = []
            net_helpers.append(self._make_file_header("network_helpers.c", "Helper Functions and L1 Fusion Kernels"))
            net_helpers.append('#include "network_internal.h"\n\n')
            net_helpers.append(helpers_body)
            net_helpers.append("\n")

            # network_workers.c: Worker structs and wrappers
            net_workers = []
            net_workers.append(self._make_file_header("network_workers.c", "Worker Functions and Argument Structures"))
            net_workers.append('#include "network_internal.h"\n')
            net_workers.append('#include "network_helpers.h"\n\n')
            net_workers.append(workers_body)
            net_workers.append("\n")

            # network_cluster.c: Entry point only
            net_cluster = []
            net_cluster.append(self._make_file_header("network_cluster.c", "Cluster Entry Point"))
            net_cluster.append('#include "network_internal.h"\n')
            net_cluster.append('#include "network_helpers.h"\n')
            net_cluster.append('#include "network_workers.h"\n\n')
            net_cluster.append(entry_body)
            net_cluster.append("\n")

            # Generate network_helpers.h header with inline function DEFINITIONS
            # (static inline functions must have definitions in header to be visible across TUs)
            helpers_h = []
            helpers_h.append("#pragma once\n")
            helpers_h.append(self._make_file_header("network_helpers.h", "Helper Function Declarations and Inline Definitions"))
            helpers_h.append('#include "network_internal.h"\n\n')
            helpers_h.append("// Golden comparison (DEBUG mode only)\n")
            helpers_h.append("#ifndef MINIMAL_OUTPUT\n")
            helpers_h.append("void compare_int8_output_impl(const char *layer_name, const int8_t *output, const int8_t *golden, size_t size);\n")
            helpers_h.append("#endif\n\n")
            # L1 fusion kernels - DEFINITIONS must be in header for static inline
            helpers_h.append("// L1 fusion kernels (static inline - definitions in header)\n")
            helpers_h.append("static inline void relu_int8_inplace_l1(int8_t *data, size_t size) {\n")
            helpers_h.append("    size_t i = pi_core_id();\n")
            helpers_h.append("    const size_t step = NUM_CORES;\n")
            helpers_h.append("    for (; i < size; i += step) {\n")
            helpers_h.append("        if (data[i] < 0) data[i] = 0;\n")
            helpers_h.append("    }\n")
            helpers_h.append("}\n\n")
            helpers_h.append("static inline void requantize_int8_inplace_l1(int8_t *data, size_t size, float scale_in, float scale_out) {\n")
            helpers_h.append("    if (fabsf(scale_in - scale_out) < 1e-12f) return;\n")
            helpers_h.append("    int8_t map_int8[256];\n")
            helpers_h.append("    for (int v = -128; v <= 127; v++) {\n")
            helpers_h.append("        float val_fp32 = (float)v * scale_in;\n")
            helpers_h.append("        int32_t val_int32 = (int32_t)lrintf(val_fp32 / scale_out);\n")
            helpers_h.append("        if (val_int32 > 127) val_int32 = 127;\n")
            helpers_h.append("        if (val_int32 < -128) val_int32 = -128;\n")
            helpers_h.append("        map_int8[(uint32_t)(v + 128)] = (int8_t)val_int32;\n")
            helpers_h.append("    }\n")
            helpers_h.append("    size_t i = pi_core_id();\n")
            helpers_h.append("    const size_t step = NUM_CORES;\n")
            helpers_h.append("    for (; i < size; i += step) {\n")
            helpers_h.append("        data[i] = map_int8[(uint32_t)((int)data[i] + 128)];\n")
            helpers_h.append("    }\n")
            helpers_h.append("}\n")

            # Strip inline function definitions from helpers_body (now in header)
            helpers_body = re.sub(
                r'// --- Fusion Kernels \(L1\) ---.*',
                '// L1 fusion kernels moved to network_helpers.h (static inline)',
                helpers_body,
                flags=re.DOTALL
            )

            # Generate network_workers.h header
            workers_h = []
            workers_h.append("#pragma once\n")
            workers_h.append(self._make_file_header("network_workers.h", "Worker Function Declarations"))
            workers_h.append('#include "network_internal.h"\n\n')
            workers_h.append("// Worker argument structures and functions declared in network_workers.c\n")
            workers_h.append("// These are static within that TU, no external declarations needed.\n")

            headers["inc/network_helpers.h"] = "".join(helpers_h)
            headers["inc/network_workers.h"] = "".join(workers_h)

            use_split = True
        else:
            # Fallback: keep everything in network_cluster.c (markers not found)
            cluster_body = cluster_text.replace("static void network_cl_entry(void *arg)", "void network_cl_entry(void *arg)", 1)

            net_cluster = []
            net_cluster.append(self._make_file_header("network_cluster.c", "Cluster Entry Point and Workers"))
            net_cluster.append('#include "network_internal.h"\n')
            net_cluster.append("\n")
            net_cluster.append(cluster_body)

            net_helpers = None
            net_workers = None
            use_split = False

        # src/net/network_fc.c (rename public entry to *_impl)
        fc_block = "".join(lines[fc_start:])
        fc_block = fc_block.replace("int network_run_test_from_l3(", "int network_run_test_from_l3_impl(", 1)

        net_fc = []
        net_fc.append(self._make_file_header("network_fc.c", "Fabric Controller Entry Point"))
        net_fc.append('#include "network_internal.h"\n\n')
        net_fc.append(fc_block)

        # src/net/network_globals.c: globals shared across modules/runtime
        net_globals = []
        net_globals.append(self._make_file_header("network_globals.c", "Global Variables and DMA Descriptors"))
        net_globals.append('#include "network_internal.h"\n\n')
        net_globals.append("pi_cl_dma_copy_t __attribute__((section(\".data\"))) g_load_dma_descs[MAX_DMA_DESCRIPTORS];\n")
        net_globals.append("pi_cl_dma_copy_t __attribute__((section(\".data\"))) g_store_dma_descs[MAX_DMA_DESCRIPTORS];\n\n")
        net_globals.append("network_cl_args_t g_network_cl_args;\n")

        sources = {
            "src/network.c": "".join(wrapper),
            "src/net/network_layers.c": "".join(net_layers),
            "src/net/network_cluster.c": "".join(net_cluster),
            "src/net/network_fc.c": "".join(net_fc),
            "src/net/network_globals.c": "".join(net_globals),
        }

        if use_split:
            sources["src/net/network_helpers.c"] = "".join(net_helpers)
            sources["src/net/network_workers.c"] = "".join(net_workers)

        return sources, headers

    def render_template(self, template_name, output_name, **kwargs):
        """Render a Mako template and save to file (or return as a string when output_name is None)."""
        template_path = self.template_dir / template_name
        module_dir = self.output_dir / "__mako_cache"
        module_dir.mkdir(parents=True, exist_ok=True)

        # Use a TemplateLookup so templates can include partials (e.g. `partials/*.mako`)
        lookup = TemplateLookup(
            directories=[str(self.template_dir)],
            module_directory=str(module_dir),
            input_encoding='utf-8',
        )
        template = lookup.get_template(template_name)

        # Add common variables
        kwargs.update({
            'network_name': self._infer_network_name(),
            'generator': self,
            'target_name': self.target.name,
            'target_display_name': self.target.display_name,
            'target_l1_total_bytes': self._get_l1_total_bytes_for_allocation(),
            'layer_specs': self.layer_specs,
            'activation_buffers': self.activation_buffers,
            'param_layers': self.param_layers,
            'input_scale': self.input_scale,
            'input_shape': self.input_shape,
            'input_numel': self.input_numel,
            'num_classes': self.num_classes,
            'weight_entries': self.weight_entries,
            'bias_entries': self.bias_entries,
            'input_entry': self.input_entry,
            'additional_input_entries': getattr(self, 'additional_input_entries', []),
            'golden_entry': self.golden_entry,
            'intermediate_entries': self.intermediate_entries,
            'sanitize_c_name': self.sanitize_c_name,
            'shared_activation_pool': self.shared_activation_pool,
            'block_activation_buffers': self.block_activation_buffers,
            'block_buffer_role_map': self.block_buffer_role_map,
            # Pipeline metadata variables
            'l2_arena_size': self.l2_arena_size,
            # Board mode: minimal prints, no golden checks, just cycles
            'board_mode': self.board_mode,
            # NE16 accelerator mode
            'use_ne16': self.use_ne16,
            'ne16_eligible_layers': self.ne16_eligible_layers,
            'max_ne16_weight_l1_size': self._calc_max_ne16_weight_size(),
            'max_ne16_bias_l1_size': self._calc_max_ne16_bias_size(),
            # SSM parameter entries
            'ssm_entries': self.ssm_entries,
            # MambaBlock parameter entries
            'mamba_block_entries': self.mamba_block_entries,
            # Alternating Attention NE16 parameter entries
            'alt_attn_ne16_entries': self.alt_attn_ne16_entries,
            # Mamba L3 streaming slab sizes
            'mamba_slab_sizes': self.mamba_slab_sizes,
            # L3 streamed golden validation
            'use_streamed_golden': self.use_streamed_golden,
            'max_golden_size': self.max_golden_size,
            'golden_chunk_size': self.golden_chunk_size,
            'total_golden_size': self.total_golden_size,
            # Llama/LLM support (auto-detected from model layers)
            'use_llama': self._has_llama_layers(),
        })

        output = template.render(**kwargs)

        if output_name is None:
            return output

        self._write_text(output_name, output)
        return output

    def _calc_max_ne16_weight_size(self):
        """Calculate the maximum packed weight size among all NE16-eligible layers."""
        if not self.ne16_eligible_layers:
            return 0
        max_size = 0
        for layer_name in self.ne16_eligible_layers:
            if layer_name in self.ne16_weight_entries:
                entry = self.ne16_weight_entries[layer_name]
                max_size = max(max_size, entry['size'])
        return max_size

    def _calc_max_ne16_bias_size(self):
        """Calculate the maximum bias correction size among all NE16-eligible layers."""
        if not self.ne16_eligible_layers:
            return 0
        max_size = 0
        for layer_name in self.ne16_eligible_layers:
            if layer_name in self.ne16_bias_entries:
                entry = self.ne16_bias_entries[layer_name]
                max_size = max(max_size, entry['size'])
        return max_size

    def _has_llama_layers(self):
        """Check if the model contains Llama-specific layers (RMSNorm, SwiGLU FFN, Llama block).

        All ops guarded by ARES_LLAMA_SUPPORT should trigger this flag.
        """
        for spec in self.layer_specs:
            op_type = spec.get('op', '')
            if op_type in ('rmsnorm', 'swiglu_ffn', 'llama_block'):
                return True
        return False

    def _infer_network_name(self):
        parent = self.output_dir.parent
        if parent and parent.name:
            return parent.name
        return "INT8Network"

    def _detect_input_quant_layers(self):
        """
        Detect input quantization layer(s).

        For single-input models: returns ['input_quant']
        For multi-input models: returns ['eeg_input_quant', 'ppg_input_quant'] etc.

        Also creates an 'input_quant' alias in network_info for compatibility
        if only branch-specific layers exist.
        """
        input_quant_layers = []

        # First, check for standard 'input_quant'
        if 'input_quant' in self.network_info:
            return ['input_quant']

        # Look for branch-specific input quantization layers
        for name in self.layer_order:
            layer_info = self.layer_info.get(name, {})
            layer_type = layer_info.get('type')
            if layer_type == 'QuantIdentity' and 'input_quant' in name:
                input_quant_layers.append(name)

        # Create 'input_quant' alias for compatibility with downstream code
        if input_quant_layers and 'input_quant' not in self.network_info:
            # Use the first input quant layer as the primary one
            first_quant = input_quant_layers[0]
            self.network_info['input_quant'] = self.network_info[first_quant].copy()
            print(f"  [Multi-input] Created 'input_quant' alias from '{first_quant}'")

        return input_quant_layers if input_quant_layers else ['input_quant']

    def _get_primary_input_scale(self):
        """Get the primary input scale (first input's scale for multi-input models)."""
        if self.input_quant_layers:
            first_input_quant = self.input_quant_layers[0]
            return self.network_info.get(first_input_quant, {}).get('scale', 1.0)
        return self.network_info.get('input_quant', {}).get('scale', 1.0)

    def _find_branch_outputs_before_concat(self, concat_idx):
        """
        Find the final output layer of each branch before a concat layer.

        For dual-input networks like drowsiness detection:
        - EEG branch ends with eeg_avgpool
        - PPG branch ends with ppg_avgpool
        - Concat merges these two

        Args:
            concat_idx: Index of the Concatenate layer in layer_order

        Returns:
            List of layer names that should be inputs to the concat
        """
        # Group layers by branch prefix
        branches = {}  # {prefix: [layers]}
        for i, name in enumerate(self.layer_order[:concat_idx]):
            prefix = self._get_branch_prefix(name)
            if prefix:
                if prefix not in branches:
                    branches[prefix] = []
                branches[prefix].append(name)

        # Find the last layer in each branch
        branch_outputs = []
        for prefix in sorted(branches.keys()):
            layers = branches[prefix]
            if layers:
                # Last layer of this branch before concat
                branch_outputs.append(layers[-1])

        return branch_outputs

    def _get_branch_prefix(self, layer_name):
        """Get branch prefix from layer name (e.g., 'eeg_' from 'eeg_conv1')."""
        if '_' in layer_name:
            parts = layer_name.split('_')
            prefix = parts[0] + '_'
            # Only consider known multi-input prefixes
            if prefix in ('eeg_', 'ppg_', 'input0_', 'input1_', 'branch0_', 'branch1_'):
                return prefix
        return ''

    def _determine_final_linear(self):
        # If a composite classifier produces the final output, no standard
        # linear layer should be treated as "final" (FP32 output).
        has_composite_classifier = any(
            self.layer_info.get(name, {}).get('type') == 'ClassificationHeadWithMLP'
            for name in self.layer_order
        )
        if has_composite_classifier:
            return None

        linear_layers = [
            name for name in self.layer_order
            if self.layer_info.get(name, {}).get('type') == 'QuantLinear'
        ]
        return linear_layers[-1] if linear_layers else None

    def _ensure_codegen_metadata(self):
        if self._metadata_ready:
            return
        self._build_layer_specs()
        self._calculate_mamba_slab_sizes()
        self._metadata_ready = True

    def _calculate_mamba_slab_sizes(self):
        """Calculate max sizes for shared Mamba weight slab (L3 streaming).

        Instead of allocating L2 memory for ALL Mamba directions at once (~1.2MB),
        we use a single shared slab that can hold ONE direction's weights (~300KB).
        Before each direction executes, we load its weights from L3 to this slab.

        For very large models (FEMBA Tiny with expand=4), even ONE direction's weights
        may exceed L2. In that case, we chunk the large projections (in_proj, out_proj).
        """
        if not self.mamba_block_entries:
            return

        # Calculate max sizes across all entries
        max_in_proj = max(e['in_proj_weight_elements'] for e in self.mamba_block_entries)
        max_conv1d_weight = max(e['conv1d_weight_elements'] for e in self.mamba_block_entries)
        max_conv1d_bias = max(e['conv1d_bias_elements'] for e in self.mamba_block_entries)
        max_x_proj = max(e['x_proj_weight_elements'] for e in self.mamba_block_entries)
        max_dt_proj = max(e['dt_proj_weight_elements'] for e in self.mamba_block_entries)
        max_dt_proj_bias = max(e['dt_proj_bias_elements'] for e in self.mamba_block_entries)
        max_A = max(e['A_elements'] for e in self.mamba_block_entries)
        max_D = max(e['D_elements'] for e in self.mamba_block_entries)
        max_out_proj = max(e['out_proj_weight_elements'] for e in self.mamba_block_entries)
        max_lut = 256  # All LUTs are 256 entries
        max_scratch = max(e['l2_scratch_size'] for e in self.mamba_block_entries)

        # Get d_model and d_inner for chunking calculations
        d_model = self.mamba_block_entries[0].get('d_model', 0)
        d_inner = self.mamba_block_entries[0].get('d_inner', 0)

        # Small weights that must always be loaded (not chunked)
        small_weights_size = (
            max_conv1d_weight * 1 +     # int8
            max_conv1d_bias * 4 +       # int32
            max_lut * 1 +               # silu_lut int8
            max_lut * 2 +               # silu_gate_lut_q13 int16
            max_lut * 2 +               # softplus_lut int16
            max_lut * 2 +               # exp_lut int16
            max_x_proj * 1 +            # int8
            max_dt_proj * 1 +           # int8
            max_dt_proj_bias * 4 +      # int32 Q16.16
            max_A * 2 +                 # int16 Q15
            max_D * 2                   # int16 Q15
        )

        # Total slab size for one direction (without chunking)
        total_slab_unchunked = (
            max_in_proj * 1 +           # int8
            small_weights_size +
            max_out_proj * 1            # int8
        )

        # L2 budget for Mamba operations (use unified constant)
        L2_MAMBA_BUDGET = self._get_l2_tiling_budget_bytes()

        # Check if chunking is needed
        total_with_scratch = total_slab_unchunked + max_scratch
        needs_chunking = total_with_scratch > L2_MAMBA_BUDGET

        if needs_chunking:
            print(f"  Mamba slab ({total_slab_unchunked // 1024:.1f} KB) + scratch ({max_scratch // 1024:.1f} KB) = {total_with_scratch // 1024:.1f} KB")
            print(f"  Exceeds L2 budget ({L2_MAMBA_BUDGET // 1024:.1f} KB) - enabling projection chunking")

            # OPTIMIZATION: in_proj and out_proj execute SEQUENTIALLY, so we can
            # SHARE the same chunk space for both. We only need max(in_proj_chunk, out_proj_chunk).
            # Budget = scratch + small_weights + proj_chunk (shared)
            available_for_proj_chunk = L2_MAMBA_BUDGET - max_scratch - small_weights_size

            # Double-buffering: need space for 2 chunk buffers (ping + pong)
            # Reduces chunk size but enables compute/DMA overlap for better throughput
            enable_double_buffer = True
            num_chunk_buffers = 2 if enable_double_buffer else 1

            # Reserve 95% for chunks, 5% margin for alignment/overhead
            # Divide by num_chunk_buffers to leave space for double-buffering
            proj_chunk_budget = int(available_for_proj_chunk * 0.95) // num_chunk_buffers

            # Calculate chunk sizes (each chunk holds a portion of output features)
            # in_proj: [d_model, 2*d_inner] - chunk along 2*d_inner dimension
            # out_proj: [d_inner, d_model] - chunk along d_model dimension
            in_proj_out_dim = 2 * d_inner
            out_proj_out_dim = d_model

            # in_proj chunk: how many output features fit in shared chunk space?
            # Weight row size = d_model bytes per output feature
            in_proj_chunk_out_features = max(1, proj_chunk_budget // d_model)
            in_proj_num_chunks = (in_proj_out_dim + in_proj_chunk_out_features - 1) // in_proj_chunk_out_features
            in_proj_chunk_bytes = in_proj_chunk_out_features * d_model

            # out_proj chunk: how many output features fit in shared chunk space?
            # Weight row size = d_inner bytes per output feature
            out_proj_chunk_out_features = max(1, proj_chunk_budget // d_inner)
            out_proj_num_chunks = (out_proj_out_dim + out_proj_chunk_out_features - 1) // out_proj_chunk_out_features
            out_proj_chunk_bytes = out_proj_chunk_out_features * d_inner

            print(f"  in_proj chunking: {in_proj_num_chunks} chunks of {in_proj_chunk_out_features} outputs ({in_proj_chunk_bytes // 1024:.1f} KB each)")
            print(f"  out_proj chunking: {out_proj_num_chunks} chunks of {out_proj_chunk_out_features} outputs ({out_proj_chunk_bytes // 1024:.1f} KB each)")

            # Shared chunk space = max of both projection chunks
            max_proj_chunk = max(in_proj_chunk_bytes, out_proj_chunk_bytes)
            max_in_proj_chunk = max_proj_chunk  # Use shared space
            max_out_proj_chunk = max_proj_chunk  # Same pointer, reused

            # Double-buffering: allocate 2x chunk space for ping-pong buffers
            # This allows overlapping L3 reads with compute
            # enable_double_buffer was set above when calculating proj_chunk_budget
            double_buffer_overhead = max_proj_chunk if enable_double_buffer else 0
            total_slab = small_weights_size + max_proj_chunk + double_buffer_overhead

            print(f"  Shared proj chunk space: {max_proj_chunk // 1024:.1f} KB")
            if enable_double_buffer:
                print(f"  Double-buffer enabled: +{double_buffer_overhead // 1024:.1f} KB (total {max_proj_chunk * 2 // 1024:.1f} KB)")
        else:
            # No chunking needed
            in_proj_num_chunks = 1
            out_proj_num_chunks = 1
            in_proj_chunk_out_features = 2 * d_inner if d_inner > 0 else 0
            out_proj_chunk_out_features = d_model
            max_in_proj_chunk = max_in_proj
            max_out_proj_chunk = max_out_proj
            total_slab = total_slab_unchunked
            enable_double_buffer = False
            max_proj_chunk = max(max_in_proj_chunk, max_out_proj_chunk)

        self.mamba_slab_sizes = {
            'in_proj_weight': max_in_proj_chunk,
            'in_proj_weight_full': max_in_proj,
            'in_proj_num_chunks': in_proj_num_chunks,
            'in_proj_chunk_out_features': in_proj_chunk_out_features,
            'conv1d_weight': max_conv1d_weight,
            'conv1d_bias': max_conv1d_bias,
            'x_proj_weight': max_x_proj,
            'dt_proj_weight': max_dt_proj,
            'dt_proj_bias': max_dt_proj_bias,
            'A': max_A,
            'D': max_D,
            'out_proj_weight': max_out_proj_chunk,
            'out_proj_weight_full': max_out_proj,
            'out_proj_num_chunks': out_proj_num_chunks,
            'out_proj_chunk_out_features': out_proj_chunk_out_features,
            'lut': max_lut,
            'scratch': max_scratch,
            'total': total_slab,
            'needs_chunking': needs_chunking,
            'enable_double_buffer': enable_double_buffer,
            'proj_chunk_bytes': max_proj_chunk,
            'd_model': d_model,
            'd_inner': d_inner,
        }

        print(f"  Mamba L3 streaming: weight slab = {total_slab // 1024:.1f} KB (one direction)")
        print(f"    vs {total_slab_unchunked * len(self.mamba_block_entries) // 1024:.1f} KB (all {len(self.mamba_block_entries)} directions, unchunked)")
        print(f"  Mamba shared scratch: {max_scratch // 1024:.1f} KB")
        print(f"    vs {max_scratch * len(self.mamba_block_entries) // 1024:.1f} KB (per-direction)")

    # ---
    # Layer Build Context Helpers
    # ---

    def _default_buffer_memory_annotation(self, name: str, comment: str, l2_required: bool = False) -> dict:
        """Return default memory annotation fields for new buffers."""
        text = f"{name} {comment}".lower()
        is_scratch = "slab" in text or "scratch" in text
        role = "scratch" if is_scratch else "activation"
        required_level = "L2" if (l2_required or is_scratch) else None
        return {
            "preferred_level": "L2",
            "required_level": required_level,
            "spill_allowed": False if required_level is not None else True,
            "streaming_role": role,
        }

    def _ctx_register_buffer(self, ctx: LayerBuildContext, name: str, ctype: str,
                             numel: int, comment: str, block_id: int = None,
                             l2_required: bool = False) -> dict:
        """
        Register an activation buffer in the build context.

        For transformer block buffers (block_id is not None), creates a shared
        pool buffer and maps the block-specific buffer name to it.
        Non-block buffers are allocated directly.

        Args:
            ctx: Layer build context with buffer tracking state
            name: Buffer name (e.g., "layer_conv1_out")
            ctype: C type (e.g., "int8_t")
            numel: Number of elements
            comment: Description for generated code
            block_id: If set, buffer is part of a transformer block
            l2_required: If True, buffer must stay in L2 (no L3 fallback)

        Returns:
            Buffer entry dictionary
        """
        if name in ctx.buffer_map:
            return ctx.buffer_map[name]

        # Transformer block buffer → map to shared pool
        if block_id is not None and 'blocks.' in name:
            parts = name.split('.')
            if len(parts) >= 3 and parts[0] == 'blocks':
                role_with_suffix = '.'.join(parts[2:])
                if role_with_suffix.endswith('_out'):
                    role = role_with_suffix[:-4]
                else:
                    role = role_with_suffix

                # Merge norm1/norm2 into single pool
                if role in ('norm1', 'norm2'):
                    role = 'norm'

                pool_buffer_name = f"block_{role.replace('.', '_')}_out_pool"

                # Create shared pool buffer if needed
                if pool_buffer_name not in ctx.buffer_map:
                    pool_c_name = self._unique_buffer_c_name(pool_buffer_name)
                    pool_mem = self._default_buffer_memory_annotation(pool_buffer_name, f"Shared pool for {role} across all transformer blocks")
                    pool_entry = {
                        'name': pool_buffer_name,
                        'c_name': pool_c_name,
                        'ctype': ctype,
                        'numel': numel,
                        'comment': f"Shared pool for {role} across all transformer blocks",
                        'is_pool': True,
                        **pool_mem,
                    }
                    ctx.buffer_map[pool_buffer_name] = pool_entry
                    self.shared_activation_pool.append(pool_entry)

                # Create block-specific entry aliasing to pool
                c_name = self._unique_buffer_c_name(name)
                block_mem = self._default_buffer_memory_annotation(name, comment, l2_required=l2_required)
                entry = {
                    'name': name,
                    'c_name': c_name,
                    'ctype': ctype,
                    'numel': numel,
                    'comment': comment,
                    'is_block_buffer': True,
                    'block_id': block_id,
                    'pool_buffer': pool_buffer_name,
                    'pool_c_name': ctx.buffer_map[pool_buffer_name]['c_name'],
                    **block_mem,
                }
                ctx.buffer_map[name] = entry

                self.block_buffer_role_map[name] = pool_buffer_name
                if block_id not in self.block_activation_buffers:
                    self.block_activation_buffers[block_id] = []
                self.block_activation_buffers[block_id].append(entry)

                return entry

        # Regular buffer allocation
        c_name = self._unique_buffer_c_name(name)
        regular_mem = self._default_buffer_memory_annotation(name, comment, l2_required=l2_required)
        entry = {
            'name': name,
            'c_name': c_name,
            'ctype': ctype,
            'numel': numel,
            'comment': comment,
            'l2_required': l2_required,
            **regular_mem,
        }
        ctx.buffer_map[name] = entry
        ctx.activation_buffers.append(entry)
        return entry

    def _ctx_buffer_c_name(self, ctx: LayerBuildContext, name: str) -> str:
        """Get the C variable name for a buffer."""
        if name not in ctx.buffer_map:
            raise ValueError(f"Unknown buffer '{name}'")
        return ctx.buffer_map[name]['c_name']

    def _ctx_attach_golden(self, ctx: LayerBuildContext, spec: dict,
                           layer_name: str, numel: int, buffer_name: str) -> None:
        """Attach golden comparison info to a layer spec if available."""
        entry = self.intermediate_layer_entries.get(layer_name)
        if entry and numel:
            spec['golden_slot'] = entry['slot']
            spec['golden_entry_index'] = entry['index']
            spec['golden_size'] = numel
            spec['compare_buffer'] = self._ctx_buffer_c_name(ctx, buffer_name)
            spec['golden_buffer'] = layer_name.replace('-', '_') + '_golden'

    def _ctx_get_mhsa_projection_scale(self, layer_name: str, proj_type: str,
                                        layer_data: dict, input_scale: float) -> float:
        """Get MHSA projection output scale with fallback and warning."""
        output_scale_key = f'{proj_type}_scale_output'
        output_scale = layer_data.get(output_scale_key)

        if output_scale is None:
            weight_scale = layer_data.get(f'{proj_type}_scale_weight', 1.0)
            fallback_scale = input_scale * weight_scale
            print(f"[WARN]  WARNING: {layer_name} missing '{output_scale_key}' - "
                  f"using fallback: {input_scale:.6f} x {weight_scale:.6f} = {fallback_scale:.6f}")
            print(f"   This may produce incorrect attention scores. "
                  f"Ensure PyTorch extractor captures projection output scales.")
            return fallback_scale

        return output_scale

    # ---
    # Layer Spec Handlers (extracted from _build_layer_specs)
    # ---

    def _handle_quant_identity(self, ctx: LayerBuildContext, layer_name: str,
                                layer_data: dict, spec: dict) -> bool:
        """Delegate QuantIdentity handling to extracted layer module."""
        return handle_quant_identity_v2(self, ctx, layer_name, layer_data, spec)

    def _handle_quant_relu(self, ctx: LayerBuildContext, layer_name: str,
                           layer_data: dict, spec: dict) -> bool:
        """Delegate QuantReLU handling to extracted layer module."""
        return handle_quant_relu_v2(self, ctx, layer_name, layer_data, spec)

    def _handle_squeeze(self, ctx: LayerBuildContext, layer_name: str,
                        layer_data: dict, spec: dict) -> bool:
        """Delegate Squeeze handling to extracted layer module."""
        return handle_squeeze_v2(self, ctx, layer_name, layer_data, spec)

    def _handle_flatten(self, ctx: LayerBuildContext, layer_name: str,
                        layer_data: dict, spec: dict) -> bool:
        """Delegate Flatten handling to extracted layer module."""
        return handle_flatten_v2(self, ctx, layer_name, layer_data, spec)

    def _handle_reshape(self, ctx: LayerBuildContext, layer_name: str,
                        layer_data: dict, spec: dict) -> bool:
        """Delegate Reshape handling to extracted layer module."""
        return handle_reshape_v2(self, ctx, layer_name, layer_data, spec)

    def _handle_permute(self, ctx: LayerBuildContext, layer_name: str,
                        layer_data: dict, spec: dict) -> bool:
        """Delegate Permute handling to extracted layer module."""
        return handle_permute_v2(self, ctx, layer_name, layer_data, spec)

    def _handle_zeropad2d(self, ctx: LayerBuildContext, layer_name: str,
                          layer_data: dict, spec: dict) -> bool:
        """Delegate ZeroPad2d handling to extracted layer module."""
        return handle_zeropad2d_v2(self, ctx, layer_name, layer_data, spec)

    def _handle_maxpool2d(self, ctx: LayerBuildContext, layer_name: str,
                          layer_data: dict, spec: dict) -> bool:
        """Delegate MaxPool2d handling to extracted layer module."""
        return handle_maxpool2d_v2(self, ctx, layer_name, layer_data, spec)

    def _handle_avgpool2d(self, ctx: LayerBuildContext, layer_name: str,
                          layer_data: dict, spec: dict, idx: int) -> bool:
        """Delegate AvgPool2d handling to extracted layer module."""
        return handle_avgpool2d_v2(self, ctx, layer_name, layer_data, spec, idx)

    def _handle_globalavgpool(self, ctx: LayerBuildContext, layer_name: str,
                              layer_data: dict, spec: dict, idx: int) -> bool:
        """Delegate GlobalAvgPool handling to extracted layer module."""
        return handle_globalavgpool_v2(self, ctx, layer_name, layer_data, spec, idx)

    def _handle_add(self, ctx: LayerBuildContext, layer_name: str,
                    layer_data: dict, spec: dict, idx: int) -> bool:
        """Delegate Add handling to extracted layer module."""
        return handle_add_v2(self, ctx, layer_name, layer_data, spec, idx)

    def _handle_concatenate(self, ctx: LayerBuildContext, layer_name: str,
                            layer_data: dict, spec: dict, idx: int) -> bool:
        """Delegate Concatenate handling to extracted layer module."""
        return handle_concatenate_v2(self, ctx, layer_name, layer_data, spec, idx)

    def _handle_mean(self, ctx: LayerBuildContext, layer_name: str,
                     layer_data: dict, spec: dict, idx: int) -> bool:
        """Delegate Mean handling to extracted layer module."""
        return handle_mean_v2(self, ctx, layer_name, layer_data, spec, idx)

    def _handle_alternating_attention(self, ctx: LayerBuildContext, layer_name: str,
                                       layer_data: dict, spec: dict, idx: int) -> bool:
        """Delegate AlternatingAttention handling to extracted layer module."""
        return handle_alternating_attention_v2(self, ctx, layer_name, layer_data, spec, idx)

    def _handle_groupnorm(self, ctx: LayerBuildContext, layer_name: str,
                          layer_data: dict, spec: dict, idx: int) -> bool:
        """Delegate GroupNorm handling to extracted layer module."""
        return handle_groupnorm_v2(self, ctx, layer_name, layer_data, spec, idx)

    def _handle_layernorm(self, ctx: LayerBuildContext, layer_name: str,
                          layer_data: dict, spec: dict, idx: int,
                          has_mamba_layers: bool = False) -> bool:
        """Delegate LayerNorm handling to extracted layer module."""
        return handle_layernorm_v2(
            self, ctx, layer_name, layer_data, spec, idx, has_mamba_layers
        )

    def _handle_rmsnorm(self, ctx: LayerBuildContext, layer_name: str,
                        layer_data: dict, spec: dict, idx: int,
                        has_mamba_layers: bool = False) -> bool:
        """Delegate RMSNorm handling to extracted layer module."""
        return handle_rmsnorm_v2(
            self, ctx, layer_name, layer_data, spec, idx, has_mamba_layers
        )

    def _handle_quantconv2d(self, ctx: LayerBuildContext, layer_name: str,
                            layer_data: dict, spec: dict, idx: int) -> bool:
        """Delegate QuantConv2d handling to extracted layer module."""
        return handle_quantconv2d_v2(self, ctx, layer_name, layer_data, spec, idx)

    def _handle_quantlinear(self, ctx: LayerBuildContext, layer_name: str,
                            layer_data: dict, spec: dict, idx: int) -> bool:
        """Delegate QuantLinear handling to extracted layer module."""
        return handle_quantlinear_v2(self, ctx, layer_name, layer_data, spec, idx)

    def _handle_conv1d_depthwise(self, ctx: LayerBuildContext, layer_name: str,
                                  layer_data: dict, spec: dict) -> bool:
        """Delegate Conv1D Depthwise handling to extracted layer module."""
        return handle_conv1d_depthwise_v2(self, ctx, layer_name, layer_data, spec)

    def _handle_adaptive_avgpool1d(self, ctx: LayerBuildContext, layer_name: str,
                                    layer_data: dict, spec: dict) -> bool:
        """Delegate AdaptiveAvgPool1d handling to extracted layer module."""
        return handle_adaptive_avgpool1d_v2(self, ctx, layer_name, layer_data, spec)

    def _handle_ssm(self, ctx: LayerBuildContext, layer_name: str,
                    layer_data: dict, spec: dict) -> bool:
        """Delegate SSM handling to extracted layer module."""
        return handle_ssm_v2(self, ctx, layer_name, layer_data, spec)

    def _handle_mambablock(self, ctx: LayerBuildContext, layer_name: str,
                           layer_data: dict, spec: dict) -> bool:
        """Delegate MambaBlock handling to extracted layer module."""
        return handle_mambablock_v2(self, ctx, layer_name, layer_data, spec)

    def _handle_mambawrapper(self, ctx: LayerBuildContext, layer_name: str,
                              layer_data: dict, spec: dict) -> bool:
        """Delegate MambaWrapper handling to extracted layer module."""
        return handle_mambawrapper_v2(self, ctx, layer_name, layer_data, spec)


    def _handle_rfft(self, ctx: LayerBuildContext, layer_name: str,
                     layer_data: dict, spec: dict, idx: int) -> bool:
        """Delegate RFFT handling to extracted layer module."""
        return handle_rfft_v2(self, ctx, layer_name, layer_data, spec, idx)

    def _handle_embedding(self, ctx: LayerBuildContext, layer_name: str,
                          layer_data: dict, spec: dict,
                          determine_weight_residency) -> bool:
        """Delegate Embedding handling to extracted layer module."""
        return handle_embedding_v2(
            self, ctx, layer_name, layer_data, spec, determine_weight_residency
        )

    def _handle_patchembed(self, ctx: LayerBuildContext, layer_name: str,
                           layer_data: dict, spec: dict,
                           determine_weight_residency) -> bool:
        """Delegate PatchEmbed handling to extracted layer module."""
        return handle_patchembed_v2(
            self, ctx, layer_name, layer_data, spec, determine_weight_residency
        )

    def _handle_positionalembedding(self, ctx: LayerBuildContext, layer_name: str,
                                     layer_data: dict, spec: dict,
                                     has_mamba_layers: bool) -> bool:
        """Delegate PositionalEmbedding handling to extracted layer module."""
        return handle_positionalembedding_v2(
            self, ctx, layer_name, layer_data, spec, has_mamba_layers
        )

    def _handle_gelu(self, ctx: LayerBuildContext, layer_name: str,
                     layer_data: dict, spec: dict, idx: int) -> bool:
        """Delegate GELU handling to extracted layer module."""
        return handle_gelu_v2(self, ctx, layer_name, layer_data, spec, idx)

    def _handle_silu(self, ctx: LayerBuildContext, layer_name: str,
                     layer_data: dict, spec: dict) -> bool:
        """Delegate SiLU handling to extracted layer module."""
        return handle_silu_v2(self, ctx, layer_name, layer_data, spec)

    def _build_layer_specs(self):
        """
        Build the execution plan (`self.layer_specs`) and activation buffers.

        This is the core codegen planning step:
        - Walks `self.layer_order` / `self.network_info` in execution order
        - Computes shapes, scales, and buffer lifetimes
        - Decides memory residency (L2 vs L3 staged/streamed)
        - Computes tiling configs via `codegen/gap9_model.py`
        - Registers activation buffers and shared pools

        The resulting `self.layer_specs`, `self.activation_buffers`, and arena
        sizing metadata drive template rendering under `codegen/templates/`.
        """
        if self.input_shape is None:
            input_layer = self.network_info.get('input_quant', {})
            shape = input_layer.get('output_shape')
            if shape:
                self.input_shape = shape
            else:
                raise ValueError("Input shape missing; run golden generation first.")

        current_shape = list(self.input_shape)
        if current_shape[0] != 1:
            raise ValueError("Only batch size 1 supported for codegen.")

        if self.input_numel is None:
            self.input_numel = self._numel(current_shape)

        self.layer_symbol_counts = {}
        self.buffer_symbol_counts = {}
        self.memory_hierarchy = {}
        self.buffer_memory_annotations = {}
        self.memory_level_report = {}
        self.memory_level_report_path = None
        self._memory_levels_ready = False
        self.l2_arena_size = 0
        self.planner = None
        self.planner_debug_dump_path = None
        activation_buffers = []
        buffer_map = {}
        buffer_scale = {}
        determine_weight_residency = self._determine_weight_residency

        def register_buffer(name, ctype, numel, comment, block_id=None, l2_required=False):
            """
            Register an activation buffer.

            For transformer block buffers (block_id is not None), creates
            a shared pool buffer and maps the block-specific buffer name to it.
            Non-block buffers are allocated directly as before.

            Args:
                l2_required: If True, this buffer MUST stay in L2 (cannot use L3 fallback).
                            Used for L3-tiled layer slabs that receive L3→L2 streaming.
            """
            if name in buffer_map:
                return buffer_map[name]

            # Detect transformer block buffers and map to shared pool
            if block_id is not None and 'blocks.' in name:
                # Extract buffer role from layer name (e.g., "blocks.0.norm1_out" → "norm1")
                # Pattern: blocks.{block_id}.{layer}_out
                parts = name.split('.')
                if len(parts) >= 3 and parts[0] == 'blocks':
                    # Everything after "blocks.{id}" but before "_out"
                    role_with_suffix = '.'.join(parts[2:])
                    # Remove "_out" suffix: "norm1_out" → "norm1", "attn.q_out" → "attn.q"
                    if role_with_suffix.endswith('_out'):
                        role = role_with_suffix[:-4]  # Remove last 4 chars ("_out")
                    else:
                        role = role_with_suffix

                    # Merge norm1 and norm2 into single norm pool (non-overlapping lifetimes)
                    if role in ('norm1', 'norm2'):
                        role = 'norm'  # Both norm1 and norm2 share the same pool

                    pool_buffer_name = f"block_{role.replace('.', '_')}_out_pool"

                    # Create or get shared pool buffer
                    if pool_buffer_name not in buffer_map:
                        pool_c_name = self._unique_buffer_c_name(pool_buffer_name)
                        pool_mem = self._default_buffer_memory_annotation(
                            pool_buffer_name,
                            f"Shared pool for {role} across all transformer blocks",
                        )
                        pool_entry = {
                            'name': pool_buffer_name,
                            'c_name': pool_c_name,
                            'ctype': ctype,
                            'numel': numel,
                            'comment': f"Shared pool for {role} across all transformer blocks",
                            'is_pool': True,
                            **pool_mem,
                        }
                        buffer_map[pool_buffer_name] = pool_entry
                        self.shared_activation_pool.append(pool_entry)

                    # Create block-specific entry that aliases to pool
                    c_name = self._unique_buffer_c_name(name)
                    block_mem = self._default_buffer_memory_annotation(
                        name, comment, l2_required=l2_required
                    )
                    entry = {
                        'name': name,
                        'c_name': c_name,
                        'ctype': ctype,
                        'numel': numel,
                        'comment': comment,
                        'is_block_buffer': True,
                        'block_id': block_id,
                        'pool_buffer': pool_buffer_name,
                        'pool_c_name': buffer_map[pool_buffer_name]['c_name'],
                        **block_mem,
                    }
                    buffer_map[name] = entry

                    # Track for buffer aliasing generation
                    self.block_buffer_role_map[name] = pool_buffer_name
                    if block_id not in self.block_activation_buffers:
                        self.block_activation_buffers[block_id] = []
                    self.block_activation_buffers[block_id].append(entry)

                    # Don't add to activation_buffers (not allocated directly)
                    return entry

            # Regular (non-block) buffer allocation
            c_name = self._unique_buffer_c_name(name)
            regular_mem = self._default_buffer_memory_annotation(
                name, comment, l2_required=l2_required
            )
            entry = {
                'name': name,
                'c_name': c_name,
                'ctype': ctype,
                'numel': numel,
                'comment': comment,
                'l2_required': l2_required,  # If True, cannot use L3 fallback
                **regular_mem,
            }
            buffer_map[name] = entry
            activation_buffers.append(entry)
            return entry

        def buffer_c_name(name):
            if name not in buffer_map:
                raise ValueError(f"Unknown buffer '{name}'")
            return buffer_map[name]['c_name']

        def _get_mhsa_projection_scale(layer_name, proj_type, layer_data, input_scale):
            """Get MHSA projection output scale with fallback and warning."""
            output_scale_key = f'{proj_type}_scale_output'
            output_scale = layer_data.get(output_scale_key)

            if output_scale is None:
                # Fallback to input x weight scale
                weight_scale = layer_data.get(f'{proj_type}_scale_weight', 1.0)
                fallback_scale = input_scale * weight_scale
                print(f"[WARN]  WARNING: {layer_name} missing '{output_scale_key}' - using fallback: {input_scale:.6f} x {weight_scale:.6f} = {fallback_scale:.6f}")
                print(f"   This may produce incorrect attention scores. Ensure PyTorch extractor captures projection output scales.")
                return fallback_scale

            return output_scale

        # Register input(s) - for multi-input models, register all input buffers
        # Map branch entry points to their input buffers
        self.branch_input_buffers = {}  # Maps 'ppg_input_quant' -> 'ppg_input_quant' buffer
        self.additional_input_entries = []  # Track additional inputs for prefetch generation

        register_buffer('input_quant', 'int8_t', self.input_numel, "Quantized network input")
        current_buffer = 'input_quant'
        current_scale = self.input_scale
        layer_output_buffer = {'input_quant': 'input_quant'}
        layer_output_scale = {'input_quant': self.input_scale}
        buffer_scale['input_quant'] = self.input_scale

        # For multi-input models, register additional input buffers
        # Build mapping from quant_layer to binary file index
        binary_input_map = {}
        if hasattr(self, 'additional_binary_input_entries'):
            for be in self.additional_binary_input_entries:
                binary_input_map[be['quant_layer']] = be['index']  # Index in binary_files array

        if len(self.input_quant_layers) > 1:
            for i, quant_layer in enumerate(self.input_quant_layers[1:], start=1):
                # Get input shape and scale for this branch
                layer_info = self.network_info.get(quant_layer, {})
                branch_shape = layer_info.get('output_shape', [1, 1, 1, 1])
                branch_numel = int(np.prod(branch_shape[1:]))  # Exclude batch
                branch_scale = layer_info.get('scale', self.input_scale)

                # Use the quant layer name as the buffer name for this branch
                buffer_name = quant_layer  # e.g., 'ppg_input_quant'
                register_buffer(buffer_name, 'int8_t', branch_numel, f"Input {i} ({quant_layer})")

                # Track mapping: branch entry point -> buffer
                self.branch_input_buffers[quant_layer] = buffer_name
                layer_output_buffer[quant_layer] = buffer_name
                layer_output_scale[quant_layer] = branch_scale
                buffer_scale[buffer_name] = branch_scale

                # Get correct index from binary file entries (handles[] array index)
                binary_file_index = binary_input_map.get(quant_layer, i)

                # Track for prefetch generation
                self.additional_input_entries.append({
                    'index': binary_file_index,  # Index in handles[] array
                    'quant_layer': quant_layer,
                    'buffer_name': buffer_name,
                    'buffer_c_name': buffer_map[buffer_name]['c_name'],
                    'numel': branch_numel,
                    'scale': branch_scale,
                    'shape': branch_shape,
                })
                print(f"  [Multi-input] Registered input buffer '{buffer_name}' ({branch_numel} elements, scale={branch_scale:.6f}, binary_index={binary_file_index})")

        # Trackers
        block_input_shape = {}
        block_input_scale = {}
        block_input_buffer = {}
        specs = []
        param_layers = []

        # Initialize build context for extracted handlers
        ctx = LayerBuildContext(
            current_shape=current_shape,
            current_scale=current_scale,
            current_buffer=current_buffer,
            layer_output_buffer=layer_output_buffer,
            layer_output_scale=layer_output_scale,
            buffer_map=buffer_map,
            buffer_scale=buffer_scale,
            activation_buffers=activation_buffers,
            specs=specs,
            param_layers=param_layers,
            block_input_shape=block_input_shape,
            block_input_scale=block_input_scale,
            block_input_buffer=block_input_buffer,
        )

        # Helper
        def attach_golden(spec, layer_name, numel, buffer_name):
            entry = self.intermediate_layer_entries.get(layer_name)
            if entry and numel:
                spec['golden_slot'] = entry['slot']
                spec['golden_entry_index'] = entry['index']
                spec['golden_size'] = numel
                spec['compare_buffer'] = buffer_c_name(buffer_name)
                spec['golden_buffer'] = layer_name.replace('-', '_') + '_golden'

        # Pre-scan: Check if model has MambaBlock/MambaWrapper layers
        # This is needed to determine if pos_embed can use mamba_shared_scratch
        has_mamba_layers = any(
            self.layer_info[name].get('type') in ('MambaBlock', 'MambaWrapper')
            for name in self.layer_order
        )

        # Main Loop
        for idx, layer_name in enumerate(self.layer_order):
            layer_data = self.layer_info[layer_name]
            layer_type = layer_data['type']

            spec = {
                'name': layer_name,
                'c_name': self._unique_layer_c_name(layer_name),
                'type': layer_type
            }

            # Extract block_id for transformer block staging
            block_id = layer_data.get('block_id')
            if block_id is not None:
                spec['block_id'] = block_id

            if layer_type == 'QuantIdentity':
                # Sync context and call extracted handler
                ctx.current_shape = current_shape
                ctx.current_scale = current_scale
                ctx.current_buffer = current_buffer
                self._handle_quant_identity(ctx, layer_name, layer_data, spec)
                # Sync back (specs list is shared, but scalars need sync)
                # Also sync shape, scale, and buffer for multi-input branch handling
                current_shape = ctx.current_shape
                current_scale = ctx.current_scale
                current_buffer = ctx.current_buffer  # Critical for branch entry points
                continue

            if layer_type == 'QuantReLU':
                # Sync context and call extracted handler
                ctx.current_shape = current_shape
                ctx.current_scale = current_scale
                ctx.current_buffer = current_buffer
                self._handle_quant_relu(ctx, layer_name, layer_data, spec)
                current_scale = ctx.current_scale
                continue

            if layer_type == 'MaxPool2d':
                ctx.current_shape = current_shape
                ctx.current_scale = current_scale
                ctx.current_buffer = current_buffer
                self._handle_maxpool2d(ctx, layer_name, layer_data, spec)
                current_buffer = ctx.current_buffer
                current_shape = ctx.current_shape
                continue

            if layer_type == 'AvgPool2d':
                ctx.current_shape = current_shape
                ctx.current_scale = current_scale
                ctx.current_buffer = current_buffer
                self._handle_avgpool2d(ctx, layer_name, layer_data, spec, idx)
                current_buffer = ctx.current_buffer
                current_shape = ctx.current_shape
                current_scale = ctx.current_scale
                continue

            if layer_type == 'GlobalAvgPool':
                ctx.current_shape = current_shape
                ctx.current_scale = current_scale
                ctx.current_buffer = current_buffer
                self._handle_globalavgpool(ctx, layer_name, layer_data, spec, idx)
                current_buffer = ctx.current_buffer
                current_shape = ctx.current_shape
                current_scale = ctx.current_scale
                layer_output_buffer[layer_name] = current_buffer
                layer_output_scale[layer_name] = current_scale
                buffer_scale[current_buffer] = current_scale
                continue

            if layer_type == 'ZeroPad2d':
                ctx.current_shape = current_shape
                ctx.current_scale = current_scale
                ctx.current_buffer = current_buffer
                self._handle_zeropad2d(ctx, layer_name, layer_data, spec)
                current_buffer = ctx.current_buffer
                current_shape = ctx.current_shape
                layer_output_buffer[layer_name] = current_buffer
                layer_output_scale[layer_name] = current_scale
                buffer_scale[current_buffer] = current_scale
                continue

            if layer_type == 'Add':
                ctx.current_shape = current_shape
                ctx.current_scale = current_scale
                ctx.current_buffer = current_buffer
                self._handle_add(ctx, layer_name, layer_data, spec, idx)
                current_buffer = ctx.current_buffer
                current_shape = ctx.current_shape
                current_scale = ctx.current_scale
                layer_output_buffer[layer_name] = current_buffer
                layer_output_scale[layer_name] = current_scale
                buffer_scale[current_buffer] = current_scale
                continue

            if layer_type == 'Concatenate':
                ctx.current_shape = current_shape
                ctx.current_scale = current_scale
                ctx.current_buffer = current_buffer
                self._handle_concatenate(ctx, layer_name, layer_data, spec, idx)
                current_buffer = ctx.current_buffer
                current_shape = ctx.current_shape
                current_scale = ctx.current_scale
                layer_output_buffer[layer_name] = current_buffer
                layer_output_scale[layer_name] = current_scale
                buffer_scale[current_buffer] = current_scale
                continue

            if layer_type == 'Mean':
                ctx.current_shape = current_shape
                ctx.current_scale = current_scale
                ctx.current_buffer = current_buffer
                self._handle_mean(ctx, layer_name, layer_data, spec, idx)
                current_buffer = ctx.current_buffer
                current_shape = ctx.current_shape
                current_scale = ctx.current_scale
                layer_output_buffer[layer_name] = current_buffer
                layer_output_scale[layer_name] = current_scale
                buffer_scale[current_buffer] = current_scale
                continue

            if layer_type == 'AlternatingAttention':
                ctx.current_shape = current_shape
                ctx.current_scale = current_scale
                ctx.current_buffer = current_buffer
                self._handle_alternating_attention(ctx, layer_name, layer_data, spec, idx)
                current_buffer = ctx.current_buffer
                current_shape = ctx.current_shape
                current_scale = ctx.current_scale
                layer_output_buffer[layer_name] = current_buffer
                layer_output_scale[layer_name] = current_scale
                buffer_scale[current_buffer] = current_scale
                continue

            if layer_type == 'MultiheadSelfAttention':
                seq_len = layer_data.get('sequence_length')
                embed_dim = layer_data.get('embed_dim')
                num_heads = layer_data.get('num_heads', 1)
                head_dim = layer_data.get('head_dim') or (embed_dim // num_heads if embed_dim and num_heads else None)
                if seq_len is None or embed_dim is None:
                    raise ValueError(f"Attention layer {layer_name} missing sequence/embedding metadata")
                pool_seq = layer_data.get('pool_sequence', 'mean')
                pool_mode = 1 if pool_seq == 'mean' else 0
                
                if pool_mode == 0:  # No pooling
                    out_dim = seq_len * embed_dim
                    out_shape = [1, seq_len, embed_dim]
                else:  # Mean pooling
                    out_dim = embed_dim
                    out_shape = [1, embed_dim]
                
                q_scale_output = layer_data.get('q_scale_output')
                k_scale_output = layer_data.get('k_scale_output')
                v_scale_output = layer_data.get('v_scale_output')

                # Force INT8 projections path (never use FP32)
                # INT8 uses 4x less memory: critical for L2 budget with large seq_len
                # If output scales missing, use input scale as reasonable default
                if q_scale_output is None:
                    q_scale_output = current_scale
                    print(f"  [MHSA] {layer_name}: q_scale_output missing, using input scale {current_scale:.6f}")
                if k_scale_output is None:
                    k_scale_output = current_scale
                    print(f"  [MHSA] {layer_name}: k_scale_output missing, using input scale {current_scale:.6f}")
                if v_scale_output is None:
                    v_scale_output = current_scale
                    print(f"  [MHSA] {layer_name}: v_scale_output missing, using input scale {current_scale:.6f}")

                use_fp32_projections = False
                
                buffer_name = f"{layer_name}_out"
                output_entry = register_buffer(buffer_name, 'int8_t', out_dim, f"{layer_name} output", spec.get('block_id'))
                scale_output = layer_data.get('scale_output', current_scale)

                # Register Params (Keep existing helper)
                def register_attention_param(prefix: str):
                    weight_key = f"{layer_name}::{prefix}"
                    weight_entry = self.weight_entries.get(weight_key)
                    bias_entry = self.bias_entries.get(weight_key)
                    if weight_entry is None:
                        raise ValueError(f"Missing {prefix} weights for attention layer {layer_name}")
                    in_features = layer_data.get(f'{prefix}_in_features', embed_dim)
                    out_features = layer_data.get(f'{prefix}_out_features', embed_dim)
                    # Use unified weight residency logic
                    weight_size_bytes = in_features * out_features
                    weight_residency = determine_weight_residency(
                        weight_size_bytes=weight_size_bytes,
                        layer_type='mhsa_projection'
                    )
                    param = {
                        'name': f"{layer_name}_{prefix}",
                        'c_name': self._unique_layer_c_name(f"{layer_name}_{prefix}"),
                        'weight_elements': in_features * out_features,
                        # MHSA projections use INT32 bias (quantized like conv/linear)
                        'bias_type': 'int32',
                        'bias_elements': out_features,
                        'weight_index': weight_entry['index'],
                        'bias_index': bias_entry['index'] if bias_entry else None,
                        'weight_residency': weight_residency,
                    }
                    if 'block_id' in spec: param['block_id'] = spec['block_id']
                    param_layers.append(param)
                    return param

                param_q = register_attention_param('q')
                param_k = register_attention_param('k')
                param_v = register_attention_param('v')
                param_o = register_attention_param('out')

                # Optional RoPE tables (Q15 sin/cos) for RoPE-enabled MHSA.
                use_rope = bool(layer_data.get('use_rope', False))
                rope_cos_param = None
                rope_sin_param = None
                if use_rope:
                    rope_cos_entry = self.weight_entries.get(f"{layer_name}::rope_cos")
                    rope_sin_entry = self.weight_entries.get(f"{layer_name}::rope_sin")
                    if rope_cos_entry is None or rope_sin_entry is None:
                        raise ValueError(f"RoPE MHSA {layer_name}: missing RoPE table binaries (rope_cos/rope_sin)")

                    if head_dim is None or (head_dim % 2) != 0:
                        raise ValueError(f"RoPE MHSA {layer_name}: head_dim must be even, got {head_dim}")

                    rope_elements = int(seq_len) * int(head_dim // 2)
                    rope_cos_param = {
                        'name': f"{layer_name}_rope_cos",
                        'c_name': self._unique_layer_c_name(f"{layer_name}_rope_cos"),
                        'weight_type': 'int16',
                        'weight_elements': rope_elements,
                        'bias_elements': 0,
                        'bias_index': None,
                        'weight_index': rope_cos_entry['index'],
                        'weight_residency': WEIGHT_RESIDENCY_L2,
                    }
                    rope_sin_param = {
                        'name': f"{layer_name}_rope_sin",
                        'c_name': self._unique_layer_c_name(f"{layer_name}_rope_sin"),
                        'weight_type': 'int16',
                        'weight_elements': rope_elements,
                        'bias_elements': 0,
                        'bias_index': None,
                        'weight_index': rope_sin_entry['index'],
                        'weight_residency': WEIGHT_RESIDENCY_L2,
                    }
                    param_layers.append(rope_cos_param)
                    param_layers.append(rope_sin_param)

                softmax_scale = layer_data.get('softmax_scale')
                if softmax_scale is None and head_dim:
                    softmax_scale = 1.0 / math.sqrt(head_dim)

                # Query KB BEFORE tiling to allow auto-application of learned configs
                self._prepare_kb_config(
                    layer_name=layer_name,
                    op_type='mhsa_int8',
                    shape={
                        'seq_len': seq_len,
                        'embed_dim': embed_dim,
                        'num_heads': num_heads,
                        'head_dim': head_dim,
                    }
                )

                # Calculate Tiling
                # Pass L2 budget for L3 Tiling check (use unified constant)
                l2_budget = self._get_l2_tiling_budget_bytes()
                tile_config = None
                memory_tier = 'L2_FULL'

                if self.enable_l1_tiling and head_dim is not None:
                    # Check for config override (auto-tuning) - extract hint to guide tiling
                    hint_tile_q = None
                    override = self._get_layer_override(layer_name)
                    if override and 'tile_config' in override:
                        tc = override['tile_config']
                        hint_tile_q = tc.get('tile_q')
                        if hint_tile_q:
                            print(f"  [TUNE] Using hint for {layer_name}: tile_q={hint_tile_q}")

                    tile_config = calculate_mhsa_tile_size(
                        seq_len=seq_len,
                        head_dim=head_dim,
                        num_heads=num_heads,
                        l1_budget=self.l1_budget_bytes,
                        l2_budget=l2_budget,
                        hint_tile_q=hint_tile_q
                    )
                    if tile_config:
                        if tile_config.l3_tiling_enabled:
                            memory_tier = 'L3_TILED'
                        else:
                            memory_tier = 'L1_TILED'

                # --- UPDATE C (MHSA): Handle L3 Tiling Buffers ---
                if memory_tier == 'L3_TILED':
                    # 1. Mark main output for L3
                    output_entry['use_l3_fallback'] = True
                    
                    # 2. Create L2 Slabs for Q, K, V, and Output
                    # Slab size = l3_seq_len * embed_dim
                    slab_elements = tile_config.l3_seq_len * embed_dim
                    
                    # Q Slab
                    q_slab = register_buffer(f"{layer_name}_q_slab", 'int8_t', slab_elements, "L2 Q Slab", None)
                    spec['q_slab_buffer'] = q_slab['c_name']
                    
                    # K Slab
                    k_slab = register_buffer(f"{layer_name}_k_slab", 'int8_t', slab_elements, "L2 K Slab", None)
                    spec['k_slab_buffer'] = k_slab['c_name']
                    
                    # V Slab
                    v_slab = register_buffer(f"{layer_name}_v_slab", 'int8_t', slab_elements, "L2 V Slab", None)
                    spec['v_slab_buffer'] = v_slab['c_name']
                    
                    # Output Slab
                    out_slab = register_buffer(f"{layer_name}_out_slab", 'int8_t', slab_elements, "L2 Output Slab", None)
                    spec['output_slab_buffer'] = out_slab['c_name']
                    input_buf_obj = buffer_map.get(current_buffer) # current_buffer is the input to this layer
                    if input_buf_obj and input_buf_obj.get('use_l3_fallback', False):
                         # Input slab size = slab_seq_len * embed_dim
                         slab_in_size = tile_config.l3_seq_len * embed_dim
                         slab_in_entry = register_buffer(f"{layer_name}_in_slab", 'int8_t', slab_in_size, "L2 Input Slab", None)
                         spec['input_slab_buffer'] = slab_in_entry['c_name']
                # -------------------------------------------------

                # Always allocate dedicated Q/K/V buffers
                q_buffer_entry = register_buffer(
                    f"{layer_name}_q", 'int8_t', seq_len * embed_dim, f"{layer_name} Q buffer", spec.get('block_id')
                )
                k_buffer_entry = register_buffer(
                    f"{layer_name}_k", 'int8_t', seq_len * embed_dim, f"{layer_name} K buffer", spec.get('block_id')
                )
                v_buffer_entry = register_buffer(
                    f"{layer_name}_v", 'int8_t', seq_len * embed_dim, f"{layer_name} V buffer", spec.get('block_id')
                )

                # In-place permutation using output buffer as scratch
                # Problem: Permutation from [seq, embed] to [heads, seq, head_dim] cannot be done
                # in-place because reads/writes overlap during iteration.
                #
                # Solution: Use the OUTPUT buffer as scratch space. It's unused until attention completes.
                # Algorithm for each of Q, K, V:
                #   1. Permute src → output_buffer (scratch)
                #   2. Memcpy output_buffer → src (now src is permuted in-place)
                #
                # Memory savings: 3 * seq_len * embed_dim = 460,800 bytes for test_20!
                # This is critical for TinyMyo to fit in L2.

                # Determine layer-level residency for prefetch (if any sub-weights are staged)
                is_staged = any(p.get('weight_residency') == WEIGHT_RESIDENCY_L3_STAGED for p in (param_q, param_k, param_v, param_o))

                # L1 weight caching: calculate if weights fit in L1 alongside inner loop buffers
                # Each projection weight is embed_dim * embed_dim bytes
                proj_weight_size = embed_dim * embed_dim
                total_proj_weights = 4 * proj_weight_size  # Q, K, V, Out

                # Inner loop L1 usage (same calculation as in C code)
                k_size = seq_len * head_dim
                v_size = seq_len * head_dim
                tile_q = tile_config.tile_q if tile_config else seq_len
                q_tile_size = tile_q * head_dim
                scores_size = tile_q * seq_len * 4  # sizeof(float)
                m_tile_size = tile_q * head_dim
                inner_loop_l1 = k_size + v_size + 2*q_tile_size + scores_size + 2*m_tile_size

                # Use target-specific L1 budget for MHSA projection caching.
                assert self.l1_budget_bytes is not None, (
                    f"Target '{self.target.name}' missing l1_budget_bytes for MHSA weight caching"
                )
                l1_budget_for_weights = self.l1_budget_bytes
                l1_available_for_weights = l1_budget_for_weights - inner_loop_l1
                l1_weight_caching_enabled = (l1_available_for_weights >= total_proj_weights) and not is_staged and not self.disable_l1_weight_caching

                if l1_weight_caching_enabled:
                    print(f"   -> MHSA '{layer_name}' L1 weight caching ENABLED "
                          f"(4 x {proj_weight_size} = {total_proj_weights} bytes, "
                          f"{l1_available_for_weights} available)")
                else:
                    reason = "L3 staged weights" if is_staged else f"need {total_proj_weights}, have {l1_available_for_weights}"
                    print(f"   -> MHSA '{layer_name}' L1 weight caching DISABLED ({reason})")

                spec.update({
                    'op': 'mhsa',
                    'input_buffer': buffer_c_name(current_buffer),
                    'output_buffer': output_entry['c_name'],
                    'seq_len': seq_len,
                    'embed_dim': embed_dim,
                    'num_heads': num_heads,
                    'head_dim': head_dim,
                    'pool_mode': pool_mode,
                    'use_fp32_projections': use_fp32_projections,
                    'scale_input': current_scale,
                    'scale_output': scale_output,
                    'q_scale_output': q_scale_output,
                    'k_scale_output': k_scale_output,
                    'v_scale_output': v_scale_output,
                    'softmax_scale': softmax_scale or 1.0,
                    'use_rope': use_rope,
                    'rope_cos_param': rope_cos_param['c_name'] if rope_cos_param else None,
                    'rope_sin_param': rope_sin_param['c_name'] if rope_sin_param else None,
                    'q_param': param_q['c_name'],
                    'k_param': param_k['c_name'],
                    'v_param': param_v['c_name'],
                    'out_param': param_o['c_name'],
                    'q_scale_weight': layer_data.get('q_scale_weight', 1.0),
                    'k_scale_weight': layer_data.get('k_scale_weight', 1.0),
                    'v_scale_weight': layer_data.get('v_scale_weight', 1.0),
                    'out_scale_weight': layer_data.get('out_scale_weight', 1.0),
                    # Sanity check: warn if output scales are missing (fallback to input x weight)
                    'scale_q': _get_mhsa_projection_scale(layer_name, 'q', layer_data, current_scale),
                    'scale_k': _get_mhsa_projection_scale(layer_name, 'k', layer_data, current_scale),
                    'scale_v': _get_mhsa_projection_scale(layer_name, 'v', layer_data, current_scale),
                    'weight_residency': WEIGHT_RESIDENCY_L3_STAGED if is_staged else WEIGHT_RESIDENCY_L2,
                    'q_weight_elements': param_q['weight_elements'],
                    'k_weight_elements': param_k['weight_elements'],
                    'v_weight_elements': param_v['weight_elements'],
                    'out_weight_elements': param_o['weight_elements'],
                    'bias_elements': param_q['bias_elements'],  # same for all projections
                    'q_buffer': q_buffer_entry['c_name'],
                    'k_buffer': k_buffer_entry['c_name'],
                    'v_buffer': v_buffer_entry['c_name'],
                    # In-place permutation using output buffer as scratch
                    # No separate Q_perm, K_perm, V_perm buffers needed - saves 450KB!
                    'use_inplace_permute': True,
                    'tile_config': tile_config.to_dict() if tile_config else None,
                    'memory_tier': memory_tier,
                    # L1 weight caching fields
                    'l1_weight_caching_enabled': l1_weight_caching_enabled,
                    'l1_proj_weight_size': proj_weight_size if l1_weight_caching_enabled else 0
                })

                attach_golden(spec, layer_name, out_dim, buffer_name)
                specs.append(spec)
                current_buffer = buffer_name
                current_shape = out_shape
                current_scale = scale_output
                layer_output_buffer[layer_name] = current_buffer
                layer_output_scale[layer_name] = current_scale
                buffer_scale[current_buffer] = current_scale
                continue

            if layer_type == 'CrossAttention':
                # Cross-Attention: learned Q queries attend to KV input sequence.
                if len(current_shape) != 3:
                    raise ValueError(f"CrossAttention {layer_name} expects [B, N, D], got {current_shape}")

                batch = int(current_shape[0])
                kv_len = int(current_shape[1])
                embed_dim = int(layer_data.get('embed_dim', current_shape[2]))
                if int(current_shape[2]) != embed_dim:
                    raise ValueError(f"CrossAttention {layer_name}: embed_dim mismatch {current_shape[2]} vs {embed_dim}")

                num_heads = int(layer_data.get('num_heads', 1))
                head_dim = int(layer_data.get('head_dim') or (embed_dim // num_heads))
                if embed_dim % num_heads != 0:
                    raise ValueError(f"CrossAttention {layer_name}: embed_dim={embed_dim} not divisible by num_heads={num_heads}")

                num_queries = layer_data.get('num_queries')
                if num_queries is None:
                    # Fallback: infer from exported query table if present
                    query_table = layer_data.get('query_embed_int8')
                    if query_table is None:
                        raise ValueError(f"CrossAttention {layer_name}: missing num_queries and query_embed_int8")
                    num_queries = int(np.array(query_table).shape[0])
                num_queries = int(num_queries)

                out_shape = [batch, num_queries, embed_dim]
                out_dim = batch * num_queries * embed_dim

                buffer_name = f"{layer_name}_out"
                output_entry = register_buffer(buffer_name, 'int8_t', out_dim, f"{layer_name} output", spec.get('block_id'))
                scale_output = layer_data.get('scale_output', current_scale)

                # Projection output scales (INT8 Path)
                q_scale_output = layer_data.get('q_scale_output')
                k_scale_output = layer_data.get('k_scale_output')
                v_scale_output = layer_data.get('v_scale_output')
                if q_scale_output is None:
                    q_scale_output = current_scale
                    print(f"  [CrossAttention] {layer_name}: q_scale_output missing, using input scale {current_scale:.6f}")
                if k_scale_output is None:
                    k_scale_output = current_scale
                    print(f"  [CrossAttention] {layer_name}: k_scale_output missing, using input scale {current_scale:.6f}")
                if v_scale_output is None:
                    v_scale_output = current_scale
                    print(f"  [CrossAttention] {layer_name}: v_scale_output missing, using input scale {current_scale:.6f}")

                query_scale = float(layer_data.get('query_scale', 1.0))

                # Register query embedding table parameter (INT8, no bias)
                query_entry = self.weight_entries.get(f"{layer_name}::query_embed")
                if query_entry is None:
                    raise ValueError(f"CrossAttention {layer_name}: missing query_embed binary entry")
                query_param = {
                    'name': f"{layer_name}_query_embed",
                    'c_name': self._unique_layer_c_name(f"{layer_name}_query_embed"),
                    'weight_elements': num_queries * embed_dim,
                    'bias_elements': 0,
                    'weight_index': query_entry['index'],
                    'bias_index': None,
                    'weight_residency': determine_weight_residency(
                        weight_size_bytes=num_queries * embed_dim,
                        layer_type='embedding'
                    ),
                }
                param_layers.append(query_param)

                # Register projection params (reuse MHSA projection residency logic)
                def register_cross_attention_param(prefix: str):
                    weight_key = f"{layer_name}::{prefix}"
                    weight_entry = self.weight_entries.get(weight_key)
                    bias_entry = self.bias_entries.get(weight_key)
                    if weight_entry is None:
                        raise ValueError(f"Missing {prefix} weights for cross-attention layer {layer_name}")
                    in_features = layer_data.get(f'{prefix}_in_features', embed_dim)
                    out_features = layer_data.get(f'{prefix}_out_features', embed_dim)
                    weight_size_bytes = in_features * out_features
                    weight_residency = determine_weight_residency(
                        weight_size_bytes=weight_size_bytes,
                        layer_type='mhsa_projection'
                    )
                    param = {
                        'name': f"{layer_name}_{prefix}",
                        'c_name': self._unique_layer_c_name(f"{layer_name}_{prefix}"),
                        'weight_elements': in_features * out_features,
                        'bias_type': 'int32',
                        'bias_elements': out_features,
                        'weight_index': weight_entry['index'],
                        'bias_index': bias_entry['index'] if bias_entry else None,
                        'weight_residency': weight_residency,
                    }
                    param_layers.append(param)
                    return param

                param_q = register_cross_attention_param('q')
                param_k = register_cross_attention_param('k')
                param_v = register_cross_attention_param('v')
                param_o = register_cross_attention_param('out')

                # Intermediate buffers (L2)
                q_buffer_entry = register_buffer(
                    f"{layer_name}_q", 'int8_t', num_queries * embed_dim, f"{layer_name} Q buffer", spec.get('block_id')
                )
                k_buffer_entry = register_buffer(
                    f"{layer_name}_k", 'int8_t', batch * kv_len * embed_dim, f"{layer_name} K buffer", spec.get('block_id')
                )
                v_buffer_entry = register_buffer(
                    f"{layer_name}_v", 'int8_t', batch * kv_len * embed_dim, f"{layer_name} V buffer", spec.get('block_id')
                )
                ctx_buffer_entry = register_buffer(
                    f"{layer_name}_ctx", 'int8_t', batch * num_queries * embed_dim, f"{layer_name} context buffer", spec.get('block_id')
                )

                softmax_scale = float(layer_data.get('softmax_scale') or (1.0 / np.sqrt(head_dim)))

                spec.update({
                    'op': 'cross_attention',
                    'input_buffer': buffer_c_name(current_buffer),
                    'output_buffer': output_entry['c_name'],
                    'batch': batch,
                    'kv_len': kv_len,
                    'num_queries': num_queries,
                    'embed_dim': embed_dim,
                    'num_heads': num_heads,
                    'head_dim': head_dim,
                    'scale_input': current_scale,
                    'query_scale': query_scale,
                    'scale_output': scale_output,
                    'q_scale_output': q_scale_output,
                    'k_scale_output': k_scale_output,
                    'v_scale_output': v_scale_output,
                    'scale_q': q_scale_output,
                    'scale_k': k_scale_output,
                    'scale_v': v_scale_output,
                    'softmax_scale': softmax_scale,
                    'query_param': query_param['c_name'],
                    'q_param': param_q['c_name'],
                    'k_param': param_k['c_name'],
                    'v_param': param_v['c_name'],
                    'out_param': param_o['c_name'],
                    'q_scale_weight': layer_data.get('q_scale_weight', 1.0),
                    'k_scale_weight': layer_data.get('k_scale_weight', 1.0),
                    'v_scale_weight': layer_data.get('v_scale_weight', 1.0),
                    'out_scale_weight': layer_data.get('out_scale_weight', 1.0),
                    'q_buffer': q_buffer_entry['c_name'],
                    'k_buffer': k_buffer_entry['c_name'],
                    'v_buffer': v_buffer_entry['c_name'],
                    'ctx_buffer': ctx_buffer_entry['c_name'],
                })
                attach_golden(spec, layer_name, out_dim, buffer_name)
                specs.append(spec)
                current_buffer = buffer_name
                current_shape = out_shape
                current_scale = scale_output
                layer_output_buffer[layer_name] = current_buffer
                layer_output_scale[layer_name] = current_scale
                buffer_scale[current_buffer] = current_scale
                continue

            if layer_type == 'GroupNorm':
                ctx.current_shape = current_shape
                ctx.current_scale = current_scale
                ctx.current_buffer = current_buffer
                self._handle_groupnorm(ctx, layer_name, layer_data, spec, idx)
                current_buffer = ctx.current_buffer
                current_scale = ctx.current_scale
                layer_output_buffer[layer_name] = current_buffer
                layer_output_scale[layer_name] = current_scale
                buffer_scale[current_buffer] = current_scale
                continue

            if layer_type == 'LayerNorm':
                ctx.current_shape = current_shape
                ctx.current_scale = current_scale
                ctx.current_buffer = current_buffer
                self._handle_layernorm(ctx, layer_name, layer_data, spec, idx, has_mamba_layers)
                current_buffer = ctx.current_buffer
                current_scale = ctx.current_scale
                layer_output_buffer[layer_name] = current_buffer
                layer_output_scale[layer_name] = current_scale
                buffer_scale[current_buffer] = current_scale
                continue

            if layer_type == 'RMSNorm':
                ctx.current_shape = current_shape
                ctx.current_scale = current_scale
                ctx.current_buffer = current_buffer
                self._handle_rmsnorm(ctx, layer_name, layer_data, spec, idx, has_mamba_layers)
                current_buffer = ctx.current_buffer
                current_scale = ctx.current_scale
                layer_output_buffer[layer_name] = current_buffer
                layer_output_scale[layer_name] = current_scale
                buffer_scale[current_buffer] = current_scale
                continue

            if layer_type == 'GELU':
                ctx.current_shape = current_shape
                ctx.current_scale = current_scale
                ctx.current_buffer = current_buffer
                self._handle_gelu(ctx, layer_name, layer_data, spec, idx)
                current_scale = ctx.current_scale
                continue

            if layer_type == 'AdaptiveAvgPool1d':
                ctx.current_shape = current_shape
                ctx.current_scale = current_scale
                ctx.current_buffer = current_buffer
                self._handle_adaptive_avgpool1d(ctx, layer_name, layer_data, spec)
                current_buffer = ctx.current_buffer
                current_shape = ctx.current_shape
                layer_output_buffer[layer_name] = current_buffer
                layer_output_scale[layer_name] = current_scale
                buffer_scale[current_buffer] = current_scale
                continue

            if layer_type == 'Squeeze':
                ctx.current_shape = current_shape
                ctx.current_scale = current_scale
                ctx.current_buffer = current_buffer
                self._handle_squeeze(ctx, layer_name, layer_data, spec)
                current_shape = ctx.current_shape
                continue

            if layer_type == 'Flatten':
                ctx.current_shape = current_shape
                ctx.current_scale = current_scale
                ctx.current_buffer = current_buffer
                self._handle_flatten(ctx, layer_name, layer_data, spec)
                current_shape = ctx.current_shape
                continue

            if layer_type == 'Permute':
                ctx.current_shape = current_shape
                ctx.current_scale = current_scale
                ctx.current_buffer = current_buffer
                self._handle_permute(ctx, layer_name, layer_data, spec)
                current_buffer = ctx.current_buffer
                current_shape = ctx.current_shape
                continue

            if layer_type == 'Reshape':
                ctx.current_shape = current_shape
                ctx.current_scale = current_scale
                ctx.current_buffer = current_buffer
                self._handle_reshape(ctx, layer_name, layer_data, spec)
                current_shape = ctx.current_shape
                continue

            if layer_type == 'QuantConv2d':
                ctx.current_shape = current_shape
                ctx.current_scale = current_scale
                ctx.current_buffer = current_buffer
                # Sync block tracking from local to context
                ctx.block_input_shape = block_input_shape
                ctx.block_input_scale = block_input_scale
                ctx.block_input_buffer = block_input_buffer
                self._handle_quantconv2d(ctx, layer_name, layer_data, spec, idx)
                # Sync block tracking back from context
                block_input_shape = ctx.block_input_shape
                block_input_scale = ctx.block_input_scale
                block_input_buffer = ctx.block_input_buffer
                current_buffer = ctx.current_buffer
                current_shape = ctx.current_shape
                current_scale = ctx.current_scale
                layer_output_buffer[layer_name] = current_buffer
                layer_output_scale[layer_name] = current_scale
                buffer_scale[current_buffer] = current_scale
                continue

            if layer_type == 'QuantLinear':
                ctx.current_shape = current_shape
                ctx.current_scale = current_scale
                ctx.current_buffer = current_buffer
                self._handle_quantlinear(ctx, layer_name, layer_data, spec, idx)
                current_buffer = ctx.current_buffer
                current_shape = ctx.current_shape
                current_scale = ctx.current_scale
                layer_output_buffer[layer_name] = current_buffer
                layer_output_scale[layer_name] = current_scale
                buffer_scale[current_buffer] = current_scale
                continue

            # Mamba layer types
            if layer_type == 'Conv1dDepthwise':
                ctx.current_shape = current_shape
                ctx.current_scale = current_scale
                ctx.current_buffer = current_buffer
                self._handle_conv1d_depthwise(ctx, layer_name, layer_data, spec)
                current_buffer = ctx.current_buffer
                current_shape = ctx.current_shape
                current_scale = ctx.current_scale
                layer_output_buffer[layer_name] = current_buffer
                layer_output_scale[layer_name] = current_scale
                buffer_scale[current_buffer] = current_scale
                continue

            if layer_type == 'SiLU':
                ctx.current_shape = current_shape
                ctx.current_scale = current_scale
                ctx.current_buffer = current_buffer
                self._handle_silu(ctx, layer_name, layer_data, spec)
                current_scale = ctx.current_scale
                continue

            if layer_type == 'SSM':
                ctx.current_shape = current_shape
                ctx.current_scale = current_scale
                ctx.current_buffer = current_buffer
                self._handle_ssm(ctx, layer_name, layer_data, spec)
                current_buffer = ctx.current_buffer
                current_shape = ctx.current_shape
                current_scale = ctx.current_scale
                layer_output_buffer[layer_name] = current_buffer
                layer_output_scale[layer_name] = current_scale
                buffer_scale[current_buffer] = current_scale
                continue

            if layer_type == 'MambaBlock':
                ctx.current_shape = current_shape
                ctx.current_scale = current_scale
                ctx.current_buffer = current_buffer
                self._handle_mambablock(ctx, layer_name, layer_data, spec)
                current_buffer = ctx.current_buffer
                current_shape = ctx.current_shape
                current_scale = ctx.current_scale
                layer_output_buffer[layer_name] = current_buffer
                layer_output_scale[layer_name] = current_scale
                buffer_scale[current_buffer] = current_scale
                continue

            if layer_type == 'MambaWrapper':
                ctx.current_shape = current_shape
                ctx.current_scale = current_scale
                ctx.current_buffer = current_buffer
                self._handle_mambawrapper(ctx, layer_name, layer_data, spec)
                current_buffer = ctx.current_buffer
                current_shape = ctx.current_shape
                current_scale = ctx.current_scale
                layer_output_buffer[layer_name] = current_buffer
                layer_output_scale[layer_name] = current_scale
                buffer_scale[current_buffer] = current_scale
                continue


            if layer_type == 'RFFT':
                ctx.current_shape = current_shape
                ctx.current_scale = current_scale
                ctx.current_buffer = current_buffer
                self._handle_rfft(ctx, layer_name, layer_data, spec, idx)
                current_buffer = ctx.current_buffer
                current_shape = ctx.current_shape
                current_scale = ctx.current_scale
                layer_output_buffer[layer_name] = current_buffer
                layer_output_scale[layer_name] = current_scale
                buffer_scale[current_buffer] = current_scale
                continue

            if layer_type == 'Embedding':
                ctx.current_shape = current_shape
                ctx.current_scale = current_scale
                ctx.current_buffer = current_buffer
                self._handle_embedding(ctx, layer_name, layer_data, spec, self._determine_weight_residency)
                current_buffer = ctx.current_buffer
                current_shape = ctx.current_shape
                current_scale = ctx.current_scale
                layer_output_buffer[layer_name] = current_buffer
                layer_output_scale[layer_name] = current_scale
                buffer_scale[current_buffer] = current_scale
                continue

            if layer_type == 'PatchEmbed':
                ctx.current_shape = current_shape
                ctx.current_scale = current_scale
                ctx.current_buffer = current_buffer
                self._handle_patchembed(ctx, layer_name, layer_data, spec, self._determine_weight_residency)
                current_buffer = ctx.current_buffer
                current_shape = ctx.current_shape
                current_scale = ctx.current_scale
                layer_output_buffer[layer_name] = current_buffer
                layer_output_scale[layer_name] = current_scale
                buffer_scale[current_buffer] = current_scale
                continue

            if layer_type == 'PositionalEmbedding':
                ctx.current_shape = current_shape
                ctx.current_scale = current_scale
                ctx.current_buffer = current_buffer
                self._handle_positionalembedding(ctx, layer_name, layer_data, spec, has_mamba_layers)
                current_buffer = ctx.current_buffer
                current_shape = ctx.current_shape
                current_scale = ctx.current_scale
                layer_output_buffer[layer_name] = current_buffer
                layer_output_scale[layer_name] = current_scale
                buffer_scale[current_buffer] = current_scale
                continue

            if layer_type == 'CrossAttentionWithSelfRefine':
                # Composite cross-attention + FFN + self-attention refinement block.
                if len(current_shape) != 3:
                    raise ValueError(f"CrossAttentionWithSelfRefine {layer_name} expects [B, N, D], got {current_shape}")

                kv_len = int(layer_data.get('kv_len', 22))
                embed_dim = int(layer_data.get('embed_dim', current_shape[2]))
                num_heads = int(layer_data.get('num_heads', 2))
                head_dim = int(layer_data.get('head_dim') or (embed_dim // num_heads))
                num_queries = int(layer_data.get('num_queries', 4))
                ff_dim = int(layer_data.get('ff_dim', 256))
                num_sa_blocks = int(layer_data.get('num_self_attn_blocks', 3))

                # Handle per-patch reshape: [B, kv_len*num_patches, D] → [B*num_patches, kv_len, D]
                total_tokens = int(current_shape[1])
                if total_tokens != kv_len:
                    if total_tokens % kv_len != 0:
                        raise ValueError(f"CrossAttentionWithSelfRefine {layer_name}: total_tokens={total_tokens} not divisible by kv_len={kv_len}")
                    num_patches = total_tokens // kv_len
                    batch = int(current_shape[0]) * num_patches
                    needs_patch_reshape = True
                else:
                    batch = int(current_shape[0])
                    num_patches = 1
                    needs_patch_reshape = False

                # Output shape: [batch, num_queries, embed_dim]
                # If patched, output is [num_patches, num_queries, embed_dim]
                out_tokens = batch * num_queries * embed_dim
                buffer_name = f"{layer_name}_out"
                output_entry = register_buffer(buffer_name, 'int8_t', out_tokens, f"{layer_name} output", spec.get('block_id'))
                scale_output = float(layer_data.get('scale_output', current_scale))

                softmax_scale = float(layer_data.get('softmax_scale') or (1.0 / np.sqrt(head_dim)))

                # Scratch buffer for intermediates (partitioned by template)
                # Max simultaneous: normed_kv + K_proj + V_proj + context + ffn_hidden
                kv_buf_size = batch * kv_len * embed_dim
                q_buf_size = batch * num_queries * embed_dim  # For SA blocks where Q is per-batch
                ffn_hidden_size = batch * num_queries * ff_dim
                scratch_size = kv_buf_size * 3 + q_buf_size + ffn_hidden_size + num_queries * embed_dim
                scratch_entry = register_buffer(
                    f"{layer_name}_scratch", 'int8_t', scratch_size,
                    f"{layer_name} scratch", spec.get('block_id')
                )

                # Register all weight parameters (deferred: allocated on-demand, not at init)
                def reg_norm_param(prefix):
                    """Register FP32 norm weight + bias."""
                    w_entry = self.weight_entries.get(f"{layer_name}::{prefix}_weight")
                    b_entry = self.weight_entries.get(f"{layer_name}::{prefix}_bias")
                    p = {
                        'name': f"{layer_name}_{prefix}",
                        'c_name': self._unique_layer_c_name(f"{layer_name}_{prefix}"),
                        'weight_type': 'fp32',
                        'weight_elements': embed_dim,
                        'bias_type': 'float',
                        'bias_elements': embed_dim,
                        'weight_index': w_entry['index'] if w_entry else None,
                        'bias_index': b_entry['index'] if b_entry else None,
                        'weight_residency': 'L2',
                        'deferred': True,
                    }
                    param_layers.append(p)
                    return p

                def reg_proj_param(prefix, in_feat=None, out_feat=None):
                    """Register INT8 projection weight + INT32 bias."""
                    in_f = in_feat or embed_dim
                    out_f = out_feat or embed_dim
                    w_entry = self.weight_entries.get(f"{layer_name}::{prefix}")
                    b_entry = self.bias_entries.get(f"{layer_name}::{prefix}")
                    weight_size = in_f * out_f
                    p = {
                        'name': f"{layer_name}_{prefix}",
                        'c_name': self._unique_layer_c_name(f"{layer_name}_{prefix}"),
                        'weight_elements': weight_size,
                        'bias_type': 'int32',
                        'bias_elements': out_f,
                        'weight_index': w_entry['index'] if w_entry else None,
                        'bias_index': b_entry['index'] if b_entry else None,
                        'weight_residency': determine_weight_residency(
                            weight_size_bytes=weight_size, layer_type='mhsa_projection'
                        ),
                        'deferred': True,
                    }
                    param_layers.append(p)
                    return p

                # Query embedding (deferred: allocated on-demand)
                qe_entry = self.weight_entries.get(f"{layer_name}::query_embed")
                query_param = {
                    'name': f"{layer_name}_query_embed",
                    'c_name': self._unique_layer_c_name(f"{layer_name}_query_embed"),
                    'weight_elements': num_queries * embed_dim,
                    'bias_elements': 0,
                    'weight_index': qe_entry['index'] if qe_entry else None,
                    'bias_index': None,
                    'weight_residency': 'L2',
                    'deferred': True,
                }
                param_layers.append(query_param)

                # Norms
                p_queries_norm = reg_norm_param('queries_norm')
                p_keys_norm = reg_norm_param('keys_norm')
                p_values_norm = reg_norm_param('values_norm')

                # Cross-attention projections
                p_q = reg_proj_param('q')
                p_k = reg_proj_param('k')
                p_v = reg_proj_param('v')
                p_out = reg_proj_param('out')

                # FFN
                p_ffn_fc1 = reg_proj_param('ffn_fc1', in_feat=embed_dim, out_feat=ff_dim)
                p_ffn_fc2 = reg_proj_param('ffn_fc2', in_feat=ff_dim, out_feat=embed_dim)

                # Self-attention blocks
                sa_params = []
                for sa_idx in range(num_sa_blocks):
                    pfx = f"sa{sa_idx}"
                    sa_p = {
                        'norm1': reg_norm_param(f"{pfx}_norm1"),
                        'norm2': reg_norm_param(f"{pfx}_norm2"),
                        'q': reg_proj_param(f"{pfx}_q"),
                        'k': reg_proj_param(f"{pfx}_k"),
                        'v': reg_proj_param(f"{pfx}_v"),
                        'out': reg_proj_param(f"{pfx}_out"),
                        'mlp_fc1': reg_proj_param(f"{pfx}_mlp_fc1", in_feat=embed_dim, out_feat=ff_dim),
                        'mlp_fc2': reg_proj_param(f"{pfx}_mlp_fc2", in_feat=ff_dim, out_feat=embed_dim),
                    }
                    sa_params.append(sa_p)

                # Collect all scale values for template
                def get_scale(key, default=None):
                    return float(layer_data.get(key, default if default is not None else current_scale))

                spec.update({
                    'op': 'cross_attn_self_refine',
                    'input_buffer': buffer_c_name(current_buffer),
                    'output_buffer': output_entry['c_name'],
                    'scratch_buffer': scratch_entry['c_name'],
                    'batch': batch,
                    'kv_len': kv_len,
                    'num_queries': num_queries,
                    'embed_dim': embed_dim,
                    'num_heads': num_heads,
                    'head_dim': head_dim,
                    'ff_dim': ff_dim,
                    'num_sa_blocks': num_sa_blocks,
                    'needs_patch_reshape': needs_patch_reshape,
                    'num_patches': num_patches,
                    'total_input_tokens': total_tokens,
                    'softmax_scale': softmax_scale,
                    'scale_input': current_scale,
                    'scale_output': scale_output,
                    # Param c_names
                    'query_embed_param': query_param['c_name'],
                    'queries_norm_param': p_queries_norm['c_name'],
                    'keys_norm_param': p_keys_norm['c_name'],
                    'values_norm_param': p_values_norm['c_name'],
                    'q_param': p_q['c_name'],
                    'k_param': p_k['c_name'],
                    'v_param': p_v['c_name'],
                    'out_param': p_out['c_name'],
                    'ffn_fc1_param': p_ffn_fc1['c_name'],
                    'ffn_fc2_param': p_ffn_fc2['c_name'],
                    # Per-operation scales
                    'query_scale': get_scale('query_scale'),
                    'queries_norm_scale_output': get_scale('queries_norm_scale_output'),
                    'keys_norm_scale_output': get_scale('keys_norm_scale_output'),
                    'values_norm_scale_output': get_scale('values_norm_scale_output'),
                    'q_scale_weight': get_scale('q_scale_weight', 1.0),
                    'k_scale_weight': get_scale('k_scale_weight', 1.0),
                    'v_scale_weight': get_scale('v_scale_weight', 1.0),
                    'out_scale_weight': get_scale('out_scale_weight', 1.0),
                    'q_scale_output': get_scale('q_scale_output'),
                    'k_scale_output': get_scale('k_scale_output'),
                    'v_scale_output': get_scale('v_scale_output'),
                    'out_scale_output': get_scale('out_scale_output'),
                    'ffn_fc1_scale_weight': get_scale('ffn_fc1_scale_weight', 1.0),
                    'ffn_fc2_scale_weight': get_scale('ffn_fc2_scale_weight', 1.0),
                    'ffn_gelu_scale': get_scale('ffn_gelu_scale'),
                    'ffn_add_scale': get_scale('ffn_add_scale'),
                })

                # SA block params and scales
                sa_block_specs = []
                for sa_idx in range(num_sa_blocks):
                    pfx = f"sa{sa_idx}"
                    sa_spec = {
                        'norm1_param': sa_params[sa_idx]['norm1']['c_name'],
                        'norm2_param': sa_params[sa_idx]['norm2']['c_name'],
                        'q_param': sa_params[sa_idx]['q']['c_name'],
                        'k_param': sa_params[sa_idx]['k']['c_name'],
                        'v_param': sa_params[sa_idx]['v']['c_name'],
                        'out_param': sa_params[sa_idx]['out']['c_name'],
                        'mlp_fc1_param': sa_params[sa_idx]['mlp_fc1']['c_name'],
                        'mlp_fc2_param': sa_params[sa_idx]['mlp_fc2']['c_name'],
                        'norm1_scale_output': get_scale(f'{pfx}_norm1_scale_output'),
                        'norm2_scale_output': get_scale(f'{pfx}_norm2_scale_output'),
                        'q_scale_weight': get_scale(f'{pfx}_q_scale_weight', 1.0),
                        'k_scale_weight': get_scale(f'{pfx}_k_scale_weight', 1.0),
                        'v_scale_weight': get_scale(f'{pfx}_v_scale_weight', 1.0),
                        'out_scale_weight': get_scale(f'{pfx}_out_scale_weight', 1.0),
                        'q_scale_output': get_scale(f'{pfx}_q_scale_output'),
                        'k_scale_output': get_scale(f'{pfx}_k_scale_output'),
                        'v_scale_output': get_scale(f'{pfx}_v_scale_output'),
                        'out_scale_output': get_scale(f'{pfx}_out_scale_output'),
                        'add1_scale': get_scale(f'{pfx}_add1_scale'),
                        'add2_scale': get_scale(f'{pfx}_add2_scale'),
                        'mlp_fc1_scale_weight': get_scale(f'{pfx}_mlp_fc1_scale_weight', 1.0),
                        'mlp_fc2_scale_weight': get_scale(f'{pfx}_mlp_fc2_scale_weight', 1.0),
                        'mlp_gelu_scale': get_scale(f'{pfx}_mlp_gelu_scale'),
                    }
                    sa_block_specs.append(sa_spec)
                spec['sa_blocks'] = sa_block_specs

                # Collect all deferred params for on-demand loading
                all_cross_attn_params = [query_param, p_queries_norm, p_keys_norm, p_values_norm,
                                         p_q, p_k, p_v, p_out, p_ffn_fc1, p_ffn_fc2]
                for sa_p in sa_params:
                    all_cross_attn_params.extend([
                        sa_p['norm1'], sa_p['norm2'],
                        sa_p['q'], sa_p['k'], sa_p['v'], sa_p['out'],
                        sa_p['mlp_fc1'], sa_p['mlp_fc2'],
                    ])
                spec['deferred_params'] = all_cross_attn_params

                attach_golden(spec, layer_name, out_tokens, buffer_name)
                specs.append(spec)

                # Update shape tracking
                if needs_patch_reshape:
                    # After cross-attn: [num_patches, num_queries, embed_dim]
                    # Need a Reshape to [1, num_patches, num_queries * embed_dim]
                    current_buffer = buffer_name
                    current_shape = [batch, num_queries, embed_dim]
                    current_scale = scale_output

                    # Emit Reshape layer: [num_patches, num_queries, embed_dim] → [1, num_patches, num_queries*embed_dim]
                    reshape_out_shape = [int(current_shape[0]) // num_patches, num_patches, num_queries * embed_dim]
                    reshape_numel = 1
                    for d in reshape_out_shape:
                        reshape_numel *= d
                    reshape_spec = {
                        'name': f"{layer_name}_reshape",
                        'c_name': self._unique_layer_c_name(f"{layer_name}_reshape"),
                        'op': 'reshape',
                        'input_buffer': output_entry['c_name'],
                        'output_buffer': output_entry['c_name'],  # In-place
                        'scale_input': current_scale,
                        'scale_output': current_scale,
                    }
                    specs.append(reshape_spec)
                    current_shape = reshape_out_shape
                else:
                    current_buffer = buffer_name
                    current_shape = [batch, num_queries, embed_dim]
                    current_scale = scale_output

                layer_output_buffer[layer_name] = current_buffer
                layer_output_scale[layer_name] = current_scale
                buffer_scale[current_buffer] = current_scale
                continue

            if layer_type == 'ClassificationHeadWithMLP':
                # Composite classification head: cross-attention pooling + MLP.
                if len(current_shape) != 3:
                    raise ValueError(f"ClassificationHeadWithMLP {layer_name} expects [B, SeqLen, D], got {current_shape}")

                batch = int(current_shape[0])
                seq_len = int(current_shape[1])
                hidden_dim = int(layer_data.get('hidden_dim', current_shape[2]))
                num_heads = int(layer_data.get('num_heads', 8))
                head_dim = int(layer_data.get('head_dim') or (hidden_dim // num_heads))
                num_classes = int(layer_data.get('num_classes', 2))
                mlp_hidden_dim = int(layer_data.get('mlp_hidden_dim', hidden_dim * 4))

                # Infer mlp_hidden_dim from fc1 weight shape if available
                fc1_w = layer_data.get('mlp_fc1_weight_int8')
                if fc1_w is not None:
                    mlp_hidden_dim = np.array(fc1_w).shape[0]

                softmax_scale = float(layer_data.get('softmax_scale') or (1.0 / np.sqrt(head_dim)))
                scale_output = float(layer_data.get('scale_output', current_scale))

                # Output: [batch, num_classes]
                out_dim = batch * num_classes
                buffer_name = f"{layer_name}_out"
                output_entry = register_buffer(buffer_name, 'int8_t', out_dim, f"{layer_name} output", spec.get('block_id'))

                # Scratch: Q_proj + K_proj + V_proj + context + mlp_hidden
                scratch_size = (batch * 1 * hidden_dim +        # Q proj [B, 1, D]
                                batch * seq_len * hidden_dim +  # K proj [B, S, D]
                                batch * seq_len * hidden_dim +  # V proj [B, S, D]
                                batch * 1 * hidden_dim +        # context [B, 1, D]
                                batch * 1 * mlp_hidden_dim)     # MLP hidden [B, 1, mlp_dim]
                scratch_entry = register_buffer(
                    f"{layer_name}_scratch", 'int8_t', scratch_size,
                    f"{layer_name} scratch", spec.get('block_id')
                )

                # Register weight parameters
                # Learned aggregation query (deferred: allocated on-demand)
                agg_entry = self.weight_entries.get(f"{layer_name}::learned_agg")
                agg_param = {
                    'name': f"{layer_name}_learned_agg",
                    'c_name': self._unique_layer_c_name(f"{layer_name}_learned_agg"),
                    'weight_elements': 1 * hidden_dim,
                    'bias_elements': 0,
                    'weight_index': agg_entry['index'] if agg_entry else None,
                    'bias_index': None,
                    'weight_residency': 'L2',
                    'deferred': True,
                }
                param_layers.append(agg_param)

                def reg_cls_proj(prefix, in_feat=None, out_feat=None):
                    in_f = in_feat or hidden_dim
                    out_f = out_feat or hidden_dim
                    w_entry = self.weight_entries.get(f"{layer_name}::{prefix}")
                    b_entry = self.bias_entries.get(f"{layer_name}::{prefix}")
                    p = {
                        'name': f"{layer_name}_{prefix}",
                        'c_name': self._unique_layer_c_name(f"{layer_name}_{prefix}"),
                        'weight_elements': in_f * out_f,
                        'bias_type': 'int32',
                        'bias_elements': out_f,
                        'weight_index': w_entry['index'] if w_entry else None,
                        'bias_index': b_entry['index'] if b_entry else None,
                        'weight_residency': determine_weight_residency(
                            weight_size_bytes=in_f * out_f, layer_type='mhsa_projection'
                        ),
                        'deferred': True,
                    }
                    param_layers.append(p)
                    return p

                p_q = reg_cls_proj('q')
                p_k = reg_cls_proj('k')
                p_v = reg_cls_proj('v')
                p_out = reg_cls_proj('out')
                p_fc1 = reg_cls_proj('mlp_fc1', in_feat=hidden_dim, out_feat=mlp_hidden_dim)
                p_fc2 = reg_cls_proj('mlp_fc2', in_feat=mlp_hidden_dim, out_feat=num_classes)

                def get_cls_scale(key, default=None):
                    return float(layer_data.get(key, default if default is not None else current_scale))

                spec.update({
                    'op': 'classification_head_mlp',
                    'input_buffer': buffer_c_name(current_buffer),
                    'output_buffer': output_entry['c_name'],
                    'scratch_buffer': scratch_entry['c_name'],
                    'batch': batch,
                    'seq_len': seq_len,
                    'hidden_dim': hidden_dim,
                    'num_heads': num_heads,
                    'head_dim': head_dim,
                    'mlp_hidden_dim': mlp_hidden_dim,
                    'num_classes': num_classes,
                    'softmax_scale': softmax_scale,
                    'scale_input': current_scale,
                    'scale_output': scale_output,
                    # Param c_names
                    'learned_agg_param': agg_param['c_name'],
                    'q_param': p_q['c_name'],
                    'k_param': p_k['c_name'],
                    'v_param': p_v['c_name'],
                    'out_param': p_out['c_name'],
                    'mlp_fc1_param': p_fc1['c_name'],
                    'mlp_fc2_param': p_fc2['c_name'],
                    # Scales
                    'agg_scale': get_cls_scale('agg_scale'),
                    'q_scale_weight': get_cls_scale('q_scale_weight', 1.0),
                    'k_scale_weight': get_cls_scale('k_scale_weight', 1.0),
                    'v_scale_weight': get_cls_scale('v_scale_weight', 1.0),
                    'out_scale_weight': get_cls_scale('out_scale_weight', 1.0),
                    'q_scale_output': get_cls_scale('q_scale_output'),
                    'k_scale_output': get_cls_scale('k_scale_output'),
                    'v_scale_output': get_cls_scale('v_scale_output'),
                    'out_scale_output': get_cls_scale('out_scale_output'),
                    'mlp_fc1_scale_weight': get_cls_scale('mlp_fc1_scale_weight', 1.0),
                    'mlp_fc2_scale_weight': get_cls_scale('mlp_fc2_scale_weight', 1.0),
                    'mlp_gelu_scale': get_cls_scale('mlp_gelu_scale'),
                })

                # Collect all deferred params for on-demand loading
                spec['deferred_params'] = [agg_param, p_q, p_k, p_v, p_out, p_fc1, p_fc2]

                attach_golden(spec, layer_name, out_dim, buffer_name)
                specs.append(spec)
                current_buffer = buffer_name
                current_shape = [batch, num_classes]
                current_scale = scale_output
                layer_output_buffer[layer_name] = current_buffer
                layer_output_scale[layer_name] = current_scale
                buffer_scale[current_buffer] = current_scale
                continue

            raise ValueError(f"Unsupported layer type: {layer_type}")

        if self.num_classes is None:
            self.num_classes = current_shape[-1]

        # Export pre-fusion checkpoint (if enabled)
        self._write_phase_checkpoint(
            CHECKPOINT_STAGE_PRE_FUSION,
            specs_override=specs,
        )

        # Apply cross-layer fusion optimizations
        if self.enable_fusion:
            specs = self._apply_layer_fusion(specs)

        # Export post-fusion checkpoint (if enabled)
        self._write_phase_checkpoint(
            CHECKPOINT_STAGE_POST_FUSION,
            specs_override=specs,
        )

        # Update intermediate_entries to use fused golden buffers
        # When fusion changes which golden file a layer should use, update the entry
        for spec in specs:
            if 'golden_slot' in spec and 'golden_buffer' in spec:
                slot = spec['golden_slot']
                fused_golden_name = spec['golden_buffer'].replace('_golden', '')

                # Check if the fused golden file exists in our intermediate entries
                if fused_golden_name in self.intermediate_layer_entries:
                    # Get the entry for the fused golden file
                    fused_entry = self.intermediate_layer_entries[fused_golden_name]

                    # Update the entry at this slot to point to the fused golden file
                    # Keep the original slot number but change the index to point to the fused file
                    updated_entry = fused_entry.copy()
                    updated_entry['slot'] = slot
                    self.intermediate_entries[slot] = updated_entry

        # Remove duplicate golden entries (keep only entries referenced by layer specs)
        referenced_paths = set()
        for spec in specs:
            if 'golden_buffer' in spec:
                # Find the entry with this golden buffer name
                golden_name = spec['golden_buffer'].replace('_golden', '')
                if golden_name in self.intermediate_layer_entries:
                    referenced_paths.add(self.intermediate_layer_entries[golden_name]['path'])

        # Rebuild intermediate_entries with only referenced files
        filtered_entries = []
        for entry in self.intermediate_entries:
            if entry['path'] in referenced_paths:
                new_entry = entry.copy()  # Make a copy to avoid modifying binary_files
                new_entry['slot'] = len(filtered_entries)  # Reassign slot numbers
                filtered_entries.append(new_entry)

        self.intermediate_entries = filtered_entries

        # Remove unused golden files from binary_files list
        kept_golden_paths = set()
        for entry in self.intermediate_entries:
            kept_golden_paths.add(entry['path'])

        self.binary_files = [f for f in self.binary_files if not (f['path'].endswith('_golden.bin') and f['path'] not in kept_golden_paths)]

        # Rebuild index field in intermediate_entries to match new positions in binary_files
        path_to_new_index = {f['path']: i for i, f in enumerate(self.binary_files)}
        for entry in self.intermediate_entries:
            entry['index'] = path_to_new_index.get(entry['path'], -1)

        # Calculate golden output sizes for L3 streaming mode
        # When total golden size exceeds L2 capacity, use on-demand L3 streaming
        # instead of bulk prefetch
        self.max_golden_size = 0
        self.total_golden_size = 0
        for entry in self.intermediate_entries:
            size = entry['size']
            self.total_golden_size += size
            if size > self.max_golden_size:
                self.max_golden_size = size

        # Auto-enable streamed golden mode when total exceeds threshold (200KB)
        STREAMED_GOLDEN_THRESHOLD = 200 * 1024
        self.use_streamed_golden = self.total_golden_size > STREAMED_GOLDEN_THRESHOLD

        # Cap staging buffer to 64KB and use chunked comparison for large goldens
        GOLDEN_CHUNK_SIZE = 64 * 1024  # 64KB chunks
        self.golden_chunk_size = min(self.max_golden_size, GOLDEN_CHUNK_SIZE)

        if self.use_streamed_golden:
            print(f"  [OK] Enabling L3 streamed golden validation ({self.total_golden_size / 1024:.1f} KB > {STREAMED_GOLDEN_THRESHOLD / 1024:.0f} KB threshold)")
            print(f"    Max single golden: {self.max_golden_size / 1024:.1f} KB")
            print(f"    Staging buffer: {self.golden_chunk_size / 1024:.1f} KB (chunked comparison)")

        # Detect transformer block boundaries for per-block weight staging
        self._mark_transformer_block_boundaries(specs)

        # Link staged layers for async prefetch
        # If a layer (Target) needs its weights in L2 but they are in L3 (L3_STAGED),
        # the PREVIOUS layer should start the prefetch.
        for i in range(1, len(specs)):
            target_spec = specs[i]
            # Check if target layer needs prefetch
            # It must have weights, and they must be L3_STAGED
            if target_spec.get('weight_residency') == WEIGHT_RESIDENCY_L3_STAGED:
                 # Find the previous layer that executes
                 prev_spec = specs[i-1]
                 
                 prev_spec['_needs_async_start_next'] = True
                 prev_spec['_next_layer_spec'] = target_spec
                 print(f"  [Prefetch] Layer '{prev_spec['name']}' will prefetch for '{target_spec['name']}'")

        # Export post-tiling checkpoint after spec graph stabilization
        self._write_phase_checkpoint(
            CHECKPOINT_STAGE_POST_TILING,
            specs_override=specs,
        )

        self.layer_specs = specs
        self.activation_buffers = activation_buffers
        self.param_layers = param_layers

        # Calculate total MACs for the network
        self.total_macs = 0
        for spec in specs:
            layer_macs = self._calculate_layer_macs(spec)
            spec['macs'] = layer_macs  # Store per-layer MACs
            self.total_macs += layer_macs
        if self.total_macs > 0:
            print(f"  [MACs] Total network MACs: {self.total_macs:,}")

        # Link mamba blocks for cross-block prefetching
        self._link_mamba_blocks(specs)

    def _link_mamba_blocks(self, specs):
        """
        Link mamba blocks for cross-block FWD small weights prefetching.

        For each mamba block, adds:
        - mamba_block_idx: Index among all mamba blocks (0, 1, 2, ...)
        - num_mamba_blocks: Total count of mamba blocks
        - next_mamba_c_name: C name of next block (None if last)
        - is_first_mamba_block: True if this is block 0
        - has_next_mamba_block: True if there's a next block

        This enables prefetching next block's FWD small weights during
        current block's REV out_proj, hiding ~417KB of DMA per block.
        """
        # Find all mamba_wrapper specs with their indices
        mamba_specs = [(i, s) for i, s in enumerate(specs) if s.get('op') == 'mamba_wrapper']
        num_mamba = len(mamba_specs)

        if num_mamba == 0:
            return

        print(f"  [MambaLink] Linking {num_mamba} mamba blocks for cross-block prefetch")

        for mamba_idx, (spec_idx, spec) in enumerate(mamba_specs):
            spec['mamba_block_idx'] = mamba_idx
            spec['num_mamba_blocks'] = num_mamba
            spec['is_first_mamba_block'] = (mamba_idx == 0)
            spec['has_next_mamba_block'] = (mamba_idx < num_mamba - 1)

            if spec['has_next_mamba_block']:
                next_spec = mamba_specs[mamba_idx + 1][1]
                spec['next_mamba_c_name'] = next_spec['c_name']
                print(f"    Block {mamba_idx} ({spec['c_name']}) -> Block {mamba_idx + 1} ({next_spec['c_name']})")
            else:
                spec['next_mamba_c_name'] = None
                print(f"    Block {mamba_idx} ({spec['c_name']}) is last block")

    def _mark_transformer_block_boundaries(self, specs):
        """
        Detect transformer block boundaries and mark layers for per-block weight staging.

        Per-block weight staging for transformers.

        Strategy:
        1. Group layers by block_id (extracted from layer names like blocks.0.norm1)
        2. Mark first layer of each block with 'block_start'
        3. Mark last layer of each block with 'block_end'
        4. Add block_id to each spec for grouping weights

        Args:
            specs: List of layer specs (modified in-place)
        """
        # Extract block_id from layer_info for each spec
        for spec in specs:
            layer_name = spec['name']
            layer_data = self.layer_info.get(layer_name, {})
            block_id = layer_data.get('block_id')
            if block_id is not None:
                spec['block_id'] = block_id

        # Group specs by block_id
        blocks = {}
        for idx, spec in enumerate(specs):
            block_id = spec.get('block_id')
            if block_id is not None:
                if block_id not in blocks:
                    blocks[block_id] = []
                blocks[block_id].append((idx, spec))

        # Mark block boundaries
        for block_id, block_specs in blocks.items():
            if len(block_specs) > 0:
                # Mark first layer of block
                first_idx, first_spec = block_specs[0]
                first_spec['block_start'] = True
                first_spec['transformer_block_id'] = block_id

                # Mark last layer of block
                last_idx, last_spec = block_specs[-1]
                last_spec['block_end'] = True
                last_spec['transformer_block_id'] = block_id

                print(f"  Transformer Block {block_id}: {len(block_specs)} layers "
                      f"(start: {first_spec['name']}, end: {last_spec['name']})")

        if blocks:
            print(f"  Total transformer blocks detected: {len(blocks)}")

    def _numel(self, shape):
        prod = 1
        for dim in shape[1:]:
            prod *= dim
        return prod

    def _as_pair(self, value):
        if isinstance(value, (list, tuple)):
            if len(value) == 1:
                return int(value[0]), int(value[0])
            return int(value[0]), int(value[1])
        return int(value), int(value)

    def _compute_output_dim(self, in_size, kernel, stride, padding):
        return math.floor((in_size + 2 * padding - kernel) / stride) + 1

    def _calculate_layer_macs(self, spec):
        """Calculate MACs (multiply-accumulate operations) for a layer.

        MACs are calculated as follows:
        - Conv2D: 2 * K_h * K_w * C_in * C_out * H_out * W_out
        - Linear: 2 * in_features * out_features * batch_tokens
        - Conv1D (depthwise): 2 * kernel_size * channels * seq_len_out
        - MHSA: 2 * (4 * d_model^2 * seq_len + 2 * seq_len^2 * d_model)
        - SSM: 2 * (d_inner * d_state * seq_len) for state update
        - LayerNorm: 5 * num_elements (mean, var, normalize)
        - Pooling/ReLU/etc: 0 (comparisons, not MACs)

        Returns:
            int: Number of MACs for this layer
        """
        op = spec.get('op', '')
        macs = 0

        if op == 'conv2d':
            # Conv2D: 2 * K_h * K_w * C_in * C_out * H_out * W_out
            k_h = spec.get('kernel_h', 1)
            k_w = spec.get('kernel_w', 1)
            c_in = spec.get('in_ch', 1)
            c_out = spec.get('out_ch', 1)
            h_out = spec.get('out_h', 1)
            w_out = spec.get('out_w', 1)
            macs = 2 * k_h * k_w * c_in * c_out * h_out * w_out

        elif op in ('linear_int8', 'linear_fp32', 'linear_ne16'):
            # Linear: 2 * in_features * out_features * batch_tokens
            in_f = spec.get('in_features', 1)
            out_f = spec.get('out_features', 1)
            batch = spec.get('batch_tokens', 1)
            macs = 2 * in_f * out_f * batch

        elif op == 'conv1d_depthwise':
            # Depthwise Conv1D: 2 * kernel_size * channels * seq_len_out
            k = spec.get('kernel_size', 1)
            c = spec.get('channels', spec.get('d_inner', 1))
            seq_out = spec.get('seq_len_out', spec.get('seq_len', 1))
            macs = 2 * k * c * seq_out

        elif op == 'mhsa':
            # MHSA: Q/K/V projections + attention scores + output projection
            # 4 linear projections (Q, K, V, O): 4 * d_model^2 * seq_len
            # Attention scores: seq_len^2 * d_model (QK^T + softmax@V)
            d_model = spec.get('d_model', 1)
            seq_len = spec.get('seq_len', 1)
            # 4 projections + 2 matmuls for attention
            macs = 2 * (4 * d_model * d_model * seq_len + 2 * seq_len * seq_len * d_model)

        elif op == 'cross_attention':
            # Similar to MHSA but with different sequence lengths
            d_model = spec.get('d_model', 1)
            seq_len_q = spec.get('seq_len_q', spec.get('seq_len', 1))
            seq_len_kv = spec.get('seq_len_kv', spec.get('seq_len', 1))
            macs = 2 * (3 * d_model * d_model * seq_len_q + seq_len_q * seq_len_kv * d_model)

        elif op == 'ssm':
            # SSM: state update and output computation
            d_inner = spec.get('d_inner', 1)
            d_state = spec.get('d_state', 1)
            seq_len = spec.get('seq_len', 1)
            # Discretization + state update + output
            macs = 2 * d_inner * d_state * seq_len * 3


        elif op in ('mamba_block', 'mamba_wrapper'):
            # Mamba block includes: in_proj, conv1d, SSM, out_proj
            d_model = spec.get('d_model', 1)
            d_inner = spec.get('d_inner', d_model * 2)
            d_state = spec.get('d_state', 16)
            seq_len = spec.get('seq_len', 1)
            kernel_size = spec.get('kernel_size', 4)
            # in_proj: 2 * d_model * 2*d_inner * seq_len
            # conv1d: 2 * kernel_size * d_inner * seq_len
            # SSM: 2 * d_inner * d_state * seq_len * 3
            # out_proj: 2 * d_inner * d_model * seq_len
            macs = 2 * (d_model * 2 * d_inner * seq_len +  # in_proj
                       kernel_size * d_inner * seq_len +    # conv1d
                       d_inner * d_state * seq_len * 3 +    # SSM
                       d_inner * d_model * seq_len)         # out_proj


        elif op == 'layernorm':
            # LayerNorm: mean, variance, normalize (5 ops per element approx)
            numel = spec.get('num_elements', 1)
            macs = 5 * numel

        elif op == 'gelu':
            # GELU approximation: ~10 ops per element
            numel = spec.get('num_elements', 1)
            macs = 10 * numel

        elif op == 'alternating_attention':
            # Alternating attention (Cerebro-style):
            # QKV projection: 3 * embed_dim * embed_dim * seq_len
            # Attention scores (Q @ K^T): seq_len * seq_len * embed_dim
            # Attention @ V: seq_len * seq_len * embed_dim
            # Output projection: embed_dim * embed_dim * seq_len
            embed_dim = spec.get('embed_dim', spec.get('d_model', 1))
            seq_len = spec.get('seq_len', 1)
            # 4 projections (Q, K, V, O) + 2 attention matmuls
            macs = 2 * (4 * embed_dim * embed_dim * seq_len + 2 * seq_len * seq_len * embed_dim)

        elif op == 'groupnorm':
            # GroupNorm: similar to LayerNorm, ~10 ops per element
            batch = spec.get('batch', 1)
            channels = spec.get('channels', 1)
            spatial = spec.get('spatial_size', 1)
            numel = batch * channels * spatial
            macs = 10 * numel

        elif op == 'softmax':
            # Softmax: exp + sum + div, ~5 ops per element
            numel = spec.get('num_elements', 1)
            macs = 5 * numel

        elif op == 'mean_pool':
            # Mean pooling: sum + divide
            seq_len = spec.get('seq_len', 1)
            features = spec.get('features', spec.get('embed_dim', 1))
            macs = 2 * seq_len * features  # sum over seq_len for each feature

        elif op in ('add', 'element_wise_add', 'residual_add'):
            # Element-wise addition: 1 op per element
            numel = spec.get('size', spec.get('num_elements', 1))
            macs = numel

        elif op == 'patch_embed':
            # Patch embedding is a Conv2D
            k_h = spec.get('patch_h', spec.get('kernel_h', 1))
            k_w = spec.get('patch_w', spec.get('kernel_w', 1))
            c_in = spec.get('in_channels', 1)
            c_out = spec.get('embed_dim', spec.get('out_channels', 1))
            h_out = spec.get('num_patches_h', 1)
            w_out = spec.get('num_patches_w', 1)
            macs = 2 * k_h * k_w * c_in * c_out * h_out * w_out


        return macs

    def _find_next_scale(self, start_idx, default_scale):
        for name in self.layer_order[start_idx + 1:]:
            layer = self.layer_info.get(name, {})
            if 'scale' in layer:
                return layer['scale']
        return default_scale

    def _generate_silu_lut(self, scale_in, scale_out):
        """Generate 256-entry INT8 SiLU lookup table.

        For each INT8 input q_in in [-128, 127], computes:
            x = q_in * scale_in
            silu = x * sigmoid(x)
            q_out = round(silu / scale_out)

        Returns:
            numpy array of shape (256,) with dtype int8
        """
        import numpy as np

        lut = np.zeros(256, dtype=np.int8)
        for q_in in range(-128, 128):
            # Dequantize
            x = q_in * scale_in

            # SiLU = x * sigmoid(x)
            sigmoid_x = 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
            silu = x * sigmoid_x

            # Quantize output
            q_out = int(np.round(silu / scale_out))
            q_out = max(-128, min(127, q_out))

            # Store at index (q_in + 128) so [-128..127] maps to [0..255]
            lut[q_in + 128] = q_out

        return lut

    def _generate_silu_gate_lut_q13(self, scale_z):
        """Generate 256-entry Q13 SiLU lookup table for gating.

        For each INT8 input z in [-128, 127], computes:
            z_float = z * scale_z
            silu = z_float * sigmoid(z_float)
            q13_out = round(silu * 8192)  # Q2.13 fixed point

        This is used for the gating operation:
            gated = (ssm_out * silu_gate_lut[z+128] + 4096) >> 13

        Returns:
            numpy array of shape (256,) with dtype int16
        """
        import numpy as np

        lut = np.zeros(256, dtype=np.int16)
        for z_i8 in range(-128, 128):
            # Dequantize
            z_float = z_i8 * scale_z

            # SiLU = z * sigmoid(z)
            sigmoid_z = 1.0 / (1.0 + np.exp(-np.clip(z_float, -20, 20)))
            silu = z_float * sigmoid_z

            # Convert to Q2.13 (multiply by 2^13 = 8192)
            q13_out = int(np.round(silu * 8192))
            q13_out = max(-32768, min(32767, q13_out))

            # Store at index (z_i8 + 128) so [-128..127] maps to [0..255]
            lut[z_i8 + 128] = q13_out

        return lut

    def _generate_exp_neg_lut_q15(self, scale_in):
        """Generate 256-entry Q15 Exp lookup table for I-Mamba discretization.

        For each INT8 input q_in in [-128, 127], computes:
            x = q_in * scale_in  (negative values expected)
            exp_val = exp(x)
            q15_out = round(exp_val * 32768)  # Q15 fixed point

        The Q15 format provides:
        - 15 fractional bits: precision down to 1/32768 ≈ 0.00003
        - Range [0, 1) maps to [0, 32767]

        For discretization: dA = exp(dt * A) where dt > 0, A < 0
        So the input (dt * A) is negative, and output is in (0, 1).

        Args:
            scale_in: Input scale, maps INT8 [-128,127] to float range
                     e.g., scale_in=0.1 covers input range [-12.8, 12.7]
                     For discretization, we mainly care about negative inputs.

        Returns:
            numpy array of shape (256,) with dtype int16 (Q15)
        """
        import numpy as np

        lut = np.zeros(256, dtype=np.int16)
        for q_in in range(-128, 128):
            # Dequantize input
            x = q_in * scale_in

            # Exp with clipping for numerical stability
            if x > 20.0:
                exp_val = 1.0  # exp(large positive) ≈ overflow, clamp to 1 for Q15
            elif x < -20.0:
                exp_val = 0.0  # exp(very negative) ≈ 0
            else:
                exp_val = np.exp(x)

            # Convert to Q15 (multiply by 2^15 = 32768)
            q15_out = int(np.round(exp_val * 32768))
            # Clamp to INT16 range (max is 32767 for exp(0) ≈ 1)
            q15_out = max(0, min(32767, q15_out))

            # Store at index (q_in + 128) so [-128..127] maps to [0..255]
            lut[q_in + 128] = q15_out

        return lut

    def _generate_softplus_lut_q8_8(self, scale_in):
        """Generate 256-entry Q8.8 Softplus lookup table for I-Mamba.

        For each INT8 input q_in in [-128, 127], computes:
            x = q_in * scale_in
            softplus = log(1 + exp(x))
            q8_8_out = round(softplus * 256)  # Q8.8 fixed point

        The Q8.8 format provides:
        - 8 integer bits: handles softplus output up to ~127
        - 8 fractional bits: precision down to 1/256 ≈ 0.004

        Args:
            scale_in: Input scale, maps INT8 [-128,127] to float range
                     e.g., scale_in=0.1 covers input range [-12.8, 12.7]

        Returns:
            numpy array of shape (256,) with dtype int16 (Q8.8)
        """
        import numpy as np

        lut = np.zeros(256, dtype=np.int16)
        for q_in in range(-128, 128):
            # Dequantize input
            x = q_in * scale_in

            # Softplus = log(1 + exp(x)) with numerical stability
            if x > 20.0:
                softplus = x  # For large x, softplus(x) ≈ x
            elif x < -20.0:
                softplus = np.exp(x)  # For very negative x, softplus(x) ≈ exp(x) ≈ 0
            else:
                softplus = np.log1p(np.exp(x))

            # Convert to Q8.8 (multiply by 2^8 = 256)
            q8_8_out = int(np.round(softplus * 256))
            # Clamp to INT16 range (though softplus is always positive)
            q8_8_out = max(0, min(32767, q8_8_out))

            # Store at index (q_in + 128) so [-128..127] maps to [0..255]
            lut[q_in + 128] = q8_8_out

        return lut

    def _register_mamba_weight(self, layer_name, param_name, num_elements):
        """Register a MAMBA weight parameter for code generation.

        Note: The actual binary file is saved in generate_binaries().
        This method just marks that we expect this weight in the execution plan.
        """
        key = f"{layer_name}::{param_name}" if '::' not in param_name else f"{layer_name}::{param_name}"
        # Check if the weight was already registered in generate_binaries()
        if key not in self.weight_entries and layer_name not in self.weight_entries:
            # Weight not found - this will be handled when weights are loaded
            print(f"  Note: MAMBA weight '{key}' will be loaded at runtime")

    def _register_mamba_param(self, layer_name, param_name, num_elements, dtype='float'):
        """Register a MAMBA FP32 parameter (like A_log, D) for code generation.

        Note: The actual binary file is saved in generate_binaries().
        This method just marks that we expect this parameter in the execution plan.
        """
        key = f"{layer_name}::{param_name}"
        # Check if the param was already registered in generate_binaries()
        if key not in self.weight_entries:
            # Param not found - this will be handled when weights are loaded
            print(f"  Note: MAMBA param '{key}' ({dtype}) will be loaded at runtime")

    def _register_mamba_lut(self, layer_name, lut_name, num_entries):
        """Register a MAMBA LUT (like SiLU LUT) for code generation.

        Note: The actual LUT is generated and saved in generate_binaries().
        This method just marks that we expect this LUT in the execution plan.
        """
        key = f"{layer_name}::{lut_name}"
        # Check if the LUT was already registered in generate_binaries()
        if key not in self.weight_entries:
            # LUT not found - will be generated when binaries are created
            print(f"  Note: MAMBA LUT '{key}' ({num_entries} entries) will be generated at runtime")

    @staticmethod
    def sanitize_c_name(name):
        """
        Sanitize a layer name to be a valid C identifier.

        Replaces dots and other invalid characters with underscores.

        Args:
            name: Layer name (may contain dots from nested modules)

        Returns:
            Valid C identifier
        """
        # Replace dots with underscores
        sanitized = name.replace('.', '_')
        # Replace any other invalid characters
        sanitized = sanitized.replace('-', '_')
        sanitized = sanitized.replace(' ', '_')
        return sanitized

    def _unique_symbol(self, base, table):
        sanitized = self.sanitize_c_name(base)
        count = table.get(sanitized, 0)
        table[sanitized] = count + 1
        if count == 0:
            return sanitized
        return f"{sanitized}_{count}"

    def _unique_layer_c_name(self, name):
        return self._unique_symbol(name, self.layer_symbol_counts)

    def _unique_buffer_c_name(self, name):
        return self._unique_symbol(name, self.buffer_symbol_counts)

    def _unique_binary_symbol(self, name):
        return self._unique_symbol(name, self.binary_symbol_counts)

    def _determine_conv2d_memory_tier(
        self,
        layer_name: str,
        in_h: int,
        in_w: int,
        in_ch: int,
        out_ch: int,
        kernel: int = None,
        stride: int = None,
        padding: int = None,
        kernel_h: int = None,
        kernel_w: int = None,
        stride_h: int = None,
        stride_w: int = None,
        pad_h: int = None,
        pad_w: int = None,
        groups: int = 1
    ) -> tuple:

        if not self.enable_l1_tiling:
            return 'L2_FULL', None

        # Check for config override (auto-tuning) - extract hints to guide tiling
        hint_tile_h = None
        hint_tile_w = None
        override = self._get_layer_override(layer_name)
        if override and 'tile_config' in override:
            tc = override['tile_config']
            # Accept both canonical tile_h/tile_w and older out_tile_* keys.
            hint_tile_h = tc.get('tile_h', tc.get('out_tile_h'))
            hint_tile_w = tc.get('tile_w', tc.get('out_tile_w'))
            hints = []
            if hint_tile_h:
                hints.append(f"tile_h={hint_tile_h}")
            if hint_tile_w:
                hints.append(f"tile_w={hint_tile_w}")
            if hints:
                print(f"  [TUNE] Using hints for {layer_name}: {', '.join(hints)}")

        # Pass L2 budget for tiling decisions (use unified constant)
        l2_budget = self._get_l2_tiling_budget_bytes()

        # Try weight caching first (L1 weights provide ~1.7x speedup)
        # Skip if L1 weight caching is disabled (for baseline benchmarking)
        if not self.disable_l1_weight_caching:
            tile_config = calculate_conv2d_tile_size_with_weights(
                in_h=in_h, in_w=in_w, in_channels=in_ch, out_channels=out_ch,
                kernel_size=kernel, stride=stride, padding=padding,
                kernel_h=kernel_h, kernel_w=kernel_w,
                stride_h=stride_h, stride_w=stride_w,
                pad_h=pad_h, pad_w=pad_w,
                l1_budget=self.l1_budget_bytes,
                l2_budget=l2_budget,
                hint_tile_h=hint_tile_h,
                hint_tile_w=hint_tile_w,
                groups=groups
            )
        else:
            tile_config = None
            print(f"  -> Layer '{layer_name}' L1 weight caching DISABLED (--no-l1-weight-caching)")

        # If weight caching succeeded, use it
        if tile_config is not None and tile_config.weight_tiling_enabled:
            triple_buf_str = " [3-BUF]" if tile_config.triple_buffer_weights else ""
            if tile_config.l3_tiling_enabled:
                print(f"  -> Layer '{layer_name}' uses L3 Tiling + L1 Weight Caching (Slab H={tile_config.l3_tile_h}, {tile_config.num_l3_tiles} slabs, {tile_config.num_out_ch_tiles} weight tiles){triple_buf_str}")
                return 'L3_TILED', tile_config
            print(f"  -> Layer '{layer_name}' uses L1 Weight Caching ({tile_config.num_out_ch_tiles} weight tiles, {tile_config.l1_weight_bytes} bytes/tile){triple_buf_str}")
            self.l1_tiled_layers.append(layer_name)
            return 'L1_TILED', tile_config

        # Fallback to standard tiling (weights in L2)
        tile_config = calculate_conv2d_tile_size(
            in_h=in_h, in_w=in_w, in_channels=in_ch, out_channels=out_ch,
            kernel_size=kernel, stride=stride, padding=padding,
            kernel_h=kernel_h, kernel_w=kernel_w,
            stride_h=stride_h, stride_w=stride_w,
            pad_h=pad_h, pad_w=pad_w,
            l1_budget=self.l1_budget_bytes,
            l2_budget=l2_budget,
            hint_tile_h=hint_tile_h,
            hint_tile_w=hint_tile_w
        )

        if tile_config is None:
            warnings.warn(f"Layer '{layer_name}': Conv2D cannot be tiled. Using L2 fallback.")
            self.l2_fallback_layers.append(layer_name)
            return 'L2_FULL', None

        if tile_config.l3_tiling_enabled:
            print(f"  -> Layer '{layer_name}' uses L3 Tiling (Slab H={tile_config.l3_tile_h}, {tile_config.num_l3_tiles} slabs)")
            return 'L3_TILED', tile_config

        self.l1_tiled_layers.append(layer_name)
        return 'L1_TILED', tile_config

    def _determine_linear_memory_tier(self, layer_name, input_features, output_features, batch_size=1, is_final=False):
        if not self.enable_l1_tiling: return 'L2_FULL', None

        # Check for config override first (auto-tuning)
        override = self._get_layer_override(layer_name)
        if override and 'tile_config' in override:
            tc = override['tile_config']
            # Accept both canonical tile_n and older tile_out_features key.
            tile_out = tc.get('tile_n', tc.get('tile_out_features', output_features))
            # Support tile_k (input feature tiling)
            tile_in = tc.get('tile_k', input_features)
            # Support tile_m (batch/token tiling for 3D linear)
            tile_m = tc.get('tile_m', tc.get('tile_batch_tokens', batch_size))
            # L1 input caching: DMA input to L1 before compute
            l1_input_cache = tc.get('l1_input_cache', False)

            # Determine if K-tiling is enabled.
            # Default: enable when tile_k < full K unless explicitly overridden by the tuner.
            k_tiling_enabled = bool(tc.get('k_tiling_enabled', tile_in < input_features))

            # Determine if M-tiling is enabled.
            # Default: enable when tile_m < full M unless explicitly overridden by the tuner.
            m_tiling_enabled = bool(tc.get('m_tiling_enabled', tile_m < batch_size))

            print(f"  [TUNE] Using override for {layer_name}: tile_out={tile_out}, tile_in={tile_in}, tile_m={tile_m}, l1_input_cache={l1_input_cache}")

            # Create tile config from override
            tile_config = LinearTileConfig()
            tile_config.tile_out_features = tile_out
            tile_config.input_features = input_features
            tile_config.num_tiles = (output_features + tile_out - 1) // tile_out

            # K-dimension tiling
            tile_config.tile_in_features = tile_in
            tile_config.num_k_tiles = (input_features + tile_in - 1) // tile_in
            tile_config.k_tiling_enabled = k_tiling_enabled

            # M-dimension tiling (batch/token tiling)
            tile_config.batch_tokens = batch_size
            tile_config.tile_batch_tokens = tile_m
            tile_config.num_m_tiles = (batch_size + tile_m - 1) // tile_m
            tile_config.m_tiling_enabled = m_tiling_enabled

            # L1 buffer size calculation
            if k_tiling_enabled:
                # With K-tiling: need accumulator buffer (INT32) + smaller weight tiles
                tile_config.l1_input_bytes = batch_size * tile_in  # Input slice
                tile_config.l1_output_bytes = batch_size * tile_out * (4 if is_final else 1)
                tile_config.l1_weight_bytes = tile_in * tile_out  # Smaller weight tile
            else:
                # Without K-tiling: normal calculation
                tile_config.l1_input_bytes = batch_size * input_features
                tile_config.l1_output_bytes = batch_size * tile_out * (4 if is_final else 1)
                tile_config.l1_weight_bytes = input_features * tile_out

            tile_config.l1_input_cache = l1_input_cache
            tile_config.l3_tiling_enabled = False

            self.l1_tiled_layers.append(layer_name)
            return 'L1_TILED', tile_config

        # Pass L2 budget for tiling decisions (use unified constant)
        l2_budget = self._get_l2_tiling_budget_bytes()

        output_element_size = 4 if is_final else 1

        tile_config = calculate_linear_tile_size(
            input_features=input_features,
            output_features=output_features,
            batch_size=batch_size,
            l1_budget=self.l1_budget_bytes,
            l2_budget=l2_budget,
            output_element_size=output_element_size
        )

        if tile_config is None:
            warnings.warn(f"Layer '{layer_name}': Linear cannot be tiled. Using L2 fallback.")
            self.l2_fallback_layers.append(layer_name)
            return 'L2_FULL', None

        if tile_config.l3_tiling_enabled:
            print(f"  -> Layer '{layer_name}' uses L3 Tiling (Slab Out={tile_config.l3_tile_out_features}, {tile_config.num_l3_tiles} slabs)")
            return 'L3_TILED', tile_config

        self.l1_tiled_layers.append(layer_name)
        return 'L1_TILED', tile_config

    def _determine_maxpool_memory_tier(
        self,
        layer_name: str,
        in_h: int,
        in_w: int,
        channels: int,
        kernel_size: int,
        stride: int,
        padding: int
    ) -> tuple:
        """
        Determine if MaxPool should use L1 tiling, L3 tiling, or L2 full-tensor execution.

        Args:
            layer_name: Name of the MaxPool layer
            in_h: Input height
            in_w: Input width
            channels: Number of channels
            kernel_size: Pooling kernel size
            stride: Pooling stride
            padding: Padding size

        Returns:
            (memory_tier, tile_config) tuple where:
            - memory_tier: 'L1_TILED', 'L3_TILED', or 'L2_FULL'
            - tile_config: MaxPoolTileConfig object if tiled, None otherwise
        """
        # If L1 tiling is disabled, use L2 for all layers
        if not self.enable_l1_tiling:
            return 'L2_FULL', None

        # Attempt to calculate tile configuration (now includes L3 tiling)
        tile_config = calculate_maxpool_tile_size(
            in_h=in_h,
            in_w=in_w,
            channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            l1_budget=self.l1_budget_bytes,
            l2_budget=self._get_l2_tiling_budget_bytes()
        )

        # If tiling is not feasible, fall back to L2
        if tile_config is None:
            warnings.warn(
                f"Layer '{layer_name}': MaxPool cannot be tiled "
                f"(in:{in_h}x{in_w}x{channels}, k:{kernel_size}). "
                f"Using L2 fallback.",
                RuntimeWarning
            )
            self.l2_fallback_layers.append(layer_name)
            return 'L2_FULL', None

        # Check if L3 tiling is enabled
        if tile_config.l3_tiling_enabled:
            print(f"  -> Layer '{layer_name}' uses L3 Tiling (Slab H={tile_config.l3_tile_h}, {tile_config.num_l3_tiles} slabs)")
            return 'L3_TILED', tile_config

        # L1 tiling is feasible
        self.l1_tiled_layers.append(layer_name)
        return 'L1_TILED', tile_config

    def _determine_avgpool_memory_tier(
        self,
        layer_name: str,
        in_h: int,
        in_w: int,
        channels: int,
        kernel_size: int,
        stride: int,
        padding: int
    ) -> tuple:
        """
        Determine if AvgPool should use L1 tiling, L3 tiling, or L2 full-tensor execution.

        Args:
            layer_name: Name of the AvgPool layer
            in_h: Input height
            in_w: Input width
            channels: Number of channels
            kernel_size: Pooling kernel size
            stride: Pooling stride
            padding: Padding size

        Returns:
            (memory_tier, tile_config) tuple where:
            - memory_tier: 'L1_TILED', 'L3_TILED', or 'L2_FULL'
            - tile_config: AvgPoolTileConfig object if tiled, None otherwise
        """
        # If L1 tiling is disabled, use L2 for all layers
        if not self.enable_l1_tiling:
            return 'L2_FULL', None

        # Attempt to calculate tile configuration (now includes L3 tiling)
        tile_config = calculate_avgpool_tile_size(
            in_h=in_h,
            in_w=in_w,
            channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            l1_budget=self.l1_budget_bytes,
            l2_budget=self._get_l2_tiling_budget_bytes()
        )

        # If tiling is not feasible, fall back to L2
        if tile_config is None:
            warnings.warn(
                f"Layer '{layer_name}': AvgPool cannot be tiled "
                f"(in:{in_h}x{in_w}x{channels}, k:{kernel_size}). "
                f"Using L2 fallback.",
                RuntimeWarning
            )
            self.l2_fallback_layers.append(layer_name)
            return 'L2_FULL', None

        # Check if L3 tiling is enabled
        if tile_config.l3_tiling_enabled:
            print(f"  -> Layer '{layer_name}' uses L3 Tiling (Slab H={tile_config.l3_tile_h}, {tile_config.num_l3_tiles} slabs)")
            return 'L3_TILED', tile_config

        # L1 tiling is feasible
        self.l1_tiled_layers.append(layer_name)
        return 'L1_TILED', tile_config

    def _determine_globalavgpool_memory_tier(
        self,
        layer_name: str,
        in_h: int,
        in_w: int,
        channels: int
    ) -> tuple:
        """
        Determine if GlobalAvgPool should use L1 tiling or L2 full-tensor execution.

        Args:
            layer_name: Name of the GlobalAvgPool layer
            in_h: Input height
            in_w: Input width
            channels: Number of channels

        Returns:
            (memory_tier, tile_config) tuple where:
            - memory_tier: 'L1_TILED' or 'L2_FULL'
            - tile_config: GlobalAvgPoolTileConfig object if L1_TILED, None otherwise
        """
        # If L1 tiling is disabled, use L2 for all layers
        if not self.enable_l1_tiling:
            return 'L2_FULL', None

        # Attempt to calculate L1 tile configuration
        tile_config = calculate_globalavgpool_tile_size(
            in_h=in_h,
            in_w=in_w,
            channels=channels,
            l1_budget=self.l1_budget_bytes
        )

        # If tiling is not feasible, fall back to L2
        if tile_config is None:
            warnings.warn(
                f"Layer '{layer_name}': GlobalAvgPool cannot be tiled for L1 "
                f"(in:{in_h}x{in_w}x{channels}). "
                f"Using L2 fallback.",
                RuntimeWarning
            )
            self.l2_fallback_layers.append(layer_name)
            return 'L2_FULL', None

        # L1 tiling is feasible
        self.l1_tiled_layers.append(layer_name)
        return 'L1_TILED', tile_config

    def _determine_elementwise_memory_tier(
        self,
        layer_name: str,
        num_elements: int,
        in_place: bool = True
    ) -> tuple:
        """
        Determine if element-wise operation should use L1 tiling or L2 direct execution.

        Element-wise operations (ReLU, GELU, Requantize) are memory-bound and benefit
        significantly from L1 tiling to hide memory latency.

        Args:
            layer_name: Name of the layer
            num_elements: Total number of elements to process
            in_place: If True, input/output share same buffer

        Returns:
            (memory_tier, tile_config) tuple where:
            - memory_tier: 'L1_TILED' or 'L2_FULL'
            - tile_config: ElementwiseTileConfig object if L1_TILED, None otherwise
        """
        # If L1 tiling is disabled, use L2 for all layers
        if not self.enable_l1_tiling:
            return 'L2_FULL', None

        # Check for config override (auto-tuning) - extract hint for tile_size
        hint_tile_size = None
        override = self._get_layer_override(layer_name)
        if override and 'tile_config' in override:
            tc = override['tile_config']
            hint_tile_size = tc.get('tile_size', tc.get('tile_elements'))
            if hint_tile_size:
                print(f"  [TUNE] Using hint for {layer_name}: tile_size={hint_tile_size}")

        # Attempt to calculate L1 tile configuration
        tile_config = calculate_elementwise_tile_size(
            num_elements=num_elements,
            l1_budget=self.l1_budget_bytes,
            in_place=in_place,
            hint_tile_size=hint_tile_size
        )

        # Element-wise ops should always be tileable (even 1 element fits)
        if tile_config is None:
            warnings.warn(
                f"Layer '{layer_name}': Element-wise op cannot be tiled for L1 "
                f"(elements:{num_elements}). Using L2 fallback.",
                RuntimeWarning
            )
            self.l2_fallback_layers.append(layer_name)
            return 'L2_FULL', None

        # L1 tiling is feasible
        self.l1_tiled_layers.append(layer_name)
        return 'L1_TILED', tile_config

    def _determine_add_memory_tier(
        self,
        layer_name: str,
        num_elements: int
    ) -> tuple:
        """
        Determine if Add operation should use L1 tiling or L2 direct execution.

        Add operation is memory-bound (reads two inputs, writes one output) and
        benefits significantly from L1 tiling.

        Args:
            layer_name: Name of the Add layer
            num_elements: Total number of elements to process

        Returns:
            (memory_tier, tile_config) tuple where:
            - memory_tier: 'L1_TILED' or 'L2_FULL'
            - tile_config: AddTileConfig object if L1_TILED, None otherwise
        """
        # If L1 tiling is disabled, use L2 for all layers
        if not self.enable_l1_tiling:
            return 'L2_FULL', None

        # Attempt to calculate L1 tile configuration
        tile_config = calculate_add_tile_size(
            num_elements=num_elements,
            l1_budget=self.l1_budget_bytes
        )

        if tile_config is None:
            warnings.warn(
                f"Layer '{layer_name}': Add cannot be tiled for L1 "
                f"(elements:{num_elements}). Using L2 fallback.",
                RuntimeWarning
            )
            self.l2_fallback_layers.append(layer_name)
            return 'L2_FULL', None

        # L1 tiling is feasible
        self.l1_tiled_layers.append(layer_name)
        return 'L1_TILED', tile_config

    def _determine_layernorm_memory_tier(
        self,
        layer_name: str,
        num_tokens: int,
        normalized_dim: int
    ) -> tuple:
        """
        Determine if LayerNorm operation should use L1 tiling or L2 direct execution.

        LayerNorm normalizes each token independently, making it naturally tileable
        by token batches. The weights (gamma, beta) are loaded once and shared.

        Args:
            layer_name: Name of the LayerNorm layer
            num_tokens: Number of tokens to normalize
            normalized_dim: Dimension of each token (embed_dim)

        Returns:
            (memory_tier, tile_config) tuple where:
            - memory_tier: 'L1_TILED' or 'L2_FULL'
            - tile_config: LayerNormTileConfig object if L1_TILED, None otherwise
        """
        # If L1 tiling is disabled, use L2 for all layers
        if not self.enable_l1_tiling:
            return 'L2_FULL', None

        # Check for config override (auto-tuning) - extract hint for tile_tokens
        hint_tile_tokens = None
        override = self._get_layer_override(layer_name)
        if override and 'tile_config' in override:
            tc = override['tile_config']
            hint_tile_tokens = tc.get('tile_tokens', tc.get('tile_m'))
            if hint_tile_tokens:
                print(f"  [TUNE] Using hint for {layer_name}: tile_tokens={hint_tile_tokens}")

        # Attempt to calculate L1 tile configuration
        tile_config = calculate_layernorm_tile_size(
            num_tokens=num_tokens,
            normalized_dim=normalized_dim,
            l1_budget=self.l1_budget_bytes,
            hint_tile_tokens=hint_tile_tokens
        )

        if tile_config is None:
            warnings.warn(
                f"Layer '{layer_name}': LayerNorm cannot be tiled for L1 "
                f"(tokens:{num_tokens}, dim:{normalized_dim}). Using L2 fallback.",
                RuntimeWarning
            )
            self.l2_fallback_layers.append(layer_name)
            return 'L2_FULL', None

        # L1 tiling is feasible
        self.l1_tiled_layers.append(layer_name)
        return 'L1_TILED', tile_config

    def _determine_concat_memory_tier(
        self,
        layer_name: str,
        num_inputs: int,
        total_channels: int,
        spatial_size: int
    ) -> tuple:
        """
        Determine if Concat operation should use L1 tiling or L2 direct execution.

        Concat copies multiple input tensors along channel dimension.
        We tile spatially, processing all channels for each spatial chunk.

        Args:
            layer_name: Name of the Concat layer
            num_inputs: Number of input tensors
            total_channels: Sum of all input channels
            spatial_size: H x W (spatial elements)

        Returns:
            (memory_tier, tile_config) tuple where:
            - memory_tier: 'L1_TILED' or 'L2_FULL'
            - tile_config: ConcatTileConfig object if L1_TILED, None otherwise
        """
        # If L1 tiling is disabled, use L2 for all layers
        if not self.enable_l1_tiling:
            return 'L2_FULL', None

        # Attempt to calculate L1 tile configuration
        tile_config = calculate_concat_tile_size(
            num_inputs=num_inputs,
            total_channels=total_channels,
            spatial_size=spatial_size,
            l1_budget=self.l1_budget_bytes
        )

        if tile_config is None:
            warnings.warn(
                f"Layer '{layer_name}': Concat cannot be tiled for L1 "
                f"(channels:{total_channels}, spatial:{spatial_size}). Using L2 fallback.",
                RuntimeWarning
            )
            self.l2_fallback_layers.append(layer_name)
            return 'L2_FULL', None

        # L1 tiling is feasible
        self.l1_tiled_layers.append(layer_name)
        return 'L1_TILED', tile_config

    def _determine_transpose2d_memory_tier(self, layer_name: str, dim1: int, dim2: int) -> tuple:
        """
        Determine memory tier for Transpose_2d operation.

        Transpose_2d ([B, D1, D2] -> [B, D2, D1]) tiles along D2 dimension.
        Each tile processes a contiguous chunk of D2 columns.

        Args:
            layer_name: Name of the layer
            dim1: First dimension (becomes dim2 in output)
            dim2: Second dimension (becomes dim1 in output)

        Returns:
            tuple: (memory_tier, tile_config or None)
        """
        if not self.enable_l1_tiling:
            return 'L2_FULL', None

        tile_config = calculate_transpose2d_tile_size(
            dim1=dim1,
            dim2=dim2,
            l1_budget=self.l1_budget_bytes
        )

        if tile_config is None:
            warnings.warn(
                f"Layer '{layer_name}': Transpose_2d cannot be tiled for L1 "
                f"(dim1:{dim1}, dim2:{dim2}). Using L2 fallback.",
                RuntimeWarning
            )
            self.l2_fallback_layers.append(layer_name)
            return 'L2_FULL', None

        # L1 tiling is feasible
        self.l1_tiled_layers.append(layer_name)
        return 'L1_TILED', tile_config

    # Cross-layer fusion

    def _apply_layer_fusion(self, specs):
        """
        Detect and apply cross-layer fusion optimizations.

        Current behavior:
        - Fusion runs exclusively through codegen/optimization/fusion/*
        - Legacy in-generator fallback path has been removed
        """
        if not FUSION_MODULE_AVAILABLE:
            raise RuntimeError(
                "Fusion module is unavailable. Legacy fallback has been removed."
            )

        try:
            registry = build_fusion_registry_v2()
            registry_errors = validate_fusion_registry_v2(registry)
            if registry_errors:
                raise RuntimeError(
                    f"Fusion registry validation failed: {registry_errors}"
                )

            fusions = detect_fusion_opportunities_v2(specs, registry=registry)
            fusion_errors = validate_fusions_v2(fusions)
            if fusion_errors:
                raise RuntimeError(
                    f"Fusion payload validation failed: {fusion_errors}"
                )
        except Exception as exc:
            raise RuntimeError(f"Fusion module execution failed: {exc}") from exc

        self.fusion_report_path = write_fusion_report_v2(
            str(self.output_dir),
            fusions,
            test_name=self.test_case_dir.name if self.test_case_dir else None,
            metadata={
                "generator": "CCodeGenerator",
                "backend": "fusion_module_v2",
                "layer_count_before_fusion": len(specs),
            },
        )

        if not fusions:
            print("  No fusion opportunities detected")
            if self.fusion_report_path:
                print(f"  [Fusion] Report: {self.fusion_report_path}")
            return specs

        print(f"\n  Detected {len(fusions)} fusion opportunities:")
        for fusion in fusions:
            print(f"    - {fusion['type']}: layers {fusion['layers']}")

        specs, tracked_fusions = transform_specs_for_fusion_v2(
            specs,
            fusions,
            fused_layers=self.fused_layers,
        )
        self.fused_layers = tracked_fusions

        if self.fusion_report_path:
            print(f"  [Fusion] Report: {self.fusion_report_path}")

        return specs


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate C code from extracted INT8 artifacts.")
    parser.add_argument(
        "--target",
        type=str,
        default="gap9",
        choices=available_targets(),
        help="Compilation target (default: gap9).",
    )
    args = parser.parse_args()

    generator = CCodeGenerator(target_name=args.target)
    generator.generate_all()


if __name__ == "__main__":
    main()
