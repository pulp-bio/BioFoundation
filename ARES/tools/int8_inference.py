# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
INT8 Inference Engine

Performs complete INT8 inference using atomic operations.
Chains together quantization, conv2d, relu, maxpool, linear, etc.
to run TRUE INT8 neural network inference.

This is the reference implementation that demonstrates:
1. How to chain atomic operations
2. How to manage quantization scales between layers
3. How to generate golden INT8 outputs for MCU verification
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add atomic_ops to path
sys.path.append(str(Path(__file__).parent.parent))

from atomic_ops import (
    quantize_linear,
    dequantize_linear,
    embedding_int8,
    groupnorm_int8_fixed_point,
    rfft40_features_int8_fixed_point,
    conv2d_int8,
    relu_int8,
    maxpool2d_int8,
    avgpool2d_int8,
    global_avgpool_int8,
    add_int8,
    concat_int8,
    flatten_int8,
    requantize_int8,
    mhsa_int8_hybrid,
    cross_attention_int8_hybrid,
    cross_attention_with_self_refine_int8,
    classification_head_with_mlp_int8,
    layernorm_int8,
    layernorm_int8_fixed_point,
    gelu_int8,
    gelu_int8_lut,
    get_builtin_gelu_lut,
    rmsnorm_int8,
    rmsnorm_int8_fixed_point,
    swiglu_ffn_int8,
)

# Alternating Attention (Cerebro transformer)
try:
    from atomic_ops.alternating_attention import alternating_attention_int8
except ImportError:
    alternating_attention_int8 = None

# MAMBA-specific atomic operations
from atomic_ops.conv1d_depthwise import conv1d_depthwise_int8, conv1d_depthwise_int8_fixedpoint
from atomic_ops.silu import (
    generate_silu_lut_int8,
    silu_lut_int8,
)
from atomic_ops.ssm import (
    ssm_layer_forward_int8_imamba,
    generate_silu_gate_lut_q13,
)


class INT8InferenceEngine:
    """
    INT8 inference engine for SimpleCNN.

    Uses atomic operations to perform TRUE INT8 inference with:
    - INT8 weights and activations
    - INT32 accumulation for Conv2D and Linear
    - Proper rescaling between layers
    """

    def __init__(self, network_info: Dict[str, Any], use_i_softmax: bool = False, softmax_lut_path: str = None,
                 use_i_gelu: bool = False, use_i_layernorm: bool = False):
        """
        Initialize inference engine with network information.

        Args:
            network_info: Dictionary from BrevitasExtractor with layer info
            use_i_softmax: Use integer-only LUT-based softmax for MHSA layers
                           (for bit-exact matching with C implementation)
            softmax_lut_path: Path to softmax LUT binary file (optional, uses builtin LUT if not provided)
            use_i_gelu: Use integer-only LUT-based GELU for transformer MLP layers
                        (for bit-exact matching with C implementation)
            use_i_layernorm: Use integer-only LayerNorm with binary search sqrt
                             (for bit-exact matching with C implementation)
        """
        self.network_info = network_info
        self.layer_info = {k: v for k, v in network_info.items() if not k.startswith('__')}
        self.layer_order = self._determine_layer_order()
        self.intermediate_outputs = {}  # Store INT8 outputs and scales for each layer
        self.output_scales = {}  # Store output scale for each layer
        self.intermediate_shapes = {}
        self.activation_cache = {}  # Cache activations for skip connections (ResNet)

        # i-Softmax support for bit-exact transformer inference
        self.use_i_softmax = use_i_softmax
        self.softmax_lut = None
        self.softmax_lut_metadata = None

        if use_i_softmax:
            if softmax_lut_path is not None:
                from atomic_ops.mhsa import load_softmax_lut
                self.softmax_lut, self.softmax_lut_metadata = load_softmax_lut(softmax_lut_path)
            else:
                # Use builtin LUT for bit-exact matching with C code
                from atomic_ops.mhsa import get_builtin_softmax_lut
                self.softmax_lut, self.softmax_lut_metadata = get_builtin_softmax_lut()

        # i-GELU support for bit-exact transformer inference
        self.use_i_gelu = use_i_gelu
        self.gelu_lut = None
        self.gelu_lut_metadata = None

        if use_i_gelu:
            self.gelu_lut, self.gelu_lut_metadata = get_builtin_gelu_lut()

        # i-LayerNorm support for bit-exact transformer inference
        self.use_i_layernorm = use_i_layernorm

    def _determine_layer_order(self) -> List[str]:
        """
        Determine the order of layers for forward pass.

        For SimpleCNN, the order is fixed:
        input_quant → conv1 → relu1 → pool1 → pool1_quant →
        conv2 → relu2 → pool2 → pool2_quant →
        flatten → pre_linear_quant → classifier

        Returns:
            List of layer names in execution order
        """
        explicit_order = self.network_info.get('__layer_order__')
        if explicit_order:
            return [name for name in explicit_order if name in self.layer_info]
        return list(self.layer_info.keys())

    def _detect_input_branches(self, inputs: list) -> dict:
        """
        Detect which input corresponds to which branch based on layer naming.

        For dual-input models like the drowsiness network:
        - input 0 (EEG) feeds layers starting with 'eeg_'
        - input 1 (PPG) feeds layers starting with 'ppg_'

        Args:
            inputs: List of input FP32 arrays

        Returns:
            Dictionary mapping branch prefix to its input array
        """
        # Find unique branch prefixes from layer names
        prefixes = set()
        for layer_name in self.layer_info.keys():
            if '_' in layer_name:
                prefix = layer_name.split('_')[0] + '_'
                # Only consider common multi-input prefixes
                if prefix in ('eeg_', 'ppg_', 'input0_', 'input1_', 'branch0_', 'branch1_', 'cnn_', 'rfft_'):
                    prefixes.add(prefix)

        # Map inputs to branches by order (first input to first branch alphabetically)
        sorted_prefixes = sorted(prefixes)
        branch_inputs = {}
        for i, prefix in enumerate(sorted_prefixes):
            if i < len(inputs):
                branch_inputs[prefix] = inputs[i]

        # If no prefixes found, default to single branch
        if not branch_inputs and len(inputs) > 0:
            branch_inputs[''] = inputs[0]

        return branch_inputs

    # Known branch prefixes for multi-input and internal-branching models
    BRANCH_PREFIXES = frozenset({
        'eeg_', 'ppg_', 'input0_', 'input1_', 'branch0_', 'branch1_', 'cnn_', 'rfft_',
    })

    def _get_branch_prefix(self, layer_name: str) -> str:
        """Get the branch prefix for a layer name."""
        if '_' in layer_name:
            prefix = layer_name.split('_')[0] + '_'
            if prefix in self.BRANCH_PREFIXES:
                return prefix
        return ''

    def _detect_internal_branches(self) -> set:
        """
        Detect internal branching in single-input models from layer name prefixes.

        Returns set of branch prefixes if >=2 distinct branches found, else empty set.
        For example, LUNA has both 'cnn_' and 'rfft_' prefixed layers that diverge
        from a shared input and merge at an Add layer.
        """
        prefixes = set()
        for layer_name in self.layer_info.keys():
            if '_' in layer_name:
                prefix = layer_name.split('_')[0] + '_'
                if prefix in self.BRANCH_PREFIXES:
                    prefixes.add(prefix)
        return prefixes if len(prefixes) >= 2 else set()

    def forward(self, x_fp32, verbose: bool = True) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Run INT8 forward pass through the network.

        Args:
            x_fp32: Input FP32 tensor [B, C, H, W] or list of tensors for multi-input models
            verbose: Print layer-by-layer progress

        Returns:
            Tuple of:
            - Output FP32 logits [B, num_classes]
            - Dictionary of intermediate INT8 outputs for each layer
        """
        if verbose:
            print("="*80)
            print("INT8 Inference Forward Pass")
            print("="*80)

        # Handle multi-input models
        is_multi_input = isinstance(x_fp32, list)
        if is_multi_input:
            if verbose:
                print(f"Multi-input model with {len(x_fp32)} inputs")
                for i, inp in enumerate(x_fp32):
                    print(f"  Input {i} shape: {inp.shape}")
            # For multi-input, we track separate branch states
            # Detect branches by naming convention (e.g., eeg_*, ppg_*)
            branch_states = {}  # {branch_prefix: (current_fp32, current_int8, current_scale)}
            branch_inputs = self._detect_input_branches(x_fp32)
            for prefix, inp in branch_inputs.items():
                branch_states[prefix] = (inp, None, 1.0)  # (fp32, int8, scale)
            current_fp32 = None
            current_int8 = None
            current_scale = 1.0
        else:
            if verbose:
                print(f"Input shape: {x_fp32.shape}")
            # Start with FP32 input
            current_fp32 = x_fp32
            current_int8 = None
            current_scale = 1.0
            branch_states = None

        # Detect internal branching for single-input models (e.g., LUNA CNN+RFFT)
        internal_branch_prefixes = set()
        fork_state = None       # Saved state at the point where branches diverge
        current_branch = None   # Which branch prefix we're currently processing
        if not is_multi_input:
            internal_branch_prefixes = self._detect_internal_branches()
            if internal_branch_prefixes:
                branch_states = {}  # Will accumulate per-branch final states
                if verbose:
                    print(f"Internal branching detected: {sorted(internal_branch_prefixes)}")

        if verbose:
            print()

        for layer_name in self.layer_order:
            if layer_name not in self.layer_info:
                print(f"Warning: Layer {layer_name} not found in network_info")
                continue

            layer_info = self.layer_info[layer_name]
            layer_type = layer_info['type']

            # Multi-input branch handling: switch to appropriate branch state
            if is_multi_input and branch_states:
                branch_prefix = self._get_branch_prefix(layer_name)
                if branch_prefix and branch_prefix in branch_states:
                    # Load state from this branch
                    current_fp32, current_int8, current_scale = branch_states[branch_prefix]
                elif layer_type == 'Add':
                    # Add layer may merge branches - handled specially below
                    pass
                elif layer_type == 'Concatenate':
                    # Concat layer merges branches - handled specially below
                    pass
                elif not branch_prefix:
                    # After concat (no prefix), use the merged state
                    pass

            # Internal branching (single-input): fork, switch, and merge
            if internal_branch_prefixes and branch_states is not None:
                branch_prefix = self._get_branch_prefix(layer_name)
                if branch_prefix and branch_prefix in internal_branch_prefixes:
                    if current_branch is None:
                        # Entering first branch — save fork state
                        fork_state = (
                            current_fp32.copy() if current_fp32 is not None else None,
                            current_int8.copy() if current_int8 is not None else None,
                            current_scale,
                        )
                        current_branch = branch_prefix
                        if verbose:
                            print(f"  [Branch] Entering '{branch_prefix}' (fork saved)")
                    elif branch_prefix != current_branch:
                        # Switching branches — save current, restore fork
                        branch_states[current_branch] = (current_fp32, current_int8, current_scale)
                        current_fp32, current_int8, current_scale = (
                            fork_state[0].copy() if fork_state[0] is not None else None,
                            fork_state[1].copy() if fork_state[1] is not None else None,
                            fork_state[2],
                        )
                        current_branch = branch_prefix
                        if verbose:
                            print(f"  [Branch] Switching to '{branch_prefix}' (restored fork)")
                    # else: continuing same branch — nothing to do

            # Track current feature shape for sanity checks
            if current_int8 is not None:
                self.intermediate_shapes[layer_name] = current_int8.shape

            # Cache activation for skip connections (ResNet)
            # When entering a residual block (first conv1), save input for skip path
            if '.conv1' in layer_name and current_int8 is not None:
                block_name = layer_name.rsplit('.conv1', 1)[0]  # e.g., "layer2.0"
                self.activation_cache[block_name + '.input'] = (current_int8.copy(), current_scale)

            if verbose:
                print(f"Layer: {layer_name} ({layer_type})")

            # Process layer based on type
            if layer_type == 'QuantIdentity':
                # Requantize INT8 tensor to a new scale
                # Use direct INT8→INT8 requantization to match C code behavior
                scale_out = layer_info.get('scale', 1.0)
                if current_int8 is not None:
                    # Requantize existing INT8 tensor to new scale
                    current_int8 = requantize_int8(current_int8, scale_in=current_scale, scale_out=scale_out)
                else:
                    # First quantization: FP32 → INT8
                    current_int8 = quantize_linear(current_fp32, scale=scale_out)
                current_fp32 = dequantize_linear(current_int8, scale=scale_out, zero_point=0)
                current_scale = scale_out

                if verbose:
                    print(f"  Input:  INT8 {current_int8.shape}, scale_in={current_scale:.6f}")
                    print(f"  Output: INT8 {current_int8.shape}, scale_out={scale_out:.6f}")
                    print(f"  Range:  [{current_int8.min()}, {current_int8.max()}]")

            elif layer_type == 'QuantConv2d':
                # INT8 Conv2D with rescaling

                # Special handling for shortcut projections in ResNet
                input_tensor = current_int8
                input_scale = current_scale

                if '.shortcut' in layer_name:
                    # Shortcut layer: use cached input to the residual block
                    block_name = layer_name.rsplit('.shortcut', 1)[0]  # e.g., "layer2.0"
                    cache_key = block_name + '.input'
                    if cache_key in self.activation_cache:
                        input_tensor, input_scale = self.activation_cache[cache_key]
                        if verbose:
                            print(f"  [Shortcut] Using cached input from '{cache_key}'")
                            print(f"  Cached shape: {input_tensor.shape}, scale: {input_scale:.6f}")
                    else:
                        print(f"  WARNING: Shortcut cache miss for '{cache_key}'")

                scale_x = input_scale
                scale_w = layer_info['scale_weight']

                # Output scale: look ahead to find next quantization layer
                # Skip through non-quantizing layers (Flatten, Permute, etc.) to find
                # the next QuantIdentity or QuantReLU that defines the output scale
                next_layer_idx = self.layer_order.index(layer_name) + 1
                scale_y = 1.0  # Default if no quantization layer found
                for idx in range(next_layer_idx, len(self.layer_order)):
                    next_layer_name = self.layer_order[idx]
                    if next_layer_name in self.layer_info:
                        next_layer = self.layer_info[next_layer_name]
                        next_type = next_layer.get('type')
                        # Found a quantization layer that defines output scale
                        if next_type in ('QuantIdentity', 'QuantReLU'):
                            scale_y = next_layer.get('scale', 1.0)
                            break

                w_int8 = layer_info['weight_int8']
                bias_fp32 = layer_info.get('bias_fp32')

                # Convert to numpy arrays if needed (JSON serialization converts arrays to lists)
                if not isinstance(w_int8, np.ndarray):
                    w_int8 = np.array(w_int8)

                # Convert bias to INT32 with proper scale
                bias_int32 = None
                if bias_fp32 is not None:
                    if not isinstance(bias_fp32, np.ndarray):
                        bias_fp32 = np.array(bias_fp32)
                    scale_bias = scale_x * scale_w
                    bias_int32 = np.round(bias_fp32 / scale_bias).astype(np.int32)

                # Run INT8 Conv2D
                try:
                    current_int8 = conv2d_int8(
                        input_tensor,
                        w_int8,
                        bias_int32,
                        scale_x=scale_x,
                        scale_w=scale_w,
                        scale_y=scale_y,
                        stride=layer_info['stride'],
                        padding=layer_info['padding'],
                        groups=layer_info.get('groups', 1),
                    )
                except IndexError as e:
                    print(f"\n[FAIL] ERROR in layer '{layer_name}':")
                    print(f"  Input shape: {input_tensor.shape}")
                    print(f"  Weight shape: {np.array(w_int8).shape}")
                    print(f"  Expected in_channels from weight: {np.array(w_int8).shape[1]}")
                    print(f"  Actual in_channels from input: {input_tensor.shape[1]}")
                    print(f"  Layer info: in_channels={layer_info['in_channels']}, out_channels={layer_info['out_channels']}")
                    raise
                current_scale = scale_y
                current_fp32 = dequantize_linear(current_int8, scale=current_scale, zero_point=0)

                if verbose:
                    print(f"  Kernel: {layer_info['kernel_size']}, "
                          f"stride={layer_info['stride']}, padding={layer_info['padding']}")
                    print(f"  Scales: x={scale_x:.6f}, w={scale_w:.6f}, y={scale_y:.6f}")
                    print(f"  Output: INT8 {current_int8.shape}, range=[{current_int8.min()}, {current_int8.max()}]")

            elif layer_type == 'QuantReLU':
                # INT8 ReLU (preserves quantization)
                current_int8 = relu_int8(current_int8, zero_point=0)

                # Requantize if output scale differs from input scale
                if 'scale' in layer_info:
                    new_scale = layer_info['scale']
                    if new_scale != current_scale:
                        # ReLU is changing scale - need to requantize INT8 values!
                        current_fp32 = dequantize_linear(current_int8, scale=current_scale, zero_point=0)
                        current_int8 = quantize_linear(current_fp32, scale=new_scale, zero_point=0)
                    current_scale = new_scale
                current_fp32 = dequantize_linear(current_int8, scale=current_scale, zero_point=0)

                if verbose:
                    print(f"  Output: INT8 {current_int8.shape}, range=[{current_int8.min()}, {current_int8.max()}]")

            elif layer_type == 'MaxPool2d':
                # INT8 MaxPool (order-preserving)
                kernel_size = layer_info['kernel_size']
                stride = layer_info['stride']
                padding = layer_info.get('padding', (0, 0))

                # For TRUE INT8, pool directly on INT8 values
                current_int8 = maxpool2d_int8(current_int8, kernel_size=kernel_size, stride=stride, padding=padding)

                # Dequantize to FP32 for next QuantIdentity layer
                current_fp32 = dequantize_linear(current_int8, scale=current_scale, zero_point=0)

                if verbose:
                    print(f"  Kernel: {kernel_size}, stride={stride}, padding={padding}")
                    print(f"  Output: INT8 {current_int8.shape}, range=[{current_int8.min()}, {current_int8.max()}]")
                    print(f"  → FP32 {current_fp32.shape} (for next QuantIdentity)")

            elif layer_type == 'ZeroPad2d':
                # ZeroPad2d - pad with zeros (scale-invariant operation)
                padding = layer_info.get('padding', (0, 0, 0, 0))
                if isinstance(padding, int):
                    padding = (padding, padding, padding, padding)
                elif len(padding) == 2:
                    padding = (padding[0], padding[0], padding[1], padding[1])
                pad_left, pad_right, pad_top, pad_bottom = padding

                # Apply padding using numpy
                # Handle both 3D [C, H, W] and 4D [B, C, H, W] inputs
                if current_int8.ndim == 4:
                    pad_width = ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right))
                elif current_int8.ndim == 3:
                    pad_width = ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right))
                else:
                    raise ValueError(f"ZeroPad2d expects 3D or 4D input, got {current_int8.ndim}D")

                current_int8 = np.pad(
                    current_int8,
                    pad_width,
                    mode='constant',
                    constant_values=0
                )
                current_fp32 = dequantize_linear(current_int8, scale=current_scale, zero_point=0)

                if verbose:
                    print(f"  Padding: left={pad_left}, right={pad_right}, top={pad_top}, bottom={pad_bottom}")
                    print(f"  Output: INT8 {current_int8.shape}, range=[{current_int8.min()}, {current_int8.max()}]")

            elif layer_type == 'AvgPool2d':
                # INT8 AvgPool with rescaling
                # Check if next layer is QuantIdentity (will be fused in C code)
                # If so, use the QuantIdentity's scale as output to match C code fusion
                kernel_size = layer_info['kernel_size']
                stride = layer_info.get('stride', kernel_size)
                padding = layer_info.get('padding', (0, 0))
                current_idx = self.layer_order.index(layer_name)
                next_idx = current_idx + 1
                scale_output = layer_info.get('scale_output', current_scale)

                if next_idx < len(self.layer_order):
                    next_layer_name = self.layer_order[next_idx]
                    next_layer_info = self.layer_info.get(next_layer_name)
                    if next_layer_info and next_layer_info.get('type') == 'QuantIdentity':
                        # Use QuantIdentity scale to match C code fusion
                        scale_output = next_layer_info.get('scale', scale_output)

                current_int8 = avgpool2d_int8(
                    current_int8,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    scale_input=current_scale,
                    scale_output=scale_output
                )
                current_scale = scale_output
                current_fp32 = dequantize_linear(current_int8, scale=current_scale, zero_point=0)

                if verbose:
                    print(f"  Kernel: {kernel_size}, stride={stride}, padding={padding}")
                    print(f"  Output: INT8 {current_int8.shape}, range=[{current_int8.min()}, {current_int8.max()}]")

            elif layer_type == 'GlobalAvgPool':
                # INT8 Global Average Pooling
                # Check if next layer is QuantIdentity (will be fused in C code)
                # If so, use the QuantIdentity's scale as output to match C code fusion
                current_idx = self.layer_order.index(layer_name)
                next_idx = current_idx + 1
                scale_output = layer_info.get('scale_output', current_scale)

                if next_idx < len(self.layer_order):
                    next_layer_name = self.layer_order[next_idx]
                    next_layer_info = self.layer_info.get(next_layer_name)
                    if next_layer_info and next_layer_info.get('type') == 'QuantIdentity':
                        # Use QuantIdentity scale to match C code fusion
                        scale_output = next_layer_info.get('scale', scale_output)

                current_int8 = global_avgpool_int8(
                    current_int8,
                    scale_input=current_scale,
                    scale_output=scale_output,
                    keepdims=True
                )
                current_scale = scale_output
                current_fp32 = dequantize_linear(current_int8, scale=current_scale, zero_point=0)

                if verbose:
                    print(f"  Output: INT8 {current_int8.shape}, range=[{current_int8.min()}, {current_int8.max()}]")

            elif layer_type == 'Add':
                # INT8 element-wise addition (two inputs)
                is_branch_merge = False

                # Internal branching merge: combine two branch outputs
                if internal_branch_prefixes and branch_states is not None and current_branch is not None:
                    # Save current (last) branch state before merging
                    branch_states[current_branch] = (current_fp32, current_int8, current_scale)

                    # Check if we have outputs from all branches
                    ready_branches = [p for p in sorted(internal_branch_prefixes)
                                      if p in branch_states and branch_states[p][1] is not None]
                    if len(ready_branches) >= 2:
                        is_branch_merge = True
                        p_a, p_b = ready_branches[0], ready_branches[1]
                        tensor_a = branch_states[p_a][1]
                        scale_a = branch_states[p_a][2]
                        tensor_b = branch_states[p_b][1]
                        scale_b = branch_states[p_b][2]

                        # Find the last layer name for each branch (for metadata)
                        name_a = name_b = None
                        for ln in reversed(self.layer_order):
                            if self._get_branch_prefix(ln) == p_a and ln in self.intermediate_outputs:
                                name_a = name_a or ln
                            if self._get_branch_prefix(ln) == p_b and ln in self.intermediate_outputs:
                                name_b = name_b or ln
                            if name_a and name_b:
                                break
                        name_a = name_a or p_a.rstrip('_')
                        name_b = name_b or p_b.rstrip('_')

                        layer_info['inputs'] = [name_a, name_b]
                        if layer_name in self.layer_info:
                            self.layer_info[layer_name]['inputs'] = [name_a, name_b]
                        if layer_name in self.network_info:
                            self.network_info[layer_name]['inputs'] = [name_a, name_b]

                        if verbose:
                            print(f"  [Branch Merge] {p_a}({name_a}) + {p_b}({name_b})")
                            print(f"  Shapes: {tensor_a.shape} + {tensor_b.shape}")

                        # Clear branch state — we're merged now
                        branch_states = None
                        fork_state = None
                        current_branch = None

                # Special handling for ResNet skip connections
                # Pattern: block_name.add combines conv2 output + shortcut output (or cached input)
                is_resnet_skip = False
                main_path_name = None
                skip_path_name = None

                if not is_branch_merge and '.add' in layer_name:
                    block_name = layer_name.rsplit('.add', 1)[0]  # e.g., "layer2.0"
                    conv2_name = block_name + '.conv2'
                    shortcut_name = block_name + '.shortcut_quant'
                    cache_key = block_name + '.input'

                    # Check if conv2 exists (main path)
                    if conv2_name in self.intermediate_outputs:
                        is_resnet_skip = True
                        main_path_name = conv2_name

                        # Check if shortcut exists (projection) or use cached input (identity)
                        if shortcut_name in self.intermediate_outputs:
                            skip_path_name = shortcut_name
                        elif cache_key in self.activation_cache:
                            # Use cached block input for identity skip connection
                            skip_path_name = cache_key

                if is_branch_merge:
                    pass  # tensor_a/b and scale_a/b already set above
                elif is_resnet_skip and main_path_name and skip_path_name:
                    # ResNet skip connection detected
                    tensor_main = self.intermediate_outputs[main_path_name]
                    scale_main = self.output_scales[main_path_name]

                    if skip_path_name in self.intermediate_outputs:
                        tensor_skip = self.intermediate_outputs[skip_path_name]
                        scale_skip = self.output_scales[skip_path_name]
                    else:
                        # Use cached activation for identity skip
                        tensor_skip, scale_skip = self.activation_cache[skip_path_name]

                    if verbose:
                        print(f"  [ResNet Skip] Main: {main_path_name}, Skip: {skip_path_name}")
                        print(f"  Main shape: {tensor_main.shape}, Skip shape: {tensor_skip.shape}")

                    # Update metadata to reflect correct inputs
                    # For skip path from cache, find the actual source layer (not block name)
                    if skip_path_name.endswith('.input'):
                        # Identity skip - find layer before block's conv1
                        block_name = skip_path_name.replace('.input', '')
                        conv1_layer = block_name + '.conv1'
                        # Find the layer immediately before conv1 in layer_order
                        try:
                            conv1_idx = self.layer_order.index(conv1_layer)
                            # The input is the output of the previous layer
                            skip_layer_name = self.layer_order[conv1_idx - 1] if conv1_idx > 0 else 'input_quant'
                        except (ValueError, IndexError):
                            skip_layer_name = skip_path_name.replace('.input', '')
                    else:
                        skip_layer_name = skip_path_name

                    corrected_inputs = [main_path_name, skip_layer_name]
                    layer_info['inputs'] = corrected_inputs
                    if layer_name in self.layer_info:
                        self.layer_info[layer_name]['inputs'] = corrected_inputs
                    if layer_name in self.network_info:
                        self.network_info[layer_name]['inputs'] = corrected_inputs

                    tensor_a = tensor_main
                    tensor_b = tensor_skip
                    scale_a = scale_main
                    scale_b = scale_skip
                else:
                    # Standard Add handling (DenseNet, etc.)
                    input_names = layer_info.get('inputs', [])
                    resolved_inputs = self._resolve_add_inputs(layer_name, input_names)
                    if len(resolved_inputs) < 2:
                        raise ValueError(f"Add layer {layer_name} could not resolve two unique inputs "
                                         f"(candidates: {input_names})")
                    name_a, name_b = resolved_inputs[:2]
                    # Update metadata to reflect the tensors actually used
                    layer_info['inputs'] = [name_a, name_b]
                    if layer_name in self.layer_info:
                        self.layer_info[layer_name]['inputs'] = [name_a, name_b]
                    if layer_name in self.network_info:
                        self.network_info[layer_name]['inputs'] = [name_a, name_b]
                    tensor_a = self.intermediate_outputs[name_a]
                    tensor_b = self.intermediate_outputs[name_b]
                    scale_a = self.output_scales[name_a]
                    scale_b = self.output_scales[name_b]
                    if tensor_a is not None and tensor_b is not None and tensor_a.shape != tensor_b.shape:
                        print(f"[INT8 Engine] Add shape mismatch in {layer_name}: "
                              f"{name_a} {tensor_a.shape} vs {name_b} {tensor_b.shape}")

                scale_output = layer_info.get('scale_output', scale_a)

                current_int8 = add_int8(
                    tensor_a,
                    tensor_b,
                    scale_x1=scale_a,
                    scale_x2=scale_b,
                    scale_output=scale_output
                )
                current_scale = scale_output
                current_fp32 = dequantize_linear(current_int8, scale=current_scale, zero_point=0)

                if verbose:
                    if is_branch_merge:
                        pass  # Already printed above
                    elif is_resnet_skip:
                        print(f"  Adding: {main_path_name} + {skip_path_name}")
                    else:
                        print(f"  Adding: {name_a} + {name_b}")
                    print(f"  Output: INT8 {current_int8.shape}, range=[{current_int8.min()}, {current_int8.max()}]")

            elif layer_type == 'Concatenate':
                # INT8 channel concatenation
                input_names = layer_info.get('inputs', [])
                tensors = []
                scales = []

                # For multi-input or internal-branching models, prefer branch states
                # The extractor may capture wrong inputs for torch.cat in dual-input networks
                if internal_branch_prefixes and branch_states is not None and current_branch is not None:
                    # Internal branching merge via concatenation
                    branch_states[current_branch] = (current_fp32, current_int8, current_scale)
                    for prefix in sorted(internal_branch_prefixes):
                        if prefix in branch_states and branch_states[prefix][1] is not None:
                            tensors.append(branch_states[prefix][1])
                            scales.append(branch_states[prefix][2])
                    if verbose:
                        print(f"  [Branch Merge] Concatenating: {sorted(internal_branch_prefixes)}")
                    branch_states = None
                    fork_state = None
                    current_branch = None
                elif is_multi_input and branch_states:
                    # Multi-input model: gather from branch states
                    for prefix in sorted(branch_states.keys()):
                        _, branch_int8, branch_scale = branch_states[prefix]
                        if branch_int8 is not None:
                            tensors.append(branch_int8)
                            scales.append(branch_scale)
                    if verbose:
                        print(f"  Merging branches: {list(branch_states.keys())}")
                elif input_names:
                    # Standard case: gather from named intermediate outputs
                    for name in input_names:
                        tensors.append(self.intermediate_outputs[name])
                        scales.append(self.output_scales[name])

                if not tensors:
                    raise ValueError(f"Concatenate layer '{layer_name}' has no inputs to merge")

                scale_output = layer_info.get('scale_output', scales[0])

                current_int8 = concat_int8(
                    tensors,
                    scales,
                    scale_output=scale_output,
                    axis=1  # Channel dimension
                )
                current_scale = scale_output
                current_fp32 = dequantize_linear(current_int8, scale=current_scale, zero_point=0)

                # After concat, clear branch states as we're now merged
                if is_multi_input and branch_states:
                    branch_states = None

                if verbose:
                    if input_names:
                        print(f"  Concatenating: {input_names}")
                    print(f"  Output: INT8 {current_int8.shape}, range=[{current_int8.min()}, {current_int8.max()}]")

            elif layer_type == 'Flatten':
                # Flatten (just reshape, preserves quantization)
                start_dim = layer_info.get('start_dim', 1)

                if current_int8 is not None:
                    # For TRUE INT8, flatten directly on INT8
                    current_int8 = flatten_int8(current_int8, start_dim=start_dim)
                    # Dequantize to FP32 for next QuantIdentity layer
                    current_fp32 = dequantize_linear(current_int8, scale=current_scale, zero_point=0)
                else:
                    # No INT8 tensor yet (e.g., Flatten before first QuantIdentity)
                    shape = current_fp32.shape
                    leading_dims = shape[:start_dim]
                    flattened = int(np.prod(shape[start_dim:]))
                    current_fp32 = current_fp32.reshape(*leading_dims, flattened)

                if verbose:
                    if current_int8 is not None:
                        print(f"  Output: INT8 {current_int8.shape}, range=[{current_int8.min()}, {current_int8.max()}]")
                        print(f"  → FP32 {current_fp32.shape} (for next QuantIdentity)")
                    else:
                        print(f"  Output: FP32 {current_fp32.shape} (pre-quantization)")

            elif layer_type == 'Permute':
                # Permute (transpose dimensions, preserves quantization)
                dims = layer_info.get('dims')

                if current_int8 is not None:
                    current_int8 = np.transpose(current_int8, dims)
                    current_fp32 = dequantize_linear(current_int8, scale=current_scale, zero_point=0)
                else:
                    current_fp32 = np.transpose(current_fp32, dims)

                if verbose:
                    print(f"  Permute dims: {dims}")
                    if current_int8 is not None:
                        print(f"  Output: INT8 {current_int8.shape}, range=[{current_int8.min()}, {current_int8.max()}]")
                    else:
                        print(f"  Output: FP32 {current_fp32.shape} (pre-quantization)")

            elif layer_type == 'Reshape':
                # Reshape (change tensor shape, preserves quantization)
                shape = layer_info.get('shape')
                if current_int8 is not None:
                    batch = current_int8.shape[0]
                    current_int8 = current_int8.reshape(batch, *shape)
                    current_fp32 = dequantize_linear(current_int8, scale=current_scale, zero_point=0)
                else:
                    batch = current_fp32.shape[0]
                    current_fp32 = current_fp32.reshape(batch, *shape)

                if verbose:
                    print(f"  Reshape: {shape}")
                    if current_int8 is not None:
                        print(f"  Output: INT8 {current_int8.shape}, range=[{current_int8.min()}, {current_int8.max()}]")
                    else:
                        print(f"  Output: FP32 {current_fp32.shape} (pre-quantization)")

            elif layer_type == 'AdaptiveAvgPool1d':
                # Adaptive average pooling to fixed output size
                # Uses pure INT32 arithmetic to match C implementation exactly
                output_size = layer_info.get('output_size')
                expected_output_shape = layer_info.get('output_shape')

                # Check if we need to transpose before pooling
                # PyTorch AdaptiveAvgPool1d expects [batch, channels, length]
                # If current is [batch, seq_len, channels] and expected output is [batch, channels, 1]
                # we need to transpose first
                needs_transpose = False
                if expected_output_shape is not None and len(current_int8.shape) == 3:
                    # Current: [B, L, M], Expected output: [B, M, 1]
                    if expected_output_shape[1] != current_int8.shape[1] and expected_output_shape[1] == current_int8.shape[2]:
                        needs_transpose = True
                        if verbose:
                            print(f"  Detected implicit transpose: {current_int8.shape} -> ", end="")
                        current_int8 = np.transpose(current_int8, (0, 2, 1))
                        if verbose:
                            print(f"{current_int8.shape}")

                # Input shape: [batch, channels, length] or [channels, length]
                input_len = current_int8.shape[-1]

                if output_size == 1:
                    # Global average pool over last dimension using INT32 arithmetic
                    # This matches the C code: (sum + count/2) / count
                    int32_input = current_int8.astype(np.int32)
                    sum_values = int32_input.sum(axis=-1, keepdims=True)
                    count = input_len
                    # Integer division with rounding, matching C truncation toward zero
                    # Python // is floor division (toward -inf), C / truncates toward 0
                    # Use np.trunc to match C behavior
                    numerator = sum_values + count // 2
                    avg_int32 = np.trunc(numerator.astype(np.float64) / count).astype(np.int32)
                    # Clip to INT8 range
                    avg_int32 = np.clip(avg_int32, -128, 127)
                    current_int8 = avg_int32.astype(np.int8)
                else:
                    # General adaptive pooling with output_size > 1
                    # Divide input evenly among output positions
                    shape_prefix = current_int8.shape[:-1]
                    output_shape = shape_prefix + (output_size,)
                    result = np.zeros(output_shape, dtype=np.int8)

                    for o in range(output_size):
                        in_start = (o * input_len) // output_size
                        in_end = ((o + 1) * input_len) // output_size
                        count = in_end - in_start

                        int32_slice = current_int8[..., in_start:in_end].astype(np.int32)
                        sum_values = int32_slice.sum(axis=-1)
                        # Use np.trunc to match C truncation toward zero
                        numerator = sum_values + count // 2
                        avg_int32 = np.trunc(numerator.astype(np.float64) / count).astype(np.int32)
                        avg_int32 = np.clip(avg_int32, -128, 127)
                        result[..., o] = avg_int32.astype(np.int8)

                    current_int8 = result

                # Scale is preserved (averaging doesn't change scale)
                current_fp32 = dequantize_linear(current_int8, scale=current_scale, zero_point=0)

                if verbose:
                    print(f"  AdaptiveAvgPool1d(output_size={output_size})")
                    print(f"  Output: INT8 {current_int8.shape}, range=[{current_int8.min()}, {current_int8.max()}]")

            elif layer_type == 'Squeeze':
                # Squeeze singleton dimensions
                dim = layer_info.get('dim')

                # Squeeze on INT8 directly
                current_int8 = np.squeeze(current_int8, axis=dim)

                # Also update FP32 for consistency
                current_fp32 = dequantize_linear(current_int8, scale=current_scale, zero_point=0)

                if verbose:
                    print(f"  Squeeze dim={dim}")
                    print(f"  Output: INT8 {current_int8.shape}, range=[{current_int8.min()}, {current_int8.max()}]")

            elif layer_type == 'QuantLinear':
                # INT8 Linear layer
                scale_x = current_scale
                scale_w = layer_info['scale_weight']

                expected_features = layer_info.get('in_features')

                # Handle input reshaping for transformers vs classifiers
                original_shape = current_int8.shape
                needs_reshape_back = False

                if current_int8.ndim > 2:
                    # Check if last dimension matches expected features (transformer case)
                    if expected_features is not None and current_int8.shape[-1] == expected_features:
                        # Transformer MLP: (batch, seq_len, features) -> (batch*seq_len, features)
                        # Then reshape back after linear to preserve sequence dimension
                        batch_seq_dims = current_int8.shape[:-1]
                        current_int8 = current_int8.reshape(-1, expected_features)
                        needs_reshape_back = True
                    else:
                        # Classifier case: flatten everything except batch dimension
                        batch = current_int8.shape[0]
                        current_int8 = current_int8.reshape(batch, -1)

                if expected_features is not None and current_int8.shape[-1] != expected_features:
                    if not needs_reshape_back:  # Only error if not already handled
                        current_int8 = current_int8.reshape(current_int8.shape[0], -1)
                        if current_int8.shape[-1] != expected_features:
                            raise ValueError(f"Flattened features {current_int8.shape[-1]} "
                                             f"do not match classifier expectation {expected_features}")
                elif expected_features is None:
                    expected_features = current_int8.shape[-1]

                w_int8 = layer_info['weight_int8']
                bias_fp32 = layer_info.get('bias_fp32')

                # Convert to numpy arrays if needed (JSON serialization converts arrays to lists)
                if not isinstance(w_int8, np.ndarray):
                    w_int8 = np.array(w_int8)
                if bias_fp32 is not None and not isinstance(bias_fp32, np.ndarray):
                    bias_fp32 = np.array(bias_fp32)

                # Check if this is the final layer (no more layers after this except maybe activation)
                current_idx = self.layer_order.index(layer_name)
                is_final_layer = current_idx == len(self.layer_order) - 1

                # Step 1: INT8 accumulation (X @ W^T)
                output_int32 = current_int8.astype(np.int32) @ w_int8.T.astype(np.int32)

                # Step 2: Rescale
                scale_combined = scale_x * scale_w

                if is_final_layer:
                    # Final layer: Output FP32 logits directly
                    current_fp32 = output_int32.astype(np.float32) * scale_combined
                    if bias_fp32 is not None:
                        current_fp32 += bias_fp32

                    # Reshape back if needed (transformer case)
                    if needs_reshape_back:
                        out_features = w_int8.shape[0]
                        new_shape = list(batch_seq_dims) + [out_features]
                        current_fp32 = current_fp32.reshape(new_shape)

                    current_int8 = None
                    current_scale = 1.0

                    if verbose:
                        print(f"  Scales: x={scale_x:.6f}, w={scale_w:.6f}")
                        print(f"  Output: FP32 {current_fp32.shape}, range=[{current_fp32.min():.3f}, {current_fp32.max():.3f}]")
                        print(f"  FP32 logits: {current_fp32[0, :3]}...")
                else:
                    # Intermediate layer: Keep INT8 for next layers
                    # Get output scale from next QuantReLU or QuantIdentity
                    scale_out = layer_info.get('scale_output', scale_combined)

                    # Bias handling (critical for bit-exact matching with C kernels):
                    #   (acc_int32 + bias_int32) * (scale_x * scale_w)  -> quantize to INT8
                    # NOT:
                    #   (acc_int32 * (scale_x * scale_w)) + bias_fp32
                    # The latter loses precision when bias magnitudes are large ("bias trap").
                    if bias_fp32 is not None:
                        bias_int32 = np.round(bias_fp32 / scale_combined).astype(np.int32)
                        output_int32 = output_int32.astype(np.int32) + bias_int32
                    else:
                        output_int32 = output_int32.astype(np.int32)

                    # Convert INT32 → FP32 (already includes bias in INT32 domain) → quantize to INT8
                    current_fp32 = output_int32.astype(np.float32) * scale_combined

                    # Quantize to INT8 for next layer
                    # Use the scale from the network info if available
                    # Otherwise, compute from current FP32 range
                    if 'scale_output' not in layer_info:
                        # Estimate scale from FP32 range
                        fp32_max = max(abs(current_fp32.min()), abs(current_fp32.max()))
                        scale_out = fp32_max / 127.0 if fp32_max > 0 else 1.0

                    current_int8 = quantize_linear(current_fp32, scale=scale_out, zero_point=0)
                    current_scale = scale_out

                    # Reshape back if needed (transformer case)
                    if needs_reshape_back:
                        out_features = w_int8.shape[0]
                        new_shape = list(batch_seq_dims) + [out_features]
                        current_int8 = current_int8.reshape(new_shape)
                        current_fp32 = current_fp32.reshape(new_shape)

                    if verbose:
                        print(f"  Scales: x={scale_x:.6f}, w={scale_w:.6f}, out={scale_out:.6f}")
                        print(f"  Output: INT8 {current_int8.shape}, range=[{current_int8.min()}, {current_int8.max()}]")

            elif layer_type in ('MultiheadSelfAttention', 'SelfAttention'):
                current_int8, current_scale, current_fp32 = self._run_attention_layer(
                    layer_name, layer_info, current_int8, current_scale, verbose=verbose
                )

            elif layer_type == 'CrossAttentionWithSelfRefine':
                current_int8, current_scale, current_fp32 = self._run_cross_attn_self_refine_layer(
                    layer_name, layer_info, current_int8, current_scale, verbose=verbose
                )

            elif layer_type == 'ClassificationHeadWithMLP':
                current_int8, current_scale, current_fp32 = self._run_classification_head_layer(
                    layer_name, layer_info, current_int8, current_scale, verbose=verbose
                )

            elif layer_type == 'CrossAttention':
                current_int8, current_scale, current_fp32 = self._run_cross_attention_layer(
                    layer_name, layer_info, current_int8, current_scale, verbose=verbose
                )

            elif layer_type == 'GroupNorm':
                # INT8 GroupNorm with affine (fixed-point mean/variance for bit-exact matching)
                weight = layer_info.get('weight')
                bias = layer_info.get('bias')
                num_groups = int(layer_info.get('num_groups', 1))

                # Convert to numpy arrays if needed (JSON serialization converts arrays to lists)
                if weight is not None and not isinstance(weight, np.ndarray):
                    weight = np.array(weight, dtype=np.float32)
                if bias is not None and not isinstance(bias, np.ndarray):
                    bias = np.array(bias, dtype=np.float32)

                # Output scale: use next QuantIdentity if present (matches C code patterns)
                current_idx = self.layer_order.index(layer_name)
                next_idx = current_idx + 1
                scale_output = current_scale

                if next_idx < len(self.layer_order):
                    next_layer_name = self.layer_order[next_idx]
                    next_layer_info = self.layer_info.get(next_layer_name)
                    if next_layer_info and next_layer_info.get('type') == 'QuantIdentity':
                        scale_output = next_layer_info.get('scale', scale_output)

                current_int8 = groupnorm_int8_fixed_point(
                    current_int8,
                    weight_fp32=weight,
                    bias_fp32=bias,
                    scale_input=current_scale,
                    scale_output=scale_output,
                    num_groups=num_groups,
                )
                current_scale = scale_output
                current_fp32 = dequantize_linear(current_int8, scale=current_scale, zero_point=0)

                if verbose:
                    print(f"  Groups: {num_groups}")
                    print(f"  Output: INT8 {current_int8.shape}, range=[{current_int8.min()}, {current_int8.max()}]")

            elif layer_type == 'LayerNorm':
                # INT8 LayerNorm with affine transformation
                weight = layer_info['weight']
                bias = layer_info['bias']
                normalized_shape = layer_info['normalized_shape']
                eps = layer_info.get('eps', 1e-5)

                # Convert to numpy arrays if needed (JSON serialization converts arrays to lists)
                if not isinstance(weight, np.ndarray):
                    weight = np.array(weight)
                if not isinstance(bias, np.ndarray):
                    bias = np.array(bias)

                # Get output scale from next QuantIdentity (if exists)
                current_idx = self.layer_order.index(layer_name)
                next_idx = current_idx + 1
                scale_output = current_scale  # Default: preserve input scale

                if next_idx < len(self.layer_order):
                    next_layer_name = self.layer_order[next_idx]
                    next_layer_info = self.layer_info.get(next_layer_name)
                    if next_layer_info and next_layer_info.get('type') == 'QuantIdentity':
                        scale_output = next_layer_info.get('scale', scale_output)

                # Apply INT8 LayerNorm
                # Convert normalized_shape to integer if it's a single-element tuple/list
                if isinstance(normalized_shape, (tuple, list)) and len(normalized_shape) == 1:
                    normalized_shape = normalized_shape[0]

                # Use integer-only LayerNorm if enabled (for bit-exact matching with C)
                if self.use_i_layernorm:
                    current_int8 = layernorm_int8_fixed_point(
                        current_int8,
                        weight,
                        bias,
                        scale_input=current_scale,
                        scale_output=scale_output,
                        normalized_shape=normalized_shape,
                        eps=eps
                    )
                    if verbose:
                        print(f"  Using i-LayerNorm (integer sqrt) for bit-exact matching")
                else:
                    current_int8 = layernorm_int8(
                        current_int8,
                        weight,
                        bias,
                        scale_input=current_scale,
                        scale_output=scale_output,
                        normalized_shape=normalized_shape,
                        eps=eps
                    )
                current_scale = scale_output
                current_fp32 = dequantize_linear(current_int8, scale=current_scale, zero_point=0)

                if verbose:
                    print(f"  Normalized shape: {normalized_shape}, eps={eps}")
                    print(f"  Output: INT8 {current_int8.shape}, range=[{current_int8.min()}, {current_int8.max()}]")

            elif layer_type == 'RMSNorm':
                # INT8 RMSNorm (used in Llama and other LLMs) - no bias, only weight
                weight = layer_info['weight']
                normalized_shape = layer_info['normalized_shape']
                eps = layer_info.get('eps', 1e-5)

                # Convert to numpy arrays if needed
                if not isinstance(weight, np.ndarray):
                    weight = np.array(weight)

                # Handle normalized_shape - can be int or tuple
                if isinstance(normalized_shape, (tuple, list)) and len(normalized_shape) == 1:
                    normalized_shape = normalized_shape[0]

                # Get output scale from next QuantIdentity (if exists)
                current_idx = self.layer_order.index(layer_name)
                next_idx = current_idx + 1
                scale_output = current_scale  # Default: preserve input scale

                if next_idx < len(self.layer_order):
                    next_layer_name = self.layer_order[next_idx]
                    next_layer_info = self.layer_info.get(next_layer_name)
                    if next_layer_info and next_layer_info.get('type') == 'QuantIdentity':
                        scale_output = next_layer_info.get('scale', scale_output)

                # Apply INT8 RMSNorm
                # Use integer-only RMSNorm if enabled (for bit-exact matching with C)
                if self.use_i_layernorm:
                    current_int8 = rmsnorm_int8_fixed_point(
                        current_int8,
                        weight,
                        scale_input=current_scale,
                        scale_output=scale_output,
                        normalized_shape=normalized_shape,
                        eps=eps
                    )
                    if verbose:
                        print(f"  Using i-RMSNorm (integer sqrt) for bit-exact matching")
                else:
                    current_int8 = rmsnorm_int8(
                        current_int8,
                        weight,
                        scale_input=current_scale,
                        scale_output=scale_output,
                        normalized_shape=normalized_shape,
                        eps=eps
                    )
                current_scale = scale_output
                current_fp32 = dequantize_linear(current_int8, scale=current_scale, zero_point=0)

                if verbose:
                    print(f"  Normalized shape: {normalized_shape}, eps={eps}")
                    print(f"  Output: INT8 {current_int8.shape}, range=[{current_int8.min()}, {current_int8.max()}]")

            elif layer_type == 'GELU':
                # INT8 GELU activation
                # Get output scale from next QuantIdentity (if exists)
                current_idx = self.layer_order.index(layer_name)
                next_idx = current_idx + 1
                scale_output = current_scale  # Default: preserve input scale

                if next_idx < len(self.layer_order):
                    next_layer_name = self.layer_order[next_idx]
                    next_layer_info = self.layer_info.get(next_layer_name)
                    if next_layer_info and next_layer_info.get('type') == 'QuantIdentity':
                        scale_output = next_layer_info.get('scale', scale_output)

                # Apply INT8 GELU (use LUT-based if enabled for bit-exact matching)
                if self.use_i_gelu:
                    current_int8 = gelu_int8_lut(
                        current_int8,
                        scale_input=current_scale,
                        scale_output=scale_output,
                        gelu_lut=self.gelu_lut,
                        lut_metadata=self.gelu_lut_metadata
                    )
                    if verbose:
                        print(f"  Using i-GELU (LUT-based) for bit-exact matching")
                else:
                    current_int8 = gelu_int8(
                        current_int8,
                        scale_input=current_scale,
                        scale_output=scale_output
                    )
                current_scale = scale_output
                current_fp32 = dequantize_linear(current_int8, scale=current_scale, zero_point=0)

                if verbose:
                    print(f"  Output: INT8 {current_int8.shape}, range=[{current_int8.min()}, {current_int8.max()}]")

            elif layer_type == 'Mean':
                # INT8 Mean pooling (typically over sequence dimension)
                dim = layer_info.get('dim', 1)
                keepdim = layer_info.get('keepdim', False)
                scale_input = current_scale
                scale_output = layer_info.get('scale_output', current_scale)

                # Compute mean over specified dimension
                # Input is INT8, compute mean in INT32, requantize to output scale
                input_fp32 = current_int8.astype(np.float32) * scale_input
                mean_fp32 = np.mean(input_fp32, axis=dim, keepdims=keepdim)
                current_int8 = np.clip(np.round(mean_fp32 / scale_output), -128, 127).astype(np.int8)
                current_scale = scale_output
                current_fp32 = dequantize_linear(current_int8, scale=current_scale, zero_point=0)

                if verbose:
                    print(f"  Mean over dim={dim}, keepdim={keepdim}")
                    print(f"  Output: INT8 {current_int8.shape}, range=[{current_int8.min()}, {current_int8.max()}]")

            elif layer_type == 'AlternatingAttention':
                # INT8 Alternating Attention (Cerebro transformer)
                if alternating_attention_int8 is None:
                    raise ImportError("alternating_attention_int8 not available from atomic_ops")

                embed_dim = layer_info.get('embed_dim')
                num_heads = layer_info.get('num_heads', 1)
                head_dim = layer_info.get('head_dim') or (embed_dim // num_heads)
                num_channels = layer_info.get('num_channels')
                temporal_len = layer_info.get('temporal_len')
                block_idx = layer_info.get('block_idx', 0)
                scaling_factor = layer_info.get('scaling_factor') or (1.0 / np.sqrt(head_dim))

                # Get weights (QKV uses combined projection, output uses proj)
                qkv_weight = np.array(layer_info['qkv_weight_int8'], dtype=np.int8)
                proj_weight = np.array(layer_info['out_weight_int8'], dtype=np.int8)

                # Biases need to be INT32 for the atomic op
                # Quantize FP32 bias to INT32 using combined scale
                scale_input = current_scale
                scale_qkv_weight = layer_info.get('qkv_scale_weight', 1.0)
                scale_out_weight = layer_info.get('out_scale_weight', 1.0)

                qkv_bias_fp32 = layer_info.get('qkv_bias_fp32')
                if qkv_bias_fp32 is not None:
                    qkv_bias_fp32 = np.array(qkv_bias_fp32, dtype=np.float32)
                    scale_bias = scale_input * scale_qkv_weight
                    qkv_bias_int32 = np.round(qkv_bias_fp32 / scale_bias).astype(np.int32)
                else:
                    qkv_bias_int32 = None

                out_bias_fp32 = layer_info.get('out_bias_fp32')
                if out_bias_fp32 is not None:
                    out_bias_fp32 = np.array(out_bias_fp32, dtype=np.float32)
                    # Output proj bias uses scale_v * scale_out_weight
                    scale_v = layer_info.get('v_scale_output', 1.0)
                    scale_bias_out = scale_v * scale_out_weight
                    proj_bias_int32 = np.round(out_bias_fp32 / scale_bias_out).astype(np.int32)
                else:
                    proj_bias_int32 = None

                # Get scales
                scale_qkv_out = layer_info.get('qkv_scale_output', 1.0)
                scale_q = layer_info.get('q_scale_output', scale_qkv_out)
                scale_k = layer_info.get('k_scale_output', scale_qkv_out)
                scale_v = layer_info.get('v_scale_output', scale_qkv_out)
                scale_output = layer_info.get('scale_output', current_scale)

                # Intermediate scales (approximations for attention computation)
                # These would ideally come from QAT but we estimate them here
                scale_attn_scores = scale_q * scale_k  # Q @ K^T scale
                scale_softmax = 1.0 / 127.0  # Standard UINT8 softmax scale
                scale_attn_out = scale_softmax * scale_v  # Softmax @ V scale

                current_int8, debug_info = alternating_attention_int8(
                    x_int8=current_int8,
                    qkv_weight_int8=qkv_weight,
                    qkv_bias_int32=qkv_bias_int32,
                    proj_weight_int8=proj_weight,
                    proj_bias_int32=proj_bias_int32,
                    scale_x=scale_input,
                    scale_qkv_weight=scale_qkv_weight,
                    scale_qkv_out=scale_qkv_out,
                    scale_q=scale_q,
                    scale_k=scale_k,
                    scale_v=scale_v,
                    scale_attn_scores=scale_attn_scores,
                    scale_softmax=scale_softmax,
                    scale_attn_out=scale_attn_out,
                    scale_proj_weight=scale_out_weight,
                    scale_output=scale_output,
                    block_idx=block_idx,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    num_channels=num_channels,
                    temporal_len=temporal_len,
                    scaling_factor=scaling_factor,
                )
                current_scale = scale_output
                current_fp32 = dequantize_linear(current_int8, scale=current_scale, zero_point=0)

                attn_type = "channel" if block_idx % 2 == 0 else "temporal"
                if verbose:
                    print(f"  AlternatingAttention block={block_idx} ({attn_type})")
                    print(f"  Output: INT8 {current_int8.shape}, range=[{current_int8.min()}, {current_int8.max()}]")

            elif layer_type == 'Conv1dDepthwise':
                # INT8 Depthwise 1D Convolution (MAMBA conv1d)
                scale_x = current_scale
                scale_w = layer_info['scale_weight']
                scale_y = layer_info['scale_output']
                kernel_size = layer_info['kernel_size']
                causal = layer_info.get('causal', True)

                w_int8 = layer_info['weight_int8']
                bias_fp32 = layer_info.get('bias_fp32')

                # Convert to numpy arrays if needed
                if not isinstance(w_int8, np.ndarray):
                    w_int8 = np.array(w_int8)

                # Convert bias to INT32 with proper scale
                bias_int32 = None
                if bias_fp32 is not None:
                    if not isinstance(bias_fp32, np.ndarray):
                        bias_fp32 = np.array(bias_fp32)
                    scale_bias = scale_x * scale_w
                    bias_int32 = np.round(bias_fp32 / scale_bias).astype(np.int32)

                # Run INT8 Conv1D Depthwise
                current_int8 = conv1d_depthwise_int8(
                    current_int8,
                    w_int8,
                    bias_int32,
                    scale_x=scale_x,
                    scale_w=scale_w,
                    scale_y=scale_y,
                    causal=causal
                )
                current_scale = scale_y
                current_fp32 = dequantize_linear(current_int8, scale=current_scale, zero_point=0)

                if verbose:
                    print(f"  Kernel: {kernel_size}, causal={causal}")
                    print(f"  Scales: x={scale_x:.6f}, w={scale_w:.6f}, y={scale_y:.6f}")
                    print(f"  Output: INT8 {current_int8.shape}, range=[{current_int8.min()}, {current_int8.max()}]")

            elif layer_type == 'SiLU':
                # INT8 SiLU activation using 256-entry LUT
                scale_input = layer_info.get('scale_input', current_scale)
                scale_output = layer_info['scale_output']

                # Generate SiLU LUT for this input/output scale combination
                silu_lut = generate_silu_lut_int8(scale_input, scale_output)

                # Apply SiLU via LUT lookup
                current_int8 = silu_lut_int8(current_int8, silu_lut)
                current_scale = scale_output
                current_fp32 = dequantize_linear(current_int8, scale=current_scale, zero_point=0)

                if verbose:
                    print(f"  SiLU LUT: scale_in={scale_input:.6f}, scale_out={scale_output:.6f}")
                    print(f"  Output: INT8 {current_int8.shape}, range=[{current_int8.min()}, {current_int8.max()}]")

            elif layer_type == 'SwiGLU':
                # INT8 SwiGLU FFN (used in Llama and other LLMs)
                # y = W2(silu(W1(x)) * W3(x))
                w1_int8 = layer_info['w1_weight_int8']
                w3_int8 = layer_info['w3_weight_int8']
                w2_int8 = layer_info['w2_weight_int8']
                scale_w1 = layer_info['scale_w1']
                scale_w3 = layer_info['scale_w3']
                scale_w2 = layer_info['scale_w2']
                scale_hidden = layer_info.get('scale_hidden', current_scale)
                scale_output = layer_info.get('scale_output', current_scale)

                # Convert to numpy arrays if needed
                if not isinstance(w1_int8, np.ndarray):
                    w1_int8 = np.array(w1_int8)
                if not isinstance(w3_int8, np.ndarray):
                    w3_int8 = np.array(w3_int8)
                if not isinstance(w2_int8, np.ndarray):
                    w2_int8 = np.array(w2_int8)

                # Optional biases
                bias1_int32 = None
                bias3_int32 = None
                bias2_int32 = None
                if 'w1_bias_fp32' in layer_info and layer_info['w1_bias_fp32'] is not None:
                    bias_fp32 = layer_info['w1_bias_fp32']
                    if not isinstance(bias_fp32, np.ndarray):
                        bias_fp32 = np.array(bias_fp32)
                    bias1_int32 = np.round(bias_fp32 / (current_scale * scale_w1)).astype(np.int32)
                if 'w3_bias_fp32' in layer_info and layer_info['w3_bias_fp32'] is not None:
                    bias_fp32 = layer_info['w3_bias_fp32']
                    if not isinstance(bias_fp32, np.ndarray):
                        bias_fp32 = np.array(bias_fp32)
                    bias3_int32 = np.round(bias_fp32 / (current_scale * scale_w3)).astype(np.int32)
                if 'w2_bias_fp32' in layer_info and layer_info['w2_bias_fp32'] is not None:
                    bias_fp32 = layer_info['w2_bias_fp32']
                    if not isinstance(bias_fp32, np.ndarray):
                        bias_fp32 = np.array(bias_fp32)
                    bias2_int32 = np.round(bias_fp32 / (scale_hidden * scale_w2)).astype(np.int32)

                # Apply INT8 SwiGLU FFN
                current_int8 = swiglu_ffn_int8(
                    current_int8,
                    w1_int8, w3_int8, w2_int8,
                    scale_input=current_scale,
                    scale_w1=scale_w1,
                    scale_w3=scale_w3,
                    scale_w2=scale_w2,
                    scale_hidden=scale_hidden,
                    scale_output=scale_output,
                    bias1_int32=bias1_int32,
                    bias3_int32=bias3_int32,
                    bias2_int32=bias2_int32
                )
                current_scale = scale_output
                current_fp32 = dequantize_linear(current_int8, scale=current_scale, zero_point=0)

                if verbose:
                    dim = w1_int8.shape[1]
                    hidden_dim = w1_int8.shape[0]
                    print(f"  SwiGLU FFN: dim={dim}, hidden_dim={hidden_dim}")
                    print(f"  Scales: w1={scale_w1:.6f}, w3={scale_w3:.6f}, w2={scale_w2:.6f}")
                    print(f"  Output: INT8 {current_int8.shape}, range=[{current_int8.min()}, {current_int8.max()}]")

            elif layer_type == 'SSM':
                # INT8 State Space Model (MAMBA core)
                current_int8, current_scale, current_fp32 = self._run_ssm_layer(
                    layer_name, layer_info, current_int8, current_scale, verbose=verbose
                )

            elif layer_type == 'MambaBlock':
                # Complete MAMBA block (in_proj, conv1d, silu, ssm, gate, out_proj)
                current_int8, current_scale, current_fp32 = self._run_mamba_block(
                    layer_name, layer_info, current_int8, current_scale, verbose=verbose
                )

            elif layer_type == 'MambaWrapper':
                # Bidirectional MAMBA: forward + flip + reverse + flip_back + add
                current_int8, current_scale, current_fp32 = self._run_mamba_wrapper(
                    layer_name, layer_info, current_int8, current_scale, verbose=verbose
                )


            elif layer_type == 'RFFT':
                # Fixed-point RFFT (patch_size=40): outputs concatenated [magnitude, phase]
                patch_size = int(layer_info.get('patch_size', 40))
                if patch_size != 40:
                    raise ValueError(f"RFFT only supports patch_size=40 currently, got {patch_size}")

                # Output scale from next QuantIdentity if present
                current_idx = self.layer_order.index(layer_name)
                next_idx = current_idx + 1
                scale_output = current_scale
                if next_idx < len(self.layer_order):
                    next_layer_name = self.layer_order[next_idx]
                    next_layer_info = self.layer_info.get(next_layer_name)
                    if next_layer_info and next_layer_info.get('type') == 'QuantIdentity':
                        scale_output = next_layer_info.get('scale', scale_output)

                current_int8 = rfft40_features_int8_fixed_point(
                    current_int8,
                    scale_input=current_scale,
                    scale_output=scale_output,
                )
                current_scale = scale_output
                current_fp32 = dequantize_linear(current_int8, scale=current_scale, zero_point=0)

                if verbose:
                    print(f"  RFFT: patch_size={patch_size}")
                    print(f"  Output: INT8 {current_int8.shape}, range=[{current_int8.min()}, {current_int8.max()}]")

            elif layer_type == 'PatchEmbed':
                # Patch embedding: Conv2D + reshape + permute
                current_int8, current_scale, current_fp32 = self._run_patch_embed(
                    layer_name, layer_info, current_int8, current_scale, verbose=verbose
                )

            elif layer_type == 'Embedding':
                # INT8 embedding lookup (gather from INT8 weight table using captured indices)
                weight_int8 = layer_info.get('weight_int8')
                indices = layer_info.get('indices')
                scale_output = layer_info.get('scale', layer_info.get('scale_output', 1.0))

                if weight_int8 is None:
                    raise ValueError(f"Embedding layer {layer_name} missing weight_int8")
                if indices is None:
                    raise ValueError(f"Embedding layer {layer_name} missing indices (capture via extractor forward hook)")

                if not isinstance(weight_int8, np.ndarray):
                    weight_int8 = np.array(weight_int8, dtype=np.int8)
                if not isinstance(indices, np.ndarray):
                    indices = np.array(indices, dtype=np.int32)
                else:
                    indices = indices.astype(np.int32, copy=False)

                current_int8 = embedding_int8(indices, weight_int8)
                current_scale = float(scale_output)
                current_fp32 = dequantize_linear(current_int8, scale=current_scale, zero_point=0)

                if verbose:
                    print(f"  Embedding: vocab={weight_int8.shape[0]}, dim={weight_int8.shape[1]}, indices={indices.shape}")
                    print(f"  Output: INT8 {current_int8.shape}, range=[{current_int8.min()}, {current_int8.max()}]")

            elif layer_type == 'PositionalEmbedding':
                # Positional embedding: add to current tensor
                # The pos_embed is stored as INT8 with its own scale
                pos_embed_int8 = layer_info.get('pos_embed_int8')
                pos_scale = layer_info.get('scale', 1.0)
                scale_equalizer = layer_info.get('scale_equalizer', current_scale)

                if pos_embed_int8 is not None:
                    if not isinstance(pos_embed_int8, np.ndarray):
                        pos_embed_int8 = np.array(pos_embed_int8, dtype=np.int8)

                    # Broadcast positional embedding over batch dimension
                    # pos_embed is [1, seq_len, d_model], current is [B, seq_len, d_model]
                    if len(pos_embed_int8.shape) == 3 and pos_embed_int8.shape[0] == 1:
                        # Broadcast: both tensors need same scale for INT8 addition
                        # Use scale_equalizer as the common scale
                        common_scale = scale_equalizer if scale_equalizer else current_scale

                        # Requantize current tensor to common scale
                        current_int8 = requantize_int8(current_int8, scale_in=current_scale, scale_out=common_scale)

                        # Requantize pos_embed to common scale
                        pos_int8 = requantize_int8(pos_embed_int8, scale_in=pos_scale, scale_out=common_scale)

                        # INT8 addition with clipping
                        result = current_int8.astype(np.int32) + pos_int8.astype(np.int32)
                        current_int8 = np.clip(result, -128, 127).astype(np.int8)

                        current_scale = common_scale
                        current_fp32 = dequantize_linear(current_int8, scale=current_scale, zero_point=0)

                        if verbose:
                            print(f"  Added positional embedding: {pos_embed_int8.shape}")
                            print(f"  Common scale: {common_scale:.6f}")
                            print(f"  Output: INT8 {current_int8.shape}, range=[{current_int8.min()}, {current_int8.max()}]")

            else:
                print(f"  Warning: Unknown layer type '{layer_type}'")
                continue

            # Store intermediate output and its scale
            self.intermediate_outputs[layer_name] = current_int8.copy() if current_int8 is not None else None
            self.output_scales[layer_name] = current_scale

            # Multi-input: save branch state after processing
            if is_multi_input and branch_states is not None:
                branch_prefix = self._get_branch_prefix(layer_name)
                if branch_prefix and branch_prefix in branch_states:
                    branch_states[branch_prefix] = (current_fp32, current_int8, current_scale)

            if verbose:
                print()

        if verbose:
            print("="*80)
            print("[PASS] INT8 Forward Pass Complete")
            print("="*80)

        return current_fp32, self.intermediate_outputs, self.output_scales

    def _resolve_add_inputs(self, layer_name: str, input_names: List[str]) -> List[str]:
        """
        Ensure Add layers operate on two distinct upstream tensors.
        Falls back to scanning previous layers with matching shapes if metadata is ambiguous.
        """
        resolved: List[str] = []
        seen = set()
        candidate_shapes: List[tuple] = []

        # First, honor the provided metadata if those tensors exist
        for name in input_names:
            tensor = self.intermediate_outputs.get(name)
            if tensor is None or name in seen:
                continue
            resolved.append(name)
            candidate_shapes.append(tuple(tensor.shape))
            seen.add(name)
            if len(resolved) == 2:
                break

        # If two inputs were found but shapes don't match, drop the second input
        if len(resolved) == 2 and candidate_shapes[0] != candidate_shapes[1]:
            seen.discard(resolved[1])
            resolved = [resolved[0]]
            candidate_shapes = [candidate_shapes[0]]

        # Fallback: walk backwards through layers to find matching shapes
        target_shape = candidate_shapes[0] if candidate_shapes else self.layer_info.get(layer_name, {}).get('output_shape')
        target_shape = tuple(target_shape) if target_shape is not None else None
        idx = self.layer_order.index(layer_name) if layer_name in self.layer_order else None
        if idx is not None:
            for prev_name in reversed(self.layer_order[:idx]):
                if prev_name in seen:
                    continue
                tensor = self.intermediate_outputs.get(prev_name)
                if tensor is None:
                    continue
                if target_shape and tuple(tensor.shape) != target_shape:
                    continue
                resolved.append(prev_name)
                seen.add(prev_name)
                if len(resolved) == 2:
                    break

        return resolved

    def _run_attention_layer(
        self,
        layer_name: str,
        layer_info: Dict[str, Any],
        current_int8: np.ndarray,
        current_scale: float,
        verbose: bool = False,
    ):
        """
        Run MHSA layer using INT8 hybrid precision implementation.

        This uses the mhsa_int8_hybrid atomic operation which implements:
        - INT8 Q/K/V projections
        - FP32 attention scores and softmax (or i-Softmax if enabled)
        - Mixed precision context computation (FP32 x INT8)
        - INT8 output projection
        - Optional sequence pooling
        """
        if current_int8 is None:
            raise ValueError(f"Attention layer {layer_name} requires INT8 input")

        # Use atomic MHSA operation with hybrid precision
        out_int8, out_scale, out_fp32 = mhsa_int8_hybrid(
            current_int8,
            layer_info,
            current_scale,
            verbose=verbose,
            use_i_softmax=self.use_i_softmax,
            softmax_lut=self.softmax_lut,
            softmax_lut_metadata=self.softmax_lut_metadata
        )

        return out_int8, out_scale, out_fp32

    def _run_cross_attn_self_refine_layer(
        self,
        layer_name: str,
        layer_info: Dict[str, Any],
        current_int8: np.ndarray,
        current_scale: float,
        verbose: bool = False,
    ):
        """Run Cross-Attention with Self-Refinement (LUNA full block).

        Handles implicit reshapes for per-patch cross-attention:
        - Input [B, num_tokens, D] is reshaped to [B*num_patches, kv_len, D]
          so the block processes each patch independently.
        - Output [B*num_patches, num_queries, D] is reshaped to
          [B, num_patches, num_queries*D] for the downstream temporal encoder.
        """
        if current_int8 is None:
            raise ValueError(f"CrossAttentionWithSelfRefine layer {layer_name} requires INT8 input")

        kv_len = layer_info['kv_len']
        num_queries = layer_info['num_queries']
        embed_dim = layer_info['embed_dim']

        # Reshape for per-patch processing: [B, N*P, D] -> [B*P, N, D]
        input_shape = current_int8.shape
        B = input_shape[0]
        total_tokens = input_shape[1]
        num_patches = total_tokens // kv_len

        if total_tokens != kv_len:
            # Need reshape: [B, kv_len * num_patches, D] -> [B*num_patches, kv_len, D]
            x = current_int8.reshape(B, kv_len, num_patches, embed_dim)
            x = np.transpose(x, (0, 2, 1, 3))  # [B, P, N, D]
            x = x.reshape(B * num_patches, kv_len, embed_dim)
            if verbose:
                print(f"  Pre-reshape: {input_shape} -> {x.shape} (per-patch)")
        else:
            x = current_int8

        out_int8, out_scale, out_fp32 = cross_attention_with_self_refine_int8(
            x,
            layer_info,
            current_scale,
            use_i_softmax=self.use_i_softmax,
        )

        # Reshape output: [B*P, Q, D] -> [B, P, Q*D]
        if total_tokens != kv_len:
            out_int8 = out_int8.reshape(B, num_patches, num_queries * embed_dim)
            out_fp32 = out_fp32.reshape(B, num_patches, num_queries * embed_dim)
            if verbose:
                print(f"  Post-reshape: [{B*num_patches}, {num_queries}, {embed_dim}] -> {out_int8.shape}")

        return out_int8, out_scale, out_fp32

    def _run_classification_head_layer(
        self,
        layer_name: str,
        layer_info: Dict[str, Any],
        current_int8: np.ndarray,
        current_scale: float,
        verbose: bool = False,
    ):
        """Run Classification Head with MLP (LUNA classifier)."""
        if current_int8 is None:
            raise ValueError(f"ClassificationHeadWithMLP layer {layer_name} requires INT8 input")

        logits_int8, logit_scale, logits_fp32 = classification_head_with_mlp_int8(
            current_int8,
            layer_info,
            current_scale,
            use_i_softmax=self.use_i_softmax,
        )

        return logits_int8, logit_scale, logits_fp32

    def _run_cross_attention_layer(
        self,
        layer_name: str,
        layer_info: Dict[str, Any],
        current_int8: np.ndarray,
        current_scale: float,
        verbose: bool = False,
    ):
        """
        Run Cross-Attention layer using INT8 hybrid precision implementation.

        Mirrors the MHSA integer-softmax path but with learned query embeddings.
        """
        if current_int8 is None:
            raise ValueError(f"Cross-Attention layer {layer_name} requires INT8 input")

        out_int8, out_scale, out_fp32 = cross_attention_int8_hybrid(
            current_int8,
            layer_info,
            current_scale,
            verbose=verbose,
            use_i_softmax=self.use_i_softmax,
        )

        return out_int8, out_scale, out_fp32

    def _run_ssm_layer(
        self,
        layer_name: str,
        layer_info: Dict[str, Any],
        current_int8: np.ndarray,
        current_scale: float,
        verbose: bool = False,
    ):
        """
        Run SSM (State Space Model) layer using integer-only implementation.

        This implements the MAMBA SSM core with:
        - INT8 input x projection to get B, C
        - INT8 dt projection through softplus to get dt
        - Q15 discretization using exp/phi1 LUTs
        - Q15 scan (state update) loop
        - INT8 output with optional D skip connection
        """
        if current_int8 is None:
            raise ValueError(f"SSM layer {layer_name} requires INT8 input")

        d_inner = layer_info['d_inner']
        d_state = layer_info['d_state']
        dt_rank = layer_info['dt_rank']

        # Extract weights
        x_proj_weight = np.array(layer_info['x_proj_weight_int8'])
        dt_proj_weight = np.array(layer_info['dt_proj_weight_int8'])
        dt_proj_bias = layer_info.get('dt_proj_bias_fp32')
        if dt_proj_bias is not None:
            dt_proj_bias = np.array(dt_proj_bias)

        A_log = np.array(layer_info['A_log_fp32'])
        D = np.array(layer_info['D_fp32'])

        # Get scales
        scale_x_proj = layer_info['x_proj_scale_weight']
        scale_dt_proj = layer_info['dt_proj_scale_weight']
        scale_output = layer_info['scale_output']

        # Run I-Mamba integer SSM path (matches C kernel behavior)
        output_int8 = ssm_layer_forward_int8_imamba(
            x_int8=current_int8,
            x_proj_weight_int8=x_proj_weight,
            dt_proj_weight_int8=dt_proj_weight,
            dt_proj_bias_fp32=dt_proj_bias,
            A_log_fp32=A_log,
            D_fp32=D,
            scale_x=current_scale,
            scale_x_proj=scale_x_proj,
            scale_dt_proj=scale_dt_proj,
            scale_output=scale_output,
            d_inner=d_inner,
            d_state=d_state,
            dt_rank=dt_rank
        )

        current_scale = scale_output
        current_fp32 = dequantize_linear(output_int8, scale=current_scale, zero_point=0)

        if verbose:
            print(f"  SSM: d_inner={d_inner}, d_state={d_state}, dt_rank={dt_rank}")
            print(f"  Output: INT8 {output_int8.shape}, range=[{output_int8.min()}, {output_int8.max()}]")

        return output_int8, current_scale, current_fp32

    def _run_mamba_block(
        self,
        layer_name: str,
        layer_info: Dict[str, Any],
        current_int8: np.ndarray,
        current_scale: float,
        verbose: bool = False,
    ):
        """
        Run complete MambaBlock layer.

        A MambaBlock implements:
        1. in_proj: Linear to 2*d_inner (split into x and z branches)
        2. conv1d: Depthwise 1D conv on x branch (causal)
        3. silu: SiLU activation after conv1d
        4. ssm: State Space Model core on x branch
        5. gate: y = ssm_output * silu(z)
        6. out_proj: Linear back to d_model
        """
        if current_int8 is None:
            raise ValueError(f"MambaBlock {layer_name} requires INT8 input")

        if verbose:
            print(f"  MambaBlock: d_model={layer_info['d_model']}, "
                  f"d_inner={layer_info['d_inner']}, d_state={layer_info['d_state']}")

        d_model = layer_info['d_model']
        d_inner = layer_info['d_inner']
        d_state = layer_info['d_state']
        dt_rank = layer_info['dt_rank']
        kernel_size = layer_info['kernel_size']

        # --- Step 1: in_proj (d_model -> 2*d_inner) ---
        in_proj_weight = np.array(layer_info['in_proj_weight_int8'])
        in_proj_bias = layer_info.get('in_proj_bias_fp32')
        if in_proj_bias is not None:
            in_proj_bias = np.array(in_proj_bias)

        scale_in_proj = layer_info['in_proj_scale_weight']
        scale_in_proj_out = layer_info['in_proj_scale_output']

        # Reshape input: [B, L, D] -> [B*L, D] for linear
        original_shape = current_int8.shape
        batch_seq = current_int8.shape[:-1]
        x_flat = current_int8.reshape(-1, d_model)

        # INT8 linear projection (match C: bias in INT32, float32 scaling)
        proj_int32 = x_flat.astype(np.int32) @ in_proj_weight.T.astype(np.int32)
        if in_proj_bias is not None:
            bias_int32 = np.round(in_proj_bias / (current_scale * scale_in_proj)).astype(np.int32)
            proj_int32 += bias_int32
        combined_scale = np.float32(current_scale * scale_in_proj / scale_in_proj_out)
        proj_fp32 = proj_int32.astype(np.float32) * combined_scale
        proj_int8 = np.clip(np.round(proj_fp32), -128, 127).astype(np.int8)
        proj_int8 = proj_int8.reshape(*batch_seq, 2 * d_inner)

        if verbose:
            print(f"    in_proj: {original_shape} -> {proj_int8.shape}")

        # Split into x and z branches
        x_branch = proj_int8[..., :d_inner]  # [B, L, d_inner]
        z_branch = proj_int8[..., d_inner:]  # [B, L, d_inner]

        # --- Step 2: conv1d on x branch ---
        # Reshape: [B, L, D] -> [B, D, L] for conv1d
        x_conv_input = np.transpose(x_branch, (0, 2, 1))  # [B, d_inner, L]

        conv1d_weight = np.array(layer_info['conv1d_weight_int8'])
        conv1d_bias = layer_info.get('conv1d_bias_fp32')
        if conv1d_bias is not None:
            conv1d_bias = np.array(conv1d_bias)
            scale_bias = scale_in_proj_out * layer_info['conv1d_scale_weight']
            conv1d_bias_int32 = np.round(conv1d_bias / scale_bias).astype(np.int32)
        else:
            conv1d_bias_int32 = None

        x_conv_out = conv1d_depthwise_int8_fixedpoint(
            x_conv_input,
            conv1d_weight,
            conv1d_bias_int32,
            scale_x=scale_in_proj_out,
            scale_w=layer_info['conv1d_scale_weight'],
            scale_y=layer_info['conv1d_scale_output'],
            causal=True
        )
        scale_after_conv = layer_info['conv1d_scale_output']

        if verbose:
            print(f"    conv1d: {x_conv_input.shape} -> {x_conv_out.shape}")

        # --- Step 3: SiLU activation ---
        # QuantSiLU only has output_quant, input uses conv1d output scale
        silu_scale_in = layer_info.get('silu_scale_input', scale_after_conv)
        silu_scale_out = layer_info['silu_scale_output']

        # Requantize if needed for SiLU input (usually not needed since QuantSiLU has no input_quant)
        if abs(scale_after_conv - silu_scale_in) > 1e-6:
            x_conv_out = requantize_int8(x_conv_out, scale_in=scale_after_conv, scale_out=silu_scale_in)

        silu_lut = generate_silu_lut_int8(silu_scale_in, silu_scale_out)
        x_silu = silu_lut_int8(x_conv_out, silu_lut)
        scale_after_silu = silu_scale_out

        # Reshape back: [B, d_inner, L] -> [B, L, d_inner]
        x_silu = np.transpose(x_silu, (0, 2, 1))

        if verbose:
            print(f"    silu: range=[{x_silu.min()}, {x_silu.max()}]")

        # --- Step 4: SSM core ---
        ssm_x_proj_weight = np.array(layer_info['ssm_x_proj_weight_int8'])
        ssm_dt_proj_weight = np.array(layer_info['ssm_dt_proj_weight_int8'])
        ssm_dt_proj_bias = layer_info.get('ssm_dt_proj_bias_fp32')
        if ssm_dt_proj_bias is not None:
            ssm_dt_proj_bias = np.array(ssm_dt_proj_bias)

        A_log = np.array(layer_info['ssm_A_log_fp32'])
        D = np.array(layer_info['ssm_D_fp32'])

        ssm_output = ssm_layer_forward_int8_imamba(
            x_int8=x_silu,
            x_proj_weight_int8=ssm_x_proj_weight,
            dt_proj_weight_int8=ssm_dt_proj_weight,
            dt_proj_bias_fp32=ssm_dt_proj_bias,
            A_log_fp32=A_log,
            D_fp32=D,
            scale_x=scale_after_silu,
            scale_x_proj=layer_info['ssm_x_proj_scale_weight'],
            scale_dt_proj=layer_info['ssm_dt_proj_scale_weight'],
            scale_output=layer_info['ssm_scale_output'],
            d_inner=d_inner,
            d_state=d_state,
            dt_rank=dt_rank
        )
        scale_after_ssm = layer_info['ssm_scale_output']

        if verbose:
            print(f"    ssm: {x_silu.shape} -> {ssm_output.shape}")

        # --- Step 5: Gate with z branch: y = ssm_output * silu(z) ---
        # Generate Q13 SiLU LUT for gating
        z_scale = scale_in_proj_out
        silu_gate_lut = generate_silu_gate_lut_q13(z_scale)

        # Apply gating: y = ssm_output * silu_q13(z) >> 13
        z_flat = z_branch.reshape(-1)
        ssm_flat = ssm_output.reshape(-1)

        # SiLU lookup on z -> Q13 result
        z_indices = z_flat.astype(np.int32) + 128
        silu_z_q13 = silu_gate_lut[z_indices]

        # Multiply and shift
        gated_i32 = ssm_flat.astype(np.int32) * silu_z_q13.astype(np.int32)
        gated_i32 = (gated_i32 + (1 << 12)) >> 13  # Round and shift

        # Clip to INT8
        gated_int8 = np.clip(gated_i32, -128, 127).astype(np.int8)
        gated_int8 = gated_int8.reshape(ssm_output.shape)

        if verbose:
            print(f"    gate: range=[{gated_int8.min()}, {gated_int8.max()}]")

        # --- Step 6: out_proj (d_inner -> d_model) ---
        out_proj_weight = np.array(layer_info['out_proj_weight_int8'])
        out_proj_bias = layer_info.get('out_proj_bias_fp32')
        if out_proj_bias is not None:
            out_proj_bias = np.array(out_proj_bias)

        scale_out_proj = layer_info['out_proj_scale_weight']
        scale_output = layer_info['scale_output']

        # Reshape: [B, L, d_inner] -> [B*L, d_inner]
        gated_flat = gated_int8.reshape(-1, d_inner)

        # INT8 linear projection
        out_int32 = gated_flat.astype(np.int32) @ out_proj_weight.T.astype(np.int32)
        if out_proj_bias is not None:
            bias_int32 = np.round(out_proj_bias / (scale_after_ssm * scale_out_proj)).astype(np.int32)
            out_int32 += bias_int32
        combined_scale = np.float32(scale_after_ssm * scale_out_proj / scale_output)
        out_fp32 = out_int32.astype(np.float32) * combined_scale
        output_int8 = np.clip(np.round(out_fp32), -128, 127).astype(np.int8)
        output_int8 = output_int8.reshape(*batch_seq, d_model)
        output_fp32 = dequantize_linear(output_int8, scale=scale_output, zero_point=0)

        if verbose:
            print(f"    out_proj: {gated_int8.shape} -> {output_int8.shape}")
            print(f"  Output: INT8 {output_int8.shape}, range=[{output_int8.min()}, {output_int8.max()}]")

        return output_int8, scale_output, output_fp32

    def _run_mamba_wrapper(
        self,
        layer_name: str,
        layer_info: Dict[str, Any],
        current_int8: np.ndarray,
        current_scale: float,
        verbose: bool = False,
    ):
        """
        Run Bidirectional MambaWrapper layer.

        Implements:
        1. Forward MambaBlock on input
        2. Flip input sequence
        3. Reverse MambaBlock on flipped input
        4. Flip reverse output back
        5. Add forward + reverse outputs with scale equalization
        """
        if current_int8 is None:
            raise ValueError(f"MambaWrapper {layer_name} requires INT8 input")

        if verbose:
            print(f"  MambaWrapper: d_model={layer_info['fwd_d_model']}, "
                  f"d_inner={layer_info['fwd_d_inner']}, d_state={layer_info['fwd_d_state']}")
            print(f"    Strategy: {layer_info.get('bidirectional_strategy', 'add')}")

        # Extract direction-specific parameters
        d_model = layer_info['fwd_d_model']
        d_inner = layer_info['fwd_d_inner']
        d_state = layer_info['fwd_d_state']
        dt_rank = layer_info['fwd_dt_rank']
        kernel_size = layer_info['fwd_kernel_size']

        # Forward direction
        fwd_info = self._extract_mamba_direction_params(layer_info, 'fwd_', d_model, d_inner, d_state, dt_rank, kernel_size)
        fwd_out_int8, fwd_scale, _ = self._run_mamba_block(
            f"{layer_name}_fwd", fwd_info, current_int8, current_scale, verbose=verbose
        )

        if verbose:
            print(f"    Forward output: range=[{fwd_out_int8.min()}, {fwd_out_int8.max()}]")

        # Flip input for reverse direction
        # Flip along sequence dimension (dim=1 for [B, L, D])
        flipped_input = np.flip(current_int8, axis=1).copy()

        if verbose:
            print(f"    Flipped input for reverse direction")

        # Reverse direction
        rev_info = self._extract_mamba_direction_params(layer_info, 'rev_', d_model, d_inner, d_state, dt_rank, kernel_size)
        rev_out_raw, rev_scale, _ = self._run_mamba_block(
            f"{layer_name}_rev", rev_info, flipped_input, current_scale, verbose=verbose
        )

        # Flip reverse output back
        rev_out_int8 = np.flip(rev_out_raw, axis=1).copy()

        if verbose:
            print(f"    Reverse output (flipped back): range=[{rev_out_int8.min()}, {rev_out_int8.max()}]")

        # Combine forward and reverse outputs
        strategy = layer_info.get('bidirectional_strategy', 'add')
        scale_output = layer_info['scale_output']
        # FEMBA-style architectures use residual connections: output = input + fwd + rev
        use_residual = layer_info.get('use_residual', True)

        if strategy == 'add':
            # Add forward and reverse outputs (and residual if enabled)
            # Match C: saturating INT8 add when scales already match
            if abs(fwd_scale - rev_scale) < 1e-6 and abs(fwd_scale - scale_output) < 1e-6:
                summed = fwd_out_int8.astype(np.int32) + rev_out_int8.astype(np.int32)
                if use_residual and abs(current_scale - scale_output) < 1e-6:
                    # Add input residual when scales match
                    summed = summed + current_int8.astype(np.int32)
                output_int8 = np.clip(summed, -128, 127).astype(np.int8)
            else:
                combined_fp32 = fwd_out_int8.astype(np.float32) * fwd_scale + \
                                rev_out_int8.astype(np.float32) * rev_scale
                if use_residual:
                    # Add input residual with proper scale conversion
                    combined_fp32 = combined_fp32 + current_int8.astype(np.float32) * current_scale
                output_int8 = quantize_linear(combined_fp32, scale=scale_output)

            if verbose and use_residual:
                print(f"    Residual added from input")
        elif strategy == 'concat':
            # Concatenate along last dimension
            output_int8 = np.concatenate([fwd_out_int8, rev_out_int8], axis=-1)
        else:
            raise ValueError(f"Unknown bidirectional strategy: {strategy}")

        output_fp32 = dequantize_linear(output_int8, scale=scale_output, zero_point=0)

        if verbose:
            print(f"  Combined output: INT8 {output_int8.shape}, range=[{output_int8.min()}, {output_int8.max()}]")

        return output_int8, scale_output, output_fp32


    def _extract_mamba_direction_params(
        self,
        layer_info: Dict[str, Any],
        prefix: str,
        d_model: int,
        d_inner: int,
        d_state: int,
        dt_rank: int,
        kernel_size: int
    ) -> Dict[str, Any]:
        """Extract direction-specific MambaBlock parameters from MambaWrapper layer_info."""
        return {
            'd_model': d_model,
            'd_inner': d_inner,
            'd_state': d_state,
            'dt_rank': dt_rank,
            'kernel_size': kernel_size,
            'in_proj_weight_int8': layer_info[f'{prefix}in_proj_weight_int8'],
            'in_proj_bias_fp32': layer_info.get(f'{prefix}in_proj_bias_fp32'),
            'in_proj_scale_weight': layer_info[f'{prefix}in_proj_scale_weight'],
            'in_proj_scale_output': layer_info[f'{prefix}in_proj_scale_output'],
            'conv1d_weight_int8': layer_info[f'{prefix}conv1d_weight_int8'],
            'conv1d_bias_fp32': layer_info.get(f'{prefix}conv1d_bias_fp32'),
            'conv1d_scale_weight': layer_info[f'{prefix}conv1d_scale_weight'],
            'conv1d_scale_output': layer_info[f'{prefix}conv1d_scale_output'],
            'silu_scale_output': layer_info[f'{prefix}silu_scale_output'],
            'ssm_x_proj_weight_int8': layer_info[f'{prefix}ssm_x_proj_weight_int8'],
            'ssm_x_proj_bias_fp32': layer_info.get(f'{prefix}ssm_x_proj_bias_fp32'),
            'ssm_x_proj_scale_weight': layer_info[f'{prefix}ssm_x_proj_scale_weight'],
            'ssm_dt_proj_weight_int8': layer_info[f'{prefix}ssm_dt_proj_weight_int8'],
            'ssm_dt_proj_bias_fp32': layer_info.get(f'{prefix}ssm_dt_proj_bias_fp32'),
            'ssm_dt_proj_scale_weight': layer_info[f'{prefix}ssm_dt_proj_scale_weight'],
            'ssm_A_log_fp32': layer_info[f'{prefix}ssm_A_log_fp32'],
            'ssm_D_fp32': layer_info[f'{prefix}ssm_D_fp32'],
            'ssm_scale_output': layer_info[f'{prefix}ssm_scale_output'],
            'out_proj_weight_int8': layer_info[f'{prefix}out_proj_weight_int8'],
            'out_proj_bias_fp32': layer_info.get(f'{prefix}out_proj_bias_fp32'),
            'out_proj_scale_weight': layer_info[f'{prefix}out_proj_scale_weight'],
            'scale_output': layer_info[f'{prefix}scale_output'],
        }

    def _run_patch_embed(
        self,
        layer_name: str,
        layer_info: Dict[str, Any],
        input_int8: np.ndarray,
        input_scale: float,
        verbose: bool = False,
    ):
        """
        Run PatchEmbed layer (Conv2D + reshape + permute).

        Args:
            layer_name: Name of the layer
            layer_info: Layer parameters from network_info
            input_int8: INT8 input tensor [B, C, H, W]
            input_scale: Input quantization scale
            verbose: Print debug info

        Returns:
            Tuple of (output_int8, output_scale, output_fp32)
            Output shape: [B, seq_len, d_model]
        """
        if verbose:
            print(f"  PatchEmbed: {layer_name}")
            print(f"    Input shape: {input_int8.shape}")

        # Get parameters
        inp_size = layer_info['inp_size']
        patch_size = layer_info['patch_size']
        stride = layer_info['stride']
        # Support both scalar and tuple stride
        if isinstance(stride, (list, tuple)):
            stride_h, stride_w = stride[0], stride[1]
        else:
            stride_h = stride_w = stride
        embed_dim = layer_info['embed_dim']
        grid_h = layer_info['grid_h']
        grid_w = layer_info['grid_w']
        seq_len = layer_info['seq_len']
        d_model = layer_info['d_model']

        # Get weights (already loaded as INT8)
        weight_int8 = layer_info['proj_weight_int8']  # [embed_dim, in_chans, kH, kW]
        weight_scale = layer_info['proj_weight_scale']
        bias = layer_info.get('proj_bias_fp32')  # FP32 bias

        scale_output = layer_info['scale_output']

        if verbose:
            print(f"    Weight shape: {weight_int8.shape}")
            print(f"    Grid: {grid_h} x {grid_w}")
            print(f"    Output: [{seq_len}, {d_model}]")

        # Convert input to FP32 for Conv2D
        input_fp32 = input_int8.astype(np.float32) * input_scale

        # Apply Conv2D using scipy or numpy
        B, C, H, W = input_fp32.shape
        out_channels = weight_int8.shape[0]
        kH, kW = weight_int8.shape[2], weight_int8.shape[3]

        # Convert weight to FP32
        weight_fp32 = weight_int8.astype(np.float32) * weight_scale

        # Compute Conv2D output: [B, embed_dim, grid_h, grid_w]
        out_h = (H - kH) // stride_h + 1
        out_w = (W - kW) // stride_w + 1
        conv_out = np.zeros((B, out_channels, out_h, out_w), dtype=np.float32)

        for b in range(B):
            for oc in range(out_channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        ih_start = oh * stride_h
                        iw_start = ow * stride_w
                        patch = input_fp32[b, :, ih_start:ih_start+kH, iw_start:iw_start+kW]
                        conv_out[b, oc, oh, ow] = np.sum(patch * weight_fp32[oc])

        # Add bias
        if bias is not None:
            conv_out += bias.reshape(1, -1, 1, 1)

        if verbose:
            print(f"    Conv output shape: {conv_out.shape}")
            print(f"    Conv output range: [{conv_out.min():.4f}, {conv_out.max():.4f}]")

        # Reshape: [B, embed_dim, grid_h, grid_w] -> [B, embed_dim * grid_h, grid_w]
        reshaped = conv_out.reshape(B, embed_dim * grid_h, grid_w)

        # Permute: [B, embed_dim * grid_h, grid_w] -> [B, grid_w, embed_dim * grid_h]
        # This gives [B, seq_len, d_model]
        output_fp32 = np.transpose(reshaped, (0, 2, 1))

        if verbose:
            print(f"    Output shape: {output_fp32.shape}")
            print(f"    Output range: [{output_fp32.min():.4f}, {output_fp32.max():.4f}]")

        # Quantize output
        output_int8 = np.clip(np.round(output_fp32 / scale_output), -128, 127).astype(np.int8)

        return output_int8, scale_output, output_fp32


def test_int8_inference():
    """Test INT8 inference engine."""
    print("="*80)
    print("Testing INT8 Inference Engine")
    print("="*80)

    # Load network info
    import json
    network_info_path = Path("golden_outputs/network_info.json")

    if not network_info_path.exists():
        print(f"[FAIL] Network info not found: {network_info_path}")
        print("   Please run tools/pytorch_extractor.py first!")
        return

    with open(network_info_path) as f:
        network_info = json.load(f)

    # Load weights back from saved files
    weights_dir = Path("golden_outputs/weights")
    for layer_name, layer_data in network_info.items():
        if 'weight_int8' in layer_data:
            weight_path = weights_dir / f"{layer_name}_weight_int8.npy"
            if weight_path.exists():
                layer_data['weight_int8'] = np.load(weight_path)

        if layer_data.get('type') == 'MultiheadSelfAttention':
            for prefix in ('q', 'k', 'v', 'out'):
                weight_key = f"{prefix}_weight_int8"
                bias_key = f"{prefix}_bias_fp32"
                weight_path = weights_dir / f"{layer_name}_{prefix}_weight_int8.npy"
                bias_path = weights_dir / f"{layer_name}_{prefix}_bias_fp32.npy"
                if weight_path.exists():
                    layer_data[weight_key] = np.load(weight_path)
                if bias_path.exists():
                    layer_data[bias_key] = np.load(bias_path)

        # Load SSM parameters
        if layer_data.get('type') == 'SSM':
            ssm_params = [
                ('x_proj_weight_int8', 'x_proj_weight_int8'),
                ('x_proj_bias_fp32', 'x_proj_bias_fp32'),
                ('dt_proj_weight_int8', 'dt_proj_weight_int8'),
                ('dt_proj_bias_fp32', 'dt_proj_bias_fp32'),
                ('A_log_fp32', 'A_log_fp32'),
                ('D_fp32', 'D_fp32'),
            ]
            for key, filename in ssm_params:
                param_path = weights_dir / f"{layer_name}_{filename}.npy"
                if param_path.exists():
                    layer_data[key] = np.load(param_path)

        # Load MambaBlock parameters
        if layer_data.get('type') == 'MambaBlock':
            mamba_params = [
                ('in_proj_weight_int8', 'in_proj_weight_int8'),
                ('in_proj_bias_fp32', 'in_proj_bias_fp32'),
                ('conv1d_weight_int8', 'conv1d_weight_int8'),
                ('conv1d_bias_fp32', 'conv1d_bias_fp32'),
                ('ssm_x_proj_weight_int8', 'ssm_x_proj_weight_int8'),
                ('ssm_x_proj_bias_fp32', 'ssm_x_proj_bias_fp32'),
                ('ssm_dt_proj_weight_int8', 'ssm_dt_proj_weight_int8'),
                ('ssm_dt_proj_bias_fp32', 'ssm_dt_proj_bias_fp32'),
                ('ssm_A_log_fp32', 'ssm_A_log_fp32'),
                ('ssm_D_fp32', 'ssm_D_fp32'),
                ('out_proj_weight_int8', 'out_proj_weight_int8'),
                ('out_proj_bias_fp32', 'out_proj_bias_fp32'),
            ]
            for key, filename in mamba_params:
                param_path = weights_dir / f"{layer_name}_{filename}.npy"
                if param_path.exists():
                    layer_data[key] = np.load(param_path)

        if 'bias_fp32' in layer_data:
            bias_path = weights_dir / f"{layer_name}_bias_fp32.npy"
            if bias_path.exists():
                layer_data['bias_fp32'] = np.load(bias_path)

    print(f"[PASS] Loaded network info with {len(network_info)} layers")
    print()

    # Create inference engine
    engine = INT8InferenceEngine(network_info)

    # Create test input
    batch_size = 1
    x_fp32 = np.random.randn(batch_size, 1, 28, 28).astype(np.float32) * 0.5

    print(f"Input: {x_fp32.shape}, range=[{x_fp32.min():.3f}, {x_fp32.max():.3f}]")
    print()

    # Run inference
    output_fp32, intermediate_outputs, output_scales = engine.forward(x_fp32, verbose=True)

    # Print results
    print("\nFinal Output:")
    print(f"  Logits shape: {output_fp32.shape}")
    print(f"  Logits: {output_fp32[0]}")
    print(f"  Predicted class: {np.argmax(output_fp32[0])}")

    print("\n" + "="*80)
    print("Intermediate INT8 Outputs Summary:")
    print("="*80)
    for layer_name, int8_output in intermediate_outputs.items():
        if int8_output is not None:
            print(f"  {layer_name:20s}: shape={str(int8_output.shape):20s} "
                  f"range=[{int8_output.min():4d}, {int8_output.max():4d}]")

    print("\n[PASS] INT8 Inference Engine test complete!")


# --- Autoregressive INT8 Inference Engine for LLM text generation ---

# Import KV cache and autoregressive MHSA
try:
    from atomic_ops.kv_cache import KVCache
    from atomic_ops.mhsa import mhsa_autoregressive_step
except ImportError:
    KVCache = None
    mhsa_autoregressive_step = None


class AutoregressiveINT8Engine:
    """
    INT8 autoregressive inference engine for Llama-style text generation.

    Runs token-by-token generation using INT8 atomic operations with a KV cache.
    Each step: embedding -> [N transformer blocks] -> norm -> logits -> argmax.

    This produces deterministic golden token sequences for GAP9 verification.
    """

    def __init__(self, model_config: dict, weights: dict, scales: dict):
        """
        Initialize autoregressive engine.

        Args:
            model_config: Model architecture parameters (dim, n_layers, n_heads, etc.)
            weights: Dictionary of INT8 weights and FP32 norm weights
            scales: Dictionary of quantization scales per layer
        """
        if KVCache is None:
            raise ImportError("KVCache not available - install atomic_ops")

        self.dim = model_config['dim']
        self.n_layers = model_config['n_layers']
        self.n_heads = model_config['n_heads']
        self.n_kv_heads = model_config['n_kv_heads']
        self.head_dim = model_config['head_dim']
        self.hidden_dim = model_config['hidden_dim']
        self.vocab_size = model_config['vocab_size']
        self.max_seq_len = model_config['max_seq_len']
        self.norm_eps = model_config.get('norm_eps', 1e-5)

        self.weights = weights
        self.scales = scales

        # Create KV cache
        self.kv_cache = KVCache(
            n_layers=self.n_layers,
            max_seq_len=self.max_seq_len,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
        )

        # Precompute RoPE frequencies
        self._precompute_rope()

    def _precompute_rope(self, theta: float = 10000.0):
        """Precompute RoPE cos/sin tables as Q15 fixed point."""
        dim = self.head_dim
        freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(np.float32) / dim))
        t = np.arange(self.max_seq_len, dtype=np.float32)
        freqs = np.outer(t, freqs)
        self.rope_cos_fp32 = np.cos(freqs).astype(np.float32)
        self.rope_sin_fp32 = np.sin(freqs).astype(np.float32)
        # Q15 format for INT8 RoPE
        self.rope_cos_q15 = np.round(self.rope_cos_fp32 * 32767).astype(np.int16)
        self.rope_sin_q15 = np.round(self.rope_sin_fp32 * 32767).astype(np.int16)

    def _embedding_lookup(self, token_id: int) -> tuple:
        """Look up embedding for a single token and return (int8_vec, scale)."""
        embed_weight = self.weights['tok_embeddings']  # [vocab, dim] INT8
        embed_scale = self.scales['tok_embeddings']

        vec_int8 = embed_weight[token_id].reshape(1, self.dim).astype(np.int8)
        return vec_int8, embed_scale

    def _rmsnorm(self, x_int8: np.ndarray, scale_in: float, layer_name: str) -> tuple:
        """Apply RMSNorm and return (output_int8, scale_out)."""
        weight_fp32 = self.weights[f'{layer_name}_weight']
        scale_out = self.scales.get(f'{layer_name}_output', scale_in)

        out = rmsnorm_int8(
            x_int8, weight_fp32, scale_in, scale_out,
            normalized_shape=self.dim, eps=self.norm_eps
        )
        return out, scale_out

    def _mhsa_autoregressive(self, x_int8: np.ndarray, scale_in: float,
                              layer_idx: int) -> tuple:
        """Run single-token MHSA with KV cache update."""
        prefix = f'layer{layer_idx}'

        layer_info = {
            'embed_dim': self.dim,
            'num_heads': self.n_heads,
            'n_kv_heads': self.n_kv_heads,
            'head_dim': self.head_dim,
            'q_weight_int8': self.weights[f'{prefix}_wq'],
            'k_weight_int8': self.weights[f'{prefix}_wk'],
            'v_weight_int8': self.weights[f'{prefix}_wv'],
            'out_weight_int8': self.weights[f'{prefix}_wo'],
            'q_bias_fp32': None,
            'k_bias_fp32': None,
            'v_bias_fp32': None,
            'out_bias_fp32': None,
            'q_scale_weight': self.scales[f'{prefix}_wq'],
            'k_scale_weight': self.scales[f'{prefix}_wk'],
            'v_scale_weight': self.scales[f'{prefix}_wv'],
            'out_scale_weight': self.scales[f'{prefix}_wo'],
            'q_scale_output': self.scales.get(f'{prefix}_q_output', 0.01),
            'k_scale_output': self.scales.get(f'{prefix}_k_output', 0.01),
            'v_scale_output': self.scales.get(f'{prefix}_v_output', 0.01),
            'scale_output': self.scales.get(f'{prefix}_attn_output', scale_in),
            'softmax_scale': 1.0 / np.sqrt(self.head_dim),
            'use_rope': True,
            'rope_cos_q15': self.rope_cos_q15,
            'rope_sin_q15': self.rope_sin_q15,
        }

        out_int8, scale_out, _ = mhsa_autoregressive_step(
            x_int8, layer_info, self.kv_cache, layer_idx, scale_in
        )
        return out_int8, scale_out

    def _swiglu_ffn(self, x_int8: np.ndarray, scale_in: float,
                     layer_idx: int) -> tuple:
        """Run SwiGLU FFN and return (output_int8, scale_out)."""
        prefix = f'layer{layer_idx}'
        scale_hidden = self.scales.get(f'{prefix}_hidden', 0.01)
        scale_out = self.scales.get(f'{prefix}_ffn_output', scale_in)

        out = swiglu_ffn_int8(
            x_int8,
            self.weights[f'{prefix}_w1'],
            self.weights[f'{prefix}_w3'],
            self.weights[f'{prefix}_w2'],
            scale_in,
            self.scales[f'{prefix}_w1'],
            self.scales[f'{prefix}_w3'],
            self.scales[f'{prefix}_w2'],
            scale_hidden,
            scale_out,
        )
        return out, scale_out

    def _residual_add(self, x_int8: np.ndarray, residual_int8: np.ndarray,
                       scale_x: float, scale_res: float) -> tuple:
        """Add residual connection and return (output_int8, scale_out)."""
        # Dequantize both to FP32, add, requantize
        x_fp32 = dequantize_linear(x_int8, scale_x)
        res_fp32 = dequantize_linear(residual_int8, scale_res)
        sum_fp32 = x_fp32 + res_fp32

        # Use the larger scale for output
        scale_out = max(scale_x, scale_res)
        max_val = np.max(np.abs(sum_fp32))
        if max_val > 0:
            scale_out = max_val / 127.0

        out_int8 = quantize_linear(sum_fp32, scale_out)
        return out_int8, scale_out

    def _compute_logits_fp32(self, x_int8: np.ndarray, scale_in: float) -> np.ndarray:
        """Compute FP32 logits using the classifier/output weight."""
        classifier_weight = self.weights['output']  # [vocab_size, dim] INT8
        classifier_scale = self.scales['output']

        # Dequantize for FP32 matmul (logits stay in FP32 for argmax)
        x_fp32 = dequantize_linear(x_int8, scale_in).reshape(1, -1)
        w_fp32 = dequantize_linear(classifier_weight, classifier_scale)

        logits = x_fp32 @ w_fp32.T  # [1, vocab_size]
        return logits.squeeze(0)

    def generate(self, prompt_tokens: list, max_new_tokens: int = 50,
                 verbose: bool = False) -> tuple:
        """
        Run autoregressive INT8 text generation.

        Processes prompt tokens one at a time (no prefill optimization for
        simplicity - matches the GAP9 implementation exactly).

        Args:
            prompt_tokens: List of integer token IDs (including BOS)
            max_new_tokens: Maximum number of new tokens to generate
            verbose: Print debug info per step

        Returns:
            Tuple of (all_tokens, intermediates) where:
                - all_tokens: List of all token IDs (prompt + generated)
                - intermediates: Dict of per-step debug info
        """
        self.kv_cache.reset()

        all_tokens = list(prompt_tokens)
        intermediates = {}
        total_steps = len(prompt_tokens) + max_new_tokens - 1

        for pos in range(total_steps):
            if pos < len(prompt_tokens):
                token = prompt_tokens[pos]
            else:
                token = all_tokens[-1]

            if verbose:
                print(f"\n--- Step {pos}: token={token} ---")

            # 1. Embedding lookup
            x_int8, scale_x = self._embedding_lookup(token)

            # 2. Transformer blocks
            for layer_idx in range(self.n_layers):
                residual_int8 = x_int8.copy()
                scale_residual = scale_x

                # Attention pre-norm
                normed, scale_normed = self._rmsnorm(
                    x_int8, scale_x, f'layer{layer_idx}_attention_norm'
                )

                # Autoregressive MHSA with KV cache
                attn_out, scale_attn = self._mhsa_autoregressive(
                    normed, scale_normed, layer_idx
                )

                # Residual add
                x_int8, scale_x = self._residual_add(
                    attn_out, residual_int8, scale_attn, scale_residual
                )

                # FFN pre-norm
                residual_int8 = x_int8.copy()
                scale_residual = scale_x

                normed, scale_normed = self._rmsnorm(
                    x_int8, scale_x, f'layer{layer_idx}_ffn_norm'
                )

                # SwiGLU FFN
                ffn_out, scale_ffn = self._swiglu_ffn(
                    normed, scale_normed, layer_idx
                )

                # Residual add
                x_int8, scale_x = self._residual_add(
                    ffn_out, residual_int8, scale_ffn, scale_residual
                )

            # 3. Advance KV cache position (after all layers processed this token)
            self.kv_cache.advance()

            # 4. Final norm + logits (only needed for sampling)
            if pos >= len(prompt_tokens) - 1:
                normed, scale_normed = self._rmsnorm(
                    x_int8, scale_x, 'final_norm'
                )

                logits = self._compute_logits_fp32(normed, scale_normed)

                # 5. Greedy argmax sampling
                next_token = int(np.argmax(logits))

                if verbose:
                    top5_idx = np.argsort(logits)[-5:][::-1]
                    print(f"  Top-5 logits: {[(int(i), f'{logits[i]:.3f}') for i in top5_idx]}")
                    print(f"  Next token: {next_token}")

                if pos >= len(prompt_tokens) - 1:
                    if pos == len(prompt_tokens) - 1:
                        # First generated token after prompt
                        pass
                    all_tokens.append(next_token)

                    # Store intermediate for debugging
                    intermediates[pos] = {
                        'x_int8': x_int8.copy(),
                        'scale_x': scale_x,
                        'logits_top5': [(int(i), float(logits[i]))
                                        for i in np.argsort(logits)[-5:][::-1]],
                        'next_token': next_token,
                    }

                    # Stop on EOS
                    if next_token == 2:
                        break

        return all_tokens, intermediates


def extract_int8_weights_from_brevitas(model) -> tuple:
    """
    Extract INT8 weights and scales from a Brevitas-quantized Llama model.

    Args:
        model: A TinyStoriesLlama (Brevitas) model instance

    Returns:
        Tuple of (weights_dict, scales_dict) for AutoregressiveINT8Engine
    """
    import torch

    weights = {}
    scales = {}

    # Embedding table (quantize FP32 to INT8)
    embed_fp32 = model.tok_embeddings.weight.detach().numpy()
    embed_scale = np.max(np.abs(embed_fp32)) / 127.0
    weights['tok_embeddings'] = np.clip(
        np.round(embed_fp32 / embed_scale), -128, 127
    ).astype(np.int8)
    scales['tok_embeddings'] = float(embed_scale)

    # Per-layer weights
    for i, layer in enumerate(model.layers):
        prefix = f'layer{i}'

        # Attention weights (extract INT8 from Brevitas)
        for proj_name, proj_module in [
            ('wq', layer.attention.wq), ('wk', layer.attention.wk),
            ('wv', layer.attention.wv), ('wo', layer.attention.wo),
        ]:
            qw = proj_module.quant_weight()
            weights[f'{prefix}_{proj_name}'] = qw.int().detach().numpy().astype(np.int8)
            scales[f'{prefix}_{proj_name}'] = float(qw.scale.item())

        # Activation scales from QuantIdentity modules
        # Use a test forward pass to get calibrated scales
        for quant_name, quant_module in [
            ('q_output', layer.attention.q_quant),
            ('k_output', layer.attention.k_quant),
            ('v_output', layer.attention.v_quant),
            ('attn_output', layer.attention.out_quant),
        ]:
            try:
                test_in = torch.randn(1, 1, model.dim)
                test_out = quant_module(test_in)
                if hasattr(test_out, 'scale'):
                    scales[f'{prefix}_{quant_name}'] = float(test_out.scale.item())
            except Exception:
                scales[f'{prefix}_{quant_name}'] = 0.01

        # Norm weights (FP32)
        weights[f'{prefix}_attention_norm_weight'] = \
            layer.attention_norm.weight.detach().numpy().astype(np.float32)
        weights[f'{prefix}_ffn_norm_weight'] = \
            layer.ffn_norm.weight.detach().numpy().astype(np.float32)

        # Norm output scales
        for quant_name, quant_module in [
            ('attention_norm_output', layer.attn_norm_quant),
            ('ffn_norm_output', layer.ffn_norm_quant),
        ]:
            try:
                test_in = torch.randn(1, 1, model.dim)
                test_out = quant_module(test_in)
                if hasattr(test_out, 'scale'):
                    scales[f'{prefix}_{quant_name}'] = float(test_out.scale.item())
            except Exception:
                scales[f'{prefix}_{quant_name}'] = 0.01

        # FFN weights
        for w_name, w_module in [
            ('w1', layer.feed_forward.w1), ('w3', layer.feed_forward.w3),
            ('w2', layer.feed_forward.w2),
        ]:
            qw = w_module.quant_weight()
            weights[f'{prefix}_{w_name}'] = qw.int().detach().numpy().astype(np.int8)
            scales[f'{prefix}_{w_name}'] = float(qw.scale.item())

        # FFN hidden scale
        try:
            test_in = torch.randn(1, 1, model.hidden_dim)
            test_out = layer.feed_forward.gate_quant(test_in)
            if hasattr(test_out, 'scale'):
                scales[f'{prefix}_hidden'] = float(test_out.scale.item())
        except Exception:
            scales[f'{prefix}_hidden'] = 0.01

        # FFN output scale
        try:
            test_in = torch.randn(1, 1, model.dim)
            test_out = layer.feed_forward.out_quant(test_in)
            if hasattr(test_out, 'scale'):
                scales[f'{prefix}_ffn_output'] = float(test_out.scale.item())
        except Exception:
            scales[f'{prefix}_ffn_output'] = 0.01

    # Final norm weight
    weights['final_norm_weight'] = model.norm.weight.detach().numpy().astype(np.float32)

    # Final norm output scale
    try:
        test_in = torch.randn(1, 1, model.dim)
        test_out = model.norm_quant(test_in)
        if hasattr(test_out, 'scale'):
            scales['final_norm_output'] = float(test_out.scale.item())
    except Exception:
        scales['final_norm_output'] = 0.01

    # Output/classifier weight (shared with embedding in original model)
    qw = model.output.quant_weight()
    weights['output'] = qw.int().detach().numpy().astype(np.int8)
    scales['output'] = float(qw.scale.item())

    return weights, scales


if __name__ == "__main__":
    test_int8_inference()
