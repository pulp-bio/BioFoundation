# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Automated Test Generation for ARES

Workflow for each test:
1. Load MNIST data
2. Train network (5 epochs)
3. Save PyTorch model (.pth)
4. Extract Brevitas quantization parameters
5. Generate golden INT8 outputs
6. Generate GAP9 C code
7. Organize in separate test directory

Usage:
    python generate_all_tests.py                    # Generate all tests
    python generate_all_tests.py --test test_1_simplecnn  # Single test
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import argparse
import sys
from pathlib import Path
import json
import numpy as np
import random
import hashlib
from typing import Dict, Any
import copy

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


def create_fp32_model(brevitas_model):
    """
    Create an FP32 equivalent model from a Brevitas model.

    This dynamically builds an FP32 model by:
    1. Identifying Brevitas layers (QuantConv2d, QuantLinear, etc.)
    2. Creating equivalent PyTorch layers (Conv2d, Linear, etc.)
    3. Copying the FP32 weights from Brevitas
    4. Reconstructing the forward pass

    Args:
        brevitas_model: A trained Brevitas model

    Returns:
        An equivalent FP32 nn.Module
    """
    from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantIdentity

    # Build a mapping of layer names to their FP32 equivalents
    layer_map = {}

    for name, module in brevitas_model.named_modules():
        if name == '':
            continue
        # Only process direct children (no dots in name)
        if '.' in name:
            continue

        if isinstance(module, QuantConv2d):
            fp32_layer = nn.Conv2d(
                module.in_channels, module.out_channels,
                module.kernel_size, module.stride, module.padding,
                module.dilation, module.groups, module.bias is not None
            )
            with torch.no_grad():
                fp32_layer.weight.copy_(module.weight.data)
                if module.bias is not None:
                    fp32_layer.bias.copy_(module.bias.data)
            layer_map[name] = fp32_layer

        elif isinstance(module, QuantLinear):
            fp32_layer = nn.Linear(module.in_features, module.out_features, module.bias is not None)
            with torch.no_grad():
                fp32_layer.weight.copy_(module.weight.data)
                if module.bias is not None:
                    fp32_layer.bias.copy_(module.bias.data)
            layer_map[name] = fp32_layer

        elif isinstance(module, QuantReLU):
            layer_map[name] = nn.ReLU()

        elif isinstance(module, QuantIdentity):
            layer_map[name] = nn.Identity()

        elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d,
                                  nn.AdaptiveAvgPool1d, nn.Flatten, nn.LayerNorm)):
            layer_map[name] = copy.deepcopy(module)

    # Create a wrapper class that mimics the original forward pass
    class FP32Model(nn.Module):
        def __init__(self, original_model, layer_map):
            super().__init__()
            self._original = original_model
            # Register layers as submodules
            for name, layer in layer_map.items():
                setattr(self, name, layer)
            self._layer_names = list(layer_map.keys())

        def forward(self, x):
            """Execute forward pass using FP32 layers."""
            # We need to match the original model's forward logic
            # This is done by tracing through attribute access
            for name in self._layer_names:
                layer = getattr(self, name)

                # Handle flatten before linear layers
                if isinstance(layer, nn.Linear) and x.dim() > 2:
                    x = x.view(x.size(0), -1)

                x = layer(x)

                # Handle QuantTensor outputs from any remaining Brevitas layers
                if hasattr(x, 'value'):
                    x = x.value

            return x

    return FP32Model(brevitas_model, layer_map)


def evaluate_fp32_from_brevitas(brevitas_model, test_loader, device):
    """
    Evaluate FP32 accuracy by running inference with quantization disabled.

    This uses a trick: we run the Brevitas model but with full precision weights.
    Since Brevitas stores FP32 weights and only quantizes during forward pass,
    we can copy the weights to an FP32-only model for baseline comparison.

    Args:
        brevitas_model: Trained Brevitas model
        test_loader: DataLoader for test data
        device: torch device

    Returns:
        float: FP32 accuracy percentage
    """
    try:
        fp32_model = create_fp32_model(brevitas_model)
        fp32_model.to(device)
        fp32_model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = fp32_model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        return 100.0 * correct / total

    except Exception as e:
        # For complex models, FP32 conversion may fail
        # Return None to indicate we couldn't measure FP32 baseline
        return None

# Import test networks (numbered 1-30)
from test_networks.test_1_simplecnn import SimpleCNN
from test_networks.test_2_tinycnn import TinyCNN
from test_networks.test_3_mlp import MLP
from test_networks.test_4_resnet_basic import ResNetBasic
from test_networks.test_5_densenet_basic import DenseNetBasic
from test_networks.test_6_multitilecnn import MultiTileCNN
from test_networks.test_7_bottleneck import BottleneckNet
from test_networks.test_8_stride2 import Stride2Net
from test_networks.test_9_padding import PaddingNet
from test_networks.test_10_resnet18 import ResNet18MNIST
from test_networks.test_11_layernorm_basic import QuantLayerNormBasic
from test_networks.test_12_gelu_basic import QuantGELUBasic
from test_networks.test_13_transformer_simple import SimpleTransformer
from test_networks.test_14_multiblock_transformer import MultiBlockTransformer
from test_networks.test_15_tinymyo_tiny import TinyMyoTinyQuant
from test_networks.test_16_mamba_conv1d import MambaConv1DTest
from test_networks.test_17_mamba_ssm import MambaSSMTest
from test_networks.test_18_mamba_block import MambaBlockTest
from test_networks.test_19_mamba_stacked import MambaStackedTest
from test_networks.test_20_bidirectional_mamba import BidirectionalMambaTest
from test_networks.test_21_femba_patchembedder import FEMBAPatchEmbedderTest
from test_networks.test_22_femba_full import FEMBAFullTest
from test_networks.test_23_femba_full_input import FEMBATinyFullInput
from test_networks.test_24_femba_full_expand2 import FEMBATiny as FEMBATinyFullExpand
from test_networks.test_25_femba_tiny_int8 import FEMBATinyInt8
from test_networks.test_26_tinymyo_8ch_400tok import TinyMyo8Ch400Quant
from test_networks.test_27_linear3d_bench import Linear3DBenchQuant
from test_networks.test_28_conv2d_remainder import Conv2DRemainderNet
from test_networks.test_29_luna_base import LUNABaseTest
from test_networks.test_30_autotune_stress import AutotuneStressNet
from test_networks.test_31_luna_full import LUNAFullTest
from test_networks.test_36_drowsiness_fusion import DrowsinessFusionNetwork
from test_networks.test_37_zeropad2d import ZeroPad2dTestNet
from test_networks.test_38_ne16_linear import NE16LinearTest
from test_networks.test_39_ne16_large import NE16LargeTest
from test_networks.test_40_depthwise_conv import DepthwiseConvNet
from test_networks.test_41_large_depthwise import LargeDepthwiseNet
from test_networks.test_42_llama_minimal import MinimalLlama
from test_networks.test_43_cerebro_original import CerebroOriginal
from test_networks.test_44_llama_swiglu import LlamaSwiGLU

# Import existing tools
from tools.pytorch_extractor import BrevitasExtractor
from tools.generate_golden_outputs import GoldenOutputGenerator
from codegen.generate_c_code import CCodeGenerator
from codegen.targets import available_targets


class TestGenerator:
    """Generate complete test suite for ARES."""

    NETWORKS = {
        # CNN Tests (1-10)
        'test_1_simplecnn': {
            'class': SimpleCNN,
            'description': 'Baseline SimpleCNN with 3x3 kernels',
            'epochs': 5,
        },
        'test_2_tinycnn': {
            'class': TinyCNN,
            'description': 'Minimal CNN with 5x5 kernels',
            'epochs': 3,
        },
        'test_3_mlp': {
            'class': MLP,
            'description': 'MLP-only network (no convolutions)',
            'epochs': 5,
        },
        'test_4_resnet_basic': {
            'class': ResNetBasic,
            'description': 'ResNet-style with Add and GlobalAvgPool',
            'epochs': 5,
        },
        'test_5_densenet_basic': {
            'class': DenseNetBasic,
            'description': 'DenseNet-style with Concatenate and AvgPool',
            'epochs': 5,
        },
        'test_6_multitilecnn': {
            'class': MultiTileCNN,
            'description': 'Large-spatial CNN (96x96) to exercise Conv2D tiling',
            'epochs': 3,
            'input_resize': 96,
        },
        'test_7_bottleneck': {
            'class': BottleneckNet,
            'description': 'Bottleneck blocks with 1x1 convolutions (ResNet-style)',
            'epochs': 5,
        },
        'test_8_stride2': {
            'class': Stride2Net,
            'description': 'Stride-2 convolutions for downsampling (alternative to pooling)',
            'epochs': 5,
        },
        'test_9_padding': {
            'class': PaddingNet,
            'description': 'Various padding values (0, 1, 2) with different kernel sizes',
            'epochs': 5,
        },
        'test_10_resnet18': {
            'class': ResNet18MNIST,
            'description': 'ResNet-18 adapted for MNIST (~700K params, tests L3 staging)',
            'epochs': 10,
        },
        # Transformer Tests (11-14)
        'test_11_layernorm_basic': {
            'class': QuantLayerNormBasic,
            'description': 'LayerNorm validation (Flatten -> LayerNorm -> Linear)',
            'epochs': 3,
        },
        'test_12_gelu_basic': {
            'class': QuantGELUBasic,
            'description': 'GELU activation validation (Linear -> GELU -> Linear)',
            'epochs': 3,
        },
        'test_13_transformer_simple': {
            'class': SimpleTransformer,
            'description': 'Simplified Transformer (1 block) with LayerNorm, MHSA, GELU',
            'epochs': 5,
            # use_ne16 disabled - NE16 is slower than SW kernels for this test
        },
        'test_14_multiblock_transformer': {
            'class': MultiBlockTransformer,
            'description': 'Multi-Block Transformer (4 blocks, 128 dim, 4 heads) for scalability validation',
            'epochs': 5,
            # use_ne16 disabled - NE16 is slower than SW kernels for this test
        },
        # Mamba/SSM Tests (15-20)
        'test_15_tinymyo_tiny': {
            'class': TinyMyoTinyQuant,
            'description': 'Tiny TinyMyo (1 block, 192 dim, 3 heads, 50 seq) for fast GVSOC validation',
            'epochs': 0,
            'custom_input_shape': (1, 1, 5, 200),
        },
        'test_16_mamba_conv1d': {
            'class': MambaConv1DTest,
            'description': 'MAMBA Conv1D depthwise + SiLU test (validates MAMBA building blocks)',
            'epochs': 0,
            'custom_input_shape': (1, 32, 64),
        },
        'test_17_mamba_ssm': {
            'class': MambaSSMTest,
            'description': 'MAMBA SSM (State Space Model) test (validates SSM core with softplus, discretization, scan)',
            'epochs': 0,
            'custom_input_shape': (1, 16, 32),
        },
        'test_18_mamba_block': {
            'class': MambaBlockTest,
            'description': 'Full MAMBA block (in_proj, conv1d, SiLU, SSM, gating, out_proj)',
            'epochs': 0,
            'custom_input_shape': (1, 16, 32),
        },
        'test_19_mamba_stacked': {
            'class': MambaStackedTest,
            'description': 'Stacked MAMBA blocks (3 blocks) to test cycle scaling',
            'epochs': 0,
            'custom_input_shape': (1, 16, 32),
        },
        'test_20_bidirectional_mamba': {
            'class': BidirectionalMambaTest,
            'description': 'Bidirectional MAMBA wrapper (fwd + flip + rev + flip + add)',
            'epochs': 0,
            'custom_input_shape': (1, 16, 32),
        },
        # FEMBA Tests (21-25)
        'test_21_femba_patchembedder': {
            'class': FEMBAPatchEmbedderTest,
            'description': 'FEMBA PatchEmbed + 2 Bi-Mamba blocks (embed_dim=35)',
            'epochs': 0,
            'custom_input_shape': (1, 1, 10, 32),
        },
        'test_22_femba_full': {
            'class': FEMBAFullTest,
            'description': 'Full FEMBA with positional embedding, residuals, LayerNorm (embed_dim=35)',
            'epochs': 0,
            'custom_input_shape': (1, 1, 10, 32),
        },
        'test_23_femba_full_input': {
            'class': FEMBATinyFullInput,
            'description': 'FEMBA Tiny with full input (22x1280), d_inner=256 cap',
            'epochs': 0,
            'custom_input_shape': (1, 1, 22, 1280),
        },
        'test_24_femba_full_expand2': {
            'class': FEMBATinyFullExpand,
            'description': 'FEMBA Tiny with full expand=2 (d_inner=770), L3 streaming',
            'epochs': 0,
            'custom_input_shape': (1, 1, 22, 1280),
        },
        'test_25_femba_tiny_int8': {
            'class': FEMBATinyInt8,
            'description': 'FEMBA Tiny INT8 (true arch: expand=4, d_state=16, ~7.6M params)',
            'epochs': 0,
            'custom_input_shape': (1, 1, 22, 1280),
        },
        # Additional Tests (26-30)
        'test_26_tinymyo_8ch_400tok': {
            'class': TinyMyo8Ch400Quant,
            'description': 'TinyMyo (8 blocks, 192 dim, 3 heads, 400 seq) with 8-channel input',
            'epochs': 0,
            'custom_input_shape': (1, 1, 8, 1000),
        },
        'test_27_linear3d_bench': {
            'class': Linear3DBenchQuant,
            'description': '3D QuantLinear benchmark (seq_len=400, 192->768->192) for fast tiling tuning',
            'epochs': 0,
            'custom_input_shape': (1, 400, 192),
        },
        'test_28_conv2d_remainder': {
            'class': Conv2DRemainderNet,
            'description': 'Conv2D im2col remainder benchmark (1x1 expand 1→3, then 7x7 3→32)',
            'epochs': 0,
        },
        'test_29_luna_base': {
            'class': LUNABaseTest,
            'description': 'LUNA_base bring-up (Embedding, GroupNorm, RFFT, RoPE-MHSA, Cross-Attention)',
            'epochs': 0,
            'custom_input_shape': (1, 22, 1280),
        },
        'test_30_autotune_stress': {
            'class': AutotuneStressNet,
            'description': 'Auto-tuner stress test with unusual linear dimensions',
            'epochs': 0,
            'custom_input_shape': (1, 64),
        },
        'test_31_luna_full': {
            'class': LUNAFullTest,
            'description': 'Full LUNA architecture (CNN+RFFT embed, NeRF channels, CrossAttn+SelfRefine, MLP head)',
            'epochs': 0,
            'custom_input_shape': (1, 22, 1280),
            'use_ne16': False,  # NE16 disabled - slower than SW kernels for these layer sizes
            'enable_codegen': True,
        },
        'test_36_drowsiness_fusion': {
            'class': DrowsinessFusionNetwork,
            'description': 'BioCAS dual-input EEG+PPG drowsiness detection with ZeroPad2d',
            'epochs': 0,
            'custom_input_shape': ((1, 1, 8, 2200), (1, 1, 2, 2200)),  # Multi-input: (eeg_shape, ppg_shape)
            # Auto-detects HWC layout (1D-style kernels) - achieves 3x speedup vs CHW
        },
        'test_37_zeropad2d': {
            'class': ZeroPad2dTestNet,
            'description': 'ZeroPad2d asymmetric padding test',
            'epochs': 0,  # No training - uses random weights
            'custom_input_shape': (1, 1, 8, 64),
        },
        # NE16 Accelerator Tests (38+)
        'test_38_ne16_linear': {
            'class': NE16LinearTest,
            'description': 'NE16 accelerator validation (Linear layers, packed weights)',
            'epochs': 0,  # No training - uses random weights for validation
            'custom_input_shape': (1, 49, 192),  # Batch 1, seq_len 49, 192 features (matches tinymyo)
            'use_ne16': True,  # Enable NE16 accelerator path
        },
        'test_39_ne16_large': {
            'class': NE16LargeTest,
            'description': 'NE16 efficiency scaling test (256x256 linear layers, ~13M MACs)',
            'epochs': 0,  # No training - uses random weights for validation
            'custom_input_shape': (1, 49, 256),  # Batch 1, seq_len 49, 256 features
            'use_ne16': True,  # Enable NE16 accelerator path
        },
        # Depthwise Convolution Tests
        'test_40_depthwise_conv': {
            'class': DepthwiseConvNet,
            'description': 'MobileNet-style depthwise separable convolutions for NE16 depthwise support',
            'epochs': 5,  # Train on MNIST
        },
        'test_41_large_depthwise': {
            'class': LargeDepthwiseNet,
            'description': 'Large depthwise separable convolutions (64x64 input, 32-256 channels) for NE16 benchmarking',
            'epochs': 0,  # No training - uses random weights for validation
            'custom_input_shape': (1, 3, 64, 64),  # RGB 64x64 input
        },
        # Llama-style Transformer Test
        'test_42_llama_minimal': {
            'class': MinimalLlama,
            'description': 'Minimal Llama block (RMSNorm + MHSA + RMSNorm + MLP) for testing Llama components',
            'epochs': 0,  # No training - uses random weights for validation
            'custom_input_shape': (1, 32, 64),  # Batch 1, seq_len 32, dim 64
        },
        # Cerebro/EEG Transformer Test
        'test_43_cerebro_original': {
            'class': CerebroOriginal,
            'description': 'Original full-sized Cerebro (6 blocks, 180 embed_dim, 5 heads) with L3 streaming',
            'epochs': 0,  # No training - uses random weights for validation
            'custom_input_shape': (1, 14, 8),  # [B, num_channels, temporal_len]
            'use_ne16': True,  # Enable NE16 with L3 streaming
        },
        # Llama with SwiGLU FFN Test
        'test_44_llama_swiglu': {
            'class': LlamaSwiGLU,
            'description': 'Llama block with SwiGLU FFN (gate=W1, up=W3, down=W2)',
            'epochs': 0,  # No training - uses random weights for validation
            'custom_input_shape': (1, 16, 64),  # Batch 1, seq_len 16, dim 64
        },
    }

    def __init__(self, output_dir='tests/outputs', mnist_root='data/mnist', board_mode=False,
                 disable_l1_weight_caching=False, skip_int8_eval=False, enable_ne16=False,
                 target_name='gap9'):
        self.output_dir = Path(output_dir)
        self.mnist_root = Path(mnist_root)
        self.board_mode = board_mode  # Board-ready code: minimal prints, no golden checks
        self.disable_l1_weight_caching = disable_l1_weight_caching  # Disable L1 weight caching for baseline benchmarks
        self.skip_int8_eval = skip_int8_eval  # Skip slow INT8 accuracy evaluation
        self.enable_ne16 = enable_ne16  # Enable NE16 accelerator for all eligible layers
        self.target_name = target_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mnist_root.mkdir(parents=True, exist_ok=True)

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if self.board_mode:
            print(f"Board mode: ENABLED (generating clean hardware code)")
        if self.enable_ne16:
            print(f"NE16 accelerator: ENABLED for all eligible layers")
        if self.skip_int8_eval:
            print(f"INT8 eval: SKIPPED (use --eval-int8 to enable)")
        print(f"Codegen target: {self.target_name}")

    def load_mnist(self, train=True, subset_size=1000, image_size=None):
        """Load MNIST dataset (or subset for faster training)."""
        transform_list = []
        if image_size:
            transform_list.append(transforms.Resize((image_size, image_size)))
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
        ])
        transform = transforms.Compose(transform_list)

        dataset = datasets.MNIST(
            root=str(self.mnist_root),
            train=train,
            download=True,
            transform=transform
        )

        # Use subset for faster iteration
        if subset_size and subset_size < len(dataset):
            indices = torch.randperm(len(dataset))[:subset_size]
            dataset = Subset(dataset, indices)

        return dataset

    def train_network(self, network, num_epochs=5, batch_size=32, image_size=None):
        """Train network on MNIST."""
        print(f"  Training for {num_epochs} epochs...")

        # Load MNIST
        train_dataset = self.load_mnist(train=True, subset_size=1000, image_size=image_size)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Optimizer and loss
        optimizer = optim.Adam(network.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        network.to(self.device)
        network.train()

        for epoch in range(num_epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = network(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

            avg_loss = total_loss / len(train_loader)
            accuracy = 100.0 * correct / total
            print(f"    Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}, acc={accuracy:.2f}%")

        network.cpu()
        print("  [OK] Training complete")

    def evaluate_fp32_accuracy(self, network, image_size=None, test_samples=1000):
        """
        Evaluate FP32 (no quantization) accuracy on test set.
        Creates an FP32 equivalent model and evaluates it.

        Returns:
            float: FP32 accuracy percentage, or None if conversion failed
        """
        test_dataset = self.load_mnist(train=False, subset_size=test_samples, image_size=image_size)
        test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

        fp32_acc = evaluate_fp32_from_brevitas(network, test_loader, self.device)
        return fp32_acc

    def evaluate_brevitas_accuracy(self, network, image_size=None, test_samples=1000):
        """
        Evaluate Brevitas (fake-quantized) accuracy on test set.
        This runs on GPU and is fast.

        Returns:
            float: Accuracy percentage
        """
        test_dataset = self.load_mnist(train=False, subset_size=test_samples, image_size=image_size)
        test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

        network.eval()
        network.to(self.device)

        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = network(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        network.cpu()
        return 100.0 * correct / total

    def evaluate_int8_accuracy(self, network_info_path, weights_dir, image_size=None,
                                test_samples=100, is_transformer=False):
        """
        Evaluate true INT8 accuracy using our inference engine.
        This is CPU-only and slow, so we use a small subset.

        Returns:
            tuple: (accuracy, max_diff_from_brevitas)
        """
        from tools.int8_inference import INT8InferenceEngine

        # Load network info
        with open(network_info_path) as f:
            network_info = json.load(f)

        # Load weights
        weights_path = Path(weights_dir)
        for layer_name, layer_data in network_info.items():
            if layer_name == '__layer_order__':
                continue
            if 'weight_int8' in layer_data:
                weight_file = weights_path / f"{layer_name}_weight_int8.npy"
                if weight_file.exists():
                    layer_data['weight_int8'] = np.load(weight_file)
            if 'bias_fp32' in layer_data:
                bias_file = weights_path / f"{layer_name}_bias_fp32.npy"
                if bias_file.exists():
                    layer_data['bias_fp32'] = np.load(bias_file)

        # Create INT8 engine
        engine = INT8InferenceEngine(
            network_info,
            use_i_softmax=is_transformer,
            use_i_gelu=is_transformer,
            use_i_layernorm=is_transformer
        )

        # Load test data
        test_dataset = self.load_mnist(train=False, subset_size=test_samples, image_size=image_size)

        correct = 0
        total = 0

        for i in range(len(test_dataset)):
            data, target = test_dataset[i]
            x_np = data.unsqueeze(0).numpy()

            # Run INT8 inference
            logits, _, _ = engine.forward(x_np, verbose=False)
            predicted = np.argmax(logits, axis=1)[0]

            if predicted == target:
                correct += 1
            total += 1

        return 100.0 * correct / total

    def save_accuracy_report(self, test_dir, test_name, brevitas_acc, int8_acc=None,
                              fp32_acc=None, brevitas_samples=1000, int8_samples=100,
                              fp32_samples=1000):
        """Save accuracy report to JSON file with degradation metrics."""
        report = {
            'test_name': test_name,
        }

        # FP32 baseline (pre-quantization)
        if fp32_acc is not None:
            report['fp32_accuracy'] = {
                'value': round(fp32_acc, 2),
                'samples': fp32_samples,
                'description': 'FP32 baseline (no quantization)'
            }

        # Brevitas (fake-quantized)
        if brevitas_acc is not None:
            report['brevitas_accuracy'] = {
                'value': round(brevitas_acc, 2),
                'samples': brevitas_samples,
                'description': 'Fake-quantized accuracy (Brevitas model on GPU)'
            }

        # True INT8
        if int8_acc is not None:
            report['int8_accuracy'] = {
                'value': round(int8_acc, 2),
                'samples': int8_samples,
                'description': 'True INT8 accuracy (Python inference engine)'
            }

        # Compute degradation metrics
        degradation = {}
        if fp32_acc is not None and brevitas_acc is not None:
            degradation['fp32_to_brevitas'] = round(brevitas_acc - fp32_acc, 2)
        if brevitas_acc is not None and int8_acc is not None:
            degradation['brevitas_to_int8'] = round(int8_acc - brevitas_acc, 2)
        if fp32_acc is not None and int8_acc is not None:
            degradation['fp32_to_int8'] = round(int8_acc - fp32_acc, 2)

        if degradation:
            report['degradation'] = degradation

        # Backward-compatible field used by existing report consumers.
        if brevitas_acc is not None and int8_acc is not None:
            report['accuracy_match'] = abs(brevitas_acc - int8_acc) < 2.0

        report_path = test_dir / 'accuracy_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        return report_path

    def extract_weights_and_golden(self, network, test_name, test_dir, image_size=None, custom_input_shape=None):
        """Extract quantized weights and generate golden outputs."""
        print("\n4. Extracting quantization parameters...")

        weights_dir = test_dir / 'golden_outputs' / 'weights'
        weights_dir.mkdir(parents=True, exist_ok=True)

        # Load sample input (reuse for golden generation)
        # Check if this is a multi-input model (tuple of shapes)
        is_multi_input = (custom_input_shape is not None and
                         isinstance(custom_input_shape, tuple) and
                         len(custom_input_shape) > 0 and
                         isinstance(custom_input_shape[0], tuple))

        if is_multi_input:
            # Multi-input case: generate multiple input tensors
            import numpy as np
            test_inputs_fp32 = []
            test_input_tensors = []
            for i, shape in enumerate(custom_input_shape):
                inp = np.random.randn(*shape).astype(np.float32)
                test_inputs_fp32.append(inp)
                test_input_tensors.append(torch.from_numpy(inp))
            test_input_fp32 = test_inputs_fp32  # List of arrays for multi-input
            test_input_tensor = tuple(test_input_tensors)  # Tuple of tensors
            print(f"  Using multi-input shapes: {[s for s in custom_input_shape]}")
        elif custom_input_shape is not None:
            # Use custom random input for models with non-MNIST input shapes
            import numpy as np
            test_input_fp32 = np.random.randn(*custom_input_shape).astype(np.float32)
            test_input_tensor = torch.from_numpy(test_input_fp32)
            print(f"  Using custom input shape: {custom_input_shape}")
        else:
            # Use MNIST input
            test_dataset = self.load_mnist(train=False, subset_size=10, image_size=image_size)
            test_input_tensor = test_dataset[3][0].unsqueeze(0)
            test_input_fp32 = test_input_tensor.numpy()

        # Extract network info using existing tool
        network.eval()
        extractor = BrevitasExtractor(network)
        network_info = extractor.extract_all(sample_input=test_input_tensor)

        # Harmonize projection shortcut input scales with block inputs
        network_info = self._fix_shortcut_scales(network_info)

        # Compute residency policy (L2 vs L3)
        print("  Computing residency policy (L2 vs L3)...")
        extractor._compute_residency_policy()

        # Save network info
        network_info_path = test_dir / 'golden_outputs' / 'network_info.json'
        serializable_info = self._to_serializable(network_info)
        # Add layer execution order (critical for ResNet skip connections)
        serializable_info['__layer_order__'] = list(network_info.keys())
        with open(network_info_path, 'w') as f:
            json.dump(serializable_info, f, indent=2)
        print(f"  [OK] Network info saved")

        # Save weights
        extractor.save_weights(str(weights_dir))
        print(f"  [OK] Weights saved to {weights_dir}")

        # Generate golden INT8 outputs
        print("\n5. Generating golden INT8 outputs...")
        # Enable i-Softmax, i-GELU, and i-LayerNorm for transformer tests (MHSA, transformer, tinymyo, layernorm)
        # This ensures bit-exact matching between Python and C code
        is_transformer = any(keyword in test_name.lower() for keyword in ['mhsa', 'transformer', 'tinymyo', 'layernorm', 'luna'])
        use_i_softmax = is_transformer
        use_i_gelu = is_transformer
        use_i_layernorm = is_transformer
        if use_i_softmax:
            print("  i-Softmax enabled for bit-exact transformer inference")
        if use_i_gelu:
            print("  i-GELU enabled for bit-exact transformer inference")
        if use_i_layernorm:
            print("  i-LayerNorm enabled for bit-exact transformer inference")
        generator = GoldenOutputGenerator(network_info, use_i_softmax=use_i_softmax, use_i_gelu=use_i_gelu, use_i_layernorm=use_i_layernorm)

        # Generate golden output for captured sample input
        test_cases = [test_input_fp32]
        output_dir = str(test_dir / 'golden_outputs' / 'test_cases')
        try:
            generator.generate_golden_outputs(test_cases, output_dir=output_dir)
        except (ValueError, KeyError, RuntimeError) as e:
            print(f"  [WARN] INT8 inference engine failed: {e}")
            print(f"  [WARN] Falling back to PyTorch model output as golden reference")
            # Use PyTorch model output directly
            self._generate_pytorch_golden(network, test_input_fp32, output_dir)

        # Update network_info with runtime output scales and re-save
        serializable_info = self._to_serializable(network_info)
        # Add layer execution order (critical for ResNet skip connections)
        serializable_info['__layer_order__'] = list(network_info.keys())
        with open(network_info_path, 'w') as f:
            json.dump(serializable_info, f, indent=2)
        print(f"  [OK] Network info updated with runtime scales")

        print(f"  [OK] Golden outputs saved to {output_dir}")
        return network_info

    def _fix_shortcut_scales(self, network_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure projection shortcuts use the block input scale (output of the previous block)
        instead of whatever current_scale happens to be when the shortcut is visited.
        """
        layer_order = list(network_info.keys())
        if not layer_order:
            return network_info

        current_scale = network_info.get('input_quant', {}).get('scale')
        block_input_scale = {}

        for name in layer_order:
            layer = network_info.get(name, {})
            if name.endswith('.conv1'):
                block_name = name.rsplit('.conv1', 1)[0]
                block_input_scale[block_name] = current_scale

            if '.shortcut' in name:
                block_name = name.rsplit('.shortcut', 1)[0]
                if block_name in block_input_scale:
                    layer['scale_input'] = block_input_scale[block_name]
                    network_info[name] = layer

            if 'scale_output' in layer:
                current_scale = layer['scale_output']
            elif 'scale' in layer:
                current_scale = layer['scale']

        return network_info

    def _to_serializable(self, obj):
        """Recursively convert numpy types to native Python types for JSON."""
        if isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._to_serializable(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._to_serializable(v) for v in obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.generic,)):
            return obj.item()
        return obj

    def _generate_pytorch_golden(self, model, test_input_fp32, output_dir: str):
        """
        Generate golden outputs using PyTorch model directly.

        This is a fallback when INT8 inference engine doesn't support
        the network architecture (e.g., Mamba blocks).

        Args:
            model: PyTorch model
            test_input_fp32: Either a single numpy array or a list of numpy arrays for multi-input models
            output_dir: Directory to save golden outputs
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Ensure model and input are on the same device
        device = next(model.parameters()).device
        model.eval()

        # Handle multi-input models
        if isinstance(test_input_fp32, list):
            # Multi-input case
            test_input_tensors = [torch.from_numpy(inp).to(device) for inp in test_input_fp32]
            with torch.no_grad():
                output = model(*test_input_tensors)
        else:
            # Single-input case
            test_input_tensor = torch.from_numpy(test_input_fp32).to(device)
            with torch.no_grad():
                output = model(test_input_tensor)

        # Handle QuantTensor output from Brevitas
        if hasattr(output, 'value'):
            output = output.value

        output_fp32 = output.cpu().numpy()

        # Save in the same format as GoldenOutputGenerator
        test_case_dir = os.path.join(output_dir, 'test_case_1')
        os.makedirs(test_case_dir, exist_ok=True)

        # Handle multi-input models
        if isinstance(test_input_fp32, list):
            # Multi-input case: save each input with numbered suffix
            input_scales = []
            input_shapes = []
            for i, inp in enumerate(test_input_fp32):
                np.save(os.path.join(test_case_dir, f'input_{i}_fp32.npy'), inp)
                # Quantize input to INT8
                scale = float(np.abs(inp).max()) / 127.0 if np.abs(inp).max() > 0 else 1.0
                inp_int8 = np.clip(np.round(inp / scale), -128, 127).astype(np.int8)
                np.save(os.path.join(test_case_dir, f'input_{i}_int8.npy'), inp_int8)
                input_scales.append(scale)
                input_shapes.append(list(inp.shape))
            # Also save first input as input_fp32.npy for compatibility
            np.save(os.path.join(test_case_dir, 'input_fp32.npy'), test_input_fp32[0])
            input_int8 = np.clip(np.round(test_input_fp32[0] / input_scales[0]), -128, 127).astype(np.int8)
            np.save(os.path.join(test_case_dir, 'input_int8.npy'), input_int8)
            input_scale = input_scales[0]
            input_shape = input_shapes
        else:
            # Single-input case
            np.save(os.path.join(test_case_dir, 'input_fp32.npy'), test_input_fp32)
            input_scale = float(np.abs(test_input_fp32).max()) / 127.0 if np.abs(test_input_fp32).max() > 0 else 1.0
            input_int8 = np.clip(np.round(test_input_fp32 / input_scale), -128, 127).astype(np.int8)
            np.save(os.path.join(test_case_dir, 'input_int8.npy'), input_int8)
            input_shape = list(test_input_fp32.shape)

        # Save output
        np.save(os.path.join(test_case_dir, 'output_fp32.npy'), output_fp32)

        # Create golden_info.json
        golden_info = {
            'input_shape': input_shape,
            'output_shape': list(output_fp32.shape),
            'input_scale': input_scale if not isinstance(test_input_fp32, list) else input_scales,
            'predicted_class': int(np.argmax(output_fp32)),
            'source': 'pytorch_fallback',
            'note': 'Generated from PyTorch model (INT8 inference engine not supported for this architecture)'
        }

        with open(os.path.join(test_case_dir, 'golden_info.json'), 'w') as f:
            json.dump(golden_info, f, indent=2)

        print(f"  [OK] PyTorch golden outputs saved to {test_case_dir}")
        print(f"    Output shape: {output_fp32.shape}")
        print(f"    Predicted class: {np.argmax(output_fp32)}")

    def generate_test(self, test_name):
        """Generate complete test for one network."""
        print(f"\n{'='*80}")
        print(f"Generating: {test_name}")
        print(f"Description: {self.NETWORKS[test_name]['description']}")
        print(f"{'='*80}")

        # Set deterministic seed for reproducibility (each test gets same initialization)
        # Use a deterministic hash (Python's hash() is randomized across runs)
        seed = int(hashlib.md5(test_name.encode()).hexdigest()[:8], 16) % (2**32)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Create output directories
        test_dir = self.output_dir / test_name
        model_dir = test_dir / 'models'
        golden_dir = test_dir / 'golden_outputs'

        # Board mode outputs directly to tests/boards/<test_name>/ (flat structure)
        if self.board_mode:
            board_dir = Path(self.output_dir).parent / 'boards' / test_name
            generated_dir = board_dir
        else:
            generated_dir = test_dir / 'generated'

        for d in [model_dir, golden_dir, generated_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # 1. Instantiate network
        print("\n1. Creating network...")
        config = self.NETWORKS[test_name]
        NetworkClass = config['class']
        model_kwargs = config.get('model_kwargs', {})
        network = NetworkClass(**model_kwargs)
        print(f"  [OK] Created {network.__class__.__name__}")
        if model_kwargs:
            print(f"  Model kwargs: {model_kwargs}")

        # Print network architecture
        total_params = sum(p.numel() for p in network.parameters())
        print(f"  Total parameters: {total_params:,}")

        # 2. Train network
        print("\n2. Training network on MNIST...")
        num_epochs = config['epochs']
        self.train_network(
            network,
            num_epochs=num_epochs,
            image_size=config.get('input_resize')
        )

        # 3. Save PyTorch model
        print("\n3. Saving model...")
        model_path = model_dir / f"{test_name}.pth"
        torch.save(network.state_dict(), model_path)
        print(f"  [OK] Saved to {model_path}")

        # 3.3. Evaluate FP32 (pre-quantization) accuracy
        fp32_acc = None
        if config.get('custom_input_shape') is None:
            print("\n3.3. Evaluating FP32 baseline accuracy...")
            fp32_acc = self.evaluate_fp32_accuracy(
                network,
                image_size=config.get('input_resize'),
                test_samples=1000
            )
            if fp32_acc is not None:
                print(f"  [OK] FP32 accuracy: {fp32_acc:.2f}% (1000 test samples)")
            else:
                print(f"  [WARN] FP32 evaluation failed (model conversion issue)")

        # 3.5. Evaluate Brevitas (fake-quantized) accuracy
        print("\n3.5. Evaluating Brevitas (fake-quantized) accuracy...")
        # Skip accuracy evaluation for models with custom input shapes (non-MNIST)
        if config.get('custom_input_shape') is None:
            brevitas_acc = self.evaluate_brevitas_accuracy(
                network,
                image_size=config.get('input_resize'),
                test_samples=1000
            )
            print(f"  [OK] Brevitas accuracy: {brevitas_acc:.2f}% (1000 test samples)")
            if fp32_acc is not None:
                drop = brevitas_acc - fp32_acc
                print(f"  → Quantization drop: {drop:+.2f}%")
        else:
            brevitas_acc = None
            print(f"  [WARN] Skipped (custom input shape, not MNIST)")

        # 4-5. Extract weights and generate golden outputs
        network_info = self.extract_weights_and_golden(
            network,
            test_name,
            test_dir,
            image_size=config.get('input_resize'),
            custom_input_shape=config.get('custom_input_shape')
        )

        # 6. Generate C code
        print(f"\n6. Generating C code for target '{self.target_name}'...")
        if config.get('enable_codegen', True):
            try:
                # NE16 configuration:
                # - If config explicitly sets 'use_ne16': True/False, use that
                # - If --enable-ne16 CLI flag is set, use True
                # - Otherwise, use "auto" (auto-detect beneficial layers)
                if 'use_ne16' in config:
                    use_ne16 = config['use_ne16']
                elif self.enable_ne16:
                    use_ne16 = True
                else:
                    use_ne16 = "auto"  # Let code generator auto-detect
                codegen = CCodeGenerator(
                    network_info_path=str(golden_dir / "network_info.json"),
                    weights_dir=str(golden_dir / "weights"),
                    test_case_dir=str(golden_dir / "test_cases" / "test_case_1"),
                    output_dir=str(generated_dir),
                    target_name=self.target_name,
                    board_mode=self.board_mode,
                    disable_l1_weight_caching=self.disable_l1_weight_caching,
                    use_hwc_layout=config.get('use_hwc_layout', "auto"),  # Auto-detect based on kernel patterns
                    use_ne16=use_ne16  # Enable NE16 accelerator for eligible layers
                )
                codegen.generate_all()
                print(f"  [OK] C code generated in {generated_dir}")
            except Exception as e:
                print(f"  [FAIL] C code generation failed: {e}")
                raise
        else:
            print(f"  [WARN] Skipped C codegen (MHSA kernels not yet supported on target '{self.target_name}')")

        # 7. Evaluate INT8 accuracy (optional, slow)
        int8_acc = None
        if brevitas_acc is not None and not self.skip_int8_eval:
            print("\n7. Evaluating true INT8 accuracy (100 samples)...")
            is_transformer = any(kw in test_name.lower() for kw in ['mhsa', 'transformer', 'tinymyo', 'layernorm', 'luna'])
            try:
                int8_acc = self.evaluate_int8_accuracy(
                    network_info_path=golden_dir / "network_info.json",
                    weights_dir=golden_dir / "weights",
                    image_size=config.get('input_resize'),
                    test_samples=100,
                    is_transformer=is_transformer
                )
                print(f"  [OK] INT8 accuracy: {int8_acc:.2f}% (100 samples)")
                acc_diff = abs(brevitas_acc - int8_acc)
                if acc_diff < 2.0:
                    print(f"  [OK] Matches Brevitas (diff: {acc_diff:.2f}%)")
                else:
                    print(f"  [WARN] Differs from Brevitas by {acc_diff:.2f}%")
            except Exception as e:
                print(f"  [WARN] INT8 evaluation failed: {e}")

        # 8. Save accuracy report
        if brevitas_acc is not None or fp32_acc is not None:
            report_path = self.save_accuracy_report(
                test_dir, test_name, brevitas_acc, int8_acc,
                fp32_acc=fp32_acc,
                brevitas_samples=1000, int8_samples=100 if int8_acc else 0,
                fp32_samples=1000 if fp32_acc else 0
            )
            print(f"\n8. Accuracy report saved to {report_path}")

        print(f"\n{'='*80}")
        print(f"[PASS] Test {test_name} generation complete!")
        print(f"\n   ACCURACY SUMMARY:")
        if fp32_acc is not None:
            print(f"   FP32 baseline:     {fp32_acc:.2f}%")
        if brevitas_acc is not None:
            print(f"   Brevitas (fake-Q): {brevitas_acc:.2f}%", end="")
            if fp32_acc is not None:
                print(f"  ({brevitas_acc - fp32_acc:+.2f}% from FP32)")
            else:
                print()
        if int8_acc is not None:
            print(f"   True INT8:         {int8_acc:.2f}%", end="")
            if brevitas_acc is not None:
                print(f"  ({int8_acc - brevitas_acc:+.2f}% from Brevitas)")
            else:
                print()
        if fp32_acc is not None and int8_acc is not None:
            total_drop = int8_acc - fp32_acc
            print(f"   ─────────────────────────────")
            print(f"   Total degradation: {total_drop:+.2f}%")
        print(f"{'='*80}")
        if self.board_mode:
            print(f"Board code: {generated_dir.absolute()}")
            print(f"\nReady for hardware deployment:")
            print(f"  cd {generated_dir}")
            print(f"  make clean all run platform=board")
        else:
            print(f"Output directory: {test_dir.absolute()}")
            if config.get('enable_codegen', True):
                print(f"\nNext: Manually test on target '{self.target_name}':")
                print(f"  cd {generated_dir}")
                print(f"  make clean all && make run")
                print(f"  Expected: 0.0% error rate\n")
            else:
                print(f"\nNext: Implement MHSA kernels before deployment on '{self.target_name}'.\n")
        return True

    def generate_all(self):
        """Generate all tests."""
        print("="*80)
        print("ARES Test Suite Generation")
        print("="*80)
        print(f"\nGenerating {len(self.NETWORKS)} test networks...")
        print(f"Output directory: {self.output_dir.absolute()}\n")

        success_count = 0
        failed_tests = []
        completed_tests = []

        for test_name in self.NETWORKS.keys():
            try:
                self.generate_test(test_name)
                success_count += 1
                completed_tests.append(test_name)
            except Exception as e:
                print(f"\n[FAIL] FAILED: {test_name}")
                print(f"   Error: {e}")
                failed_tests.append(test_name)
                import traceback
                traceback.print_exc()

        print("\n" + "="*80)
        print(f"Test Generation Summary: {success_count}/{len(self.NETWORKS)} successful")
        print("="*80)

        if failed_tests:
            print(f"\n[WARN]  Failed tests: {', '.join(failed_tests)}")
        else:
            print("\n[PASS] ALL TESTS GENERATED SUCCESSFULLY")

        print(f"\nOutputs saved to: {self.output_dir.absolute()}")
        print("\nNext steps:")
        print("  1. Review generated C code in tests/outputs/*/generated/")
        print(f"  2. Manually test each on target '{self.target_name}':")
        for test_name in self.NETWORKS.keys():
            if test_name not in failed_tests:
                print(f"     cd tests/outputs/{test_name}/generated && make clean all && make run")
        print("  3. Verify 0.0% error rate for all tests")
        return completed_tests


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate ARES test suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_all_tests.py                    # Generate all 30+ tests
  python generate_all_tests.py --test test_1_simplecnn   # Single test
  python generate_all_tests.py --output-dir ./my_tests   # Custom output dir
        """
    )
    parser.add_argument('--test', type=str,
                       choices=list(TestGenerator.NETWORKS.keys()),
                       help='Generate single test')
    parser.add_argument('--output-dir', type=str, default='tests/outputs',
                       help='Output directory (default: tests/outputs)')
    parser.add_argument('--mnist-root', type=str, default='data/mnist',
                       help='MNIST data directory (default: data/mnist)')
    parser.add_argument('--skip-gvsoc', action='store_true',
                       help='Skip automatic gvsoc regression after codegen')
    parser.add_argument(
        '--target',
        type=str,
        default='gap9',
        choices=available_targets(),
        help='Codegen target (default: gap9)',
    )
    parser.add_argument('--gap-env', type=str,
                       default=str(Path("tools/gap9_env_gvsoc.sh")),
                       help='Path to target environment script for gvsoc runs (default: tools/gap9_env_gvsoc.sh)')
    parser.add_argument('--board', action='store_true',
                       help='Generate board-ready code (minimal prints, no golden checks, clean timing)')
    parser.add_argument('--no-l1-weight-caching', action='store_true',
                       help='Disable L1 weight caching optimization (for baseline benchmarking)')
    parser.add_argument('--skip-int8-eval', action='store_true',
                       help='Skip slow INT8 accuracy evaluation (default: enabled)')
    parser.add_argument('--eval-int8', action='store_true',
                       help='Enable INT8 accuracy evaluation (100 samples, slow)')
    parser.add_argument('--auto-tune', action='store_true',
                       help='Run auto-tuner on bottleneck layers after GVSOC validation (default: ON)')
    parser.add_argument('--skip-auto-tune', action='store_true',
                       help='Skip auto-tuning (for CI or quick iteration)')
    parser.add_argument('--tune-max-iter', type=int, default=10,
                       help='Max iterations per layer for auto-tuning (default: 10)')
    parser.add_argument('--tune-min-cycles', type=int, default=500000,
                       help='Only tune layers with more than this many cycles (default: 500000)')
    parser.add_argument('--tune-regenerate', action='store_true', default=True,
                       help='Regenerate code after tuning with KB configs (default: True)')
    parser.add_argument('--enable-ne16', action='store_true',
                       help='Enable NE16 hardware accelerator for all eligible Linear layers')

    args = parser.parse_args()

    # By default, skip INT8 eval unless explicitly requested with --eval-int8
    skip_int8 = not args.eval_int8

    generator = TestGenerator(
        output_dir=args.output_dir,
        mnist_root=args.mnist_root,
        board_mode=args.board,
        disable_l1_weight_caching=args.no_l1_weight_caching,
        skip_int8_eval=skip_int8,
        enable_ne16=args.enable_ne16,
        target_name=args.target,
    )

    completed = []
    if args.test:
        generator.generate_test(args.test)
        completed = [args.test]
    else:
        completed = generator.generate_all()

    if not args.skip_gvsoc and completed:
        from tests.run_gap9_projects import run_gap9_projects

        gap_env_path = Path(args.gap_env).expanduser()
        outputs_dir = Path(args.output_dir)
        runnable = [t for t in completed if TestGenerator.NETWORKS[t].get('enable_codegen', True)]
        if runnable:
            print("\n============================================================")
            print(f"Running {args.target} gvsoc regression for generated tests")
            print("============================================================")
            results = run_gap9_projects(
                gap_env=gap_env_path,
                outputs_dir=outputs_dir,
                tests=runnable,
            )
            failed = [r for r in results if not r.success]
            if failed:
                print("\nSome gvsoc runs failed:")
                for res in failed:
                    print(f"  - {res.test_name}: see {res.log_path}")
                sys.exit(1)
            else:
                print("\nAll gvsoc tests passed.")

            # Run auto-tuner unless skipped
            # Auto-tune is ON by default (use --skip-auto-tune to disable)
            run_auto_tune = args.auto_tune or not args.skip_auto_tune
            if run_auto_tune and not args.skip_auto_tune:
                print("\n============================================================")
                print("Running Auto-Tuner on bottleneck layers")
                print("============================================================")
                try:
                    from codegen.optimization import AutoTuner
                    for test_name in runnable:
                        print(f"\n--- Auto-tuning {test_name} ---")
                        tuner = AutoTuner(
                            test_name=test_name,
                            max_iterations=args.tune_max_iter,
                            verbose=True
                        )
                        if args.tune_regenerate:
                            # Use tune_and_regenerate for full workflow
                            tuner.tune_and_regenerate(
                                min_cycles_threshold=args.tune_min_cycles,
                                verify_improvement=True
                            )
                        else:
                            # Just tune, don't regenerate
                            tuner.tune_all(min_cycles_threshold=args.tune_min_cycles)
                    print("\nAuto-tuning complete. Results recorded to knowledge base.")
                except Exception as e:
                    print(f"Warning: Auto-tuner failed: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print(f"\n(No {args.target} runs triggered — new tests pending kernel support.)")
