# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Measure Quantized Accuracy

This script measures the "True Quantized Accuracy" of a network by running 
the full validation dataset through the bit-exact INT8 inference engine.

It compares:
1. Original PyTorch (FP32) Accuracy
2. True INT8 Accuracy (using tools/int8_inference.py)

Usage:
    python measure_quantized_accuracy.py --test test_1_simplecnn
"""
import argparse
import sys
import json
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from tests.generate_all_tests import TestGenerator
from tools.int8_inference import INT8InferenceEngine


def load_network_info_and_weights(test_dir):
    """Load network info and weights from the golden_outputs directory."""
    network_info_path = test_dir / 'golden_outputs' / 'network_info.json'
    if not network_info_path.exists():
        raise FileNotFoundError(f"Network info not found at {network_info_path}. Run generate_all_tests.py first.")

    with open(network_info_path) as f:
        network_info = json.load(f)

    weights_dir = test_dir / 'golden_outputs' / 'weights'
    
    # Reload weights into the dictionary (similar to test_int8_inference)
    for layer_name, layer_data in network_info.items():
        if layer_name == '__layer_order__': # Skip the special layer order list
            continue
        # Standard weights
        if 'weight_int8' in layer_data:
            weight_path = weights_dir / f"{layer_name}_weight_int8.npy"
            if weight_path.exists():
                layer_data['weight_int8'] = np.load(weight_path)
        
        # MHSA weights
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

        # SSM / Mamba weights
        if layer_data.get('type') in ('SSM', 'MambaBlock'):
             # List all potential parameter files for these complex blocks
            param_suffixes = [
                'x_proj_weight_int8', 'x_proj_bias_fp32',
                'dt_proj_weight_int8', 'dt_proj_bias_fp32',
                'A_log_fp32', 'D_fp32',
                'in_proj_weight_int8', 'in_proj_bias_fp32',
                'conv1d_weight_int8', 'conv1d_bias_fp32',
                'ssm_x_proj_weight_int8', 'ssm_x_proj_bias_fp32',
                'ssm_dt_proj_weight_int8', 'ssm_dt_proj_bias_fp32',
                'ssm_A_log_fp32', 'ssm_D_fp32',
                'out_proj_weight_int8', 'out_proj_bias_fp32',
            ]
            for suffix in param_suffixes:
                 param_path = weights_dir / f"{layer_name}_{suffix}.npy"
                 if param_path.exists():
                    layer_data[suffix] = np.load(param_path)

        # Standard bias
        if 'bias_fp32' in layer_data:
            bias_path = weights_dir / f"{layer_name}_bias_fp32.npy"
            if bias_path.exists():
                layer_data['bias_fp32'] = np.load(bias_path)

    return network_info

def evaluate_fp32(network, dataloader, device):
    """Evaluate standard PyTorch FP32 accuracy."""
    network.eval()
    network.to(device)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Evaluating FP32"):
            data, target = data.to(device), target.to(device)
            output = network(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
    return 100.0 * correct / total

import multiprocessing
import os

def _init_worker(network_info, use_i_softmax, use_i_gelu, use_i_layernorm):
    """Initialize the global engine in the worker process."""
    global _global_int8_engine
    # Re-seed random for safety, though we are deterministic
    np.random.seed(os.getpid())
    
    _global_int8_engine = INT8InferenceEngine(
        network_info,
        use_i_softmax=use_i_softmax,
        use_i_gelu=use_i_gelu,
        use_i_layernorm=use_i_layernorm
    )

def _process_batch(args):
    """Worker function to process a single batch."""
    x_batch, y_batch = args
    global _global_int8_engine
    
    # Run inference (verbose=False to keep stdout clean)
    logits_fp32, _, _ = _global_int8_engine.forward(x_batch, verbose=False)
    
    predicted = np.argmax(logits_fp32, axis=1)
    correct = (predicted == y_batch).sum()
    total = len(y_batch)
    return correct, total

def evaluate_int8(engine_config, dataloader):
    """
    Evaluate True INT8 accuracy using multiprocessing.
    
    Args:
        engine_config: Tuple of (network_info, use_i_softmax, use_i_gelu, use_i_layernorm)
        dataloader: PyTorch dataloader
    """
    # prepare data for multiprocessing
    batches = []
    for data, target in dataloader:
        batches.append((data.numpy(), target.numpy()))
    
    # Determine number of workers
    num_workers = min(multiprocessing.cpu_count(), len(batches))
    # Cap at 16 to avoid excessive memory usage if network is huge
    num_workers = min(num_workers, 16)
    
    print(f"   Parallelizing INT8 inference across {num_workers} CPU cores...")
    
    correct = 0
    total = 0
    
    with multiprocessing.Pool(processes=num_workers, initializer=_init_worker, initargs=engine_config) as pool:
        results = list(tqdm(pool.imap(_process_batch, batches), total=len(batches), desc="Evaluating INT8"))
        
    for c, t in results:
        correct += c
        total += t
        
    return 100.0 * correct / total

def main():
    parser = argparse.ArgumentParser(description='Measure True Quantized Accuracy')
    parser.add_argument('--test', type=str, required=True, 
                        choices=list(TestGenerator.NETWORKS.keys()),
                        help='Name of the test to evaluate')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for evaluation')
    parser.add_argument('--subset', type=int, default=1000,
                        help='Number of test samples to use (default: 1000)')
    
    args = parser.parse_args()
    
    print(f"Evaluating {args.test}...")
    
    # 1. Setup paths and config
    generator = TestGenerator() # Initialize to get paths
    test_config = generator.NETWORKS[args.test]
    test_dir = generator.output_dir / args.test
    
    if not test_dir.exists():
        print(f"[FAIL] Test directory {test_dir} does not exist.")
        print(f"   Please run: python tests/generate_all_tests.py --test {args.test}")
        sys.exit(1)

    # 2. Load Data
    # Use the generator's data loading logic to ensure consistency
    print(f"Loading data (subset={args.subset})...")
    dataset = generator.load_mnist(train=False, subset_size=args.subset, 
                                 image_size=test_config.get('input_resize'))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 3. FP32 Evaluation
    print("\n1. Measuring Baseline FP32 Accuracy...")
    # Load the saved model
    model_path = test_dir / 'models' / f"{args.test}.pth"
    if not model_path.exists():
         print(f"[FAIL] Model file not found: {model_path}")
         sys.exit(1)
         
    NetworkClass = test_config['class']
    model_kwargs = test_config.get('model_kwargs', {})
    network = NetworkClass(**model_kwargs)
    network.load_state_dict(torch.load(model_path, map_location=generator.device))
    
    fp32_acc = evaluate_fp32(network, dataloader, generator.device)
    print(f"   FP32 Accuracy: {fp32_acc:.2f}%")

    # 4. INT8 Evaluation
    print("\n2. Measuring True INT8 Accuracy...")
    try:
        network_info = load_network_info_and_weights(test_dir)
        
        # Configure special flags for transformers
        is_transformer = any(keyword in args.test.lower() for keyword in ['mhsa', 'transformer', 'tinymyo', 'layernorm'])
        
        # Prepare configuration for workers
        # (network_info, use_i_softmax, use_i_gelu, use_i_layernorm)
        engine_config = (
            network_info,
            is_transformer,
            is_transformer,
            is_transformer
        )
        
        int8_acc = evaluate_int8(engine_config, dataloader)
        print(f"   INT8 Accuracy: {int8_acc:.2f}%")
        
        # 5. Report
        print("\n" + "="*40)
        print("Results Summary")
        print("="*40)
        print(f"Test Network:   {args.test}")
        print(f"Samples:        {args.subset}")
        print(f"FP32 Accuracy:  {fp32_acc:.2f}%")
        print(f"INT8 Accuracy:  {int8_acc:.2f}%")
        
        drop = fp32_acc - int8_acc
        print(f"Accuracy Drop:  {drop:.2f}%")
        
        if drop > 1.0:
            print("\n[WARN]  Warning: Significant accuracy drop (>1.0%)")
        else:
            print("\n[PASS] Quantization successful")
            
    except Exception as e:
        print(f"\n[FAIL] INT8 Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
