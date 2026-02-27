# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Golden INT8 Output Generator

Generates golden reference INT8 outputs for MCU verification.

For each test case:
1. Run INT8 inference through the network
2. Save INT8 intermediate outputs for every layer
3. Save final FP32 outputs
4. Create a verification package for the MCU

These golden outputs allow the MCU to verify that its INT8 computation
matches our reference implementation exactly.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List
import sys

sys.path.append(str(Path(__file__).parent))
from int8_inference import INT8InferenceEngine


class GoldenOutputGenerator:
    """Generate golden INT8 outputs for MCU verification."""

    def __init__(self, network_info: Dict[str, Any], use_i_softmax: bool = False, softmax_lut_path: str = None,
                 use_i_gelu: bool = False, use_i_layernorm: bool = False):
        """
        Initialize generator with network information.

        Args:
            network_info: Dictionary from BrevitasExtractor
            use_i_softmax: Use integer-only LUT-based softmax for MHSA layers
            softmax_lut_path: Path to softmax LUT binary file (optional, uses builtin LUT if not provided)
            use_i_gelu: Use integer-only LUT-based GELU for transformer MLP layers
            use_i_layernorm: Use integer-only LayerNorm with binary search sqrt
        """
        self.raw_network_info = network_info
        self.layer_names = [k for k in network_info.keys() if not k.startswith('__')]
        self.network_info = {k: network_info[k] for k in self.layer_names}
        self.engine = INT8InferenceEngine(
            network_info,
            use_i_softmax=use_i_softmax,
            softmax_lut_path=softmax_lut_path,
            use_i_gelu=use_i_gelu,
            use_i_layernorm=use_i_layernorm
        )

    def generate_test_cases(self, num_cases: int = 5) -> List[np.ndarray]:
        """
        Generate test input cases.

        Args:
            num_cases: Number of test cases to generate

        Returns:
            List of FP32 input tensors [1, 1, 28, 28]
        """
        test_cases = []

        # Test case 1: All zeros
        test_cases.append(np.zeros((1, 1, 28, 28), dtype=np.float32))

        # Test case 2: All ones
        test_cases.append(np.ones((1, 1, 28, 28), dtype=np.float32))

        # Test case 3: Random normal
        test_cases.append(np.random.randn(1, 1, 28, 28).astype(np.float32) * 0.5)

        # Test case 4: Edge values (simulating normalized MNIST)
        edge_case = np.random.rand(1, 1, 28, 28).astype(np.float32)
        edge_case = (edge_case - 0.5) * 2.0  # Range: [-1, 1]
        test_cases.append(edge_case)

        # Test case 5: Pattern (checkerboard)
        checkerboard = np.zeros((1, 1, 28, 28), dtype=np.float32)
        checkerboard[0, 0, ::2, ::2] = 1.0
        checkerboard[0, 0, 1::2, 1::2] = 1.0
        test_cases.append(checkerboard)

        return test_cases[:num_cases]

    def generate_golden_outputs(
        self,
        test_cases: List[np.ndarray],
        output_dir: str = "golden_outputs/test_cases/"
    ):
        """
        Generate golden INT8 outputs for test cases.

        Args:
            test_cases: List of FP32 input tensors
            output_dir: Directory to save golden outputs
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("="*80)
        print("Generating Golden INT8 Outputs")
        print("="*80)
        print(f"Number of test cases: {len(test_cases)}")
        print(f"Output directory: {output_dir}")
        print()

        all_results = []

        for i, x_fp32 in enumerate(test_cases):
            print(f"Test Case {i+1}/{len(test_cases)}")
            print("-"*80)

            # Run INT8 inference
            output_fp32, intermediate_outputs, output_scales = self.engine.forward(x_fp32, verbose=False)

            # Update network info with observed output scales
            for layer_name, scale_val in output_scales.items():
                if layer_name in self.network_info:
                    self.network_info[layer_name]['scale_output'] = float(scale_val)

            # Save test case
            test_case_dir = output_dir / f"test_case_{i+1}"
            test_case_dir.mkdir(exist_ok=True)

            # Save input(s)
            is_multi_input = isinstance(x_fp32, list)
            if is_multi_input:
                # Multi-input model: save each input separately
                for j, inp in enumerate(x_fp32):
                    np.save(test_case_dir / f"input{j}_fp32.npy", inp)
                    print(f"  Input {j}: {inp.shape}, range=[{inp.min():.3f}, {inp.max():.3f}]")
                input_shapes = [list(inp.shape) for inp in x_fp32]
            else:
                np.save(test_case_dir / "input_fp32.npy", x_fp32)
                print(f"  Input: {x_fp32.shape}, range=[{x_fp32.min():.3f}, {x_fp32.max():.3f}]")
                input_shapes = list(x_fp32.shape)

            # Save output
            np.save(test_case_dir / "output_fp32.npy", output_fp32)
            predicted_class = np.argmax(output_fp32[0])
            print(f"  Output: {output_fp32.shape}, predicted class={predicted_class}")

            # Save intermediate INT8 outputs
            intermediate_dir = test_case_dir / "intermediate_int8"
            intermediate_dir.mkdir(exist_ok=True)

            metadata = {
                'test_case_id': i + 1,
                'input_shape': input_shapes,  # Single shape or list of shapes for multi-input
                'output_shape': list(output_fp32.shape),
                'predicted_class': int(predicted_class),
                'layers': {}
            }

            for layer_name, int8_output in intermediate_outputs.items():
                if int8_output is not None:
                    # Save INT8 output
                    layer_file = intermediate_dir / f"{layer_name}_int8.npy"
                    np.save(layer_file, int8_output)

                    # Add to metadata
                    metadata['layers'][layer_name] = {
                        'shape': list(int8_output.shape),
                        'min': int(int8_output.min()),
                        'max': int(int8_output.max()),
                        'file': str(layer_file.relative_to(test_case_dir))
                    }

                    print(f"    {layer_name:20s}: shape={str(int8_output.shape):20s} "
                          f"range=[{int8_output.min():4d}, {int8_output.max():4d}]")

            # Save metadata
            with open(test_case_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            all_results.append(metadata)
            print()

        # Save summary
        summary = {
            'num_test_cases': len(test_cases),
            'network_layers': len(self.layer_names),
            'test_cases': all_results
        }

        with open(output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        # Save updated network_info back to file (includes corrected Add inputs for ResNet)
        # output_dir is like "golden_outputs/test_cases/", so network_info.json is in parent
        network_info_path = output_dir.parent / "network_info.json"
        # Create a copy without numpy arrays (they're already saved as .npy files)
        network_info_clean = {}
        for key, value in self.network_info.items():
            if isinstance(value, dict):
                layer_clean = {k: v for k, v in value.items()
                               if not isinstance(v, np.ndarray)}
                network_info_clean[key] = layer_clean
            else:
                network_info_clean[key] = value
        with open(network_info_path, 'w') as f:
            json.dump(network_info_clean, f, indent=2)
        print(f"[OK] Network info updated with runtime scales")

        print("="*80)
        print(f"[PASS] Generated {len(test_cases)} golden test cases")
        print(f"[PASS] Saved to {output_dir}")
        print("="*80)

    def create_c_header(
        self,
        test_case_id: int = 1,
        output_file: str = "golden_outputs/golden_reference.h"
    ):
        """
        Create C header file with golden outputs for MCU verification.

        Args:
            test_case_id: Which test case to use (1-indexed)
            output_file: Path to output C header file
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Load test case
        test_case_dir = Path(f"golden_outputs/test_cases/test_case_{test_case_id}")
        if not test_case_dir.exists():
            print(f"[FAIL] Test case {test_case_id} not found at {test_case_dir}")
            return

        # Load metadata
        with open(test_case_dir / "metadata.json") as f:
            metadata = json.load(f)

        # Load input
        input_fp32 = np.load(test_case_dir / "input_fp32.npy")

        # Start building C header
        lines = []
        lines.append("/*")
        lines.append(" * Golden INT8 Reference Outputs")
        lines.append(" * ")
        lines.append(f" * Generated for Test Case {test_case_id}")
        lines.append(f" * Network: SimpleCNN (INT8 quantized)")
        lines.append(" * ")
        lines.append(" * This file contains INT8 intermediate outputs for each layer.")
        lines.append(" * Use these to verify your MCU implementation is correct.")
        lines.append(" */")
        lines.append("")
        lines.append("#ifndef GOLDEN_REFERENCE_H")
        lines.append("#define GOLDEN_REFERENCE_H")
        lines.append("")
        lines.append("#include <stdint.h>")
        lines.append("")

        # Add input
        input_quantized = self._quantize_for_c(input_fp32, scale=1.0)
        lines.append(f"// Input: shape={metadata['input_shape']}")
        lines.append(f"#define INPUT_SIZE {input_quantized.size}")
        lines.append(f"const int8_t golden_input[{input_quantized.size}] = {{")
        lines.extend(self._format_array_for_c(input_quantized.flatten(), indent=4))
        lines.append("};")
        lines.append("")

        # Add intermediate outputs
        for layer_name, layer_info in metadata['layers'].items():
            layer_file = test_case_dir / layer_info['file']
            int8_output = np.load(layer_file)

            safe_name = layer_name.replace('.', '_')
            lines.append(f"// Layer: {layer_name}")
            lines.append(f"// Shape: {layer_info['shape']}, Range: [{layer_info['min']}, {layer_info['max']}]")
            lines.append(f"#define {safe_name.upper()}_SIZE {int8_output.size}")
            lines.append(f"const int8_t golden_{safe_name}[{int8_output.size}] = {{")
            lines.extend(self._format_array_for_c(int8_output.flatten(), indent=4))
            lines.append("};")
            lines.append("")

        lines.append("#endif // GOLDEN_REFERENCE_H")
        lines.append("")

        # Write to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))

        print(f"[PASS] Created C header: {output_file}")
        print(f"   Test case: {test_case_id}")
        print(f"   Input size: {input_quantized.size} bytes")
        print(f"   Layers: {len(metadata['layers'])}")

    def _quantize_for_c(self, x_fp32: np.ndarray, scale: float) -> np.ndarray:
        """Quantize FP32 to INT8 for C header."""
        return np.clip(np.round(x_fp32 / scale), -128, 127).astype(np.int8)

    def _format_array_for_c(self, arr: np.ndarray, indent: int = 4, values_per_line: int = 16) -> List[str]:
        """Format numpy array as C array initializer."""
        lines = []
        indent_str = ' ' * indent

        for i in range(0, len(arr), values_per_line):
            chunk = arr[i:i+values_per_line]
            values = ', '.join(f"{int(v):4d}" for v in chunk)
            if i + values_per_line < len(arr):
                lines.append(f"{indent_str}{values},")
            else:
                lines.append(f"{indent_str}{values}")

        return lines


def main():
    """Main function to generate golden outputs."""
    print("="*80)
    print("Golden INT8 Output Generator")
    print("="*80)
    print()

    # Load network info
    network_info_path = Path("golden_outputs/network_info.json")
    if not network_info_path.exists():
        print(f"[FAIL] Network info not found: {network_info_path}")
        print("   Please run tools/pytorch_extractor.py first!")
        return

    with open(network_info_path) as f:
        network_info = json.load(f)

    # Load weights
    weights_dir = Path("golden_outputs/weights")
    for layer_name, layer_data in network_info.items():
        if 'weight_int8' in layer_data:
            weight_path = weights_dir / f"{layer_name}_weight_int8.npy"
            if weight_path.exists():
                layer_data['weight_int8'] = np.load(weight_path)

        if 'bias_fp32' in layer_data:
            bias_path = weights_dir / f"{layer_name}_bias_fp32.npy"
            if bias_path.exists():
                layer_data['bias_fp32'] = np.load(bias_path)

    print(f"[PASS] Loaded network with {len(network_info)} layers")
    print()

    # Create generator
    generator = GoldenOutputGenerator(network_info)

    # Generate test cases
    test_cases = generator.generate_test_cases(num_cases=5)

    # Generate golden outputs
    generator.generate_golden_outputs(test_cases)

    # Create C header for first test case
    print()
    generator.create_c_header(test_case_id=1)

    print()
    print("="*80)
    print("[PASS] Golden Output Generation Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
