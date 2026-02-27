#!/usr/bin/env python
# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
ARES Auto-Tuner CLI

Automatically tunes layer configurations by running GVSOC iterations
and finding optimal settings. Results are recorded to the knowledge base.

Usage:
    # Show what would be tuned (dry run)
    python tools/auto_tune.py --test test_41_luna_base --dry-run

    # Tune all bottleneck layers
    python tools/auto_tune.py --test test_41_luna_base --tune-all

    # Tune specific layers
    python tools/auto_tune.py --test test_41_luna_base --layers freq_fc1 cross_attn_unify

    # Tune with more iterations (slower but more thorough)
    python tools/auto_tune.py --test test_41_luna_base --tune-all --max-iter 50

    # Tune only layers with >500K cycles
    python tools/auto_tune.py --test test_41_luna_base --tune-all --min-cycles 500000

Examples:
    # Quick tune of TinyMyo bottlenecks
    python tools/auto_tune.py --test test_36_tinymyo_8ch_400tok --tune-all --max-iter 10

    # Thorough tune of LUNA attention layers
    python tools/auto_tune.py --test test_41_luna_base --layers cross_attn_unify freq_fc1 --max-iter 30
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from codegen.optimization import AutoTuner


def main():
    parser = argparse.ArgumentParser(
        description="ARES Auto-Tuner - automatically optimize layer configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--test", "-t",
        required=True,
        help="Test network name (e.g., test_41_luna_base)"
    )

    parser.add_argument(
        "--tune-all", "-a",
        action="store_true",
        help="Tune all bottleneck layers"
    )

    parser.add_argument(
        "--layers", "-l",
        nargs="+",
        help="Specific layers to tune"
    )

    parser.add_argument(
        "--max-iter", "-m",
        type=int,
        default=20,
        help="Maximum configurations to try per layer (default: 20)"
    )

    parser.add_argument(
        "--min-cycles",
        type=int,
        default=100000,
        help="Only tune layers with more than this many cycles (default: 100000)"
    )

    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show tuning plan without running"
    )

    parser.add_argument(
        "--no-kb",
        action="store_true",
        help="Don't record results to knowledge base"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Verbose output (default: True)"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )

    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug mode with full tracebacks on errors"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.tune_all and not args.layers and not args.dry_run:
        parser.error("Specify --tune-all, --layers, or --dry-run")

    verbose = args.verbose and not args.quiet

    # Create tuner
    try:
        tuner = AutoTuner(
            test_name=args.test,
            max_iterations=args.max_iter,
            verbose=verbose,
            debug=args.debug
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    if args.dry_run:
        tuner.show_tuning_plan()
        return 0

    # Run tuning
    try:
        if args.tune_all:
            results = tuner.tune_all(
                min_cycles_threshold=args.min_cycles
            )
        else:
            # Tune specific layers
            results = []
            for layer_name in args.layers:
                # Get layer shape from network_info
                shape = tuner._get_layer_shape(layer_name)
                if shape is None:
                    print(f"Warning: Could not determine shape for {layer_name}")
                    continue

                # Infer op_type from layer name
                op_type = infer_op_type(layer_name)

                result = tuner.tune_layer(
                    layer_name=layer_name,
                    op_type=op_type,
                    shape=shape,
                    record_to_kb=not args.no_kb
                )
                results.append(result)

        # Print final summary
        if verbose and results:
            print(f"\n{'='*60}")
            print("AUTO-TUNING COMPLETE")
            print(f"{'='*60}")
            print(f"Layers tuned: {len(results)}")
            print(f"Results recorded to KB: {sum(1 for r in results if r.recorded_to_kb)}")

            # Show KB location
            print(f"\nKnowledge base: codegen/optimization/data/knowledge_base.json")
            print("Run 'python tools/knowledge_base_cli.py list' to see all entries")

        return 0

    except KeyboardInterrupt:
        print("\nTuning interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


def infer_op_type(layer_name: str) -> str:
    """Infer operation type from layer name."""
    name_lower = layer_name.lower()

    if 'conv' in name_lower:
        return 'conv2d_int8'
    elif 'linear' in name_lower or 'fc' in name_lower:
        return 'linear_int8'
    elif 'mhsa' in name_lower or 'attention' in name_lower:
        return 'mhsa_int8'
    elif 'norm' in name_lower:
        return 'layernorm_int8'
    elif 'gelu' in name_lower:
        return 'gelu_int8'
    elif 'ssm' in name_lower or 'mamba' in name_lower:
        return 'ssm_int8'

    return 'unknown'


if __name__ == "__main__":
    sys.exit(main())
