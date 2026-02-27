#!/usr/bin/env python
# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Record a manual optimization to the ARES knowledge base.

Use this tool when you've manually optimized a layer and want to save
the configuration for future use.

Usage:
    # Record a successful Linear optimization
    python tools/record_optimization.py \
        --op-type linear_int8 \
        --shape '{"M": 400, "N": 768, "K": 192}' \
        --tile-config '{"tile_m": 32, "tile_n": 64}' \
        --measured-macs-per-cycle 10.5 \
        --source "manual_tune_tinymyo" \
        --notes "Tuned for TinyMyo fc1 layer"

    # Record a Conv2D optimization with compile flags
    python tools/record_optimization.py \
        --op-type conv2d_int8 \
        --shape '{"kernel_h": 3, "kernel_w": 3, "in_channels": 64, "out_channels": 128}' \
        --kernel-config '{"outch_unroll": 4}' \
        --compile-flags '{"CONV2D_IM2COL_OUTCH_UNROLL": 4}' \
        --measured-macs-per-cycle 9.0

    # Record a negative result (optimization that didn't work)
    python tools/record_optimization.py \
        --op-type mhsa_int8 \
        --negative \
        --attempted-config '{"fuse_softmax_av": true}' \
        --result "regressed +45%" \
        --notes "Caused 45% regression on test_36"

Examples:
    # Quick recording from profiling session
    python tools/record_optimization.py \
        --op-type linear_int8 \
        --shape '{"M": 32, "N": 256, "K": 64}' \
        --measured-macs-per-cycle 11.2 \
        --source "profile_session_20260106"
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from codegen.optimization import KnowledgeBase


def main():
    parser = argparse.ArgumentParser(
        description="Record an optimization to the ARES knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument(
        "--op-type", "-o",
        required=True,
        help="Operation type (e.g., linear_int8, conv2d_int8, mhsa_int8)"
    )

    # Shape/config arguments
    parser.add_argument(
        "--shape", "-s",
        type=json.loads,
        default={},
        help="Layer shape as JSON (e.g., '{\"M\": 32, \"N\": 256}')"
    )

    parser.add_argument(
        "--tile-config",
        type=json.loads,
        default={},
        help="Tile configuration as JSON"
    )

    parser.add_argument(
        "--kernel-config",
        type=json.loads,
        default={},
        help="Kernel configuration as JSON"
    )

    parser.add_argument(
        "--pipeline-config",
        type=json.loads,
        default={},
        help="Pipeline configuration as JSON"
    )

    parser.add_argument(
        "--compile-flags",
        type=json.loads,
        default={},
        help="Compile flags as JSON"
    )

    # Performance measurements
    parser.add_argument(
        "--measured-cycles",
        type=int,
        help="Measured total cycles"
    )

    parser.add_argument(
        "--measured-macs-per-cycle",
        type=float,
        help="Measured MACs per cycle"
    )

    parser.add_argument(
        "--measured-overlap",
        type=float,
        help="Measured DMA/compute overlap ratio (0-1)"
    )

    # Metadata
    parser.add_argument(
        "--source",
        default="manual_optimization",
        help="Source identifier (e.g., 'manual_tune_test41')"
    )

    parser.add_argument(
        "--test-network",
        default="",
        help="Test network used for validation"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.9,
        help="Confidence score 0-1 (default: 0.9)"
    )

    parser.add_argument(
        "--notes",
        default="",
        help="Additional notes about this optimization"
    )

    parser.add_argument(
        "--description",
        default="",
        help="Human-readable description"
    )

    # Negative result recording
    parser.add_argument(
        "--negative",
        action="store_true",
        help="Record a negative result (optimization that didn't work)"
    )

    parser.add_argument(
        "--attempted-config",
        type=json.loads,
        help="Configuration that was attempted (for negative results)"
    )

    parser.add_argument(
        "--result",
        help="Result description for negative entry (e.g., 'regressed +45%')"
    )

    # Output control
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be recorded without saving"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Load knowledge base
    kb = KnowledgeBase()

    if args.negative:
        # Record negative result
        if not args.attempted_config:
            parser.error("--attempted-config required for negative results")
        if not args.result:
            parser.error("--result required for negative results")

        if args.verbose or args.dry_run:
            print("\nRecording negative result:")
            print(f"  Op type: {args.op_type}")
            print(f"  Attempted config: {args.attempted_config}")
            print(f"  Result: {args.result}")
            print(f"  Notes: {args.notes}")

        if not args.dry_run:
            kb.record_negative(
                op_type=args.op_type,
                attempted_config=args.attempted_config,
                result=args.result,
                source=args.source,
                notes=args.notes
            )
            kb.save()
            print(f"\n[OK] Negative result recorded to knowledge base")

    else:
        # Record optimization entry
        if not args.shape and not args.tile_config:
            parser.error("At least --shape or --tile-config required")

        # Generate description if not provided
        description = args.description
        if not description:
            if args.tile_config:
                description = f"Manual optimization: {args.tile_config}"
            else:
                description = f"Manual optimization for {args.op_type}"

        if args.verbose or args.dry_run:
            print("\nRecording optimization:")
            print(f"  Op type: {args.op_type}")
            print(f"  Shape: {args.shape}")
            print(f"  Tile config: {args.tile_config}")
            print(f"  Kernel config: {args.kernel_config}")
            print(f"  Pipeline config: {args.pipeline_config}")
            print(f"  Compile flags: {args.compile_flags}")
            print(f"  Measured cycles: {args.measured_cycles}")
            print(f"  Measured MACs/cycle: {args.measured_macs_per_cycle}")
            print(f"  Source: {args.source}")
            print(f"  Description: {description}")

        if not args.dry_run:
            kb.record(
                op_type=args.op_type,
                shape=args.shape,
                tile_config=args.tile_config,
                kernel_config=args.kernel_config,
                pipeline_config=args.pipeline_config,
                compile_flags=args.compile_flags,
                measured_cycles=args.measured_cycles,
                measured_macs_per_cycle=args.measured_macs_per_cycle,
                measured_overlap_ratio=args.measured_overlap,
                source=args.source,
                test_network=args.test_network,
                confidence=args.confidence,
                notes=args.notes,
                description=description
            )
            kb.save()
            print(f"\n[OK] Optimization recorded to knowledge base")
            print(f"  Total entries: {len(kb.entries)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
