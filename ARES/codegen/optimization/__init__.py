# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
ARES Optimization Module

This module provides the knowledge base infrastructure for ARES auto-tuning.
It stores learned optimization configurations and provides lookup/matching
functionality for generating optimized code.

Usage:
    from codegen.optimization import KnowledgeBase

    # Load knowledge base
    kb = KnowledgeBase()

    # Lookup config for a layer
    config = kb.lookup("linear_int8", {"M": 400, "N": 768, "K": 192})
    if config:
        print(f"Found config with {config.performance.macs_per_cycle} MACs/cycle")
        print(f"Tile config: {config.tile_config.config}")

    # Record a new optimization
    kb.record(
        op_type="linear_int8",
        shape={"M": 32, "N": 256, "K": 64},
        tile_config={"tile_m": 32, "tile_n": 64},
        measured_macs_per_cycle=10.5,
        source="auto_tuner_run_001"
    )
    kb.save()

    # Parse and analyze profiles
    from codegen.optimization import ProfileParser, PerformanceAnalyzer

    parser = ProfileParser()
    profile = parser.parse_log("gvsoc_run.log")

    analyzer = PerformanceAnalyzer()
    suggestions = analyzer.analyze(profile)

    # Auto-tune layers
    from codegen.optimization import AutoTuner

    tuner = AutoTuner(test_name="test_41_luna_base")
    tuner.tune_all()
"""

__version__ = "1.0.0"

# Main class
from .knowledge_base import KnowledgeBase

# Data classes
from .config_schema import (
    OpType,
    ShapePattern,
    TileConfig,
    KernelConfig,
    PipelineConfig,
    CompileFlags,
    PerformanceMetrics,
    OptimizationEntry,
    NegativeResult,
    PerformanceBaseline,
)

# Shape matching utilities
from .shape_matching import (
    value_matches,
    shape_matches,
    shape_distance,
    pattern_specificity,
    find_matching_entries,
    find_best_match,
    check_negative_results,
    extract_shape_from_layer,
)

# Profile parsing and analysis
from .profile_parser import (
    LayerProfile,
    NetworkProfile,
    ProfileParser,
    parse_gvsoc_log,
    parse_profile_csv,
)

from .analyzer import (
    OptimizationSuggestion,
    PerformanceAnalyzer,
    analyze_profile,
)

# Search space and auto-tuning
from .search_space import (
    TuningConfig,
    SearchSpace,
    get_search_space,
)

from .auto_tuner import (
    TuningResult,
    AutoTuner,
    run_auto_tuner,
)

# Fusion module
from . import fusion

# Checkpoint scaffolding (fusion/checkpoint workstream)
from . import checkpoints

__all__ = [
    # Main class
    "KnowledgeBase",
    # Enums
    "OpType",
    # Data classes
    "ShapePattern",
    "TileConfig",
    "KernelConfig",
    "PipelineConfig",
    "CompileFlags",
    "PerformanceMetrics",
    "OptimizationEntry",
    "NegativeResult",
    "PerformanceBaseline",
    # Shape matching
    "value_matches",
    "shape_matches",
    "shape_distance",
    "pattern_specificity",
    "find_matching_entries",
    "find_best_match",
    "check_negative_results",
    "extract_shape_from_layer",
    # Profile parsing
    "LayerProfile",
    "NetworkProfile",
    "ProfileParser",
    "parse_gvsoc_log",
    "parse_profile_csv",
    # Analysis
    "OptimizationSuggestion",
    "PerformanceAnalyzer",
    "analyze_profile",
    # Search space
    "TuningConfig",
    "SearchSpace",
    "get_search_space",
    # Auto-tuner
    "TuningResult",
    "AutoTuner",
    "run_auto_tuner",
    # Fusion/checkpoint scaffolding modules
    "fusion",
    "checkpoints",
]
