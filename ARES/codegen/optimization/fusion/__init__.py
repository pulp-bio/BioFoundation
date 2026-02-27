# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Fusion helpers and exports."""

from .checks import (
    EXPECTED_DEFAULT_PATTERN_ORDER,
    EXPECTED_FUSION_TYPES,
    validate_default_registry,
    validate_fusion_payload,
    validate_fusions,
)
from .matcher import detect_fusion_opportunities
from .registry import FusionPattern, FusionRegistry, build_default_registry
from .report import write_fusion_report
from .transformer import FusionTransformer, transform_specs_for_fusion

__all__ = [
    "EXPECTED_DEFAULT_PATTERN_ORDER",
    "EXPECTED_FUSION_TYPES",
    "FusionPattern",
    "FusionRegistry",
    "FusionTransformer",
    "build_default_registry",
    "detect_fusion_opportunities",
    "transform_specs_for_fusion",
    "validate_default_registry",
    "validate_fusion_payload",
    "validate_fusions",
    "write_fusion_report",
]
