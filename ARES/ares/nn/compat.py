# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Compatibility entrypoints for ares.nn users."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from tools.model_compatibility_core import scan_model_modules
except ImportError:  # pragma: no cover - fallback when running from repo root package layouts
    from model_compatibility_core import scan_model_modules  # type: ignore


def check_compatibility(
    model: nn.Module,
    strict: bool = True,
    strict_warnings: bool = False,
) -> Dict[str, Any]:
    """
    Check ARES extractor compatibility for a model.

    Args:
        model: PyTorch/Brevitas model.
        strict: If True, raise ValueError on incompatibility.
        strict_warnings: If True, treat warnings as strict failures.

    Returns:
        Dict report with `compatible`, counts, and findings.
    """
    report = scan_model_modules(model.eval())
    report_dict = report.to_dict()

    incompatible = not report.compatible or (strict_warnings and len(report.warnings) > 0)
    if strict and incompatible:
        raise ValueError(
            "Model is not fully compatible with current ARES extraction support. "
            "Use tools/check_model_compatibility.py for a full text report."
        )
    return report_dict
