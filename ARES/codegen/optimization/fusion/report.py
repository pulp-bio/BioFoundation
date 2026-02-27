# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Fusion debug artifact writer utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

FusionSpec = Dict[str, Any]


def write_fusion_report(
    output_dir: str,
    fusions: Sequence[FusionSpec],
    *,
    test_name: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> str:
    """
    Write fusion report JSON artifact for debugging.

    Returns:
        Path to written report.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "test_name": test_name,
        "fusion_count": len(fusions),
        "fusions": list(fusions),
        "metadata": dict(metadata or {}),
    }

    report_path = root / "fusion_report.json"
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return str(report_path)

