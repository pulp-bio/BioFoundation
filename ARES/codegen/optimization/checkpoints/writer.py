# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint writer/loader utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

from .schema import build_checkpoint_payload


def write_checkpoint(path: str, payload: Mapping[str, Any]) -> str:
    """Write checkpoint payload as canonical JSON and return file path."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    return str(out_path)


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Load checkpoint payload from JSON file."""
    in_path = Path(path)
    return json.loads(in_path.read_text())


def write_standard_wave_checkpoints(
    output_dir: str,
    *,
    pre_fusion_state: Mapping[str, Any],
    post_fusion_state: Mapping[str, Any],
    metadata: Optional[Mapping[str, Any]] = None,
) -> MutableMapping[str, str]:
    """
    Write Wave checkpoint files currently in scope for fusion/checkpoint work.

    Returns:
        Mapping of checkpoint stage name to file path.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    pre_path = write_checkpoint(
        str(root / "pre_fusion.json"),
        build_checkpoint_payload(
            stage="pre_fusion",
            state=pre_fusion_state,
            metadata=metadata,
        ),
    )
    post_path = write_checkpoint(
        str(root / "post_fusion.json"),
        build_checkpoint_payload(
            stage="post_fusion",
            state=post_fusion_state,
            metadata=metadata,
        ),
    )

    return {
        "pre_fusion": pre_path,
        "post_fusion": post_path,
    }

