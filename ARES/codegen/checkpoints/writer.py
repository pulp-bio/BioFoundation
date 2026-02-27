# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint writer/loader helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .schema import STAGE_TO_FILENAME, build_checkpoint_payload

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def _to_jsonable(value: Any) -> Any:
    """Normalize objects into JSON-serializable primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]

    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)

    # Fall back to string representation for unsupported objects (e.g. planner objects).
    return str(value)


def write_checkpoint(path: str, payload: Mapping[str, Any]) -> str:
    """Write checkpoint payload as canonical JSON and return path."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_to_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n")
    return str(out_path)


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Load checkpoint payload from JSON file."""
    return json.loads(Path(path).read_text())


def checkpoint_path_for_stage(output_dir: str, stage: str) -> str:
    """Return canonical checkpoint file path for a stage."""
    if stage not in STAGE_TO_FILENAME:
        raise ValueError(f"Unsupported checkpoint stage: {stage}")
    return str(Path(output_dir) / STAGE_TO_FILENAME[stage])


def write_stage_checkpoint(
    output_dir: str,
    *,
    stage: str,
    state: Mapping[str, Any],
    metadata: Optional[Mapping[str, Any]] = None,
) -> str:
    """Build and write one stage checkpoint into output_dir."""
    payload = build_checkpoint_payload(stage=stage, state=state, metadata=metadata)
    return write_checkpoint(checkpoint_path_for_stage(output_dir, stage), payload)


class CheckpointManager:
    """Manage stage checkpoint export for one codegen run."""

    def __init__(self, output_dir: str, base_metadata: Optional[Mapping[str, Any]] = None) -> None:
        self.output_dir = str(output_dir)
        self.base_metadata: Dict[str, Any] = dict(base_metadata or {})

    def write_stage(
        self,
        *,
        stage: str,
        state: Mapping[str, Any],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> str:
        merged_metadata = dict(self.base_metadata)
        if metadata:
            merged_metadata.update(dict(metadata))

        return write_stage_checkpoint(
            self.output_dir,
            stage=stage,
            state=state,
            metadata=merged_metadata,
        )

