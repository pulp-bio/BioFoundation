# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Replay helpers for checkpointed codegen state."""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .schema import CHECKPOINT_STAGE_POST_MEMORY_PLAN, STAGE_TO_FILENAME
from .writer import checkpoint_path_for_stage, load_checkpoint


def load_checkpoint_bundle(output_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load all stage checkpoint files that exist in output_dir."""
    bundle: Dict[str, Dict[str, Any]] = {}
    root = Path(output_dir)
    for stage, filename in STAGE_TO_FILENAME.items():
        path = root / filename
        if path.exists():
            bundle[stage] = load_checkpoint(str(path))
    return bundle


def restore_generator_from_checkpoint(
    generator: Any,
    checkpoint_payload: Mapping[str, Any],
    *,
    mark_metadata_ready: bool = True,
) -> Dict[str, Any]:
    """
    Restore generator attributes from checkpoint state.

    This allows isolating later-stage generation issues without rebuilding
    layer specs or memory plans from scratch.
    """
    state = dict(checkpoint_payload.get("state", {}))

    for attr in ("layer_specs", "activation_buffers", "shared_activation_pool", "param_layers"):
        if attr in state:
            setattr(generator, attr, state[attr])

    if "l2_arena_size" in state:
        generator.l2_arena_size = state["l2_arena_size"]

    planner_offsets = state.get("planner_offsets")
    planner_lifetimes = state.get("planner_lifetimes")
    if planner_offsets is not None or planner_lifetimes is not None:
        planner_obj = getattr(generator, "planner", None)
        if planner_obj is None:
            planner_obj = types.SimpleNamespace()
        planner_obj.offsets = planner_offsets or {}
        planner_obj.lifetimes = planner_lifetimes or {}
        planner_obj.total_size = state.get("l2_arena_size", getattr(planner_obj, "total_size", 0))
        generator.planner = planner_obj

    if mark_metadata_ready and hasattr(generator, "_metadata_ready"):
        generator._metadata_ready = True

    return state


def replay_from_checkpoint_file(
    generator: Any,
    checkpoint_path: str,
    *,
    mark_metadata_ready: bool = True,
) -> Dict[str, Any]:
    """Load one checkpoint file and restore generator state from it."""
    payload = load_checkpoint(checkpoint_path)
    return restore_generator_from_checkpoint(
        generator,
        payload,
        mark_metadata_ready=mark_metadata_ready,
    )


def replay_stage_from_directory(
    generator: Any,
    output_dir: str,
    *,
    stage: str = CHECKPOINT_STAGE_POST_MEMORY_PLAN,
    mark_metadata_ready: bool = True,
) -> Dict[str, Any]:
    """Restore generator state from a stage checkpoint under output_dir."""
    checkpoint_path = checkpoint_path_for_stage(output_dir, stage)
    return replay_from_checkpoint_file(
        generator,
        checkpoint_path,
        mark_metadata_ready=mark_metadata_ready,
    )

