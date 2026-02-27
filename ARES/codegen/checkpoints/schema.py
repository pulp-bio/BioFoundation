# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Schema definitions for stage checkpoint payloads."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Mapping, Optional

CHECKPOINT_SCHEMA_VERSION = "1.0"

CHECKPOINT_STAGE_PRE_FUSION = "pre_fusion"
CHECKPOINT_STAGE_POST_FUSION = "post_fusion"
CHECKPOINT_STAGE_POST_TILING = "post_tiling"
CHECKPOINT_STAGE_POST_MEMORY_PLAN = "post_memory_plan"

CHECKPOINT_STAGES = (
    CHECKPOINT_STAGE_PRE_FUSION,
    CHECKPOINT_STAGE_POST_FUSION,
    CHECKPOINT_STAGE_POST_TILING,
    CHECKPOINT_STAGE_POST_MEMORY_PLAN,
)

STAGE_TO_FILENAME = {
    CHECKPOINT_STAGE_PRE_FUSION: "pre_fusion.json",
    CHECKPOINT_STAGE_POST_FUSION: "post_fusion.json",
    CHECKPOINT_STAGE_POST_TILING: "post_tiling.json",
    CHECKPOINT_STAGE_POST_MEMORY_PLAN: "post_memory_plan.json",
}


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def build_checkpoint_payload(
    *,
    stage: str,
    state: Mapping[str, Any],
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a checkpoint payload with deterministic schema fields."""
    if stage not in CHECKPOINT_STAGES:
        raise ValueError(f"Unsupported checkpoint stage: {stage}")

    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "stage": stage,
        "created_utc": _utc_now_iso(),
        "state": dict(state),
        "metadata": dict(metadata or {}),
    }
