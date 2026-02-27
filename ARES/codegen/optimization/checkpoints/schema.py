# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Schema helpers for Wave checkpoint payloads."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Mapping, Optional

CHECKPOINT_SCHEMA_VERSION = "1.0"


def build_checkpoint_payload(
    *,
    stage: str,
    state: Mapping[str, Any],
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build deterministic checkpoint payload with schema and timestamp fields."""
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "stage": stage,
        "created_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "state": dict(state),
        "metadata": dict(metadata or {}),
    }

