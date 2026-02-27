# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Stage checkpoint export and replay utilities."""

from .replay import (
    load_checkpoint_bundle,
    replay_from_checkpoint_file,
    replay_stage_from_directory,
    restore_generator_from_checkpoint,
)
from .schema import (
    CHECKPOINT_SCHEMA_VERSION,
    CHECKPOINT_STAGE_POST_FUSION,
    CHECKPOINT_STAGE_POST_MEMORY_PLAN,
    CHECKPOINT_STAGE_POST_TILING,
    CHECKPOINT_STAGE_PRE_FUSION,
    CHECKPOINT_STAGES,
    STAGE_TO_FILENAME,
    build_checkpoint_payload,
)
from .writer import (
    CheckpointManager,
    checkpoint_path_for_stage,
    load_checkpoint,
    write_checkpoint,
    write_stage_checkpoint,
)

__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CHECKPOINT_STAGE_PRE_FUSION",
    "CHECKPOINT_STAGE_POST_FUSION",
    "CHECKPOINT_STAGE_POST_TILING",
    "CHECKPOINT_STAGE_POST_MEMORY_PLAN",
    "CHECKPOINT_STAGES",
    "STAGE_TO_FILENAME",
    "build_checkpoint_payload",
    "checkpoint_path_for_stage",
    "load_checkpoint",
    "write_checkpoint",
    "write_stage_checkpoint",
    "CheckpointManager",
    "load_checkpoint_bundle",
    "restore_generator_from_checkpoint",
    "replay_from_checkpoint_file",
    "replay_stage_from_directory",
]
