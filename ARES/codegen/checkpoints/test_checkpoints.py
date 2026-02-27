# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for codegen.checkpoints stage export and replay."""

import tempfile
import unittest
from pathlib import Path

from .replay import load_checkpoint_bundle, restore_generator_from_checkpoint
from .schema import (
    CHECKPOINT_SCHEMA_VERSION,
    CHECKPOINT_STAGE_POST_FUSION,
    CHECKPOINT_STAGE_POST_MEMORY_PLAN,
    CHECKPOINT_STAGE_POST_TILING,
    CHECKPOINT_STAGE_PRE_FUSION,
)
from .writer import CheckpointManager, load_checkpoint


class _DummyGenerator:
    def __init__(self):
        self.layer_specs = []
        self.activation_buffers = []
        self.shared_activation_pool = []
        self.param_layers = []
        self.l2_arena_size = 0
        self.planner = None
        self._metadata_ready = False


class TestCheckpointWriter(unittest.TestCase):
    def test_write_all_standard_stage_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(
                tmpdir,
                base_metadata={"test_name": "unit"},
            )

            state = {
                "layer_specs": [{"name": "conv1"}],
                "activation_buffers": [{"name": "buf0", "numel": 16}],
                "shared_activation_pool": [],
                "l2_arena_size": 1024,
                "planner_offsets": {"buf0": 0},
            }

            pre_path = manager.write_stage(stage=CHECKPOINT_STAGE_PRE_FUSION, state=state)
            post_fusion_path = manager.write_stage(stage=CHECKPOINT_STAGE_POST_FUSION, state=state)
            post_tiling_path = manager.write_stage(stage=CHECKPOINT_STAGE_POST_TILING, state=state)
            post_memory_path = manager.write_stage(stage=CHECKPOINT_STAGE_POST_MEMORY_PLAN, state=state)

            self.assertTrue(Path(pre_path).exists())
            self.assertTrue(Path(post_fusion_path).exists())
            self.assertTrue(Path(post_tiling_path).exists())
            self.assertTrue(Path(post_memory_path).exists())

            payload = load_checkpoint(post_memory_path)
            self.assertEqual(payload["schema_version"], CHECKPOINT_SCHEMA_VERSION)
            self.assertEqual(payload["stage"], CHECKPOINT_STAGE_POST_MEMORY_PLAN)
            self.assertEqual(payload["metadata"]["test_name"], "unit")
            self.assertEqual(payload["state"]["l2_arena_size"], 1024)


class TestCheckpointReplay(unittest.TestCase):
    def test_restore_generator_from_payload(self):
        generator = _DummyGenerator()
        payload = {
            "state": {
                "layer_specs": [{"name": "conv1"}],
                "activation_buffers": [{"name": "buf0", "numel": 16}],
                "shared_activation_pool": [{"name": "pool0", "numel": 8}],
                "param_layers": ["conv1"],
                "l2_arena_size": 2048,
                "planner_offsets": {"buf0": 0},
                "planner_lifetimes": {"buf0": {"start": 0, "end": 1, "size": 16}},
            }
        }

        restore_generator_from_checkpoint(generator, payload)

        self.assertEqual(len(generator.layer_specs), 1)
        self.assertEqual(generator.l2_arena_size, 2048)
        self.assertTrue(generator._metadata_ready)
        self.assertIsNotNone(generator.planner)
        self.assertEqual(generator.planner.offsets["buf0"], 0)

    def test_load_checkpoint_bundle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)
            for stage in (
                CHECKPOINT_STAGE_PRE_FUSION,
                CHECKPOINT_STAGE_POST_FUSION,
                CHECKPOINT_STAGE_POST_TILING,
                CHECKPOINT_STAGE_POST_MEMORY_PLAN,
            ):
                manager.write_stage(stage=stage, state={"layer_specs": []})

            bundle = load_checkpoint_bundle(tmpdir)
            self.assertEqual(set(bundle.keys()), {
                CHECKPOINT_STAGE_PRE_FUSION,
                CHECKPOINT_STAGE_POST_FUSION,
                CHECKPOINT_STAGE_POST_TILING,
                CHECKPOINT_STAGE_POST_MEMORY_PLAN,
            })


if __name__ == "__main__":
    unittest.main()
