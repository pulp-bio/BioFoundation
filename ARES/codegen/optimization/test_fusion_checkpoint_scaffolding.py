# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for fusion/checkpoint scaffolding in codegen.optimization."""

import os
import tempfile
import unittest

from .checkpoints import (
    CHECKPOINT_SCHEMA_VERSION,
    load_checkpoint,
    write_standard_wave_checkpoints,
)
from .fusion import (
    EXPECTED_DEFAULT_PATTERN_ORDER,
    FusionTransformer,
    build_default_registry,
    detect_fusion_opportunities,
    validate_default_registry,
    validate_fusions,
    write_fusion_report,
)


class TestFusionRegistryScaffolding(unittest.TestCase):
    def test_default_registry_order(self) -> None:
        registry = build_default_registry()
        self.assertEqual(
            [pattern.name for pattern in registry.patterns()],
            list(EXPECTED_DEFAULT_PATTERN_ORDER),
        )
        self.assertEqual(validate_default_registry(registry), [])


class TestFusionMatcherScaffolding(unittest.TestCase):
    def test_detect_conv_relu_quant(self) -> None:
        specs = [
            {"name": "conv1", "op": "conv2d", "output_buffer": "buf0"},
            {"name": "relu1", "op": "relu", "buffer": "buf0"},
            {"name": "quant1", "op": "requantize", "buffer": "buf0"},
        ]
        fusions = detect_fusion_opportunities(specs)
        self.assertEqual(len(fusions), 1)
        self.assertEqual(fusions[0]["type"], "conv_relu_quant")
        self.assertEqual(validate_fusions(fusions), [])

    def test_detect_conv_relu_fallback_for_multitile_maxpool_chain(self) -> None:
        specs = [
            {
                "name": "conv1",
                "op": "conv2d",
                "output_buffer": "buf0",
                "tile_config": {"num_tiles": 2},
            },
            {"name": "relu1", "op": "relu", "buffer": "buf0"},
            {"name": "pool1", "op": "maxpool", "input_buffer": "buf0"},
        ]
        fusions = detect_fusion_opportunities(specs)
        self.assertEqual(len(fusions), 1)
        self.assertEqual(fusions[0]["type"], "conv_relu")
        self.assertEqual(fusions[0]["layers"], [0, 1])


class TestFusionTransformerScaffolding(unittest.TestCase):
    def test_transform_conv_relu_quant(self) -> None:
        specs = [
            {
                "name": "conv1",
                "op": "conv2d",
                "output_buffer": "buf0",
                "golden_slot": 1,
                "golden_buffer": "conv1_golden",
                "c_name": "conv1",
                "compare_buffer": "conv1_compare",
            },
            {
                "name": "relu1",
                "op": "relu",
                "buffer": "buf0",
                "scale": 0.25,
                "golden_slot": 2,
                "golden_buffer": "relu1_golden",
                "c_name": "relu1",
                "compare_buffer": "relu1_compare",
            },
            {
                "name": "quant1",
                "op": "requantize",
                "buffer": "buf0",
                "scale_in": 0.25,
                "scale_out": 0.125,
                "golden_slot": 3,
                "golden_buffer": "quant1_golden",
                "c_name": "quant1",
                "compare_buffer": "quant1_compare",
            },
        ]
        fusions = detect_fusion_opportunities(specs)

        transformer = FusionTransformer()
        fused_layers = []
        fused_specs, skip_indices, tracked = transformer.transform(specs, fusions, fused_layers)

        self.assertEqual(skip_indices, {1, 2})
        self.assertEqual(len(fused_specs), 1)

        conv_spec = fused_specs[0]
        self.assertTrue(conv_spec["fusion_relu"])
        self.assertTrue(conv_spec["fusion_quant"])
        self.assertEqual(conv_spec["quant_scale_in"], 0.25)
        self.assertEqual(conv_spec["quant_scale_out"], 0.125)
        self.assertEqual(conv_spec["golden_slot"], 3)
        self.assertEqual(conv_spec["golden_buffer"], "quant1_golden")
        self.assertEqual(conv_spec["golden_c_name"], "quant1")
        self.assertEqual(conv_spec["compare_buffer"], "quant1_compare")
        self.assertEqual(tracked, ["conv1+relu1+quant1 (conv_relu_quant)"])


class TestFusionCheckpointArtifacts(unittest.TestCase):
    def test_write_fusion_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fusions = [{"type": "conv_relu", "layers": [0, 1], "base_layer": 0}]
            report_path = write_fusion_report(tmpdir, fusions, test_name="unit_test")
            self.assertTrue(os.path.exists(report_path))
            payload = load_checkpoint(report_path)
            self.assertEqual(payload["fusion_count"], 1)
            self.assertEqual(payload["test_name"], "unit_test")

    def test_write_standard_wave_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_standard_wave_checkpoints(
                tmpdir,
                pre_fusion_state={"layer_count": 12},
                post_fusion_state={"layer_count": 8},
                metadata={"test": "unit"},
            )
            self.assertIn("pre_fusion", paths)
            self.assertIn("post_fusion", paths)

            pre = load_checkpoint(paths["pre_fusion"])
            post = load_checkpoint(paths["post_fusion"])
            self.assertEqual(pre["schema_version"], CHECKPOINT_SCHEMA_VERSION)
            self.assertEqual(post["schema_version"], CHECKPOINT_SCHEMA_VERSION)
            self.assertEqual(pre["stage"], "pre_fusion")
            self.assertEqual(post["stage"], "post_fusion")
            self.assertEqual(pre["state"]["layer_count"], 12)
            self.assertEqual(post["state"]["layer_count"], 8)


if __name__ == "__main__":
    unittest.main()

