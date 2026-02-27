# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for ARES optimization knowledge base.

Run with:
    python -m pytest codegen/optimization/test_knowledge_base.py -v

Or directly:
    python codegen/optimization/test_knowledge_base.py
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from .config_schema import (
    OpType,
    ShapePattern,
    TileConfig,
    KernelConfig,
    PipelineConfig,
    CompileFlags,
    PerformanceMetrics,
    OptimizationEntry,
    NegativeResult,
)
from .shape_matching import (
    value_matches,
    shape_matches,
    shape_distance,
    pattern_specificity,
    find_best_match,
    check_negative_results,
    extract_shape_from_layer,
)
from .knowledge_base import KnowledgeBase


class TestValueMatches(unittest.TestCase):
    """Tests for value_matches function."""

    def test_none_matches_anything(self):
        self.assertTrue(value_matches(42, None))
        self.assertTrue(value_matches("hello", None))
        self.assertTrue(value_matches(None, None))

    def test_exact_int_match(self):
        self.assertTrue(value_matches(42, 42))
        self.assertFalse(value_matches(42, 43))

    def test_range_match(self):
        self.assertTrue(value_matches(50, [0, 100]))
        self.assertTrue(value_matches(0, [0, 100]))
        self.assertTrue(value_matches(100, [0, 100]))
        self.assertFalse(value_matches(-1, [0, 100]))
        self.assertFalse(value_matches(101, [0, 100]))

    def test_bool_match(self):
        self.assertTrue(value_matches(True, True))
        self.assertTrue(value_matches(False, False))
        self.assertFalse(value_matches(True, False))

    def test_string_match(self):
        self.assertTrue(value_matches("hello", "hello"))
        self.assertFalse(value_matches("hello", "world"))


class TestShapeMatches(unittest.TestCase):
    """Tests for shape_matches function."""

    def test_exact_match(self):
        actual = {"M": 32, "N": 256, "K": 64}
        pattern = {"M": 32, "N": 256, "K": 64}
        self.assertTrue(shape_matches(actual, pattern))

    def test_range_match(self):
        actual = {"M": 32, "N": 256, "K": 64}
        pattern = {"M": [1, 64], "N": [128, 512], "K": [32, 128]}
        self.assertTrue(shape_matches(actual, pattern))

    def test_partial_pattern(self):
        actual = {"M": 32, "N": 256, "K": 64, "extra": 100}
        pattern = {"M": 32}
        self.assertTrue(shape_matches(actual, pattern))

    def test_no_match(self):
        actual = {"M": 32, "N": 256, "K": 64}
        pattern = {"M": [100, 200]}
        self.assertFalse(shape_matches(actual, pattern))

    def test_missing_key(self):
        actual = {"M": 32}
        pattern = {"M": 32, "N": 256}
        self.assertFalse(shape_matches(actual, pattern))


class TestShapeDistance(unittest.TestCase):
    """Tests for shape_distance function."""

    def test_exact_match_zero_distance(self):
        actual = {"M": 32, "N": 256}
        pattern = {"M": 32, "N": 256}
        self.assertEqual(shape_distance(actual, pattern), 0.0)

    def test_range_center_low_distance(self):
        actual = {"M": 50}
        pattern = {"M": [0, 100]}
        # 50 is at center, should have low distance
        dist = shape_distance(actual, pattern)
        self.assertLess(dist, 0.1)

    def test_range_edge_higher_distance(self):
        actual = {"M": 100}
        pattern = {"M": [0, 100]}
        # 100 is at edge, should have higher distance than center
        dist = shape_distance(actual, pattern)
        self.assertGreater(dist, 0.4)

    def test_no_match_infinite_distance(self):
        actual = {"M": 200}
        pattern = {"M": [0, 100]}
        self.assertEqual(shape_distance(actual, pattern), float('inf'))


class TestPatternSpecificity(unittest.TestCase):
    """Tests for pattern_specificity function."""

    def test_exact_match_high_specificity(self):
        pattern = {"M": 32, "N": 256}
        specificity = pattern_specificity(pattern)
        self.assertGreater(specificity, 3.0)

    def test_any_match_low_specificity(self):
        pattern = {"M": None, "N": None}
        specificity = pattern_specificity(pattern)
        self.assertLess(specificity, 1.0)

    def test_narrow_range_higher_than_wide(self):
        narrow = {"M": [30, 35]}
        wide = {"M": [0, 1000]}
        self.assertGreater(
            pattern_specificity(narrow),
            pattern_specificity(wide)
        )


class TestOptimizationEntry(unittest.TestCase):
    """Tests for OptimizationEntry dataclass."""

    def test_from_dict(self):
        data = {
            "op_type": "linear_int8",
            "shape_pattern": {"M": [1, 64], "N": [64, 256]},
            "tile_config": {"tile_m": 32, "tile_n": 64},
            "kernel_config": {"fast_qround": True},
            "measured_macs_per_cycle": 10.5,
            "confidence": 0.9,
            "source": "test_source",
        }
        entry = OptimizationEntry.from_dict(data)

        self.assertEqual(entry.op_type, "linear_int8")
        self.assertEqual(entry.shape_pattern.pattern, {"M": [1, 64], "N": [64, 256]})
        self.assertEqual(entry.tile_config.get("tile_m"), 32)
        self.assertEqual(entry.kernel_config.get("fast_qround"), True)
        self.assertEqual(entry.performance.macs_per_cycle, 10.5)
        self.assertEqual(entry.confidence, 0.9)

    def test_to_dict(self):
        entry = OptimizationEntry(
            op_type="linear_int8",
            shape_pattern=ShapePattern({"M": 32}),
            tile_config=TileConfig({"tile_m": 32}),
            performance=PerformanceMetrics(macs_per_cycle=10.0),
            confidence=0.9,
            source="test",
        )
        data = entry.to_dict()

        self.assertEqual(data["op_type"], "linear_int8")
        self.assertEqual(data["shape_pattern"], {"M": 32})
        self.assertEqual(data["tile_config"], {"tile_m": 32})
        self.assertAlmostEqual(data["measured_macs_per_cycle"], 10.0)


class TestKnowledgeBase(unittest.TestCase):
    """Tests for KnowledgeBase class."""

    def setUp(self):
        """Create a temporary knowledge base for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_kb_path = os.path.join(self.temp_dir, "test_kb.json")

        # Create test data
        test_data = {
            "version": "1.0",
            "last_updated": "2026-01-06",
            "entries": [
                {
                    "op_type": "linear_int8",
                    "description": "Small linear",
                    "shape_pattern": {"M": [1, 64], "N": [64, 256], "K": [64, 256]},
                    "tile_config": {"tile_m": 32},
                    "measured_macs_per_cycle": 10.5,
                    "confidence": 0.9,
                    "source": "test",
                },
                {
                    "op_type": "linear_int8",
                    "description": "Large linear",
                    "shape_pattern": {"M": [1, 64], "N": [256, 1024], "K": [256, 1024]},
                    "tile_config": {"tile_m": 16},
                    "measured_macs_per_cycle": 9.0,
                    "confidence": 0.85,
                    "source": "test",
                },
                {
                    "op_type": "conv2d_int8",
                    "description": "3x3 conv",
                    "shape_pattern": {"kernel_h": 3, "kernel_w": 3},
                    "measured_macs_per_cycle": 8.0,
                    "confidence": 0.8,
                    "source": "test",
                },
            ],
            "negative_results": [
                {
                    "op_type": "mhsa_int8",
                    "attempted_config": {"fuse_softmax_av_only": True},
                    "result": "regressed",
                },
            ],
            "compile_flag_reference": {},
            "performance_baselines": {},
        }

        with open(self.temp_kb_path, "w") as f:
            json.dump(test_data, f)

        self.kb = KnowledgeBase(self.temp_kb_path)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_entries(self):
        self.assertEqual(len(self.kb.entries), 3)
        self.assertEqual(len(self.kb.negative_results), 1)

    def test_lookup_linear_small(self):
        config = self.kb.lookup("linear_int8", {"M": 32, "N": 128, "K": 128})
        self.assertIsNotNone(config)
        self.assertEqual(config.tile_config.get("tile_m"), 32)
        self.assertAlmostEqual(config.performance.macs_per_cycle, 10.5)

    def test_lookup_linear_large(self):
        config = self.kb.lookup("linear_int8", {"M": 32, "N": 512, "K": 512})
        self.assertIsNotNone(config)
        self.assertEqual(config.tile_config.get("tile_m"), 16)

    def test_lookup_no_match(self):
        config = self.kb.lookup("linear_int8", {"M": 1000, "N": 10000, "K": 10000})
        self.assertIsNone(config)

    def test_lookup_wrong_op_type(self):
        config = self.kb.lookup("mhsa_int8", {"M": 32, "N": 128, "K": 128})
        self.assertIsNone(config)

    def test_lookup_all(self):
        # Shape that matches both linear entries
        matches = self.kb.lookup_all("linear_int8", {"M": 32, "N": 200, "K": 200})
        # Should match at least one (small linear pattern covers N=[64,256])
        self.assertGreaterEqual(len(matches), 1)

    def test_record_entry(self):
        initial_count = len(self.kb.entries)
        self.kb.record(
            op_type="gelu_int8",
            shape={"size": [1000, 100000]},
            measured_macs_per_cycle=5.0,
            source="test_record",
        )
        self.assertEqual(len(self.kb.entries), initial_count + 1)
        self.assertEqual(self.kb.entries[-1].op_type, "gelu_int8")

    def test_check_negative(self):
        # Should find negative result
        neg = self.kb.check_negative("mhsa_int8", {"fuse_softmax_av_only": True})
        self.assertIsNotNone(neg)
        self.assertEqual(neg.result, "regressed")

        # Should not find negative result
        neg = self.kb.check_negative("mhsa_int8", {"different_config": True})
        self.assertIsNone(neg)

    def test_save_and_reload(self):
        # Record new entry
        self.kb.record(
            op_type="test_op",
            shape={"x": 100},
            measured_macs_per_cycle=1.0,
            source="save_test",
        )
        self.kb.save()

        # Reload
        kb2 = KnowledgeBase(self.temp_kb_path)
        self.assertEqual(len(kb2.entries), len(self.kb.entries))

        # Check new entry exists
        test_entries = [e for e in kb2.entries if e.op_type == "test_op"]
        self.assertEqual(len(test_entries), 1)

    def test_export_summary(self):
        summary = self.kb.export_summary()
        self.assertIn("linear_int8", summary)
        self.assertIn("conv2d_int8", summary)
        self.assertIn("Negative Results", summary)

    def test_get_entries_by_op_type(self):
        linear_entries = self.kb.get_entries_by_op_type("linear_int8")
        self.assertEqual(len(linear_entries), 2)

        conv_entries = self.kb.get_entries_by_op_type("conv2d_int8")
        self.assertEqual(len(conv_entries), 1)

    def test_get_best_entry(self):
        best = self.kb.get_best_entry("linear_int8")
        self.assertIsNotNone(best)
        # Should be the one with highest MACs/cycle (10.5)
        self.assertAlmostEqual(best.performance.macs_per_cycle, 10.5)


class TestExtractShapeFromLayer(unittest.TestCase):
    """Tests for extract_shape_from_layer function."""

    def test_linear_shape(self):
        layer_info = {
            "batch_tokens": 400,
            "out_features": 768,
            "in_features": 192,
        }
        shape = extract_shape_from_layer(layer_info, "linear_int8")
        self.assertEqual(shape["M"], 400)
        self.assertEqual(shape["N"], 768)
        self.assertEqual(shape["K"], 192)

    def test_conv2d_shape(self):
        layer_info = {
            "kernel_size": [3, 3],
            "in_channels": 64,
            "out_channels": 128,
            "stride": 1,
            "padding": 1,
        }
        shape = extract_shape_from_layer(layer_info, "conv2d_int8")
        self.assertEqual(shape["kernel_h"], 3)
        self.assertEqual(shape["kernel_w"], 3)
        self.assertEqual(shape["in_channels"], 64)
        self.assertEqual(shape["out_channels"], 128)

    def test_mhsa_shape(self):
        layer_info = {
            "seq_len": 400,
            "embed_dim": 192,
            "num_heads": 3,
            "head_dim": 64,
        }
        shape = extract_shape_from_layer(layer_info, "mhsa_int8")
        self.assertEqual(shape["seq_len"], 400)
        self.assertEqual(shape["embed_dim"], 192)
        self.assertEqual(shape["num_heads"], 3)
        self.assertEqual(shape["head_dim"], 64)


class TestCompileFlags(unittest.TestCase):
    """Tests for CompileFlags dataclass."""

    def test_to_makefile_string(self):
        flags = CompileFlags({
            "LINEAR_INT8_INPUT_L1_CACHE": 1,
            "CONV2D_IM2COL_OUTCH_UNROLL": 4,
            "DEBUG_MODE": True,
        })
        makefile_str = flags.to_makefile_string()
        self.assertIn("-DLINEAR_INT8_INPUT_L1_CACHE=1", makefile_str)
        self.assertIn("-DCONV2D_IM2COL_OUTCH_UNROLL=4", makefile_str)
        self.assertIn("-DDEBUG_MODE=1", makefile_str)


class TestRealKnowledgeBase(unittest.TestCase):
    """Tests using the real knowledge base."""

    def test_load_real_kb(self):
        """Test that the real knowledge base loads correctly."""
        kb = KnowledgeBase()
        self.assertGreater(len(kb.entries), 0)
        self.assertGreater(len(kb.negative_results), 0)

    def test_lookup_tinymyo_linear(self):
        """Test lookup for TinyMyo-style linear layer."""
        kb = KnowledgeBase()
        config = kb.lookup("linear_int8", {"M": 400, "N": 768, "K": 192})
        if config:
            self.assertIsNotNone(config.performance.macs_per_cycle)
            print(f"Found config: {config.description}")
            print(f"  MACs/cycle: {config.performance.macs_per_cycle}")


def run_tests():
    """Run all tests."""
    unittest.main(module=__name__, exit=False, verbosity=2)


if __name__ == "__main__":
    run_tests()
