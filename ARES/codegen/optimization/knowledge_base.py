# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Knowledge Base for ARES optimization configurations.

This module provides the KnowledgeBase class for storing, querying,
and recording optimization configurations learned from past tuning runs.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from .config_schema import (
    OptimizationEntry,
    NegativeResult,
    PerformanceBaseline,
    ShapePattern,
    TileConfig,
    KernelConfig,
    PipelineConfig,
    CompileFlags,
    PerformanceMetrics,
)
from .shape_matching import (
    find_best_match,
    find_matching_entries,
    check_negative_results,
    extract_shape_from_layer,
)


# Default path to knowledge base JSON
DEFAULT_KB_PATH = Path(__file__).parent / "data" / "knowledge_base.json"


class KnowledgeBase:
    """
    Stores and retrieves optimization configurations.

    Usage:
        kb = KnowledgeBase()

        # Lookup config for a layer
        config = kb.lookup("linear_int8", {"M": 32, "N": 256, "K": 64})
        if config:
            # Use config.tile_config, config.kernel_config, etc.
            pass

        # After successful optimization, record it
        kb.record(
            op_type="linear_int8",
            shape={"M": 32, "N": 256, "K": 64},
            tile_config={"tile_m": 32, "tile_n": 64},
            kernel_config={"fast_qround": True},
            pipeline_config={"enable_overlap": True},
            measured_cycles=100000,
            measured_macs_per_cycle=10.5,
            source="auto_tuner_run_001"
        )
        kb.save()
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Load knowledge base from JSON file.

        Args:
            db_path: Path to knowledge base JSON file.
                     If None, uses default location.
        """
        self.db_path = Path(db_path) if db_path else DEFAULT_KB_PATH
        self.entries: List[OptimizationEntry] = []
        self.negative_results: List[NegativeResult] = []
        self.compile_flag_reference: Dict[str, Dict[str, Any]] = {}
        self.performance_baselines: Dict[str, PerformanceBaseline] = {}
        self.version: str = "1.0"
        self.last_updated: str = ""
        self.source_documents: List[str] = []

        self._load()

    def _load(self) -> None:
        """Load knowledge base from JSON file."""
        if not self.db_path.exists():
            # Initialize empty KB
            self.last_updated = datetime.now().strftime("%Y-%m-%d")
            return

        with open(self.db_path, "r") as f:
            data = json.load(f)

        self.version = data.get("version", "1.0")
        self.last_updated = data.get("last_updated", "")
        self.source_documents = data.get("source_documents", [])

        # Load entries
        self.entries = [
            OptimizationEntry.from_dict(e)
            for e in data.get("entries", [])
        ]

        # Load negative results
        self.negative_results = [
            NegativeResult.from_dict(n)
            for n in data.get("negative_results", [])
        ]

        # Load compile flag reference
        self.compile_flag_reference = data.get("compile_flag_reference", {})

        # Load performance baselines
        self.performance_baselines = {
            name: PerformanceBaseline.from_dict(name, baseline)
            for name, baseline in data.get("performance_baselines", {}).items()
        }

    def save(self) -> None:
        """Persist knowledge base to disk."""
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": self.version,
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "description": "ARES optimization knowledge base - learned configurations from past tuning runs",
            "source_documents": self.source_documents,
            "entries": [e.to_dict() for e in self.entries],
            "negative_results": [n.to_dict() for n in self.negative_results],
            "compile_flag_reference": self.compile_flag_reference,
            "performance_baselines": {
                name: {
                    "total_cycles": b.total_cycles,
                    "macs_per_total_cycle": b.macs_per_total_cycle,
                    "notes": b.notes,
                }
                for name, b in self.performance_baselines.items()
            },
        }

        with open(self.db_path, "w") as f:
            json.dump(data, f, indent=2)

    def lookup(self,
               op_type: str,
               shape: Dict[str, Any],
               min_confidence: float = 0.0) -> Optional[OptimizationEntry]:
        """
        Find best matching config for given op type and shape.

        Args:
            op_type: Operation type (e.g., "linear_int8", "conv2d_int8")
            shape: Dictionary of layer dimensions
            min_confidence: Minimum confidence threshold (0-1)

        Returns:
            Best matching OptimizationEntry, or None if no suitable match found.

        Example:
            >>> kb = KnowledgeBase()
            >>> config = kb.lookup("linear_int8", {"M": 400, "N": 768, "K": 192})
            >>> if config:
            ...     print(f"Found config with {config.performance.macs_per_cycle} MACs/cycle")
        """
        return find_best_match(shape, self.entries, op_type, min_confidence)

    def lookup_all(self,
                   op_type: str,
                   shape: Dict[str, Any],
                   min_confidence: float = 0.0) -> List[Tuple[OptimizationEntry, float]]:
        """
        Return all matching configs, sorted by score (best first).

        Args:
            op_type: Operation type
            shape: Dictionary of layer dimensions
            min_confidence: Minimum confidence threshold

        Returns:
            List of (entry, score) tuples, sorted by score descending
        """
        matches = find_matching_entries(shape, self.entries, op_type)
        return [(entry, score) for entry, score in matches
                if entry.confidence >= min_confidence]

    def lookup_for_layer(self,
                         layer_info: Dict[str, Any],
                         op_type: str,
                         min_confidence: float = 0.0) -> Optional[OptimizationEntry]:
        """
        Lookup optimization config for a layer using layer info dict.

        This is a convenience method that extracts the relevant shape
        dimensions from a layer info dictionary.

        Args:
            layer_info: Layer information dictionary (from network_info.json)
            op_type: Operation type
            min_confidence: Minimum confidence threshold

        Returns:
            Best matching OptimizationEntry, or None
        """
        shape = extract_shape_from_layer(layer_info, op_type)
        return self.lookup(op_type, shape, min_confidence)

    def record(self,
               op_type: str,
               shape: Dict[str, Any],
               tile_config: Optional[Dict[str, Any]] = None,
               kernel_config: Optional[Dict[str, Any]] = None,
               pipeline_config: Optional[Dict[str, Any]] = None,
               compile_flags: Optional[Dict[str, Any]] = None,
               measured_cycles: Optional[int] = None,
               measured_macs_per_cycle: Optional[float] = None,
               measured_overlap_ratio: Optional[float] = None,
               source: str = "unknown",
               test_network: str = "",
               confidence: float = 0.9,
               notes: str = "",
               description: str = "") -> OptimizationEntry:
        """
        Record a new optimization result.

        Args:
            op_type: Operation type
            shape: Dictionary of layer dimensions (will be stored as pattern)
            tile_config: Tiling configuration
            kernel_config: Kernel variant configuration
            pipeline_config: DMA/compute pipeline configuration
            compile_flags: C compile-time flags
            measured_cycles: Total cycles measured
            measured_macs_per_cycle: MACs/cycle performance
            measured_overlap_ratio: DMA/compute overlap ratio (0-1)
            source: Source identifier (e.g., "auto_tuner_run_001")
            test_network: Test network used for validation
            confidence: Confidence score (0-1)
            notes: Additional notes
            description: Human-readable description

        Returns:
            The created OptimizationEntry
        """
        entry = OptimizationEntry(
            op_type=op_type,
            shape_pattern=ShapePattern(pattern=shape),
            tile_config=TileConfig(config=tile_config or {}),
            kernel_config=KernelConfig(config=kernel_config or {}),
            pipeline_config=PipelineConfig.from_dict(pipeline_config or {}),
            compile_flags=CompileFlags(flags=compile_flags or {}),
            performance=PerformanceMetrics(
                macs_per_cycle=measured_macs_per_cycle,
                overlap_ratio=measured_overlap_ratio,
                total_cycles=measured_cycles,
            ),
            description=description,
            source=source,
            test_network=test_network,
            confidence=confidence,
            notes=notes,
        )

        self.entries.append(entry)
        return entry

    def record_negative(self,
                        op_type: str,
                        attempted_config: Dict[str, Any],
                        result: str,
                        source: str = "",
                        notes: str = "") -> NegativeResult:
        """
        Record a negative result (optimization that didn't work).

        Args:
            op_type: Operation type
            attempted_config: Configuration that was tried
            result: Result description (e.g., "regressed +45.7%")
            source: Source document or experiment
            notes: Additional notes

        Returns:
            The created NegativeResult
        """
        neg = NegativeResult(
            op_type=op_type,
            attempted_config=attempted_config,
            result=result,
            source=source,
            notes=notes,
        )

        self.negative_results.append(neg)
        return neg

    def check_negative(self,
                       op_type: str,
                       proposed_config: Dict[str, Any]) -> Optional[NegativeResult]:
        """
        Check if a proposed config matches any negative results.

        This helps avoid re-trying optimizations that are known to fail.

        Args:
            op_type: Operation type
            proposed_config: Configuration being considered

        Returns:
            Matching NegativeResult if found, None otherwise
        """
        result = check_negative_results(
            op_type,
            proposed_config,
            [n.to_dict() for n in self.negative_results]
        )
        if result:
            return NegativeResult.from_dict(result)
        return None

    def get_baseline(self, test_name: str) -> Optional[PerformanceBaseline]:
        """Get performance baseline for a test network."""
        return self.performance_baselines.get(test_name)

    def update_baseline(self,
                        test_name: str,
                        total_cycles: int,
                        macs_per_total_cycle: float,
                        notes: str = "") -> None:
        """Update or add a performance baseline."""
        self.performance_baselines[test_name] = PerformanceBaseline(
            test_name=test_name,
            total_cycles=total_cycles,
            macs_per_total_cycle=macs_per_total_cycle,
            notes=notes,
        )

    def get_compile_flag_info(self, flag_name: str) -> Optional[Dict[str, Any]]:
        """Get documentation for a compile flag."""
        return self.compile_flag_reference.get(flag_name)

    def get_entries_by_op_type(self, op_type: str) -> List[OptimizationEntry]:
        """Get all entries for a specific op type."""
        return [e for e in self.entries if e.op_type == op_type]

    def get_best_entry(self, op_type: str) -> Optional[OptimizationEntry]:
        """Get the highest-performing entry for an op type."""
        entries = self.get_entries_by_op_type(op_type)
        if not entries:
            return None

        # Sort by MACs/cycle (descending)
        entries_with_perf = [
            e for e in entries
            if e.performance.macs_per_cycle is not None
        ]

        if entries_with_perf:
            return max(entries_with_perf,
                       key=lambda e: e.performance.macs_per_cycle)

        # Fall back to highest confidence
        return max(entries, key=lambda e: e.confidence)

    def export_summary(self) -> str:
        """Export human-readable summary of known optimizations."""
        lines = [
            "# ARES Knowledge Base Summary",
            f"Version: {self.version}",
            f"Last Updated: {self.last_updated}",
            f"Total Entries: {len(self.entries)}",
            f"Negative Results: {len(self.negative_results)}",
            "",
            "## Entries by Op Type",
            "",
        ]

        # Group by op type
        by_type: Dict[str, List[OptimizationEntry]] = {}
        for entry in self.entries:
            if entry.op_type not in by_type:
                by_type[entry.op_type] = []
            by_type[entry.op_type].append(entry)

        for op_type in sorted(by_type.keys()):
            entries = by_type[op_type]
            lines.append(f"### {op_type} ({len(entries)} entries)")
            lines.append("")

            for entry in entries:
                desc = entry.description or "No description"
                perf = ""
                if entry.performance.macs_per_cycle is not None:
                    perf = f" [{entry.performance.macs_per_cycle:.1f} MACs/cycle]"
                elif entry.performance.improvement is not None:
                    perf = f" [{entry.performance.improvement}]"

                lines.append(f"- **{desc}**{perf}")
                lines.append(f"  - Shape: {entry.shape_pattern.pattern}")
                lines.append(f"  - Source: {entry.source}")
                lines.append(f"  - Confidence: {entry.confidence:.0%}")
                lines.append("")

        # Negative results
        if self.negative_results:
            lines.append("## Negative Results (Don't Try)")
            lines.append("")
            for neg in self.negative_results:
                lines.append(f"- **{neg.op_type}**: {neg.result}")
                if neg.notes:
                    lines.append(f"  - {neg.notes}")
            lines.append("")

        # Performance baselines
        if self.performance_baselines:
            lines.append("## Performance Baselines")
            lines.append("")
            lines.append("| Test Network | Total Cycles | MACs/cycle |")
            lines.append("|--------------|--------------|------------|")
            for name, baseline in sorted(self.performance_baselines.items()):
                cycles = f"{baseline.total_cycles:,}"
                macs = f"{baseline.macs_per_total_cycle:.2f}"
                lines.append(f"| {name} | {cycles} | {macs} |")
            lines.append("")

        return "\n".join(lines)

    def __len__(self) -> int:
        """Return number of entries in knowledge base."""
        return len(self.entries)

    def __repr__(self) -> str:
        return f"KnowledgeBase(entries={len(self.entries)}, negative={len(self.negative_results)})"
