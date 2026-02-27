# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Configuration schema for ARES optimization knowledge base.

This module defines the data structures used to represent optimization
configurations, shape patterns, and performance metrics.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from enum import Enum


class OpType(Enum):
    """Supported operation types for optimization."""
    LINEAR_INT8 = "linear_int8"
    CONV2D_INT8 = "conv2d_int8"
    MHSA_INT8 = "mhsa_int8"
    CROSS_ATTENTION_INT8 = "cross_attention_int8"
    LAYERNORM_INT8 = "layernorm_int8"
    GELU_INT8 = "gelu_int8"
    GROUPNORM_INT8 = "groupnorm_int8"
    RFFT_INT8 = "rfft_int8"
    SSM_INT8 = "ssm_int8"
    ADD_INT8 = "add_int8"
    IDENTITY_REQUANT = "identity_requant"
    EMBEDDING = "embedding"
    ROPE = "rope"

    @classmethod
    def from_string(cls, s: str) -> "OpType":
        """Convert string to OpType enum."""
        for member in cls:
            if member.value == s:
                return member
        raise ValueError(f"Unknown op type: {s}")


# Type alias for shape pattern values
# Can be: int (exact), [min, max] (range), None (any), or bool
ShapeValue = Union[int, List[int], None, bool]


@dataclass
class ShapePattern:
    """
    Pattern for matching layer shapes.

    Values can be:
      - None or omitted: matches anything
      - int: exact match required
      - [min, max]: value must be in range (inclusive)
      - bool: for boolean flags like 'fits_in_l1'
    """
    pattern: Dict[str, ShapeValue] = field(default_factory=dict)

    def __post_init__(self):
        # Ensure pattern is a dict
        if not isinstance(self.pattern, dict):
            self.pattern = {}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ShapePattern":
        """Create ShapePattern from dictionary."""
        return cls(pattern=d)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.pattern.copy()


@dataclass
class TileConfig:
    """
    Tiling configuration for a layer.

    Different operations use different tile dimensions:
    - Linear: tile_m, tile_n, tile_k
    - Conv2D: tile_h, tile_w, tile_c
    - MHSA: tile_q, tile_seq
    """
    config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TileConfig":
        """Create TileConfig from dictionary."""
        return cls(config=d if d else {})

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.config.copy()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value."""
        return self.config.get(key, default)


@dataclass
class KernelConfig:
    """
    Kernel variant selection and configuration.

    Controls which kernel implementation to use and its parameters.
    """
    config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KernelConfig":
        """Create KernelConfig from dictionary."""
        return cls(config=d if d else {})

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.config.copy()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value."""
        return self.config.get(key, default)


@dataclass
class PipelineConfig:
    """
    DMA/compute pipeline configuration.

    Controls how DMA transfers are overlapped with computation.
    """
    enable_overlap: bool = True
    prefetch_depth: int = 2
    double_buffer: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineConfig":
        """Create PipelineConfig from dictionary."""
        if not d:
            return cls()
        return cls(
            enable_overlap=d.get("enable_overlap", True),
            prefetch_depth=d.get("prefetch_depth", 2),
            double_buffer=d.get("double_buffer", True),
            extra={k: v for k, v in d.items()
                   if k not in ("enable_overlap", "prefetch_depth", "double_buffer")}
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "enable_overlap": self.enable_overlap,
            "prefetch_depth": self.prefetch_depth,
            "double_buffer": self.double_buffer,
        }
        result.update(self.extra)
        return result


@dataclass
class CompileFlags:
    """
    C compile-time flags for an optimization.

    These are passed to the C compiler as -D flags.
    """
    flags: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CompileFlags":
        """Create CompileFlags from dictionary."""
        return cls(flags=d if d else {})

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.flags.copy()

    def to_makefile_string(self) -> str:
        """Convert to Makefile CFLAGS format."""
        parts = []
        for key, value in self.flags.items():
            if isinstance(value, bool):
                if value:
                    parts.append(f"-D{key}=1")
            else:
                parts.append(f"-D{key}={value}")
        return " ".join(parts)


@dataclass
class PerformanceMetrics:
    """
    Measured performance metrics for an optimization.
    """
    macs_per_cycle: Optional[float] = None
    overlap_ratio: Optional[float] = None  # 0.0 to 1.0
    total_cycles: Optional[int] = None
    compute_cycles: Optional[int] = None
    dma_cycles: Optional[int] = None
    improvement: Optional[str] = None  # Human-readable, e.g., "27% vs baseline"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PerformanceMetrics":
        """Create PerformanceMetrics from entry dictionary."""
        return cls(
            macs_per_cycle=d.get("measured_macs_per_cycle"),
            overlap_ratio=d.get("measured_overlap_ratio"),
            total_cycles=d.get("measured_cycles"),
            improvement=d.get("measured_improvement"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        if self.macs_per_cycle is not None:
            result["measured_macs_per_cycle"] = self.macs_per_cycle
        if self.overlap_ratio is not None:
            result["measured_overlap_ratio"] = self.overlap_ratio
        if self.total_cycles is not None:
            result["measured_cycles"] = self.total_cycles
        if self.improvement is not None:
            result["measured_improvement"] = self.improvement
        return result


@dataclass
class OptimizationEntry:
    """
    A single entry in the knowledge base.

    Represents a learned optimization configuration for a specific
    operation type and shape pattern.
    """
    op_type: str
    shape_pattern: ShapePattern
    tile_config: TileConfig = field(default_factory=TileConfig)
    kernel_config: KernelConfig = field(default_factory=KernelConfig)
    pipeline_config: PipelineConfig = field(default_factory=PipelineConfig)
    compile_flags: CompileFlags = field(default_factory=CompileFlags)
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)

    # Metadata
    description: str = ""
    source: str = "unknown"
    test_network: str = ""
    confidence: float = 1.0  # 0-1, how confident we are in this config
    notes: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OptimizationEntry":
        """Create OptimizationEntry from JSON dictionary."""
        return cls(
            op_type=d.get("op_type", "unknown"),
            shape_pattern=ShapePattern.from_dict(d.get("shape_pattern", {})),
            tile_config=TileConfig.from_dict(d.get("tile_config")),
            kernel_config=KernelConfig.from_dict(d.get("kernel_config")),
            pipeline_config=PipelineConfig.from_dict(d.get("pipeline_config")),
            compile_flags=CompileFlags.from_dict(d.get("compile_flags")),
            performance=PerformanceMetrics.from_dict(d),
            description=d.get("description", ""),
            source=d.get("source", "unknown"),
            test_network=d.get("test_network", ""),
            confidence=d.get("confidence", 1.0),
            notes=d.get("notes", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result = {
            "op_type": self.op_type,
            "shape_pattern": self.shape_pattern.to_dict(),
        }
        if self.description:
            result["description"] = self.description
        if self.tile_config.config:
            result["tile_config"] = self.tile_config.to_dict()
        if self.kernel_config.config:
            result["kernel_config"] = self.kernel_config.to_dict()
        if self.pipeline_config.extra or not self.pipeline_config.enable_overlap:
            result["pipeline_config"] = self.pipeline_config.to_dict()
        if self.compile_flags.flags:
            result["compile_flags"] = self.compile_flags.to_dict()
        result.update(self.performance.to_dict())
        result["source"] = self.source
        if self.test_network:
            result["test_network"] = self.test_network
        result["confidence"] = self.confidence
        if self.notes:
            result["notes"] = self.notes
        return result


@dataclass
class NegativeResult:
    """
    A record of an optimization that was tried but didn't work.

    This helps avoid repeating failed experiments.
    """
    op_type: str
    attempted_config: Dict[str, Any]
    result: str  # e.g., "regressed", "regressed +45.7%"
    source: str = ""
    notes: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NegativeResult":
        """Create NegativeResult from JSON dictionary."""
        return cls(
            op_type=d.get("op_type", "unknown"),
            attempted_config=d.get("attempted_config", {}),
            result=d.get("result", "unknown"),
            source=d.get("source", ""),
            notes=d.get("notes", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result = {
            "op_type": self.op_type,
            "attempted_config": self.attempted_config,
            "result": self.result,
        }
        if self.source:
            result["source"] = self.source
        if self.notes:
            result["notes"] = self.notes
        return result


@dataclass
class PerformanceBaseline:
    """Reference performance baseline for a test network."""
    test_name: str
    total_cycles: int
    macs_per_total_cycle: float
    notes: str = ""

    @classmethod
    def from_dict(cls, name: str, d: Dict[str, Any]) -> "PerformanceBaseline":
        """Create PerformanceBaseline from JSON dictionary."""
        return cls(
            test_name=name,
            total_cycles=d.get("total_cycles", 0),
            macs_per_total_cycle=d.get("macs_per_total_cycle", 0.0),
            notes=d.get("notes", ""),
        )
