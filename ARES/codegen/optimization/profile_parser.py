# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Profile Parser for GVSOC output.

Parses GVSOC run logs and profile_suite.py CSV outputs into structured
LayerProfile objects for analysis.

Usage:
    from codegen.optimization import ProfileParser

    parser = ProfileParser()

    # Parse a GVSOC log file
    profiles = parser.parse_log("gvsoc_run.log")

    # Parse a layers.csv from profile_suite.py
    profiles = parser.parse_csv("layers.csv")

    for p in profiles:
        print(f"{p.name}: {p.macs_per_cycle:.2f} MACs/cycle")
"""

import re
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class LayerProfile:
    """Parsed performance data for a single layer."""
    name: str
    op_type: str
    total_cycles: int
    compute_cycles: int = 0
    dma_load_cycles: int = 0
    dma_store_cycles: int = 0
    idle_cycles: int = 0
    macs: Optional[int] = None

    # Additional metrics
    l1_tiled: bool = False
    memory_tier: str = "L2_FULL"
    tile_count: int = 1

    @property
    def macs_per_cycle(self) -> float:
        """Compute efficiency in MACs per cycle."""
        if self.macs and self.compute_cycles > 0:
            return self.macs / self.compute_cycles
        elif self.macs and self.total_cycles > 0:
            return self.macs / self.total_cycles
        return 0.0

    @property
    def macs_per_total_cycle(self) -> float:
        """Compute efficiency including all overhead."""
        if self.macs and self.total_cycles > 0:
            return self.macs / self.total_cycles
        return 0.0

    @property
    def overlap_ratio(self) -> float:
        """DMA/compute overlap ratio (1.0 = perfect overlap)."""
        if self.total_cycles > 0 and self.idle_cycles >= 0:
            return 1.0 - (self.idle_cycles / self.total_cycles)
        return 0.0

    @property
    def is_bottleneck(self) -> bool:
        """Check if this layer is a performance bottleneck."""
        return self.overlap_ratio < 0.7 or self.macs_per_cycle < 5.0

    @property
    def is_significant(self) -> bool:
        """Check if this layer consumes significant cycles."""
        return self.total_cycles > 100000

    def __repr__(self) -> str:
        return (f"LayerProfile({self.name}, {self.op_type}, "
                f"cycles={self.total_cycles}, MACs/cyc={self.macs_per_cycle:.2f})")


@dataclass
class NetworkProfile:
    """Complete profile for a network run."""
    test_name: str
    layers: List[LayerProfile] = field(default_factory=list)
    total_cycles: int = 0
    total_macs: int = 0

    @property
    def macs_per_cycle(self) -> float:
        """Overall network efficiency."""
        if self.total_macs and self.total_cycles > 0:
            return self.total_macs / self.total_cycles
        return 0.0

    @property
    def bottleneck_layers(self) -> List[LayerProfile]:
        """Return layers that are performance bottlenecks."""
        return [l for l in self.layers if l.is_bottleneck and l.is_significant]

    def get_layer(self, name: str) -> Optional[LayerProfile]:
        """Get a layer profile by name."""
        for layer in self.layers:
            if layer.name == name:
                return layer
        return None


class ProfileParser:
    """Parse GVSOC output into structured profile data."""

    # Regex patterns for parsing GVSOC output
    LAYER_START_PATTERN = re.compile(
        r'CL:\s*(?:Layer\s+)?(\d+)[\s:]+([^\s:]+)\s*(?:\(([^)]+)\))?'
    )
    CYCLES_PATTERN = re.compile(
        r'(?:cycles|Cycles)[\s:=]+(\d[\d,]*)'
    )
    LAYER_CYCLES_PATTERN = re.compile(
        r'(\S+)\s+cycles:\s*(\d[\d,]*)'
    )
    # PERF format: "PERF layer_name : total=X compute=Y dma_load=Z dma_store=W idle=I overlap=P%"
    PERF_LINE_PATTERN = re.compile(
        r'PERF\s+(\S+)\s+:\s+total=\s*(\d+)\s+compute=\s*(\d+)\s+'
        r'dma_load=\s*(\d+)\s+dma_store=\s*(\d+)\s+idle=\s*(\d+)\s+'
        r'overlap=\s*([\d.]+)%'
    )
    TOTAL_CYCLES_PATTERN = re.compile(
        r'Total\s+(?:network\s+)?cycles[\s:=]+(\d[\d,]*)',
        re.IGNORECASE
    )
    MACS_PATTERN = re.compile(
        r'MACs[\s:=]+(\d[\d,]*)'
    )
    MACS_PER_CYCLE_PATTERN = re.compile(
        r'MACs/(?:cycle|cyc)[\s:=]+([\d.]+)'
    )
    OVERLAP_PATTERN = re.compile(
        r'overlap[\s:=]+([\d.]+)%?'
    )
    TILE_PATTERN = re.compile(
        r'(\d+)\s*tiles?\s*\((\d+)x(\d+)\)'
    )
    L1_TILED_PATTERN = re.compile(
        r'\[L1[_\s]?TILED\]|\[PIPELINED\]|L1\s+double.?buffer'
    )

    # Op type detection patterns - ORDER MATTERS! More specific patterns first.
    # E.g., "quant" must come before "linear" so "pre_linear_quant" → requantize, not linear
    # Activations (gelu, relu) must come before linear/fc patterns so "mlp_gelu" → gelu, not linear
    OP_TYPE_MAP = {
        # Requantize patterns (check first - often embedded in other names)
        'requant': 'requantize',
        'quant': 'requantize',
        # Activation patterns (check BEFORE linear/fc so "mlp_gelu" → gelu, not linear)
        'gelu': 'gelu_int8',
        'relu': 'relu',
        # Convolution patterns
        'conv2d': 'conv2d_int8',
        'conv': 'conv2d_int8',
        # Linear patterns (after activations!)
        'mlp_fc': 'linear_int8',
        'freq_fc': 'linear_int8',
        'classifier': 'linear_int8',
        'linear': 'linear_int8',
        'fc': 'linear_int8',
        # Pooling patterns
        'maxpool': 'maxpool_int8',
        'avgpool': 'avgpool_int8',
        'global_pool': 'avgpool_int8',
        'pool': 'maxpool_int8',  # Default pool to maxpool (most common)
        # Attention patterns
        'cross_attn': 'mhsa_int8',
        'attention': 'mhsa_int8',
        'mhsa': 'mhsa_int8',
        'attn': 'mhsa_int8',
        # Normalization patterns
        'layernorm': 'layernorm_int8',
        'groupnorm': 'layernorm_int8',
        'final_norm': 'layernorm_int8',
        'norm': 'layernorm_int8',
        # Other ops
        'add': 'add_int8',
        'concat': 'concat',
        'ssm': 'ssm_int8',
        'mamba': 'ssm_int8',
        'rfft': 'rfft_int8',
        'embed': 'embedding',
    }

    def __init__(self):
        self.verbose = False

    def parse_log(self, log_path: str) -> NetworkProfile:
        """
        Parse a GVSOC run log file.

        Args:
            log_path: Path to gvsoc_run.log or similar

        Returns:
            NetworkProfile with parsed layer data
        """
        log_path = Path(log_path)
        if not log_path.exists():
            raise FileNotFoundError(f"Log file not found: {log_path}")

        with open(log_path, 'r') as f:
            content = f.read()

        # Extract test name from path
        test_name = log_path.parent.parent.name if 'generated' in str(log_path) else 'unknown'

        profile = NetworkProfile(test_name=test_name)

        # Parse layer-by-layer data
        layers_data = self._extract_layer_data(content)

        for name, data in layers_data.items():
            # Skip summary/non-layer entries
            if name.lower() in self.SUMMARY_NAMES:
                continue

            layer = LayerProfile(
                name=name,
                op_type=data.get('op_type', self._infer_op_type(name)),
                total_cycles=data.get('cycles', 0),
                compute_cycles=data.get('compute_cycles', data.get('cycles', 0)),
                dma_load_cycles=data.get('dma_load_cycles', 0),
                dma_store_cycles=data.get('dma_store_cycles', 0),
                idle_cycles=data.get('idle_cycles', 0),
                macs=data.get('macs'),
                l1_tiled=data.get('l1_tiled', False),
                tile_count=data.get('tile_count', 1),
            )
            profile.layers.append(layer)

        # Parse total cycles
        total_match = self.TOTAL_CYCLES_PATTERN.search(content)
        if total_match:
            profile.total_cycles = self._parse_number(total_match.group(1))
        else:
            profile.total_cycles = sum(l.total_cycles for l in profile.layers)

        # Calculate total MACs
        profile.total_macs = sum(l.macs or 0 for l in profile.layers)

        return profile

    # Summary line names to exclude (not actual layers)
    SUMMARY_NAMES = {
        'total', 'cluster', 'compute', 'dma', 'idle', 'summary',
        'performance', 'cycles', 'load', 'store'
    }

    def _extract_layer_data(self, content: str) -> Dict[str, Dict]:
        """Extract per-layer performance data from log content."""
        layers = {}
        current_layer = None

        for line in content.split('\n'):
            # Check for PERF format line (highest priority - most complete data)
            perf_match = self.PERF_LINE_PATTERN.search(line)
            if perf_match:
                layer_name = perf_match.group(1)
                # Skip summary lines (not actual layers)
                if layer_name.lower() in self.SUMMARY_NAMES:
                    continue
                layers[layer_name] = {
                    'op_type': self._infer_op_type(layer_name),
                    'cycles': int(perf_match.group(2)),
                    'compute_cycles': int(perf_match.group(3)),
                    'dma_load_cycles': int(perf_match.group(4)),
                    'dma_store_cycles': int(perf_match.group(5)),
                    'idle_cycles': int(perf_match.group(6)),
                    'overlap_ratio': float(perf_match.group(7)) / 100.0,
                }
                continue

            # Check for layer start
            layer_match = self.LAYER_START_PATTERN.search(line)
            if layer_match:
                current_layer = layer_match.group(2)
                op_hint = layer_match.group(3) if layer_match.group(3) else ''
                if current_layer not in layers:
                    layers[current_layer] = {
                        'op_type': self._infer_op_type(current_layer, op_hint),
                    }

            # Check for layer cycles line
            layer_cycles_match = self.LAYER_CYCLES_PATTERN.search(line)
            if layer_cycles_match:
                layer_name = layer_cycles_match.group(1)
                cycles = self._parse_number(layer_cycles_match.group(2))
                if layer_name not in layers:
                    layers[layer_name] = {'op_type': self._infer_op_type(layer_name)}
                layers[layer_name]['cycles'] = cycles

            # Check for cycles in current layer context
            if current_layer and current_layer in layers:
                cycles_match = self.CYCLES_PATTERN.search(line)
                if cycles_match and 'cycles' not in layers[current_layer]:
                    layers[current_layer]['cycles'] = self._parse_number(cycles_match.group(1))

                macs_match = self.MACS_PATTERN.search(line)
                if macs_match:
                    layers[current_layer]['macs'] = self._parse_number(macs_match.group(1))

                if self.L1_TILED_PATTERN.search(line):
                    layers[current_layer]['l1_tiled'] = True

                tile_match = self.TILE_PATTERN.search(line)
                if tile_match:
                    layers[current_layer]['tile_count'] = int(tile_match.group(1))

        return layers

    def _infer_op_type(self, layer_name: str, hint: str = '') -> str:
        """Infer operation type from layer name."""
        name_lower = layer_name.lower()
        hint_lower = hint.lower()

        for keyword, op_type in self.OP_TYPE_MAP.items():
            if keyword in name_lower or keyword in hint_lower:
                return op_type

        return 'unknown'

    def _parse_number(self, s: str) -> int:
        """Parse a number string, handling commas."""
        return int(s.replace(',', ''))

    def parse_csv(self, csv_path: str) -> NetworkProfile:
        """
        Parse a layers.csv from profile_suite.py.

        Args:
            csv_path: Path to layers.csv file

        Returns:
            NetworkProfile with parsed layer data
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Extract test name from path
        test_name = csv_path.parent.name if csv_path.parent.name.startswith('test_') else 'unknown'

        profile = NetworkProfile(test_name=test_name)

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                layer = LayerProfile(
                    name=row.get('layer', row.get('name', 'unknown')),
                    op_type=row.get('op_type', row.get('type', 'unknown')),
                    total_cycles=int(row.get('cycles', row.get('total_cycles', 0))),
                    compute_cycles=int(row.get('compute_cycles', row.get('cycles', 0))),
                    macs=int(row.get('macs', 0)) if row.get('macs') else None,
                )
                profile.layers.append(layer)

        profile.total_cycles = sum(l.total_cycles for l in profile.layers)
        profile.total_macs = sum(l.macs or 0 for l in profile.layers)

        return profile

    def parse_profile_macs_output(self, output: str) -> NetworkProfile:
        """
        Parse output from tools/profile_macs.py.

        Args:
            output: String output from profile_macs.py

        Returns:
            NetworkProfile with parsed layer data
        """
        profile = NetworkProfile(test_name='unknown')

        # Parse markdown table format
        in_table = False
        for line in output.split('\n'):
            if '|' not in line:
                continue

            parts = [p.strip() for p in line.split('|')]
            parts = [p for p in parts if p]

            if len(parts) < 3:
                continue

            # Skip header separator
            if parts[0].startswith('-'):
                in_table = True
                continue

            # Skip header
            if parts[0].lower() in ('layer', 'name'):
                continue

            if in_table and len(parts) >= 4:
                try:
                    layer = LayerProfile(
                        name=parts[0],
                        op_type=self._infer_op_type(parts[0]),
                        total_cycles=self._parse_number(parts[1]) if parts[1].replace(',', '').isdigit() else 0,
                        macs=self._parse_number(parts[2]) if parts[2].replace(',', '').isdigit() else None,
                    )
                    profile.layers.append(layer)
                except (ValueError, IndexError):
                    continue

        profile.total_cycles = sum(l.total_cycles for l in profile.layers)
        profile.total_macs = sum(l.macs or 0 for l in profile.layers)

        return profile


def parse_gvsoc_log(log_path: str) -> NetworkProfile:
    """Convenience function to parse a GVSOC log."""
    parser = ProfileParser()
    return parser.parse_log(log_path)


def parse_profile_csv(csv_path: str) -> NetworkProfile:
    """Convenience function to parse a profile CSV."""
    parser = ProfileParser()
    return parser.parse_csv(csv_path)
