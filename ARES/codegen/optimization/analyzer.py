# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Performance Analyzer for ARES optimization.

Analyzes layer profiles and generates optimization suggestions based on
known patterns and bottleneck detection.

Usage:
    from codegen.optimization import ProfileParser, PerformanceAnalyzer

    parser = ProfileParser()
    profile = parser.parse_log("gvsoc_run.log")

    analyzer = PerformanceAnalyzer()
    suggestions = analyzer.analyze(profile)

    for s in suggestions:
        print(f"[P{s.priority}] {s.layer_name}: {s.issue}")
        print(f"    Suggestion: {s.suggestion}")
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .profile_parser import LayerProfile, NetworkProfile
from .knowledge_base import KnowledgeBase


@dataclass
class OptimizationSuggestion:
    """A suggested optimization for a layer."""
    layer_name: str
    op_type: str
    issue: str
    suggestion: str
    expected_improvement: str
    priority: int  # 1=high, 2=medium, 3=low
    kb_entry: Optional[str] = None  # Reference to relevant KB entry

    def __repr__(self) -> str:
        return f"[P{self.priority}] {self.layer_name}: {self.issue}"


class PerformanceAnalyzer:
    """Analyze profiles and suggest optimizations."""

    # Thresholds for different metrics
    LOW_OVERLAP_THRESHOLD = 0.7  # Below this is considered poor overlap
    LOW_EFFICIENCY_THRESHOLD = 5.0  # MACs/cycle below this is poor
    HIGH_EFFICIENCY_TARGET = 10.0  # MACs/cycle target for transformer layers
    SIGNIFICANT_CYCLES_THRESHOLD = 100000  # Cycles below this aren't worth optimizing

    # Op-specific targets
    OP_TARGETS = {
        'linear_int8': 10.0,  # Good Linear should hit 9-12 MACs/cycle
        'conv2d_int8': 8.0,   # Conv2D with SIMD should hit 8-12 MACs/cycle
        'mhsa_int8': 6.0,     # MHSA typically 6-8 MACs/cycle
        'layernorm_int8': 4.0,
        'gelu_int8': 3.0,
        'ssm_int8': 2.0,
    }

    def __init__(self, knowledge_base: Optional[KnowledgeBase] = None):
        """
        Initialize analyzer.

        Args:
            knowledge_base: Optional KB for looking up known optimizations
        """
        self.kb = knowledge_base or KnowledgeBase()

    def analyze(self, profile: NetworkProfile) -> List[OptimizationSuggestion]:
        """
        Analyze network profile and return optimization suggestions.

        Args:
            profile: NetworkProfile from ProfileParser

        Returns:
            List of OptimizationSuggestion, sorted by priority
        """
        suggestions = []

        for layer in profile.layers:
            layer_suggestions = self._analyze_layer(layer, profile)
            suggestions.extend(layer_suggestions)

        # Sort by priority (1 = highest)
        suggestions.sort(key=lambda x: (x.priority, -len(x.layer_name)))

        return suggestions

    def _analyze_layer(self, layer: LayerProfile,
                       profile: NetworkProfile) -> List[OptimizationSuggestion]:
        """Analyze a single layer for optimization opportunities."""
        suggestions = []

        # Skip layers with negligible cycle count
        if layer.total_cycles < self.SIGNIFICANT_CYCLES_THRESHOLD:
            return suggestions

        # Check for low DMA/compute overlap
        if layer.overlap_ratio < self.LOW_OVERLAP_THRESHOLD:
            suggestions.append(self._suggest_overlap_fix(layer))

        # Check for low compute efficiency
        target = self.OP_TARGETS.get(layer.op_type, self.LOW_EFFICIENCY_THRESHOLD)
        if layer.macs_per_cycle < target * 0.7:  # Below 70% of target
            suggestions.append(self._suggest_efficiency_fix(layer, target))

        # Check for excessive idle time
        if layer.idle_cycles > layer.compute_cycles:
            suggestions.append(self._suggest_idle_fix(layer))

        # Check KB for known better configs
        kb_suggestion = self._check_kb_for_better_config(layer)
        if kb_suggestion:
            suggestions.append(kb_suggestion)

        return suggestions

    def _suggest_overlap_fix(self, layer: LayerProfile) -> OptimizationSuggestion:
        """Suggest fix for low DMA/compute overlap."""
        issue = f"Low overlap ratio: {layer.overlap_ratio:.1%}"

        if layer.op_type == 'mhsa_int8':
            suggestion = ("Enable pipelined QK+Softmax+AV fusion. "
                         "Check MHSA_FUSE_QK_SOFTMAX_AV flag.")
            improvement = "20-40% cycle reduction"
        elif layer.op_type in ('linear_int8', 'linear_fp32'):
            suggestion = ("Enable L1 input caching and double-buffered weight tiling. "
                         "Check LINEAR_INT8_INPUT_L1_CACHE flag.")
            improvement = "15-25% cycle reduction"
        elif layer.op_type == 'conv2d_int8':
            suggestion = ("Enable L1 double-buffer tiling with async DMA. "
                         "Check tile sizes fit in L1.")
            improvement = "20-30% cycle reduction"
        else:
            suggestion = "Enable DMA/compute pipelining or increase prefetch depth."
            improvement = "Up to 40% cycle reduction"

        return OptimizationSuggestion(
            layer_name=layer.name,
            op_type=layer.op_type,
            issue=issue,
            suggestion=suggestion,
            expected_improvement=improvement,
            priority=1  # High priority - overlap issues are critical
        )

    def _suggest_efficiency_fix(self, layer: LayerProfile,
                                target: float) -> OptimizationSuggestion:
        """Suggest fix for low compute efficiency."""
        issue = f"Low efficiency: {layer.macs_per_cycle:.1f} MACs/cycle (target: {target:.1f})"

        if layer.op_type == 'linear_int8':
            suggestion = ("Try SIMD-optimized kernel, adjust tile sizes for L1 cache, "
                         "or enable fast qround via fcvt.w.s.")
            improvement = f"{target/max(layer.macs_per_cycle, 0.1):.1f}x throughput improvement"
        elif layer.op_type == 'conv2d_int8':
            suggestion = ("Enable outch_unroll=4, try 2-pixel unroll for 1x1 conv, "
                         "or use im2col+SIMD optimization.")
            improvement = "2-3x throughput improvement"
        elif layer.op_type == 'mhsa_int8':
            suggestion = ("Tune tile_q to match L1 model, enable head-contiguous projections, "
                         "or use QK key-loop unroll by 4.")
            improvement = "1.5-2x throughput improvement"
        else:
            suggestion = "Profile compute kernel and check for memory bottlenecks."
            improvement = "Variable improvement"

        return OptimizationSuggestion(
            layer_name=layer.name,
            op_type=layer.op_type,
            issue=issue,
            suggestion=suggestion,
            expected_improvement=improvement,
            priority=1 if layer.macs_per_cycle < target * 0.5 else 2
        )

    def _suggest_idle_fix(self, layer: LayerProfile) -> OptimizationSuggestion:
        """Suggest fix for excessive idle time."""
        idle_ratio = layer.idle_cycles / max(layer.total_cycles, 1)
        issue = f"Excessive idle: {layer.idle_cycles:,} cycles ({idle_ratio:.0%} of total)"

        if layer.op_type in ('linear_int8', 'linear_fp32'):
            suggestion = ("Reduce output tile size to improve core occupancy, "
                         "or enable L1 weight tiling.")
        elif layer.op_type == 'conv2d_int8':
            suggestion = ("Adjust spatial tile size, or use smaller output channel tiles "
                         "to reduce idle between DMA transfers.")
        else:
            suggestion = ("Reduce tile size to improve occupancy, or check "
                         "for unnecessary synchronization barriers.")

        return OptimizationSuggestion(
            layer_name=layer.name,
            op_type=layer.op_type,
            issue=issue,
            suggestion=suggestion,
            expected_improvement="Up to 50% cycle reduction",
            priority=2
        )

    def _check_kb_for_better_config(self, layer: LayerProfile) -> Optional[OptimizationSuggestion]:
        """Check if KB has a known better config for this layer."""
        if self.kb is None:
            return None

        # We can't do shape-based lookup without shape info, so this is limited
        # Just check if there's a better known config for this op type
        best = self.kb.get_best_entry(layer.op_type)
        if best is None:
            return None

        best_macs = best.performance.macs_per_cycle or 0
        if best_macs > layer.macs_per_cycle * 1.2:  # 20% better
            return OptimizationSuggestion(
                layer_name=layer.name,
                op_type=layer.op_type,
                issue=f"KB has better config: {best_macs:.1f} vs {layer.macs_per_cycle:.1f} MACs/cycle",
                suggestion=f"Apply KB entry '{best.description}' - {best.source}",
                expected_improvement=f"+{(best_macs/max(layer.macs_per_cycle, 0.1) - 1)*100:.0f}% efficiency",
                priority=2,
                kb_entry=best.description
            )

        return None

    def generate_report(self, profile: NetworkProfile,
                        output_path: str) -> str:
        """
        Generate markdown report with analysis.

        Args:
            profile: NetworkProfile to analyze
            output_path: Path to write report

        Returns:
            Report content as string
        """
        suggestions = self.analyze(profile)

        lines = [
            "# Performance Analysis Report",
            "",
            f"**Network:** {profile.test_name}",
            f"**Total Cycles:** {profile.total_cycles:,}",
            f"**Total MACs:** {profile.total_macs:,}",
            f"**Overall Efficiency:** {profile.macs_per_cycle:.2f} MACs/cycle",
            "",
        ]

        # Summary
        bottlenecks = profile.bottleneck_layers
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- Total layers analyzed: {len(profile.layers)}")
        lines.append(f"- Bottleneck layers: {len(bottlenecks)}")
        lines.append(f"- Optimization suggestions: {len(suggestions)}")
        lines.append(f"- High priority issues: {sum(1 for s in suggestions if s.priority == 1)}")
        lines.append("")

        # Top bottlenecks table
        if bottlenecks:
            lines.append("## Top Bottleneck Layers")
            lines.append("")
            lines.append("| Layer | Op Type | Cycles | MACs/cycle | Overlap | Issues |")
            lines.append("|-------|---------|--------|------------|---------|--------|")

            for layer in sorted(bottlenecks, key=lambda x: -x.total_cycles)[:10]:
                issues = []
                if layer.overlap_ratio < self.LOW_OVERLAP_THRESHOLD:
                    issues.append("low overlap")
                target = self.OP_TARGETS.get(layer.op_type, self.LOW_EFFICIENCY_THRESHOLD)
                if layer.macs_per_cycle < target * 0.7:
                    issues.append("low efficiency")

                lines.append(
                    f"| {layer.name} | {layer.op_type} | {layer.total_cycles:,} | "
                    f"{layer.macs_per_cycle:.1f} | {layer.overlap_ratio:.0%} | "
                    f"{', '.join(issues) or 'N/A'} |"
                )
            lines.append("")

        # Optimization suggestions
        if suggestions:
            lines.append("## Optimization Suggestions")
            lines.append("")

            for priority in [1, 2, 3]:
                priority_suggestions = [s for s in suggestions if s.priority == priority]
                if not priority_suggestions:
                    continue

                priority_name = {1: "High", 2: "Medium", 3: "Low"}[priority]
                lines.append(f"### {priority_name} Priority")
                lines.append("")

                for s in priority_suggestions:
                    lines.append(f"**{s.layer_name}** ({s.op_type})")
                    lines.append(f"- Issue: {s.issue}")
                    lines.append(f"- Suggestion: {s.suggestion}")
                    lines.append(f"- Expected improvement: {s.expected_improvement}")
                    if s.kb_entry:
                        lines.append(f"- KB reference: {s.kb_entry}")
                    lines.append("")

        # Layer breakdown
        lines.append("## All Layers")
        lines.append("")
        lines.append("| Layer | Op Type | Cycles | % Total | MACs | MACs/cycle |")
        lines.append("|-------|---------|--------|---------|------|------------|")

        for layer in sorted(profile.layers, key=lambda x: -x.total_cycles):
            pct = (layer.total_cycles / max(profile.total_cycles, 1)) * 100
            macs_str = f"{layer.macs:,}" if layer.macs else "N/A"
            lines.append(
                f"| {layer.name} | {layer.op_type} | {layer.total_cycles:,} | "
                f"{pct:.1f}% | {macs_str} | {layer.macs_per_cycle:.2f} |"
            )
        lines.append("")

        content = "\n".join(lines)

        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(content)

        return content

    def print_suggestions(self, suggestions: List[OptimizationSuggestion]) -> None:
        """Print suggestions to console."""
        if not suggestions:
            print("No optimization suggestions.")
            return

        print(f"\n{'='*60}")
        print("OPTIMIZATION SUGGESTIONS")
        print(f"{'='*60}\n")

        for priority in [1, 2, 3]:
            priority_suggestions = [s for s in suggestions if s.priority == priority]
            if not priority_suggestions:
                continue

            priority_name = {1: "HIGH", 2: "MEDIUM", 3: "LOW"}[priority]
            print(f"{priority_name} PRIORITY:")
            print("-" * 40)

            for s in priority_suggestions:
                print(f"\n  [{s.op_type}] {s.layer_name}")
                print(f"    Issue: {s.issue}")
                print(f"    Fix: {s.suggestion}")
                print(f"    Expected: {s.expected_improvement}")

            print()


def analyze_profile(profile: NetworkProfile,
                    output_path: Optional[str] = None) -> List[OptimizationSuggestion]:
    """
    Convenience function to analyze a profile and optionally write report.

    Args:
        profile: NetworkProfile to analyze
        output_path: Optional path to write markdown report

    Returns:
        List of optimization suggestions
    """
    analyzer = PerformanceAnalyzer()
    suggestions = analyzer.analyze(profile)

    if output_path:
        analyzer.generate_report(profile, output_path)

    return suggestions
