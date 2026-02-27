# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Default pipeline runner and pass wrappers."""

import time
from typing import Iterable

from ..memory.annotation_pass import MemoryLevelAnnotationPass
from .context import PipelineContext
from .pass_base import CodegenPass


class CodegenPipeline:
    """Run a list of passes in order while collecting timing diagnostics."""

    def __init__(self, passes: Iterable[CodegenPass]):
        self.passes = list(passes)

    def run(self, context: PipelineContext) -> PipelineContext:
        print("\n[PipelineV2] Running default codegen pipeline...")
        for pipeline_pass in self.passes:
            stage_name = pipeline_pass.name
            start_s = time.perf_counter()
            pipeline_pass.run(context)
            elapsed_s = time.perf_counter() - start_s
            context.add_timing(stage_name, elapsed_s)
            context.diagnostics.setdefault("stages", []).append(
                {"name": stage_name, "elapsed_s": elapsed_s}
            )
            print(f"  [PipelineV2] {stage_name}: {elapsed_s:.3f}s")
        return context


class ExtractModelPass(CodegenPass):
    """Wrapper for model extraction/init state collection."""

    name = "extract_model"

    def run(self, context: PipelineContext) -> None:
        generator = context.generator
        context.diagnostics["layer_count"] = len(getattr(generator, "layer_order", []))
        context.diagnostics["input_count"] = len(getattr(generator, "input_quant_layers", []))


class BuildLayerSpecsPass(CodegenPass):
    """Wrapper around layer spec construction."""

    name = "build_layer_specs"

    def run(self, context: PipelineContext) -> None:
        generator = context.generator
        generator._ensure_codegen_metadata()
        context.diagnostics["layer_specs_count"] = len(getattr(generator, "layer_specs", []))


class FusionBoundaryPass(CodegenPass):
    """Stage boundary for fusion diagnostics.

    Fusion is currently applied inside `_build_layer_specs()`. This pass keeps
    an explicit stage boundary without changing behavior.
    """

    # Keep stage identifier stable for existing perf/checkpoint tooling.
    name = "legacy_fusion"

    def run(self, context: PipelineContext) -> None:
        generator = context.generator
        context.diagnostics["fusion_enabled"] = bool(getattr(generator, "enable_fusion", False))
        context.diagnostics["fused_layers_count"] = len(getattr(generator, "fused_layers", []))


class MemoryPlanningPass(CodegenPass):
    """Wrapper around L3 fallback + memory planner flow."""

    # Keep stage identifier stable for existing perf/checkpoint tooling.
    name = "legacy_memory_planning"

    def run(self, context: PipelineContext) -> None:
        generator = context.generator
        generator._ensure_codegen_metadata()
        generator._prepare_memory_levels()
        generator._run_memory_planner()
        context.diagnostics["l2_arena_size"] = getattr(generator, "l2_arena_size", 0)


class EmitCodePass(CodegenPass):
    """Wrapper for code emission steps."""

    name = "emit_code"

    def run(self, context: PipelineContext) -> None:
        generator = context.generator
        generator._emit_headers_from_current_state()
        generator.generate_sources()
        generator.generate_makefile()


def build_default_pipeline() -> CodegenPipeline:
    """Build the behavior-preserving default pipeline skeleton."""
    return CodegenPipeline(
        [
            ExtractModelPass(),
            BuildLayerSpecsPass(),
            FusionBoundaryPass(),
            MemoryLevelAnnotationPass(),
            MemoryPlanningPass(),
            EmitCodePass(),
        ]
    )


def run_default_pipeline(generator) -> PipelineContext:
    """Run the default pipeline for a generator instance."""
    context = PipelineContext(generator=generator)
    return build_default_pipeline().run(context)


# Compatibility aliases for older call sites.
LegacyFusionPass = FusionBoundaryPass
LegacyMemoryPlanningPass = MemoryPlanningPass


def build_phase1_compat_pipeline() -> CodegenPipeline:
    """Compatibility wrapper for previous pipeline constructor name."""
    return build_default_pipeline()


def run_phase1_compat_pipeline(generator) -> PipelineContext:
    """Compatibility wrapper for previous pipeline entrypoint name."""
    return run_default_pipeline(generator)
