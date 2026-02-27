# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Memory-level annotations and report emission."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..pipeline.context import PipelineContext
from ..pipeline.pass_base import CodegenPass
from .memory_levels import MemoryHierarchy, MemoryLevel, as_memory_level

_WEIGHT_LEVEL_MAP = {
    "L2": MemoryLevel.L2,
    "L3_STAGED": MemoryLevel.L3,
    "L3_TILED": MemoryLevel.L3,
    "MAMBA_SCRATCH": MemoryLevel.L2,
}


def _collect_unique_buffers(generator: Any) -> List[Dict[str, Any]]:
    unique: Dict[str, Dict[str, Any]] = {}

    for buf in getattr(generator, "activation_buffers", []) or []:
        name = buf.get("name")
        if name:
            unique[name] = buf

    for buf in getattr(generator, "shared_activation_pool", []) or []:
        name = buf.get("name")
        if name:
            unique[name] = buf

    for block_buffers in (getattr(generator, "block_activation_buffers", {}) or {}).values():
        for buf in block_buffers:
            name = buf.get("name")
            if name:
                unique[name] = buf

    return list(unique.values())


def _collect_input_buffer_names(generator: Any) -> Set[str]:
    names: Set[str] = {"input_quant"}

    for buffer_name in (getattr(generator, "branch_input_buffers", {}) or {}).values():
        if buffer_name:
            names.add(buffer_name)

    for entry in getattr(generator, "additional_input_entries", []) or []:
        buffer_name = entry.get("buffer_name")
        if buffer_name:
            names.add(buffer_name)

    return names


def _iter_spec_inputs(spec: Dict[str, Any]) -> Iterable[str]:
    for key in ("input_buffer", "input1_buffer", "input2_buffer"):
        value = spec.get(key)
        if isinstance(value, str) and value:
            yield value

    values = spec.get("input_buffers")
    if isinstance(values, list):
        for value in values:
            if isinstance(value, str) and value:
                yield value

    for key in ("q_buffer", "k_buffer", "v_buffer", "ctx_buffer"):
        value = spec.get(key)
        if isinstance(value, str) and value:
            yield value


def _collect_output_buffer_names(specs: Sequence[Dict[str, Any]]) -> Set[str]:
    produced: Set[str] = set()
    consumed: Set[str] = set()

    for spec in specs:
        out = spec.get("output_buffer")
        if isinstance(out, str) and out:
            produced.add(out)
        consumed.update(_iter_spec_inputs(spec))

    terminal = produced - consumed
    if terminal:
        return terminal

    for spec in reversed(specs):
        out = spec.get("output_buffer")
        if isinstance(out, str) and out:
            return {out}

    return set()


def _is_scratch_name(name: str, comment: str) -> bool:
    tag = f"{name} {comment}".lower()
    scratch_markers = ("scratch", "workspace", "slab", "stash", "temp", "tmp")
    return any(marker in tag for marker in scratch_markers)


def _infer_streaming_role(
    name: str,
    comment: str,
    is_input: bool,
    is_output: bool,
) -> str:
    if is_input:
        return "input"
    if is_output:
        return "output"
    if _is_scratch_name(name, comment):
        return "scratch"
    if "weight" in name.lower() or "weight" in comment.lower():
        return "weight"
    return "activation"


def _resolve_levels(
    buffer_entry: Dict[str, Any],
    role: str,
    is_input: bool,
    is_output: bool,
) -> Dict[str, Any]:
    preferred = as_memory_level(buffer_entry.get("preferred_level"))
    required = as_memory_level(buffer_entry.get("required_level"))
    spill_allowed = buffer_entry.get("spill_allowed")
    l2_required = bool(buffer_entry.get("l2_required", False))
    use_l3_fallback = bool(buffer_entry.get("use_l3_fallback", False))

    if use_l3_fallback:
        preferred = MemoryLevel.L3
        if required == MemoryLevel.L2:
            required = None
        spill_allowed = True
    elif l2_required:
        preferred = MemoryLevel.L2
        required = MemoryLevel.L2
        spill_allowed = False
    elif is_input or is_output:
        preferred = preferred or MemoryLevel.L2
        required = required or MemoryLevel.L2
        spill_allowed = False
    elif role == "scratch":
        preferred = preferred or MemoryLevel.L2
        required = required or MemoryLevel.L2
        spill_allowed = False
    else:
        preferred = preferred or MemoryLevel.L2
        if spill_allowed is None:
            spill_allowed = True

    if required is not None:
        preferred = required

    return {
        "preferred_level": preferred.value if preferred else None,
        "required_level": required.value if required else None,
        "spill_allowed": bool(spill_allowed),
    }


def _annotate_buffer(
    buffer_entry: Dict[str, Any],
    hierarchy: MemoryHierarchy,
    input_buffers: Set[str],
    output_buffers: Set[str],
) -> Dict[str, Any]:
    name = str(buffer_entry.get("name", ""))
    c_name = str(buffer_entry.get("c_name", ""))
    comment = str(buffer_entry.get("comment", ""))
    is_input = name in input_buffers or c_name in input_buffers
    is_output = name in output_buffers or c_name in output_buffers

    role = _infer_streaming_role(name=name, comment=comment, is_input=is_input, is_output=is_output)
    levels = _resolve_levels(buffer_entry, role=role, is_input=is_input, is_output=is_output)
    preferred = as_memory_level(levels["preferred_level"]) or MemoryLevel.L2

    annotation = {
        "name": name,
        "c_name": c_name,
        "ctype": buffer_entry.get("ctype"),
        "numel": buffer_entry.get("numel"),
        "preferred_level": levels["preferred_level"],
        "required_level": levels["required_level"],
        "spill_allowed": levels["spill_allowed"],
        "streaming_role": role,
        "is_pool": bool(buffer_entry.get("is_pool", False)),
        "is_block_buffer": bool(buffer_entry.get("is_block_buffer", False)),
        "use_l3_fallback": bool(buffer_entry.get("use_l3_fallback", False)),
        "path_from_L2": hierarchy.path_names(MemoryLevel.L2, preferred),
    }

    buffer_entry["preferred_level"] = annotation["preferred_level"]
    buffer_entry["required_level"] = annotation["required_level"]
    buffer_entry["spill_allowed"] = annotation["spill_allowed"]
    buffer_entry["streaming_role"] = annotation["streaming_role"]

    return annotation


def _layer_weight_annotation(spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    residency = spec.get("weight_residency")
    has_weights = residency is not None or spec.get("weight_index") is not None
    if not has_weights:
        return None

    preferred_level = _WEIGHT_LEVEL_MAP.get(str(residency), MemoryLevel.L2)
    if residency == "MAMBA_SCRATCH":
        required_level = MemoryLevel.L2
        spill_allowed = False
        streaming_role = "scratch"
    elif residency in ("L3_STAGED", "L3_TILED"):
        required_level = None
        spill_allowed = True
        streaming_role = "weight_stream"
    else:
        required_level = MemoryLevel.L2 if residency == "L2" else None
        spill_allowed = False
        streaming_role = "weight"

    return {
        "residency": residency,
        "preferred_level": preferred_level.value,
        "required_level": required_level.value if required_level else None,
        "spill_allowed": spill_allowed,
        "streaming_role": streaming_role,
    }


def _layer_scratch_buffers(spec: Dict[str, Any]) -> List[str]:
    scratch = []
    for key, value in spec.items():
        if not isinstance(value, str) or not value:
            continue
        if not key.endswith("_buffer"):
            continue
        if "slab" in key or "scratch" in key:
            scratch.append(value)
    return scratch


def _build_layer_annotation(
    spec: Dict[str, Any],
    buffer_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    inputs = [buf for buf in (_iter_spec_inputs(spec)) if buf in buffer_lookup]
    outputs = []
    out = spec.get("output_buffer")
    if isinstance(out, str) and out and out in buffer_lookup:
        outputs.append(out)
    scratch = [buf for buf in _layer_scratch_buffers(spec) if buf in buffer_lookup]

    layer_annotation = {
        "name": spec.get("name"),
        "op": spec.get("op"),
        "inputs": [buffer_lookup[name] for name in inputs],
        "outputs": [buffer_lookup[name] for name in outputs],
        "scratch": [buffer_lookup[name] for name in scratch],
        "weights": _layer_weight_annotation(spec),
    }
    spec["memory_annotation"] = layer_annotation
    return layer_annotation


def _count(values: Iterable[Optional[str]]) -> Dict[str, int]:
    counter = Counter((value if value is not None else "UNSPECIFIED") for value in values)
    return dict(sorted(counter.items()))


def _resolve_target_memory_caps(generator: Any) -> Tuple[Optional[int], int, Optional[int]]:
    """Resolve memory hierarchy capacities from the active target/generator."""
    target = getattr(generator, "target", None)
    if target is None:
        raise ValueError("Generator is missing target; cannot resolve memory hierarchy.")

    if hasattr(target, "validate_required_capabilities"):
        target.validate_required_capabilities()

    memory = getattr(target, "memory", None)
    if memory is None:
        raise ValueError(
            f"Target '{getattr(target, 'name', 'unknown')}' does not expose memory capabilities."
        )

    # Honor explicit generator override for L1 when present.
    l1_bytes = getattr(generator, "l1_budget_bytes", None)
    if l1_bytes is None:
        l1_bytes = memory.l1_bytes

    l2_bytes = memory.l2_bytes
    l3_bytes = memory.l3_bytes

    if l2_bytes is None:
        raise ValueError(
            f"Target '{getattr(target, 'name', 'unknown')}' must define memory.l2_bytes."
        )

    return l1_bytes, l2_bytes, l3_bytes


def annotate_generator_memory_levels(generator: Any) -> Dict[str, Any]:
    """Annotate generator buffers/specs with explicit memory-level metadata."""
    l1_bytes, l2_bytes, l3_bytes = _resolve_target_memory_caps(generator)

    hierarchy = MemoryHierarchy.standard_3_level(
        l1_bytes=l1_bytes,
        l2_bytes=l2_bytes,
        l3_bytes=l3_bytes,
    )

    input_buffers = _collect_input_buffer_names(generator)
    output_buffers = _collect_output_buffer_names(getattr(generator, "layer_specs", []) or [])
    buffer_entries = _collect_unique_buffers(generator)

    buffer_annotations: Dict[str, Dict[str, Any]] = {}
    c_name_lookup: Dict[str, Dict[str, Any]] = {}
    for buffer_entry in buffer_entries:
        annotation = _annotate_buffer(
            buffer_entry=buffer_entry,
            hierarchy=hierarchy,
            input_buffers=input_buffers,
            output_buffers=output_buffers,
        )
        buffer_annotations[annotation["name"]] = annotation
        c_name = annotation.get("c_name")
        if c_name:
            c_name_lookup[c_name] = annotation

    buffer_lookup = dict(buffer_annotations)
    buffer_lookup.update(c_name_lookup)

    layer_annotations = [
        _build_layer_annotation(spec=spec, buffer_lookup=buffer_lookup)
        for spec in (getattr(generator, "layer_specs", []) or [])
    ]

    report = {
        "phase": "phase3_memory_level_model",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "memory_hierarchy": hierarchy.to_dict(),
        "summary": {
            "buffer_count": len(buffer_annotations),
            "layer_count": len(layer_annotations),
            "preferred_level_counts": _count(
                ann.get("preferred_level") for ann in buffer_annotations.values()
            ),
            "required_level_counts": _count(
                ann.get("required_level") for ann in buffer_annotations.values()
            ),
            "streaming_role_counts": _count(
                ann.get("streaming_role") for ann in buffer_annotations.values()
            ),
        },
        "buffers": sorted(buffer_annotations.values(), key=lambda ann: ann["name"]),
        "layers": layer_annotations,
    }

    report_path = Path(generator.output_dir) / "memory_level_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    generator.memory_hierarchy = hierarchy.to_dict()
    generator.buffer_memory_annotations = buffer_annotations
    generator.memory_level_report = report
    generator.memory_level_report_path = str(report_path)
    generator._memory_levels_ready = True
    return report


class MemoryLevelAnnotationPass(CodegenPass):
    """Pipeline pass that materializes memory-level annotations."""

    name = "memory_level_annotation"

    def run(self, context: PipelineContext) -> None:
        generator = context.generator
        generator._ensure_codegen_metadata()
        generator.detect_l3_activation_fallback()
        report = annotate_generator_memory_levels(generator)
        context.buffer_annotations = getattr(generator, "buffer_memory_annotations", {})
        context.memory_level_report = report
        context.memory_hierarchy = report.get("memory_hierarchy", {})
        context.diagnostics["memory_level_report"] = str(
            getattr(generator, "memory_level_report_path", "")
        )
        context.diagnostics["memory_level_buffers"] = report["summary"]["buffer_count"]
