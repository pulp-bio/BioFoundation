# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Core model-compatibility checks for ARES extractor support."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

import torch.nn as nn
from brevitas import nn as qnn

try:
    from tools import pytorch_extractor as pe
except ImportError:  # pragma: no cover - fallback when run from tools/
    import pytorch_extractor as pe  # type: ignore

try:
    from tools.ares_support_registry import (
        get_composite_layer_types,
        get_optype_mapping,
        get_replacement_suggestions,
        get_supported_layer_types,
        get_warning_rules,
    )
except ImportError:  # pragma: no cover - fallback when run from tools/
    from ares_support_registry import (  # type: ignore
        get_composite_layer_types,
        get_optype_mapping,
        get_replacement_suggestions,
        get_supported_layer_types,
        get_warning_rules,
    )


@dataclass
class CompatibilityFinding:
    """Single compatibility finding for a module in named_modules()."""

    module_name: str
    class_name: str
    status: str  # supported | warning | unsupported
    reason: str
    suggestion: str = ""
    detected_type: Optional[str] = None
    mapped_optype: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_name": self.module_name,
            "class_name": self.class_name,
            "status": self.status,
            "reason": self.reason,
            "suggestion": self.suggestion,
            "detected_type": self.detected_type,
            "mapped_optype": self.mapped_optype,
        }


@dataclass
class CompatibilityReport:
    """Compatibility report for a model."""

    model_name: str
    findings: List[CompatibilityFinding] = field(default_factory=list)
    scanned_modules: int = 0
    skipped_modules: int = 0
    supported_types: Sequence[str] = field(default_factory=get_supported_layer_types)

    @property
    def supported(self) -> List[CompatibilityFinding]:
        return [f for f in self.findings if f.status == "supported"]

    @property
    def warnings(self) -> List[CompatibilityFinding]:
        return [f for f in self.findings if f.status == "warning"]

    @property
    def unsupported(self) -> List[CompatibilityFinding]:
        return [f for f in self.findings if f.status == "unsupported"]

    @property
    def compatible(self) -> bool:
        return len(self.unsupported) == 0

    @property
    def exit_code(self) -> int:
        return 0 if self.compatible else 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "compatible": self.compatible,
            "exit_code": self.exit_code,
            "scanned_modules": self.scanned_modules,
            "skipped_modules": self.skipped_modules,
            "supported_count": len(self.supported),
            "warning_count": len(self.warnings),
            "unsupported_count": len(self.unsupported),
            "supported_types": list(self.supported_types),
            "findings": [f.to_dict() for f in self.findings],
        }


_WARNING_RULES = {rule["class_name"]: rule for rule in get_warning_rules()}
_REPLACEMENT_SUGGESTIONS = get_replacement_suggestions()
_COMPOSITE_LAYER_TYPES = get_composite_layer_types()
_OPTYPE_MAPPING = get_optype_mapping()


def _is_rmsnorm_module(module: nn.Module) -> bool:
    """Match extractor RMSNorm behavior (nn.RMSNorm or class-name fallback)."""
    rmsnorm_cls = getattr(nn, "RMSNorm", None)
    if rmsnorm_cls is not None and isinstance(module, rmsnorm_cls):
        return True
    return module.__class__.__name__ == "RMSNorm"


def detect_layer_type(module: nn.Module) -> Optional[str]:
    """
    Detect extractor-supported layer type for a module.

    Mirrors tools/pytorch_extractor.py dispatch order and naming.
    """
    # Brevitas quantized primitives.
    if isinstance(module, qnn.QuantIdentity):
        return "QuantIdentity"
    if isinstance(module, qnn.QuantConv2d):
        return "QuantConv2d"
    if isinstance(module, qnn.QuantReLU):
        return "QuantReLU"

    # Standard PyTorch layers explicitly handled by the extractor.
    if isinstance(module, nn.MaxPool2d):
        return "MaxPool2d"
    if isinstance(module, nn.AvgPool2d):
        return "AvgPool2d"
    if isinstance(module, nn.AdaptiveAvgPool2d):
        return "GlobalAvgPool"
    if isinstance(module, nn.AdaptiveAvgPool1d):
        return "AdaptiveAvgPool1d"
    if isinstance(module, nn.ZeroPad2d):
        return "ZeroPad2d"
    if module.__class__.__name__ == "Squeeze":
        return "Squeeze"
    if isinstance(module, nn.LayerNorm):
        return "LayerNorm"
    if _is_rmsnorm_module(module):
        return "RMSNorm"
    if isinstance(module, nn.GroupNorm):
        return "GroupNorm"
    if isinstance(module, nn.Embedding):
        return "Embedding"
    if isinstance(module, nn.GELU):
        return "GELU"
    if isinstance(module, nn.Flatten):
        return "Flatten"

    # Class-name detectors for lightweight helper modules.
    if module.__class__.__name__ == "Permute":
        return "Permute"
    if module.__class__.__name__ == "Reshape":
        return "Reshape"
    if module.__class__.__name__ in ("RFFT", "RFFTFeatures", "RFFTFeature"):
        return "RFFT"

    # Brevitas quantized linear.
    if isinstance(module, qnn.QuantLinear):
        return "QuantLinear"

    # Custom elementwise modules.
    if pe._is_quant_add(module):
        return "Add"
    if pe._is_quant_concatenate(module):
        return "Concatenate"
    if pe._is_quant_mean(module):
        return "Mean"

    # Custom attention/composite blocks.
    if pe._is_quant_self_attention(module) or pe._is_quant_multihead_attention(module) or pe._is_quant_rope_self_attention(module):
        return "MultiheadSelfAttention"
    if pe._is_quant_cross_attention(module):
        return "CrossAttention"
    if pe._is_quant_alternating_attention(module):
        return "AlternatingAttention"

    # Sequence/Mamba family custom blocks.
    if pe._is_quant_conv1d_depthwise(module):
        return "Conv1dDepthwise"
    if pe._is_quant_silu(module):
        return "SiLU"
    if pe._is_quant_ssm(module):
        return "SSM"
    if pe._is_quant_mamba_wrapper(module):
        return "MambaWrapper"
    if pe._is_quant_patch_embed(module):
        return "PatchEmbed"
    if pe._is_quant_mamba_block(module):
        return "MambaBlock"
    return None


def is_composite_layer(layer_type: Optional[str]) -> bool:
    """Return True if extractor treats this layer as a composite block."""
    return layer_type in _COMPOSITE_LAYER_TYPES


def should_skip_module(name: str, skip_prefixes: Set[str]) -> bool:
    """Return True when module name is in a skipped sub-tree."""
    for prefix in skip_prefixes:
        if name.startswith(prefix):
            return True
    return False


def _to_tuple2(value: Any) -> Sequence[int]:
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, tuple):
        return value
    return tuple(value)


def _conv_dilation_supported(module: nn.Module) -> bool:
    dilation = _to_tuple2(getattr(module, "dilation", (1, 1)))
    return dilation == (1, 1)


def _is_container(module: nn.Module) -> bool:
    return isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict))


def _is_leaf(module: nn.Module) -> bool:
    return len(list(module.children())) == 0


def classify_module(name: str, module: nn.Module) -> Optional[CompatibilityFinding]:
    """
    Classify a module as supported, warning, unsupported, or ignored.

    Returns None for containers/structural modules that should not be reported.
    """
    class_name = module.__class__.__name__
    layer_type = detect_layer_type(module)

    if layer_type is not None:
        if layer_type == "QuantConv2d" and not _conv_dilation_supported(module):
            return CompatibilityFinding(
                module_name=name,
                class_name=class_name,
                status="unsupported",
                reason=f"Dilated convolution not supported for extraction (dilation={getattr(module, 'dilation', None)}).",
                suggestion="Use dilation=1 convolution or restructure the model with supported operations.",
                detected_type=layer_type,
                mapped_optype=_OPTYPE_MAPPING.get(layer_type),
            )

        return CompatibilityFinding(
            module_name=name,
            class_name=class_name,
            status="supported",
            reason="Extractor has explicit support for this module type.",
            detected_type=layer_type,
            mapped_optype=_OPTYPE_MAPPING.get(layer_type),
        )

    # Explicit warnings
    if class_name in _WARNING_RULES:
        rule = _WARNING_RULES[class_name]
        return CompatibilityFinding(
            module_name=name,
            class_name=class_name,
            status="warning",
            reason=rule["reason"],
            suggestion=rule["suggestion"],
        )

    if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
        return CompatibilityFinding(
            module_name=name,
            class_name=class_name,
            status="warning",
            reason="Dropout is inactive in eval mode and not represented as a runtime op.",
            suggestion=_REPLACEMENT_SUGGESTIONS["Dropout"],
        )

    # Explicit unsupported
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        suggestion = _REPLACEMENT_SUGGESTIONS.get(class_name, _REPLACEMENT_SUGGESTIONS["BatchNorm2d"])
        return CompatibilityFinding(
            module_name=name,
            class_name=class_name,
            status="unsupported",
            reason="BatchNorm is not extractable as a standalone op in current ARES flow.",
            suggestion=suggestion,
        )

    if isinstance(module, nn.MultiheadAttention):
        return CompatibilityFinding(
            module_name=name,
            class_name=class_name,
            status="unsupported",
            reason="PyTorch nn.MultiheadAttention is not directly extractable.",
            suggestion=_REPLACEMENT_SUGGESTIONS["MultiheadAttention"],
        )

    if class_name in _REPLACEMENT_SUGGESTIONS:
        return CompatibilityFinding(
            module_name=name,
            class_name=class_name,
            status="unsupported",
            reason=f"{class_name} is not currently in extractor-supported layer set.",
            suggestion=_REPLACEMENT_SUGGESTIONS[class_name],
        )

    # Ignore structural containers and non-leaf helper modules.
    if _is_container(module) or not _is_leaf(module):
        return None

    # For unknown leaf modules, report unsupported only if they are likely compute-bearing.
    has_params = next(module.parameters(recurse=False), None) is not None
    if has_params:
        return CompatibilityFinding(
            module_name=name,
            class_name=class_name,
            status="unsupported",
            reason="Unknown parameterized leaf module. Extractor has no explicit handling for this class.",
            suggestion="Replace with a supported ARES layer/block or add extraction/runtime support.",
        )

    # Unknown parameter-free leaf: warning only (often helper shape modules).
    return CompatibilityFinding(
        module_name=name,
        class_name=class_name,
        status="warning",
        reason="Unknown parameter-free leaf module; extractor may ignore it.",
        suggestion="Use explicit supported shape/layout layers (Flatten/Squeeze/Reshape/Permute) when possible.",
    )


def scan_model_modules(model: nn.Module) -> CompatibilityReport:
    """Run compatibility scan over model.named_modules()."""
    report = CompatibilityReport(model_name=model.__class__.__name__)
    skip_prefixes: Set[str] = set()

    for name, module in model.named_modules():
        if name == "":
            continue
        if should_skip_module(name, skip_prefixes):
            report.skipped_modules += 1
            continue

        report.scanned_modules += 1
        finding = classify_module(name, module)
        if finding is None:
            continue
        report.findings.append(finding)

        # Skip internals for recognized layers with submodules to avoid false positives
        # from quantizer internals and composite implementations.
        layer_type = finding.detected_type
        if layer_type is not None and len(list(module.children())) > 0:
            skip_prefixes.add(f"{name}.")

        # Keep explicit parity for composite layer behavior from extractor.
        if is_composite_layer(layer_type):
            skip_prefixes.add(f"{name}.")

    return report


def summarize_report(report: CompatibilityReport) -> str:
    """Format a human-readable compatibility summary."""
    lines: List[str] = []
    lines.append(f"ARES Compatibility Report for {report.model_name}")
    lines.append("=" * (len(lines[-1])))
    lines.append(f"Scanned modules: {report.scanned_modules}")
    lines.append(f"Skipped modules: {report.skipped_modules}")
    lines.append(f"Supported:      {len(report.supported)}")
    lines.append(f"Warnings:       {len(report.warnings)}")
    lines.append(f"Unsupported:    {len(report.unsupported)}")

    if report.unsupported:
        lines.append("")
        lines.append("Unsupported modules:")
        for f in report.unsupported:
            lines.append(f"  - {f.module_name}: {f.class_name}")
            lines.append(f"    reason: {f.reason}")
            if f.suggestion:
                lines.append(f"    suggestion: {f.suggestion}")

    if report.warnings:
        lines.append("")
        lines.append("Warnings:")
        for f in report.warnings:
            lines.append(f"  - {f.module_name}: {f.class_name}")
            lines.append(f"    reason: {f.reason}")
            if f.suggestion:
                lines.append(f"    suggestion: {f.suggestion}")

    lines.append("")
    result = "COMPATIBLE" if report.compatible else "INCOMPATIBLE"
    lines.append(f"Result: {result}")
    return "\n".join(lines)
