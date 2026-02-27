# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Check whether a PyTorch/Brevitas model is extractor-compatible in ARES."""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Type

import torch.nn as nn

# Ensure repo root is importable when running as `python tools/...`.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ares_support_registry import get_supported_layer_types
from tools.model_compatibility_core import scan_model_modules, summarize_report


def _parse_init_kwargs(values: Optional[list[str]]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if not values:
        return kwargs
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --init-kwarg '{item}'. Expected KEY=VALUE.")
        key, value = item.split("=", 1)
        kwargs[key] = ast.literal_eval(value)
    return kwargs


def _load_module_from_path(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_model_class(module, model_class: Optional[str]) -> Type[nn.Module]:
    if model_class:
        cls = getattr(module, model_class, None)
        if cls is None or not inspect.isclass(cls) or not issubclass(cls, nn.Module):
            raise ValueError(f"Class '{model_class}' not found or not an nn.Module in {module.__name__}.")
        return cls

    candidates = []
    for _, obj in vars(module).items():
        if inspect.isclass(obj) and issubclass(obj, nn.Module) and obj.__module__ == module.__name__:
            candidates.append(obj)

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise ValueError(
            f"No nn.Module subclass found in {module.__name__}. "
            "Use --model-class to select a class explicitly."
        )

    names = ", ".join(c.__name__ for c in candidates)
    raise ValueError(
        f"Multiple nn.Module classes found in {module.__name__}: {names}. "
        "Use --model-class to choose one."
    )


def _instantiate_model(module, model_class: Optional[str], init_kwargs: Dict[str, Any]) -> nn.Module:
    """
    Instantiate model from module using ARES checker conventions.

    Convention:
    - If `--model-class` is omitted and the module provides `create_model()`,
      that factory is used (when no `--init-kwarg` is passed).
    - Otherwise, instantiate the selected/discovered nn.Module class.
    """
    if model_class is None and hasattr(module, "create_model") and callable(module.create_model) and not init_kwargs:
        model = module.create_model()
        if not isinstance(model, nn.Module):
            raise ValueError(f"{module.__name__}.create_model() did not return nn.Module.")
        return model
    cls = _find_model_class(module, model_class)
    return cls(**init_kwargs)


def load_model_from_pyfile(path: str, model_class: Optional[str], init_kwargs: Dict[str, Any]) -> nn.Module:
    """
    Load and instantiate a model from an arbitrary Python file path.

    Args:
        path: Path to `.py` file containing model class and/or `create_model()`.
        model_class: Optional class name (subclass of nn.Module).
        init_kwargs: Constructor kwargs parsed from `--init-kwarg`.
    """
    module = _load_module_from_path(Path(path).resolve())
    return _instantiate_model(module, model_class, init_kwargs)


def load_model_from_test_network(test_network: str, model_class: Optional[str], init_kwargs: Dict[str, Any]) -> nn.Module:
    """
    Load and instantiate a model from `tests/test_networks`.

    Accepts short names like `test_1_simplecnn` and resolves both
    `tests.test_networks.*` and `test_networks.*` import styles.
    """
    module_name = test_network
    if module_name.endswith(".py"):
        module_name = module_name[:-3]
    if not module_name.startswith("test_"):
        module_name = f"test_{module_name}"
    if module_name.startswith("tests.test_networks.") or module_name.startswith("test_networks."):
        module = importlib.import_module(module_name)
    else:
        try:
            module = importlib.import_module(f"tests.test_networks.{module_name}")
        except ModuleNotFoundError:
            module = importlib.import_module(f"test_networks.{module_name}")
    return _instantiate_model(module, model_class, init_kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate model compatibility with current ARES extractor support."
    )
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--model-file", type=str, help="Path to Python file containing an nn.Module.")
    source.add_argument("--test-network", type=str, help="Test-network module name (e.g. test_1_simplecnn).")
    parser.add_argument("--model-class", type=str, default=None, help="Model class name in the module.")
    parser.add_argument(
        "--init-kwarg",
        action="append",
        default=[],
        help="Model constructor argument as KEY=VALUE (VALUE parsed with Python literal_eval).",
    )
    parser.add_argument("--json-out", type=str, default=None, help="Optional path to write machine-readable JSON report.")
    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="Treat warnings as incompatible (exit code 1).",
    )
    parser.add_argument(
        "--list-supported-types",
        action="store_true",
        help="Print supported extractor layer type names and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list_supported_types:
        print("Supported extractor layer types:")
        for name in get_supported_layer_types():
            print(f"  - {name}")
        return 0

    if not args.model_file and not args.test_network:
        print("ERROR: provide either --model-file or --test-network.")
        return 2

    init_kwargs = _parse_init_kwargs(args.init_kwarg)

    if args.model_file:
        model = load_model_from_pyfile(args.model_file, args.model_class, init_kwargs)
    else:
        model = load_model_from_test_network(args.test_network, args.model_class, init_kwargs)

    model.eval()
    report = scan_model_modules(model)
    print(summarize_report(report))

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report.to_dict(), indent=2))
        print(f"\nWrote JSON report to: {out_path}")

    if args.strict_warnings and report.warnings:
        return 1
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
