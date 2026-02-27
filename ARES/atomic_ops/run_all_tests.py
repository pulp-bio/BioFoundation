#!/usr/bin/env python3
# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Run all atomic operation unit tests.

Usage:
    python atomic_ops/run_all_tests.py
    python atomic_ops/run_all_tests.py --verbose
    python atomic_ops/run_all_tests.py conv2d linear
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import pkgutil
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterable, List, Sequence, Tuple


def _ensure_repo_root_on_syspath() -> None:
    """
    Allow running as `python atomic_ops/run_all_tests.py` by adding the repo root
    to sys.path (so `import atomic_ops` resolves as a package).
    """
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def _discover_modules() -> List[str]:
    import atomic_ops  # noqa: PLC0415

    modules: List[str] = []
    for module_info in pkgutil.iter_modules(atomic_ops.__path__):
        name = module_info.name
        if name in {"__init__", "constants", "run_all_tests"}:
            continue
        modules.append(name)
    return sorted(modules)


def _normalize_requested_modules(requested: Sequence[str], available: Sequence[str]) -> List[str]:
    if not requested:
        return list(available)

    normalized: List[str] = []
    available_set = set(available)
    for raw in requested:
        name = raw
        if name.startswith("atomic_ops."):
            name = name[len("atomic_ops."):]
        if name not in available_set:
            raise SystemExit(f"Unknown op/module '{raw}'. Available: {' '.join(available)}")
        normalized.append(name)
    return normalized


def _iter_test_functions(module: ModuleType) -> Iterable[Tuple[str, Callable[[], object]]]:
    for name, obj in vars(module).items():
        if not callable(obj) or not name.startswith("test_"):
            continue

        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            continue

        # Only run tests that can be called with no required positional params.
        required = [
            p
            for p in sig.parameters.values()
            if p.default is inspect._empty
            and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if required:
            continue

        yield name, obj  # type: ignore[misc]


def run_tests(modules: Sequence[str], verbose: bool) -> int:
    total = 0
    failed = 0

    for mod_name in modules:
        full_name = f"atomic_ops.{mod_name}"
        try:
            module = importlib.import_module(full_name)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] import {full_name}: {exc}")
            if verbose:
                traceback.print_exc()
            continue

        tests = list(_iter_test_functions(module))
        if not tests:
            if verbose:
                print(f"[SKIP] {full_name}: no test_*() functions found")
            continue

        for test_name, test_fn in tests:
            total += 1
            label = f"{mod_name}.{test_name}"
            try:
                test_fn()
                if verbose:
                    print(f"[PASS] {label}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"[FAIL] {label}: {exc}")
                if verbose:
                    traceback.print_exc()

    passed = total - failed
    print(f"\nAtomic ops: {passed}/{total} tests passed")
    return 0 if failed == 0 else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run atomic_ops unit tests.")
    parser.add_argument(
        "ops",
        nargs="*",
        help="Optional subset of atomic op modules (e.g. conv2d linear). Defaults to all.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-test PASS lines and tracebacks.")
    args = parser.parse_args(argv)

    _ensure_repo_root_on_syspath()
    available = _discover_modules()
    selected = _normalize_requested_modules(args.ops, available)
    return run_tests(selected, verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
