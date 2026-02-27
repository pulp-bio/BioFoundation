# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Validate coverage of docs/SUPPORTED_OPERATIONS.md against registry metadata."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ares_support_registry import get_optype_mapping, get_supported_layer_types


def validate_doc_covers_all_types(doc_path: Path) -> List[str]:
    """Return extractor layer types missing from the document."""
    text = doc_path.read_text()
    missing: List[str] = []
    for layer_type in get_supported_layer_types():
        if re.search(rf"\b{re.escape(layer_type)}\b", text) is None:
            missing.append(layer_type)
    return missing


def validate_optype_links(doc_path: Path) -> List[str]:
    """Return OpType symbols expected from registry but missing in document."""
    text = doc_path.read_text()
    missing: List[str] = []
    for _, mapping_value in get_optype_mapping().items():
        for symbol in re.findall(r"\bOP_[A-Z0-9_]+\b", mapping_value):
            if re.search(rf"\b{re.escape(symbol)}\b", text) is None:
                missing.append(symbol)
    return sorted(set(missing))


def run_validation(doc_path: Path) -> Tuple[List[str], List[str]]:
    return validate_doc_covers_all_types(doc_path), validate_optype_links(doc_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate supported-operations documentation coverage.")
    parser.add_argument(
        "--doc",
        default=str(REPO_ROOT / "docs/SUPPORTED_OPERATIONS.md"),
        help="Path to supported operations markdown document.",
    )
    args = parser.parse_args()

    doc_path = Path(args.doc)
    if not doc_path.exists():
        print(f"ERROR: document not found: {doc_path}")
        return 1

    missing_types, missing_optypes = run_validation(doc_path)

    if not missing_types and not missing_optypes:
        print(f"Validation OK: {doc_path}")
        return 0

    print(f"Validation FAILED: {doc_path}")
    if missing_types:
        print("Missing extractor types:")
        for name in missing_types:
            print(f"  - {name}")
    if missing_optypes:
        print("Missing OpType symbols:")
        for name in missing_optypes:
            print(f"  - {name}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
