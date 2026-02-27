#!/usr/bin/env python3
# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Merge one or more `profiling/gvsoc_*` CSV outputs into a single directory.\n"
            "Later inputs override earlier ones on key collisions."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Input profiling run directories (e.g. profiling/gvsoc_2025.../)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for merged CSVs",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tests_by_name: dict[str, dict[str, str]] = {}
    layers_by_key: dict[tuple[str, str], dict[str, str]] = {}
    ssm_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    subops_by_key: dict[tuple[str, str], dict[str, str]] = {}

    sources: list[str] = []

    for run_dir in args.inputs:
        run_dir = run_dir.resolve()
        sources.append(str(run_dir))

        for row in _read_csv(run_dir / "tests.csv"):
            name = row.get("test", "").strip()
            if not name:
                continue
            tests_by_name[name] = row

        for row in _read_csv(run_dir / "layers.csv"):
            test = row.get("test", "").strip()
            layer = row.get("layer", "").strip()
            if not test or not layer:
                continue
            layers_by_key[(test, layer)] = row

        for row in _read_csv(run_dir / "ssm_phases.csv"):
            test = row.get("test", "").strip()
            ordinal = row.get("ordinal", "").strip()
            phase = row.get("phase", "").strip()
            if not test or not ordinal or not phase:
                continue
            ssm_by_key[(test, ordinal, phase)] = row

        for row in _read_csv(run_dir / "subops.csv"):
            test = row.get("test", "").strip()
            item = row.get("item", "").strip()
            if not test or not item:
                continue
            subops_by_key[(test, item)] = row

    merged_tests = [tests_by_name[k] for k in sorted(tests_by_name.keys())]
    merged_layers = [layers_by_key[k] for k in sorted(layers_by_key.keys())]
    merged_ssm = [ssm_by_key[k] for k in sorted(ssm_by_key.keys())]
    merged_subops = [subops_by_key[k] for k in sorted(subops_by_key.keys())]

    _write_csv(out_dir / "tests.csv", merged_tests)
    _write_csv(out_dir / "layers.csv", merged_layers)
    _write_csv(out_dir / "ssm_phases.csv", merged_ssm)
    _write_csv(out_dir / "subops.csv", merged_subops)

    (out_dir / "sources.txt").write_text("\n".join(sources) + "\n", encoding="utf-8")

    print("Wrote:")
    if (out_dir / "tests.csv").exists():
        print(f"  - {out_dir / 'tests.csv'}")
    if (out_dir / "layers.csv").exists():
        print(f"  - {out_dir / 'layers.csv'}")
    if (out_dir / "ssm_phases.csv").exists():
        print(f"  - {out_dir / 'ssm_phases.csv'}")
    if (out_dir / "subops.csv").exists():
        print(f"  - {out_dir / 'subops.csv'}")
    print(f"  - {out_dir / 'sources.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

