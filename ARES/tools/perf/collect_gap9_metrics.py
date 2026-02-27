#!/usr/bin/env python3
# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


PASS_PATTERNS = (
    "Test PASSED",
    "TEST PASSED",
    "Network test completed successfully.",
    "Golden validation DISABLED - marking test as PASSED",
)

FAIL_PATTERNS = (
    "Test FAILED",
    "TEST FAILED",
    "[FAIL] Test FAILED",
)

FATAL_PATTERNS = (
    "Invalid access from the UDMA",
    "Accessing memory while it is down",
    "Cluster stack overflow",
    "Fatal error",
    "FATAL:",
    "Segmentation fault",
    "Bus error",
    "Abort signal",
    "double free",
    "heap corruption",
    "stack smashing",
    "out of memory",
    "malloc failed",
    "assertion failed",
)

RE_CLUSTER_CYCLES = re.compile(r"^\s*CL:\s*Total cluster cycles:\s*([0-9,]+)\s*$", re.MULTILINE)
RE_TOTAL_CYCLES = re.compile(r"^\s*Total cycles:\s*([0-9,]+)\s*$", re.MULTILINE)
RE_TOTAL_MACS = re.compile(r"^\s*-\s*Total MACs:\s*([0-9,]+)\s*$", re.MULTILINE)
RE_MAX_ERROR = re.compile(r"^\s*Max error:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*$", re.MULTILINE)
RE_MAX_DIFF = re.compile(r"max_diff\s*=\s*([0-9]+)")


@dataclass(frozen=True)
class LogParseResult:
    log_path: str
    status: str
    cycles: int | None
    cycles_source: str | None
    max_error: float | None
    max_error_source: str | None
    has_fatal: bool


@dataclass(frozen=True)
class TestMetrics:
    test: str
    status: str
    cycles: int | None
    cycles_source: str | None
    macs: int | None
    macs_source: str | None
    macs_per_cycle: float | None
    max_error: float | None
    max_error_source: str | None
    log_path: str | None
    log_files: list[str]
    notes: list[str]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _parse_int(value: str) -> int:
    return int(value.replace(",", "").strip())


def _parse_log(log_path: Path) -> LogParseResult:
    text = _read_text(log_path)
    lower = text.lower()

    has_fail = any(p.lower() in lower for p in FAIL_PATTERNS)
    has_fatal = any(p.lower() in lower for p in FATAL_PATTERNS)
    has_pass = any(p.lower() in lower for p in PASS_PATTERNS)

    if has_fail or has_fatal:
        status = "FAIL"
    elif has_pass:
        status = "PASS"
    else:
        status = "UNKNOWN"

    cluster_matches = RE_CLUSTER_CYCLES.findall(text)
    perf_matches = RE_TOTAL_CYCLES.findall(text)
    cycles: int | None = None
    cycles_source: str | None = None
    if cluster_matches:
        cycles = _parse_int(cluster_matches[-1])
        cycles_source = "cluster_cycles"
    elif perf_matches:
        cycles = _parse_int(perf_matches[-1])
        cycles_source = "perf_total_cycles"

    max_error: float | None = None
    max_error_source: str | None = None
    max_error_matches = RE_MAX_ERROR.findall(text)
    if max_error_matches:
        max_error = float(max_error_matches[-1])
        max_error_source = "max_error_line"
    else:
        max_diff_matches = RE_MAX_DIFF.findall(text)
        if max_diff_matches:
            max_error = float(max(int(v) for v in max_diff_matches))
            max_error_source = "max_diff_line"

    return LogParseResult(
        log_path=str(log_path),
        status=status,
        cycles=cycles,
        cycles_source=cycles_source,
        max_error=max_error,
        max_error_source=max_error_source,
        has_fatal=has_fatal,
    )


def _log_priority(path: Path) -> tuple[int, str]:
    name = path.name
    if name == "gvsoc_run.log":
        return (0, name)
    if name.startswith("gvsoc_run_validation"):
        return (1, name)
    if name.startswith("gvsoc_run_perf"):
        return (2, name)
    if name == "gvsoc_run_autotune.log":
        return (3, name)
    return (4, name)


def _discover_tests(outputs_dir: Path, selected_tests: list[str] | None) -> list[str]:
    if selected_tests:
        return selected_tests

    tests: list[str] = []
    if not outputs_dir.exists():
        return tests
    for child in sorted(outputs_dir.iterdir()):
        if child.is_dir() and child.name.startswith("test_"):
            tests.append(child.name)
    return tests


def _parse_total_macs(test_dir: Path) -> tuple[int | None, str | None]:
    candidates = [
        test_dir / "generated" / "optimization_report.md",
        test_dir / "optimization_report.md",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        text = _read_text(candidate)
        match = RE_TOTAL_MACS.search(text)
        if match:
            return _parse_int(match.group(1)), str(candidate)
    return None, None


def _build_test_metrics(test_dir: Path) -> TestMetrics:
    test_name = test_dir.name
    generated_dir = test_dir / "generated"
    notes: list[str] = []

    logs = sorted(generated_dir.glob("gvsoc_run*.log"), key=_log_priority) if generated_dir.exists() else []
    parsed_logs = [_parse_log(path) for path in logs]

    status = "MISSING_LOG"
    if parsed_logs:
        if any(entry.status == "FAIL" for entry in parsed_logs):
            status = "FAIL"
        elif any(entry.status == "PASS" for entry in parsed_logs):
            status = "PASS"
        else:
            status = "UNKNOWN"
    else:
        notes.append("No gvsoc_run*.log files found")

    cycles: int | None = None
    cycles_source: str | None = None
    cycles_log_path: str | None = None
    for entry in parsed_logs:
        if entry.cycles is None:
            continue
        cycles = entry.cycles
        cycles_source = entry.cycles_source
        cycles_log_path = entry.log_path
        break

    max_error: float | None = None
    max_error_source: str | None = None
    max_error_log_path: str | None = None
    max_error_candidates = [entry for entry in parsed_logs if entry.max_error is not None]
    if max_error_candidates:
        best = max(max_error_candidates, key=lambda item: float(item.max_error or -math.inf))
        max_error = best.max_error
        max_error_source = best.max_error_source
        max_error_log_path = best.log_path

    macs, macs_source = _parse_total_macs(test_dir)
    macs_per_cycle: float | None = None
    if macs is not None and cycles is not None and cycles > 0:
        macs_per_cycle = macs / cycles
    elif macs is not None and cycles in (None, 0):
        notes.append("MACs found but cycles missing or zero")

    log_path = cycles_log_path or max_error_log_path or (parsed_logs[0].log_path if parsed_logs else None)
    if any(entry.has_fatal for entry in parsed_logs):
        notes.append("Fatal error pattern found in at least one log")

    return TestMetrics(
        test=test_name,
        status=status,
        cycles=cycles,
        cycles_source=cycles_source,
        macs=macs,
        macs_source=macs_source,
        macs_per_cycle=macs_per_cycle,
        max_error=max_error,
        max_error_source=max_error_source,
        log_path=log_path,
        log_files=[str(path) for path in logs],
        notes=notes,
    )


def _write_csv(path: Path, rows: list[TestMetrics]) -> None:
    fieldnames = [
        "test",
        "status",
        "cycles",
        "cycles_source",
        "macs",
        "macs_source",
        "macs_per_cycle",
        "max_error",
        "max_error_source",
        "log_path",
        "log_files",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            data = asdict(row)
            data["log_files"] = ";".join(data["log_files"])
            data["notes"] = ";".join(data["notes"])
            writer.writerow(data)


def _print_summary(rows: list[TestMetrics]) -> None:
    pass_count = sum(1 for row in rows if row.status == "PASS")
    fail_count = sum(1 for row in rows if row.status == "FAIL")
    unknown_count = len(rows) - pass_count - fail_count
    print(f"Collected metrics for {len(rows)} tests: PASS={pass_count}, FAIL={fail_count}, OTHER={unknown_count}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collect GAP9 validation/performance metrics from tests/outputs.\n"
            "Extracts pass/fail, cycles, MACs, MACs/cycle, and max error."
        )
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("tests/outputs"),
        help="Directory containing test outputs (default: tests/outputs)",
    )
    parser.add_argument(
        "--tests",
        nargs="*",
        help="Optional explicit test names (default: discover all test_* directories)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON output path",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional CSV output path",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if any test is missing logs, cycles, or MACs",
    )
    args = parser.parse_args()

    tests = _discover_tests(args.outputs_dir, args.tests)
    if not tests:
        print(f"No test directories found under {args.outputs_dir}", file=sys.stderr)
        return 2

    rows: list[TestMetrics] = []
    for test_name in tests:
        test_dir = args.outputs_dir / test_name
        if not test_dir.exists():
            rows.append(
                TestMetrics(
                    test=test_name,
                    status="MISSING_TEST_DIR",
                    cycles=None,
                    cycles_source=None,
                    macs=None,
                    macs_source=None,
                    macs_per_cycle=None,
                    max_error=None,
                    max_error_source=None,
                    log_path=None,
                    log_files=[],
                    notes=[f"Missing test directory: {test_dir}"],
                )
            )
            continue
        rows.append(_build_test_metrics(test_dir))

    rows.sort(key=lambda item: item.test)
    _print_summary(rows)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "outputs_dir": str(args.outputs_dir),
        "tests": [asdict(row) for row in rows],
        "summary": {
            "total": len(rows),
            "pass": sum(1 for row in rows if row.status == "PASS"),
            "fail": sum(1 for row in rows if row.status == "FAIL"),
            "other": sum(1 for row in rows if row.status not in {"PASS", "FAIL"}),
        },
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote JSON metrics to {args.out}")
    else:
        print(json.dumps(payload, indent=2))

    if args.csv_out:
        _write_csv(args.csv_out, rows)
        print(f"Wrote CSV metrics to {args.csv_out}")

    if args.strict:
        for row in rows:
            if row.status in {"MISSING_TEST_DIR", "MISSING_LOG"}:
                return 1
            if row.cycles is None or row.macs is None:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
