#!/usr/bin/env python3
# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_HOTSPOT_TESTS = [
    "test_6_multitilecnn",
    "test_9_padding",
    "test_13_transformer_simple",
    "test_14_multiblock_transformer",
    "test_15_tinymyo_tiny",
    "test_23_femba_full_input",
    "test_26_tinymyo_8ch_400tok",
    "test_27_linear3d_bench",
    "test_29_luna_base",
    "test_31_luna_full",
    "test_36_drowsiness_fusion",
]


@dataclass(frozen=True)
class Thresholds:
    max_cycle_regression_pct: float
    hotspot_cycle_regression_pct: float
    max_macs_drift_pct: float
    max_macs_per_cycle_drop_pct: float
    max_geomean_macs_per_cycle_drop_pct: float
    max_error_increase_abs: float
    fail_on_missing_candidate: bool
    require_baseline_pass: bool
    require_candidate_metrics: bool


def _load_metrics(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "tests" in data and isinstance(data["tests"], list):
        rows = data["tests"]
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError(f"Unsupported metrics JSON format in {path}")

    by_test: dict[str, dict] = {}
    for row in rows:
        test_name = row.get("test")
        if not test_name:
            continue
        by_test[str(test_name)] = row
    return by_test


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _compare_single(
    test_name: str,
    base_row: dict | None,
    cand_row: dict | None,
    thresholds: Thresholds,
    hotspot_tests: set[str],
) -> tuple[dict, list[float]]:
    regressions: list[str] = []
    warnings: list[str] = []
    ratios_for_geomean: list[float] = []

    base_status = (base_row or {}).get("status")
    cand_status = (cand_row or {}).get("status")
    base_cycles = _to_int((base_row or {}).get("cycles"))
    cand_cycles = _to_int((cand_row or {}).get("cycles"))
    base_macs = _to_int((base_row or {}).get("macs"))
    cand_macs = _to_int((cand_row or {}).get("macs"))
    base_mpc = _to_float((base_row or {}).get("macs_per_cycle"))
    cand_mpc = _to_float((cand_row or {}).get("macs_per_cycle"))
    base_err = _to_float((base_row or {}).get("max_error"))
    cand_err = _to_float((cand_row or {}).get("max_error"))

    if cand_row is None:
        if thresholds.fail_on_missing_candidate:
            regressions.append("Missing candidate metrics")
        else:
            warnings.append("Missing candidate metrics")
    if base_row is None:
        warnings.append("Missing baseline metrics")

    if thresholds.require_baseline_pass and base_row is not None and base_status == "PASS":
        if cand_status != "PASS":
            regressions.append(f"Status regression: baseline=PASS, candidate={cand_status}")

    cycle_regression_pct: float | None = None
    cycle_threshold = thresholds.hotspot_cycle_regression_pct if test_name in hotspot_tests else thresholds.max_cycle_regression_pct
    if base_cycles is not None and base_cycles > 0 and cand_cycles is not None:
        cycle_regression_pct = 100.0 * (cand_cycles - base_cycles) / base_cycles
        if cycle_regression_pct > cycle_threshold:
            regressions.append(f"Cycle regression {cycle_regression_pct:.2f}% > {cycle_threshold:.2f}%")
    elif thresholds.require_candidate_metrics and base_cycles is not None and cand_cycles is None:
        regressions.append("Candidate cycles missing")

    macs_drift_pct: float | None = None
    if base_macs is not None and base_macs != 0 and cand_macs is not None:
        macs_drift_pct = 100.0 * (cand_macs - base_macs) / base_macs
        if abs(macs_drift_pct) > thresholds.max_macs_drift_pct:
            regressions.append(
                f"MAC drift {macs_drift_pct:.6f}% exceeds +/-{thresholds.max_macs_drift_pct:.6f}%"
            )
    elif thresholds.require_candidate_metrics and base_macs is not None and cand_macs is None:
        regressions.append("Candidate MACs missing")

    mpc_drop_pct: float | None = None
    if base_mpc is not None and base_mpc > 0 and cand_mpc is not None:
        ratio = cand_mpc / base_mpc
        geomean_ratio = ratio if ratio > 0 else 1e-12
        ratios_for_geomean.append(geomean_ratio)
        mpc_drop_pct = 100.0 * (1.0 - ratio)
        if mpc_drop_pct > thresholds.max_macs_per_cycle_drop_pct:
            regressions.append(f"MACs/cycle drop {mpc_drop_pct:.2f}% > {thresholds.max_macs_per_cycle_drop_pct:.2f}%")
    elif thresholds.require_candidate_metrics and base_mpc is not None and cand_mpc is None:
        regressions.append("Candidate MACs/cycle missing")

    max_error_delta: float | None = None
    if base_err is not None and cand_err is not None:
        max_error_delta = cand_err - base_err
        if max_error_delta > thresholds.max_error_increase_abs:
            regressions.append(
                f"Max error increase {max_error_delta:.6f} > {thresholds.max_error_increase_abs:.6f}"
            )
    elif thresholds.require_candidate_metrics and base_err is not None and cand_err is None:
        warnings.append("Candidate max_error missing")

    result = {
        "test": test_name,
        "regression": bool(regressions),
        "reasons": regressions,
        "warnings": warnings,
        "baseline": {
            "status": base_status,
            "cycles": base_cycles,
            "macs": base_macs,
            "macs_per_cycle": base_mpc,
            "max_error": base_err,
        },
        "candidate": {
            "status": cand_status,
            "cycles": cand_cycles,
            "macs": cand_macs,
            "macs_per_cycle": cand_mpc,
            "max_error": cand_err,
        },
        "deltas": {
            "cycle_regression_pct": cycle_regression_pct,
            "macs_drift_pct": macs_drift_pct,
            "macs_per_cycle_drop_pct": mpc_drop_pct,
            "max_error_delta": max_error_delta,
            "is_hotspot": test_name in hotspot_tests,
        },
    }
    return result, ratios_for_geomean


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare GAP9 metrics (candidate vs baseline) with thresholds.")
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline metrics JSON")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate metrics JSON")
    parser.add_argument("--out", type=Path, default=None, help="Optional output JSON path")
    parser.add_argument(
        "--max-cycle-regression-pct",
        type=float,
        default=5.0,
        help="Max allowed cycle regression percentage per test (default: 5.0)",
    )
    parser.add_argument(
        "--hotspot-cycle-regression-pct",
        type=float,
        default=3.0,
        help="Stricter max cycle regression percentage for hotspot tests (default: 3.0)",
    )
    parser.add_argument(
        "--max-macs-drift-pct",
        type=float,
        default=0.0,
        help="Max allowed absolute MAC drift percentage (default: 0.0)",
    )
    parser.add_argument(
        "--max-macs-per-cycle-drop-pct",
        type=float,
        default=5.0,
        help="Max allowed MACs/cycle drop percentage per test (default: 5.0)",
    )
    parser.add_argument(
        "--max-geomean-macs-per-cycle-drop-pct",
        type=float,
        default=3.0,
        help="Max allowed geometric-mean MACs/cycle drop percentage (default: 3.0)",
    )
    parser.add_argument(
        "--max-error-increase-abs",
        type=float,
        default=0.05,
        help="Max allowed absolute max-error increase (default: 0.05)",
    )
    parser.add_argument(
        "--hotspot-tests",
        nargs="*",
        default=None,
        help="Override hotspot test list. Default uses refactor plan list.",
    )
    parser.add_argument(
        "--allow-missing-candidate",
        action="store_true",
        help="Do not treat missing candidate test rows as regression",
    )
    parser.add_argument(
        "--ignore-baseline-status",
        action="store_true",
        help="Do not enforce baseline PASS -> candidate PASS status gate",
    )
    parser.add_argument(
        "--allow-missing-metrics",
        action="store_true",
        help="Do not fail missing candidate cycles/macs/mpc for baseline-known tests",
    )
    args = parser.parse_args()

    thresholds = Thresholds(
        max_cycle_regression_pct=args.max_cycle_regression_pct,
        hotspot_cycle_regression_pct=args.hotspot_cycle_regression_pct,
        max_macs_drift_pct=args.max_macs_drift_pct,
        max_macs_per_cycle_drop_pct=args.max_macs_per_cycle_drop_pct,
        max_geomean_macs_per_cycle_drop_pct=args.max_geomean_macs_per_cycle_drop_pct,
        max_error_increase_abs=args.max_error_increase_abs,
        fail_on_missing_candidate=not args.allow_missing_candidate,
        require_baseline_pass=not args.ignore_baseline_status,
        require_candidate_metrics=not args.allow_missing_metrics,
    )

    baseline_by_test = _load_metrics(args.baseline)
    candidate_by_test = _load_metrics(args.candidate)
    hotspot_tests = set(args.hotspot_tests) if args.hotspot_tests is not None else set(DEFAULT_HOTSPOT_TESTS)

    all_tests = sorted(set(baseline_by_test.keys()) | set(candidate_by_test.keys()))
    results: list[dict] = []
    geomean_ratios: list[float] = []
    for test_name in all_tests:
        result, ratio_values = _compare_single(
            test_name=test_name,
            base_row=baseline_by_test.get(test_name),
            cand_row=candidate_by_test.get(test_name),
            thresholds=thresholds,
            hotspot_tests=hotspot_tests,
        )
        results.append(result)
        geomean_ratios.extend(ratio_values)

    geomean_ratio: float | None = None
    geomean_drop_pct: float | None = None
    geomean_regression = False
    if geomean_ratios:
        geomean_ratio = math.exp(sum(math.log(value) for value in geomean_ratios) / len(geomean_ratios))
        geomean_drop_pct = 100.0 * (1.0 - geomean_ratio)
        geomean_regression = geomean_drop_pct > thresholds.max_geomean_macs_per_cycle_drop_pct

    regression_count = sum(1 for item in results if item["regression"])
    if geomean_regression:
        regression_count += 1

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_path": str(args.baseline),
        "candidate_path": str(args.candidate),
        "thresholds": {
            "max_cycle_regression_pct": thresholds.max_cycle_regression_pct,
            "hotspot_cycle_regression_pct": thresholds.hotspot_cycle_regression_pct,
            "max_macs_drift_pct": thresholds.max_macs_drift_pct,
            "max_macs_per_cycle_drop_pct": thresholds.max_macs_per_cycle_drop_pct,
            "max_geomean_macs_per_cycle_drop_pct": thresholds.max_geomean_macs_per_cycle_drop_pct,
            "max_error_increase_abs": thresholds.max_error_increase_abs,
            "fail_on_missing_candidate": thresholds.fail_on_missing_candidate,
            "require_baseline_pass": thresholds.require_baseline_pass,
            "require_candidate_metrics": thresholds.require_candidate_metrics,
            "hotspot_tests": sorted(hotspot_tests),
        },
        "summary": {
            "total_tests": len(all_tests),
            "tests_with_regressions": sum(1 for item in results if item["regression"]),
            "geomean_macs_per_cycle_ratio": geomean_ratio,
            "geomean_macs_per_cycle_drop_pct": geomean_drop_pct,
            "geomean_regression": geomean_regression,
            "gate_passed": regression_count == 0,
        },
        "results": results,
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote comparison JSON to {args.out}")
    else:
        print(json.dumps(payload, indent=2))

    print(
        f"Compared {len(all_tests)} tests: regressions={sum(1 for item in results if item['regression'])}, "
        f"geomean_regression={'yes' if geomean_regression else 'no'}"
    )
    if geomean_ratio is not None:
        print(f"Geomean MACs/cycle ratio: {geomean_ratio:.6f}")

    if regression_count != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
