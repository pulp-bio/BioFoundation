#!/usr/bin/env python3
# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _fmt_int(value: object) -> str:
    if value is None:
        return "-"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "-"


def _fmt_float(value: object, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_pct(value: object, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return "-"


def _build_report(data: dict, title: str) -> str:
    summary = data.get("summary", {})
    thresholds = data.get("thresholds", {})
    results = data.get("results", [])
    regressions = [item for item in results if item.get("regression")]

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Baseline: `{data.get('baseline_path', '-')}`")
    lines.append(f"- Candidate: `{data.get('candidate_path', '-')}`")
    lines.append(f"- Gate result: **{'PASS' if summary.get('gate_passed') else 'FAIL'}**")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total tests compared: {summary.get('total_tests', 0)}")
    lines.append(f"- Tests with regressions: {summary.get('tests_with_regressions', 0)}")
    lines.append(
        f"- Geomean MACs/cycle ratio: {_fmt_float(summary.get('geomean_macs_per_cycle_ratio'), digits=6)} "
        f"(drop {_fmt_pct(summary.get('geomean_macs_per_cycle_drop_pct'))})"
    )
    lines.append(f"- Geomean regression: {'yes' if summary.get('geomean_regression') else 'no'}")
    lines.append("")

    lines.append("## Thresholds")
    lines.append("")
    lines.append(f"- Max cycle regression: {thresholds.get('max_cycle_regression_pct', '-')}%")
    lines.append(f"- Hotspot cycle regression: {thresholds.get('hotspot_cycle_regression_pct', '-')}%")
    lines.append(f"- Max MAC drift: +/-{thresholds.get('max_macs_drift_pct', '-')}%")
    lines.append(f"- Max MACs/cycle drop: {thresholds.get('max_macs_per_cycle_drop_pct', '-')}%")
    lines.append(f"- Max geomean MACs/cycle drop: {thresholds.get('max_geomean_macs_per_cycle_drop_pct', '-')}%")
    lines.append(f"- Max error increase (abs): {thresholds.get('max_error_increase_abs', '-')}")
    lines.append("")

    lines.append("## Regressions")
    lines.append("")
    if not regressions:
        lines.append("No threshold violations detected.")
        lines.append("")
    else:
        lines.append("| Test | Reasons | Cycle Δ | MAC/cycle Δ | Max error Δ |")
        lines.append("|------|---------|---------|-------------|-------------|")
        for item in regressions:
            deltas = item.get("deltas", {})
            reasons = "; ".join(item.get("reasons", [])) or "-"
            lines.append(
                "| {test} | {reasons} | {cycle_delta} | {mpc_delta} | {err_delta} |".format(
                    test=item.get("test", "-"),
                    reasons=reasons.replace("|", "\\|"),
                    cycle_delta=_fmt_pct(deltas.get("cycle_regression_pct")),
                    mpc_delta=_fmt_pct(deltas.get("macs_per_cycle_drop_pct")),
                    err_delta=_fmt_float(deltas.get("max_error_delta"), digits=6),
                )
            )
        lines.append("")

    lines.append("## Test Matrix")
    lines.append("")
    lines.append("| Test | Baseline Status | Candidate Status | Baseline Cycles | Candidate Cycles | Baseline MAC/cycle | Candidate MAC/cycle | Baseline Max Error | Candidate Max Error |")
    lines.append("|------|-----------------|------------------|-----------------|------------------|--------------------|---------------------|--------------------|---------------------|")
    for item in sorted(results, key=lambda row: row.get("test", "")):
        base = item.get("baseline", {})
        cand = item.get("candidate", {})
        lines.append(
            "| {test} | {bs} | {cs} | {bc} | {cc} | {bmpc} | {cmpc} | {be} | {ce} |".format(
                test=item.get("test", "-"),
                bs=base.get("status", "-"),
                cs=cand.get("status", "-"),
                bc=_fmt_int(base.get("cycles")),
                cc=_fmt_int(cand.get("cycles")),
                bmpc=_fmt_float(base.get("macs_per_cycle")),
                cmpc=_fmt_float(cand.get("macs_per_cycle")),
                be=_fmt_float(base.get("max_error"), digits=6),
                ce=_fmt_float(cand.get("max_error"), digits=6),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit markdown report from compare_gap9_metrics JSON output.")
    parser.add_argument("--compare-json", type=Path, required=True, help="Path to comparison JSON")
    parser.add_argument("--out", type=Path, required=True, help="Markdown output path")
    parser.add_argument(
        "--title",
        default="GAP9 Refactor Regression Report",
        help="Report title (default: GAP9 Refactor Regression Report)",
    )
    args = parser.parse_args()

    data = json.loads(args.compare_json.read_text(encoding="utf-8"))
    report = _build_report(data, title=args.title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report + "\n", encoding="utf-8")
    print(f"Wrote markdown report to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
