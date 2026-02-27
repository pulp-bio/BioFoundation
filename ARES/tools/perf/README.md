# GAP9 Perf Tooling

This folder provides refactor-gate tooling for validation and performance comparisons.

## Tools

1. `collect_gap9_metrics.py`
- Scans `tests/outputs/test_*/generated/gvsoc_run*.log`.
- Extracts per-test:
  - `status` (`PASS` / `FAIL` / `UNKNOWN`)
  - `cycles`
  - `macs`
  - `macs_per_cycle`
  - `max_error`

2. `compare_gap9_metrics.py`
- Compares candidate vs baseline metrics JSON.
- Applies threshold gates:
  - per-test cycles regression
  - per-test MAC drift
  - per-test MACs/cycle drop
  - geomean MACs/cycle drop
  - max-error increase
- Returns non-zero on regression.

3. `emit_perf_report_md.py`
- Converts comparison JSON into a markdown regression report.

## One-command Entrypoint

Use `scripts/validate_refactor_phase.sh` for Tier 0/1/2 runs and perf compare.

Examples:

```bash
# Tier 0 run + metrics collection only (no baseline compare)
scripts/validate_refactor_phase.sh --tier 0

# Tier 0 run + compare against frozen baseline metrics file
scripts/validate_refactor_phase.sh \
  --tier 0 \
  --baseline-metrics tests/outputs/_refactor_validation/baseline_metrics_tier0.json

# Tier 1 using existing logs only, with compare required
scripts/validate_refactor_phase.sh \
  --tier 1 \
  --skip-run \
  --baseline-metrics tests/outputs/_refactor_validation/baseline_metrics_tier1.json \
  --require-compare
```

## Direct Usage

```bash
# 1) Collect
python3 tools/perf/collect_gap9_metrics.py \
  --outputs-dir tests/outputs \
  --out tests/outputs/_refactor_validation/candidate_metrics.json

# 2) Compare
python3 tools/perf/compare_gap9_metrics.py \
  --baseline tests/outputs/_refactor_validation/baseline_metrics.json \
  --candidate tests/outputs/_refactor_validation/candidate_metrics.json \
  --out tests/outputs/_refactor_validation/compare.json

# 3) Emit markdown report
python3 tools/perf/emit_perf_report_md.py \
  --compare-json tests/outputs/_refactor_validation/compare.json \
  --out tests/outputs/_refactor_validation/report.md
```

## Output JSON Shape

Collector output (`collect_gap9_metrics.py`):

```json
{
  "generated_at": "...",
  "outputs_dir": "tests/outputs",
  "tests": [
    {
      "test": "test_1_simplecnn",
      "status": "PASS",
      "cycles": 904014,
      "macs": 2063488,
      "macs_per_cycle": 2.282577,
      "max_error": 0.0
    }
  ],
  "summary": {
    "total": 1,
    "pass": 1,
    "fail": 0,
    "other": 0
  }
}
```

Comparer output (`compare_gap9_metrics.py`) includes:

- `thresholds`
- `summary` (`gate_passed`, geomean metrics, regression counts)
- `results[]` with per-test baseline/candidate values, deltas, and regression reasons.
