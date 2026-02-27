#!/usr/bin/env python3
# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Automate GAP9 (gvsoc) builds/runs for generated test projects.
Provides both CLI usage and a reusable run_gap9_projects() helper.
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


# Fatal patterns that indicate a crash/memory error even if "Test PASSED" appears.
# These catch "pass then crash" scenarios where validation succeeds but cleanup fails.
FATAL_PATTERNS = [
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
]


DEFAULT_TESTS = [
    # CNN tests (1-10)
    "test_1_simplecnn",
    "test_2_tinycnn",
    "test_3_mlp",
    "test_4_resnet_basic",
    "test_5_densenet_basic",
    "test_6_multitilecnn",
    "test_7_bottleneck",
    "test_8_stride2",
    "test_9_padding",
    "test_10_resnet18",
    # Transformer tests (11-14)
    "test_11_layernorm_basic",
    "test_12_gelu_basic",
    "test_13_transformer_simple",
    "test_14_multiblock_transformer",
    # Mamba/SSM tests (15-20)
    "test_15_tinymyo_tiny",
    "test_16_mamba_conv1d",
    "test_17_mamba_ssm",
    "test_18_mamba_block",
    "test_19_mamba_stacked",
    "test_20_bidirectional_mamba",
    # FEMBA tests (21-25)
    "test_21_femba_patchembedder",
    "test_22_femba_full",
    "test_23_femba_full_input",
    "test_24_femba_full_expand2",
    "test_25_femba_tiny_int8",
    # Additional tests (26-31)
    "test_26_tinymyo_8ch_400tok",
    "test_27_linear3d_bench",
    "test_28_conv2d_remainder",
    "test_29_luna_base",
    "test_30_autotune_stress",
    "test_31_luna_full",
    "test_36_drowsiness_fusion",
    "test_37_zeropad2d",
]


@dataclass
class Gap9RunResult:
    test_name: str
    project_dir: Path
    log_path: Path
    success: bool
    error: Optional[str] = None


def run_command(command: str) -> subprocess.CompletedProcess:
    """Run a shell command via bash -lc and capture combined output."""
    return subprocess.run(
        ["bash", "-lc", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def check_fatal_patterns(log_content: str) -> Optional[str]:
    """Check log content for fatal patterns indicating crashes/memory errors.

    Returns the first matching fatal pattern if found, None otherwise.
    This catches "pass then crash" scenarios where validation succeeds
    but the program crashes during cleanup or exits abnormally.
    """
    log_lower = log_content.lower()
    for pattern in FATAL_PATTERNS:
        if pattern.lower() in log_lower:
            return pattern
    return None


def build_and_run(project_dir: Path, gap_env: Path) -> Path:
    """Build and run a single GAP9 project in gvsoc, writing a log file.

    Success is determined by:
    1. No fatal patterns (crashes, memory errors) in output - checked FIRST
    2. "Test PASSED" appears in output
    3. "Test FAILED" does NOT appear in output

    Note: gvsoc can return non-zero even on success, so we prioritize
    log content analysis over return codes.
    """
    log_path = project_dir / "gvsoc_run.log"
    cmd = (
        f"set -eo pipefail && "
        f"source {gap_env} && "
        f"cd {project_dir} && "
        f"rm -rf BUILD && "
        f"make clean && "
        f"make all -j platform=gvsoc MINIMAL_OUTPUT=0 && "
        f"make run platform=gvsoc MINIMAL_OUTPUT=0"
    )
    result = run_command(cmd)
    log_path.write_text(result.stdout)

    log_content = result.stdout

    # FIRST: Check for fatal patterns (crashes, memory errors)
    # This catches "pass then crash" scenarios where validation succeeds
    # but the program crashes during cleanup
    fatal_pattern = check_fatal_patterns(log_content)
    if fatal_pattern:
        raise RuntimeError(
            f"Fatal error detected: '{fatal_pattern}' in {project_dir}. "
            f"See {log_path} for details."
        )

    # Check for test pass/fail markers in log content
    test_passed = "Test PASSED" in log_content or "TEST PASSED" in log_content
    test_failed = "Test FAILED" in log_content or "TEST FAILED" in log_content

    if test_passed and not test_failed:
        return log_path  # Success based on log content
    elif test_failed:
        raise RuntimeError(
            f"Test failed for {project_dir}. See {log_path} for details."
        )
    elif result.returncode != 0:
        # No explicit pass/fail in log, fall back to return code
        raise RuntimeError(
            f"Build/run failed for {project_dir}. See {log_path} for details."
        )
    return log_path


def discover_projects(outputs_root: Path, requested: Sequence[str]) -> List[Path]:
    """Determine which generated projects to run."""
    tests = requested if requested else DEFAULT_TESTS
    projects = []
    for test_name in tests:
        project_dir = outputs_root / test_name / "generated"
        if not project_dir.exists():
            print(f"[skip] {test_name}: generated project not found at {project_dir}")
            continue
        if not (project_dir / "Makefile").exists():
            print(f"[skip] {test_name}: Makefile missing in {project_dir}")
            continue
        projects.append(project_dir)
    return projects


def run_gap9_projects(
    gap_env: Path,
    outputs_dir: Path,
    tests: Optional[Iterable[str]] = None,
) -> List[Gap9RunResult]:
    """Build/run the requested GAP9 projects and return per-test results."""
    if not gap_env.exists():
        raise FileNotFoundError(f"GAP9 environment script not found: {gap_env}")

    test_list = list(tests) if tests else []
    projects = discover_projects(outputs_dir, test_list)

    results: List[Gap9RunResult] = []
    for project_dir in projects:
        test_name = project_dir.parent.name
        print(f"\n=== [{test_name}] Building/Running ===")
        try:
            log_path = build_and_run(project_dir, gap_env)
            print(f"[pass] {test_name}: see log {log_path}")
            results.append(
                Gap9RunResult(
                    test_name=test_name,
                    project_dir=project_dir,
                    log_path=log_path,
                    success=True,
                )
            )
        except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
            print(f"[fail] {test_name}: {exc}")
            results.append(
                Gap9RunResult(
                    test_name=test_name,
                    project_dir=project_dir,
                    log_path=project_dir / "gvsoc_run.log",
                    success=False,
                    error=str(exc),
                )
            )
    return results


def main():
    parser = argparse.ArgumentParser(description="Run GAP9 gvsoc tests for generated projects.")
    parser.add_argument(
        "--gap-env",
        type=Path,
        default=Path("tools/gap9_env_gvsoc.sh"),
        help="Path to GAP9 environment setup script (default: tools/gap9_env_gvsoc.sh)",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("tests/outputs"),
        help="Root directory containing test outputs (default: tests/outputs)",
    )
    parser.add_argument(
        "--tests",
        nargs="*",
        help=f"Subset of tests to run (default: {' '.join(DEFAULT_TESTS)})",
    )
    args = parser.parse_args()

    projects = discover_projects(args.outputs_dir, args.tests or [])
    if not projects:
        print("No generated projects found to run.")
        sys.exit(1)

    print("============================================================")
    print("Running GAP9 (gvsoc) regression for generated projects")
    print("Environment script:", args.gap_env)
    print("Projects:", ", ".join(str(p) for p in projects))
    print("============================================================")

    results = run_gap9_projects(args.gap_env, args.outputs_dir, args.tests)

    failures = [r for r in results if not r.success]
    print("\n============================================================")
    if failures:
        print("FAILED tests:")
        for res in failures:
            print(f"  - {res.test_name} ({res.project_dir}) -> {res.error}")
        sys.exit(1)
    else:
        print("All requested tests PASSED.")


if __name__ == "__main__":
    main()
