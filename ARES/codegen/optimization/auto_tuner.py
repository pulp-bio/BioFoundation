# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Auto-Tuner Framework for ARES.

Provides automated tuning of layer configurations by running GVSOC
iterations and finding optimal settings.

Usage:
    from codegen.optimization import AutoTuner

    tuner = AutoTuner(test_name="test_41_luna_base")

    # Tune a specific layer
    best_config, best_profile = tuner.tune_layer(
        layer_name="freq_fc1",
        op_type="linear_int8",
        shape={'M': 400, 'N': 768, 'K': 192}
    )

    # Or tune all layers
    results = tuner.tune_all()
"""

import os
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from .knowledge_base import KnowledgeBase
from .profile_parser import ProfileParser, LayerProfile, NetworkProfile
from .search_space import SearchSpace, TuningConfig
from .analyzer import PerformanceAnalyzer


@dataclass
class TuningResult:
    """Result of tuning a single layer."""
    layer_name: str
    op_type: str
    shape: Dict[str, int]
    best_config: TuningConfig
    best_cycles: int
    best_macs_per_cycle: float
    configs_tried: int
    total_iterations: int
    recorded_to_kb: bool = False


class AutoTuner:
    """
    Auto-tuning framework for ARES.

    Iteratively tests configurations on GVSOC to find optimal settings,
    then records successful configs to the knowledge base.
    """

    # Known operation types supported by the tuner
    KNOWN_OP_TYPES = {
        'linear_int8', 'linear_fp32', 'conv2d_int8', 'conv1d_int8',
        'mhsa_int8', 'layernorm_int8', 'gelu_int8', 'ssm_int8',
        'maxpool_int8', 'avgpool_int8', 'add_int8', 'concat',
        'embedding', 'rfft_int8', 'cross_attention_int8',
        'groupnorm_int8', 'rope_int8', 'identity_requant',
    }

    def __init__(self,
                 test_name: str,
                 max_iterations: int = 20,
                 test_dir: Optional[str] = None,
                 knowledge_base: Optional[KnowledgeBase] = None,
                 verbose: bool = True,
                 debug: bool = False):
        """
        Initialize auto-tuner.

        Args:
            test_name: Name of test network (e.g., "test_41_luna_base")
            max_iterations: Maximum configurations to try per layer
            test_dir: Override test directory path
            knowledge_base: Knowledge base instance (created if None)
            verbose: Print progress information
            debug: Enable debug mode with full tracebacks
        """
        self.test_name = test_name
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.debug = debug

        # Paths
        if test_dir:
            self.test_dir = Path(test_dir)
        else:
            self.test_dir = Path(f"tests/outputs/{test_name}")
        self.generated_dir = self.test_dir / "generated"

        # Validate paths exist
        if not self.test_dir.exists():
            raise FileNotFoundError(
                f"Test directory not found: {self.test_dir}\n"
                f"Run 'python tests/generate_all_tests.py --test {test_name}' first."
            )
        if not self.generated_dir.exists():
            raise FileNotFoundError(
                f"Generated directory not found: {self.generated_dir}\n"
                f"Run 'python tests/generate_all_tests.py --test {test_name}' first."
            )
        if not (self.generated_dir / "Makefile").exists():
            raise FileNotFoundError(
                f"Makefile not found in {self.generated_dir}\n"
                f"Run code generation before auto-tuning."
            )

        # Components
        self.kb = knowledge_base or KnowledgeBase()
        self.parser = ProfileParser()
        self.search_space = SearchSpace()
        self.analyzer = PerformanceAnalyzer(self.kb)

        # State
        self.results: List[TuningResult] = []
        self.baseline_profile: Optional[NetworkProfile] = None

        # GAP9 environment script path
        self.project_root = Path(__file__).parent.parent.parent
        self.gap9_env_script = self.project_root / "tools" / "gap9_env_gvsoc.sh"

        # Validate GAP9 env script exists
        if not self.gap9_env_script.exists():
            raise FileNotFoundError(
                f"GAP9 environment script not found: {self.gap9_env_script}"
            )

    def tune_layer(self,
                   layer_name: str,
                   op_type: str,
                   shape: Dict[str, int],
                   record_to_kb: bool = True) -> TuningResult:
        """
        Tune a single layer by trying different configurations.

        Args:
            layer_name: Name of layer to tune
            op_type: Operation type
            shape: Layer shape parameters
            record_to_kb: Whether to record best config to knowledge base

        Returns:
            TuningResult with best configuration found
        """
        # Validate op_type
        if op_type not in self.KNOWN_OP_TYPES:
            import warnings
            warnings.warn(
                f"Unknown op_type '{op_type}' for layer {layer_name}. "
                f"Known types: {', '.join(sorted(self.KNOWN_OP_TYPES))}. "
                f"Using default search space which may be ineffective."
            )

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Tuning {layer_name} ({op_type})")
            print(f"Shape: {shape}")
            print(f"{'='*60}")

        # Get candidate configurations
        candidates = self.search_space.get_candidates(
            op_type, shape, self.max_iterations
        )

        if self.verbose:
            print(f"Generated {len(candidates)} candidate configurations")

        # Get baseline if not already cached
        if self.baseline_profile is None:
            if self.verbose:
                print("Running baseline profile...")
            self.baseline_profile = self._run_and_profile()

        baseline_layer = self.baseline_profile.get_layer(layer_name) if self.baseline_profile else None
        baseline_cycles = baseline_layer.total_cycles if baseline_layer else 0

        if self.verbose and baseline_layer:
            print(f"Baseline: {baseline_cycles:,} cycles, {baseline_layer.macs_per_cycle:.2f} MACs/cycle")

        # Test each configuration
        results: List[Tuple[TuningConfig, int, float]] = []

        for i, config in enumerate(candidates):
            if self.verbose:
                print(f"\n[{i+1}/{len(candidates)}] Testing: {config.description}")

            # Check if this config is in negative results (known to fail)
            proposed_config = {
                'tile_config': config.tile_config,
                'kernel_config': config.kernel_config,
            }
            neg_result = self.kb.check_negative(op_type, proposed_config)
            if neg_result:
                if self.verbose:
                    print(f"  Skipping (known failure: {neg_result.result})")
                continue

            try:
                # Generate code with this config
                if not self._generate_with_config(layer_name, op_type, config):
                    if self.verbose:
                        print("  Code generation failed, skipping")
                    # Record as negative result
                    self._record_negative(op_type, config, "codegen_failed",
                                         f"Code generation failed for {layer_name}")
                    continue

                # Build (pass config for compile flags like l1_input_cache)
                if not self._build(config):
                    if self.verbose:
                        print("  Build failed, skipping")
                    # Record as negative result
                    self._record_negative(op_type, config, "build_failed",
                                         f"Build failed for {layer_name}")
                    continue

                # Run and profile
                profile = self._run_and_profile()
                if profile is None:
                    if self.verbose:
                        print("  Run failed, skipping")
                    # Record as negative result
                    self._record_negative(op_type, config, "run_failed",
                                         f"GVSOC run failed for {layer_name}")
                    continue

                # Get layer result
                layer_profile = profile.get_layer(layer_name)
                if layer_profile is None:
                    if self.verbose:
                        print(f"  Layer {layer_name} not found in profile")
                    continue

                cycles = layer_profile.total_cycles
                macs_per_cycle = layer_profile.macs_per_cycle

                # Check for regression (>20% worse than baseline)
                if baseline_cycles > 0 and cycles > baseline_cycles * 1.2:
                    if self.verbose:
                        regression = ((cycles - baseline_cycles) / baseline_cycles * 100)
                        print(f"  Regression: {cycles:,} cycles (+{regression:.1f}%), recording as negative")
                    self._record_negative(op_type, config, "regressed",
                                         f"Performance regressed by {regression:.1f}% for {layer_name}")
                    continue

                results.append((config, cycles, macs_per_cycle))

                if self.verbose:
                    improvement = ((baseline_cycles - cycles) / baseline_cycles * 100) if baseline_cycles > 0 else 0
                    print(f"  Result: {cycles:,} cycles ({improvement:+.1f}%), {macs_per_cycle:.2f} MACs/cycle")

            except subprocess.TimeoutExpired as e:
                if self.verbose:
                    print(f"  Timeout: GVSOC timed out after {e.timeout}s")
                self._record_negative(op_type, config, "timeout",
                                     f"GVSOC timeout ({e.timeout}s) for {layer_name}")
                continue
            except subprocess.CalledProcessError as e:
                stderr_snippet = e.stderr[:200] if e.stderr else "no stderr"
                if self.verbose:
                    print(f"  Subprocess error (exit {e.returncode}): {stderr_snippet}")
                self._record_negative(op_type, config, "subprocess_error",
                                     f"Exit {e.returncode}: {stderr_snippet}")
                continue
            except FileNotFoundError as e:
                if self.verbose:
                    print(f"  File not found: {e}")
                self._record_negative(op_type, config, "file_not_found", str(e))
                continue
            except Exception as e:
                if self.verbose:
                    print(f"  Error: {type(e).__name__}: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()
                # Record exception as negative result
                self._record_negative(op_type, config, "exception",
                                     f"{type(e).__name__}: {str(e)[:100]}")
                continue

        if not results:
            if self.verbose:
                print("\nNo successful configurations found!")
            return TuningResult(
                layer_name=layer_name,
                op_type=op_type,
                shape=shape,
                best_config=candidates[0] if candidates else TuningConfig({}, {}, {}, {}, "default"),
                best_cycles=baseline_cycles,
                best_macs_per_cycle=baseline_layer.macs_per_cycle if baseline_layer else 0,
                configs_tried=len(candidates),
                total_iterations=len(results),
                recorded_to_kb=False
            )

        # Find best (lowest cycles)
        best_config, best_cycles, best_macs = min(results, key=lambda x: x[1])

        if self.verbose:
            improvement = ((baseline_cycles - best_cycles) / baseline_cycles * 100) if baseline_cycles > 0 else 0
            print(f"\n{'='*60}")
            print(f"Best config: {best_config.description}")
            print(f"Cycles: {best_cycles:,} ({improvement:+.1f}% vs baseline)")
            print(f"MACs/cycle: {best_macs:.2f}")
            print(f"{'='*60}")

        # Record to knowledge base if we have results
        recorded = False
        if record_to_kb and best_cycles > 0:
            self.kb.record(
                op_type=op_type,
                shape=shape,
                tile_config=best_config.tile_config,
                kernel_config=best_config.kernel_config,
                pipeline_config=best_config.pipeline_config,
                compile_flags=best_config.compile_flags,
                measured_cycles=best_cycles,
                measured_macs_per_cycle=best_macs,
                source=f"auto_tuner_{self.test_name}_{datetime.now().strftime('%Y%m%d')}",
                test_network=self.test_name,
                confidence=0.9,
                description=f"Auto-tuned {layer_name}: {best_config.description}"
            )
            self.kb.save()
            recorded = True
            if self.verbose:
                print("Recorded to knowledge base.")

        result = TuningResult(
            layer_name=layer_name,
            op_type=op_type,
            shape=shape,
            best_config=best_config,
            best_cycles=best_cycles,
            best_macs_per_cycle=best_macs,
            configs_tried=len(candidates),
            total_iterations=len(results),
            recorded_to_kb=recorded
        )
        self.results.append(result)

        # Save KB to persist any negative results even if no positive results
        if not recorded and self.kb.negative_results:
            self.kb.save()

        return result

    def tune_all(self,
                 layers_to_tune: Optional[List[str]] = None,
                 min_cycles_threshold: int = 100000) -> List[TuningResult]:
        """
        Tune all (or specified) layers in the network.

        Args:
            layers_to_tune: Specific layers to tune (None = auto-detect bottlenecks)
            min_cycles_threshold: Only tune layers with more than this many cycles

        Returns:
            List of TuningResult for each tuned layer
        """
        if self.verbose:
            print(f"\n{'#'*60}")
            print(f"AUTO-TUNING: {self.test_name}")
            print(f"{'#'*60}")

        # Get baseline profile to identify bottlenecks
        # First, rebuild with ENABLE_PERF=1 to ensure performance counters are enabled
        if self.verbose:
            print("\nRebuilding with ENABLE_PERF=1...")
        if not self._build():
            raise RuntimeError("Failed to build with ENABLE_PERF=1")

        if self.verbose:
            print("Running baseline profile...")
        self.baseline_profile = self._run_and_profile()

        if self.baseline_profile is None:
            raise RuntimeError("Failed to get baseline profile")

        # Determine layers to tune
        if layers_to_tune:
            layers = [(l, self.baseline_profile.get_layer(l)) for l in layers_to_tune]
            layers = [(name, layer) for name, layer in layers if layer is not None]
        else:
            # Auto-detect: tune layers above threshold, sorted by cycles
            layers = [
                (l.name, l) for l in self.baseline_profile.layers
                if l.total_cycles >= min_cycles_threshold
            ]
            layers.sort(key=lambda x: -x[1].total_cycles)

        if self.verbose:
            print(f"\nLayers to tune: {len(layers)}")
            for name, layer in layers[:10]:
                print(f"  - {name}: {layer.total_cycles:,} cycles, {layer.macs_per_cycle:.2f} MACs/cycle")

        # Tune each layer
        results = []
        for layer_name, layer in layers:
            # Extract shape from network_info
            shape = self._get_layer_shape(layer_name)
            if shape is None:
                if self.verbose:
                    print(f"\nSkipping {layer_name}: could not determine shape")
                continue

            result = self.tune_layer(
                layer_name=layer_name,
                op_type=layer.op_type,
                shape=shape
            )
            results.append(result)

        # Print summary
        if self.verbose:
            self._print_tuning_summary(results)

        return results

    def tune_and_regenerate(self,
                           min_cycles_threshold: int = 500000,
                           verify_improvement: bool = True) -> bool:
        """
        Convenience method that tunes bottleneck layers and regenerates code.

        This method:
        1. Profiles current code with GVSOC
        2. Identifies bottleneck layers (> min_cycles_threshold)
        3. Tunes those layers
        4. Regenerates code with KB configs
        5. Optionally verifies improvement

        Args:
            min_cycles_threshold: Only tune layers with more than this many cycles
            verify_improvement: Run GVSOC after regeneration to verify

        Returns:
            True if regeneration happened and code was updated
        """
        if self.verbose:
            print(f"\n{'#'*60}")
            print(f"AUTO-TUNE AND REGENERATE: {self.test_name}")
            print(f"{'#'*60}")

        # Run tuning
        results = self.tune_all(min_cycles_threshold=min_cycles_threshold)

        # Check if any configs were recorded to KB
        recorded_count = sum(1 for r in results if r.recorded_to_kb)

        if recorded_count == 0:
            if self.verbose:
                print("\nNo new configurations recorded to KB.")
                print("Code regeneration not needed.")
            return False

        if self.verbose:
            print(f"\n{recorded_count} new configuration(s) recorded to KB.")
            print("Regenerating code with learned optimizations...")

        # Regenerate code
        regenerated = self._regenerate_code()

        if not regenerated:
            if self.verbose:
                print("Code regeneration failed.")
            return False

        if verify_improvement and self.verbose:
            print("\nVerifying improvement with GVSOC...")
            new_profile = self._run_and_profile()
            if new_profile and self.baseline_profile:
                self._compare_profiles(self.baseline_profile, new_profile)

        return True

    def _regenerate_code(self) -> bool:
        """Regenerate C code using current KB entries."""
        try:
            # Import code generator
            import sys
            sys.path.insert(0, str(self.project_root))
            from codegen.generate_c_code import CCodeGenerator

            # Load network info
            network_info_path = self.test_dir / "network_info.json"
            if not network_info_path.exists():
                if self.verbose:
                    print(f"Error: network_info.json not found at {network_info_path}")
                return False

            with open(network_info_path, 'r') as f:
                network_info = json.load(f)

            # Regenerate code
            generator = CCodeGenerator(
                network_info=network_info,
                output_dir=self.generated_dir,
                enable_l1_tiling=True
            )
            generator.generate_all()

            if self.verbose:
                print(f"Code regenerated successfully at {self.generated_dir}")

            # Rebuild
            if not self._build():
                if self.verbose:
                    print("Rebuild failed after code regeneration")
                return False

            return True

        except Exception as e:
            if self.verbose:
                print(f"Regeneration error: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return False

    def _compare_profiles(self,
                         before: 'NetworkProfile',
                         after: 'NetworkProfile') -> None:
        """Compare before/after profiles and print improvement summary."""
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON")
        print("="*60)

        total_before = sum(l.total_cycles for l in before.layers)
        total_after = sum(l.total_cycles for l in after.layers)

        if total_before > 0:
            improvement = ((total_before - total_after) / total_before * 100)
            print(f"Total cycles: {total_before:,} -> {total_after:,} ({improvement:+.1f}%)")

        # Show per-layer changes
        print("\nPer-layer changes:")
        for layer_before in before.layers:
            layer_after = after.get_layer(layer_before.name)
            if layer_after and layer_before.total_cycles > 100000:
                change = ((layer_before.total_cycles - layer_after.total_cycles)
                         / layer_before.total_cycles * 100)
                if abs(change) > 1:  # Only show significant changes
                    print(f"  {layer_before.name}: {layer_before.total_cycles:,} -> "
                          f"{layer_after.total_cycles:,} ({change:+.1f}%)")

    def show_tuning_plan(self) -> None:
        """Show what would be tuned without actually running."""
        print(f"\n{'='*60}")
        print(f"TUNING PLAN: {self.test_name}")
        print(f"{'='*60}")

        # Try existing log file first
        print("\nGetting baseline profile...")
        profile = self._get_existing_profile()

        if profile is None:
            # No existing log, run GVSOC to get profile
            print("No existing profile found, running GVSOC...")
            if not self._build():
                print("Build failed. Check GAP9 SDK setup.")
                return
            profile = self._run_and_profile()

        if profile is None:
            print("Failed to get baseline profile.")
            return

        # Show layers that would be tuned
        print(f"\nLayers that would be tuned (cycles > 100,000):\n")
        print(f"{'Layer':<40} {'Op Type':<15} {'Cycles':>12} {'MACs/cyc':>10}")
        print("-" * 80)

        tunable = [l for l in profile.layers if l.total_cycles >= 100000]
        tunable.sort(key=lambda x: -x.total_cycles)

        for layer in tunable:
            shape = self._get_layer_shape(layer.name)
            candidates = len(self.search_space.get_candidates(layer.op_type, shape or {}))
            print(f"{layer.name:<40} {layer.op_type:<15} {layer.total_cycles:>12,} {layer.macs_per_cycle:>10.2f}  ({candidates} configs)")

        print(f"\nTotal layers to tune: {len(tunable)}")
        print(f"Max iterations per layer: {self.max_iterations}")
        print(f"Estimated total iterations: {len(tunable) * self.max_iterations}")

    def _get_existing_profile(self) -> Optional[NetworkProfile]:
        """
        Get profile from existing gvsoc_run.log file.

        This is used for --dry-run mode to avoid needing to run GVSOC.
        Prefers PERF logs (gvsoc_run_perf*.log) over regular logs.
        """
        # Try to find a PERF log first (has detailed cycle breakdown)
        perf_logs = list(self.generated_dir.glob("gvsoc_run_perf*.log"))
        if perf_logs:
            # Use most recently modified
            log_path = max(perf_logs, key=lambda p: p.stat().st_mtime)
            if self.verbose:
                print(f"Using performance log: {log_path.name}")
        else:
            log_path = self.generated_dir / "gvsoc_run.log"

        if not log_path.exists():
            return None

        try:
            profile = self.parser.parse_log(str(log_path))
            profile.test_name = self.test_name

            # Check if we got any layer data
            if not profile.layers:
                if self.verbose:
                    print(f"Warning: No layer data found in {log_path.name}")
                    print("The log may have been run without ENABLE_PERF=1")
                return None

            return profile
        except Exception as e:
            if self.verbose:
                print(f"Error parsing log: {e}")
            return None

    def _generate_with_config(self, layer_name: str, op_type: str,
                              config: TuningConfig) -> bool:
        """
        Regenerate code with specific config for one layer.

        Writes a config override JSON file and calls the code generator.

        Returns:
            True if code generation succeeded, False otherwise
        """
        # Write config override JSON
        overrides = {
            layer_name: {
                'tile_config': config.tile_config,
                'kernel_config': config.kernel_config,
                'pipeline_config': config.pipeline_config,
            }
        }

        override_file = self.test_dir / "layer_config_overrides.json"
        with open(override_file, 'w') as f:
            json.dump(overrides, f, indent=2)

        if self.verbose:
            print(f"    Wrote overrides to {override_file}")

        # Regenerate code with overrides
        try:
            # Import here to avoid circular imports
            import sys
            sys.path.insert(0, str(self.project_root))
            from codegen.generate_c_code import CCodeGenerator

            # Paths relative to test_dir
            network_info_path = self.test_dir / "golden_outputs" / "network_info.json"
            weights_dir = self.test_dir / "golden_outputs" / "weights"
            test_case_dir = self.test_dir / "golden_outputs" / "test_cases" / "test_case_3"
            if not test_case_dir.exists():
                test_case_dir = self.test_dir / "golden_outputs" / "test_cases" / "test_case_1"

            generator = CCodeGenerator(
                network_info_path=str(network_info_path),
                weights_dir=str(weights_dir),
                test_case_dir=str(test_case_dir),
                output_dir=str(self.generated_dir),
                layer_config_overrides=overrides
            )

            generator.generate_all()

            if self.verbose:
                print(f"    Code regenerated with config: {config.description}")

            return True

        except Exception as e:
            if self.verbose:
                print(f"    Code generation failed: {e}")
            return False

    def _config_to_extra_cflags(self, config: TuningConfig) -> str:
        """Convert TuningConfig to EXTRA_CFLAGS string for make.

        Maps tile_config and compile_flags to -D defines.
        """
        defines = []

        # Map l1_input_cache from tile_config to LINEAR_INT8_INPUT_L1_CACHE
        if 'l1_input_cache' in config.tile_config:
            l1_cache = 1 if config.tile_config['l1_input_cache'] else 0
            defines.append(f"-DLINEAR_INT8_INPUT_L1_CACHE={l1_cache}")

        # Add explicit compile_flags from config
        for key, value in config.compile_flags.items():
            if isinstance(value, bool):
                defines.append(f"-D{key}={1 if value else 0}")
            else:
                defines.append(f"-D{key}={value}")

        return ' '.join(defines)

    def _build(self, config: Optional[TuningConfig] = None) -> bool:
        """Run make clean all with GAP9 environment.

        Args:
            config: Optional TuningConfig to extract compile flags from.
        """
        try:
            # Build EXTRA_CFLAGS from config
            extra_cflags = ""
            if config:
                extra_cflags = self._config_to_extra_cflags(config)
                if extra_cflags and self.verbose:
                    print(f"    EXTRA_CFLAGS: {extra_cflags}")

            # Build command that sources the GAP9 environment first
            cmd = f"source {self.gap9_env_script} && make clean all platform=gvsoc ENABLE_PERF=1 MINIMAL_OUTPUT=1"
            if extra_cflags:
                cmd += f' EXTRA_CFLAGS="{extra_cflags}"'

            result = subprocess.run(
                ["bash", "-c", cmd],
                cwd=self.generated_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception as e:
            if self.verbose:
                print(f"Build error: {e}")
            return False

    def _run_and_profile(self) -> Optional[NetworkProfile]:
        """Run on GVSOC and parse output with GAP9 environment."""
        try:
            # Run command that sources the GAP9 environment first
            cmd = f"source {self.gap9_env_script} && make run platform=gvsoc ENABLE_PERF=1 MINIMAL_OUTPUT=1"

            result = subprocess.run(
                ["bash", "-c", cmd],
                cwd=self.generated_dir,
                capture_output=True,
                text=True,
                timeout=600
            )

            # Save full output to log file for debugging
            log_path = self.generated_dir / "gvsoc_run_autotune.log"
            with open(log_path, 'w') as f:
                f.write(f"=== AUTO-TUNER GVSOC RUN LOG ===\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Test: {self.test_name}\n")
                f.write(f"Exit code: {result.returncode}\n\n")
                f.write(f"=== STDOUT ===\n{result.stdout}\n\n")
                f.write(f"=== STDERR ===\n{result.stderr}\n")

            if result.returncode != 0:
                if self.verbose:
                    print(f"GVSOC run failed (exit code {result.returncode})")
                    print(f"Full log saved to: {log_path}")
                    if result.stderr:
                        print(f"stderr (first 500 chars): {result.stderr[:500]}")
                return None

            # Parse the PERF output from stdout
            profile = self._parse_perf_output(result.stdout)
            if profile:
                profile.test_name = self.test_name

            return profile

        except subprocess.TimeoutExpired as e:
            if self.verbose:
                print(f"GVSOC run timed out after {e.timeout}s")
            return None
        except Exception as e:
            if self.verbose:
                print(f"Run error: {type(e).__name__}: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return None

    def _parse_perf_output(self, output: str) -> Optional[NetworkProfile]:
        """Parse PERF lines from GVSOC stdout."""
        profile = NetworkProfile(test_name=self.test_name)

        # Use the parser's extraction method directly
        layers_data = self.parser._extract_layer_data(output)

        for name, data in layers_data.items():
            if name.lower() in self.parser.SUMMARY_NAMES:
                continue

            # Calculate MACs if we have shape info
            macs = data.get('macs')
            if macs is None:
                shape = self._get_layer_shape(name)
                if shape:
                    macs = self._calculate_macs(name, shape)

            layer = LayerProfile(
                name=name,
                op_type=self.parser._infer_op_type(name),
                total_cycles=data.get('cycles', 0),
                compute_cycles=data.get('compute_cycles', data.get('cycles', 0)),
                dma_load_cycles=data.get('dma_load_cycles', 0),
                dma_store_cycles=data.get('dma_store_cycles', 0),
                idle_cycles=data.get('idle_cycles', 0),
                macs=macs,
            )
            profile.layers.append(layer)

        profile.total_cycles = sum(l.total_cycles for l in profile.layers)
        profile.total_macs = sum(l.macs or 0 for l in profile.layers)

        return profile if profile.layers else None

    def _calculate_macs(self, layer_name: str, shape: Dict[str, int]) -> Optional[int]:
        """Calculate MACs for a layer based on its shape."""
        op_type = self.parser._infer_op_type(layer_name)

        if op_type == 'conv2d_int8':
            # Conv2D MACs = out_h * out_w * out_channels * in_channels * kernel_h * kernel_w
            in_h = shape.get('in_h', 0)
            in_w = shape.get('in_w', 0)
            in_ch = shape.get('in_channels', 0)
            out_ch = shape.get('out_channels', 0)
            kh = shape.get('kernel_h', 3)
            kw = shape.get('kernel_w', 3)
            stride = shape.get('stride', 1)
            # Approximate output size
            out_h = in_h // stride
            out_w = in_w // stride
            return out_h * out_w * out_ch * in_ch * kh * kw

        elif op_type == 'linear_int8':
            # Linear MACs = M * N * K
            M = shape.get('M', 1)
            N = shape.get('N', 0)
            K = shape.get('K', 0)
            return M * N * K

        elif op_type == 'mhsa_int8':
            # MHSA MACs (approximate): Q/K/V projections + attention + output projection
            seq = shape.get('seq_len', 0)
            dim = shape.get('embed_dim', 0)
            heads = shape.get('num_heads', 1)
            head_dim = shape.get('head_dim', dim // heads if heads > 0 else 0)
            # Q/K/V projections: 3 * seq * dim * dim
            # Attention: seq * seq * head_dim * heads
            # Output proj: seq * dim * dim
            proj_macs = 4 * seq * dim * dim
            attn_macs = seq * seq * head_dim * heads
            return proj_macs + attn_macs

        return None

    def _get_layer_shape(self, layer_name: str) -> Optional[Dict[str, int]]:
        """Extract layer shape from network_info.json."""
        network_info_path = self.test_dir / "golden_outputs" / "network_info.json"

        if not network_info_path.exists():
            return None

        try:
            with open(network_info_path) as f:
                network_info = json.load(f)

            layer_info = network_info.get(layer_name, {})

            # Extract shape based on layer type
            layer_type = layer_info.get('type', '')

            if 'Linear' in layer_type:
                return {
                    'M': layer_info.get('batch_tokens', 1),
                    'N': layer_info.get('out_features', 0),
                    'K': layer_info.get('in_features', 0),
                }
            elif 'Conv2d' in layer_type:
                output_shape = layer_info.get('output_shape', [1, 1, 1, 1])
                return {
                    'in_h': layer_info.get('in_h', output_shape[2] if len(output_shape) > 2 else 1),
                    'in_w': layer_info.get('in_w', output_shape[3] if len(output_shape) > 3 else 1),
                    'in_channels': layer_info.get('in_channels', 0),
                    'out_channels': layer_info.get('out_channels', output_shape[1] if len(output_shape) > 1 else 0),
                    'kernel_h': layer_info.get('kernel_size', [3, 3])[0] if isinstance(layer_info.get('kernel_size'), list) else layer_info.get('kernel_size', 3),
                    'kernel_w': layer_info.get('kernel_size', [3, 3])[1] if isinstance(layer_info.get('kernel_size'), list) else layer_info.get('kernel_size', 3),
                }
            elif 'Attention' in layer_type or 'MHSA' in layer_type:
                return {
                    'seq_len': layer_info.get('sequence_length', layer_info.get('seq_len', 0)),
                    'embed_dim': layer_info.get('embed_dim', 0),
                    'num_heads': layer_info.get('num_heads', 1),
                    'head_dim': layer_info.get('head_dim', 0),
                }
            elif 'LayerNorm' in layer_type:
                return {
                    'num_tokens': layer_info.get('num_tokens', 1),
                    'embed_dim': layer_info.get('normalized_shape', [0])[0] if isinstance(layer_info.get('normalized_shape'), list) else layer_info.get('normalized_shape', 0),
                }
            elif 'SSM' in layer_type:
                return {
                    'seq_len': layer_info.get('seq_len', 0),
                    'd_inner': layer_info.get('d_inner', 0),
                    'd_state': layer_info.get('d_state', 16),
                    'dt_rank': layer_info.get('dt_rank', 0),
                }

            return {}

        except Exception:
            return None

    def _record_negative(self, op_type: str, config: TuningConfig,
                         result: str, notes: str) -> None:
        """
        Record a negative result to the knowledge base.

        This tracks configurations that failed or regressed, so we can
        avoid trying them again in future tuning runs.

        Args:
            op_type: Operation type (e.g., "linear_int8")
            config: The TuningConfig that failed
            result: Short result code (e.g., "codegen_failed", "regressed")
            notes: Detailed notes about the failure
        """
        attempted_config = {
            'tile_config': config.tile_config,
            'kernel_config': config.kernel_config,
            'pipeline_config': config.pipeline_config,
            'description': config.description,
        }
        self.kb.record_negative(
            op_type=op_type,
            attempted_config=attempted_config,
            result=result,
            source=f"auto_tuner_{self.test_name}",
            notes=notes
        )
        if self.verbose:
            print(f"    [KB] Recorded negative: {result}")

    def _print_tuning_summary(self, results: List[TuningResult]) -> None:
        """Print summary of tuning results."""
        print(f"\n{'#'*60}")
        print("TUNING SUMMARY")
        print(f"{'#'*60}\n")

        total_baseline = sum(r.best_cycles for r in results)  # Approximate
        total_improved = 0

        print(f"{'Layer':<35} {'Config':<25} {'Cycles':>12} {'MACs/cyc':>10} {'KB':>4}")
        print("-" * 90)

        for r in results:
            kb_marker = "[OK]" if r.recorded_to_kb else ""
            print(f"{r.layer_name:<35} {r.best_config.description:<25} {r.best_cycles:>12,} {r.best_macs_per_cycle:>10.2f} {kb_marker:>4}")
            total_improved += r.best_cycles

        print("-" * 90)
        print(f"Total layers tuned: {len(results)}")
        print(f"Configs recorded to KB: {sum(1 for r in results if r.recorded_to_kb)}")


def run_auto_tuner(test_name: str,
                   layers: Optional[List[str]] = None,
                   max_iterations: int = 20) -> List[TuningResult]:
    """
    Convenience function to run auto-tuning.

    Args:
        test_name: Test network name
        layers: Specific layers to tune (None = auto-detect)
        max_iterations: Max iterations per layer

    Returns:
        List of tuning results
    """
    tuner = AutoTuner(test_name=test_name, max_iterations=max_iterations)
    return tuner.tune_all(layers_to_tune=layers)
