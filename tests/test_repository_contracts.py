#*----------------------------------------------------------------------------*
#* Copyright (C) 2026 ETH Zurich, Switzerland                                 *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Author:  BioFoundation Contributors                                       *
#*----------------------------------------------------------------------------*

import ast
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_TARGET_PREFIXES = ("criterion.", "data_module.", "datasets.", "models.", "schedulers.", "tasks.")

# Derived rather than listed so that adding a task file cannot silently opt out of the
# shared batch-adapter contract below.
TASK_FILES = tuple(sorted(path.name for path in (ROOT / "tasks").glob("*.py")))


class RepositoryContractsTest(unittest.TestCase):
    def test_local_hydra_targets_resolve(self):
        target_pattern = re.compile(
            r"^\s*_target_:\s*(?:'([^']+)'|\"([^\"]+)\"|([^\s#]+))",
            re.MULTILINE,
        )
        for config_path in (ROOT / "config").rglob("*.yaml"):
            matches = target_pattern.findall(config_path.read_text(encoding="utf-8"))
            for match in matches:
                target = next(value for value in match if value)
                if "CHANGEME" in target or not target.startswith(LOCAL_TARGET_PREFIXES):
                    continue

                module_name, attribute = target.rsplit(".", 1)
                module_path = ROOT / f"{module_name.replace('.', '/')}.py"
                self.assertTrue(module_path.is_file(), f"{config_path}: {target}")

                tree = ast.parse(module_path.read_text(encoding="utf-8"))
                definitions = {
                    node.name
                    for node in tree.body
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                }
                self.assertIn(attribute, definitions, f"{config_path}: {target}")

    def test_hydra_package_directives_stay_on_the_first_line(self):
        misplaced = []
        for config_path in (ROOT / "config").rglob("*.yaml"):
            lines = config_path.read_text(encoding="utf-8").splitlines()
            directive_lines = [index for index, line in enumerate(lines) if "@package" in line]
            if directive_lines and directive_lines != [0]:
                misplaced.append(str(config_path.relative_to(ROOT)))
        self.assertEqual(misplaced, [])

    def test_training_steps_use_the_shared_batch_adapter(self):
        for filename in TASK_FILES:
            path = ROOT / "tasks" / filename
            tree = ast.parse(path.read_text(encoding="utf-8"))
            methods = {
                node.name: node
                for class_node in tree.body
                if isinstance(class_node, ast.ClassDef)
                for node in class_node.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            for method_name in ("training_step", "validation_step", "test_step"):
                if method_name not in methods:
                    continue
                calls_adapter = any(
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "as_signal_batch"
                    for node in ast.walk(methods[method_name])
                )
                self.assertTrue(calls_adapter, f"{filename}:{method_name}")

    def test_cli_validates_environment_before_hydra_starts(self):
        tree = ast.parse((ROOT / "run_train.py").read_text(encoding="utf-8"))
        main_guard = next(
            node
            for node in tree.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
        )
        call_names = [
            node.value.func.id
            for node in main_guard.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
        ]
        self.assertLess(call_names.index("require_environment"), call_names.index("run"))

    def test_onboarding_docs_do_not_link_to_missing_local_paths(self):
        link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
        paths = (
            ROOT / "README.md",
            ROOT / "CONTRIBUTING.md",
            ROOT / "models" / "README.md",
            ROOT / "docs" / "README.md",
            ROOT / "docs" / "TRAINING.md",
            ROOT / "docs" / "CITATIONS.md",
            *(ROOT / "docs" / "model").glob("*.md"),
        )
        for path in paths:
            for link in link_pattern.findall(path.read_text(encoding="utf-8")):
                if "://" in link or link.startswith(("#", "mailto:")):
                    continue
                local_path = (path.parent / link.split("#", 1)[0]).resolve()
                self.assertTrue(local_path.exists(), f"{path.relative_to(ROOT)}: {link}")

    def test_source_and_config_files_keep_the_apache_header(self):
        source_roots = (
            ROOT / "ARES",
            ROOT / "biofoundation",
            ROOT / "criterion",
            ROOT / "data_module",
            ROOT / "datasets",
            ROOT / "make_datasets",
            ROOT / "models",
            ROOT / "schedulers",
            ROOT / "tasks",
            ROOT / "tests",
            ROOT / "util",
        )
        paths = list(ROOT.glob("*.py"))
        for source_root in source_roots:
            paths.extend(source_root.rglob("*.py"))
        paths.extend((ROOT / "config").rglob("*.yaml"))
        paths.extend((ROOT / ".github").rglob("*.yml"))

        missing = [
            str(path.relative_to(ROOT))
            for path in paths
            if "SPDX-License-Identifier: Apache-2.0" not in path.read_text(encoding="utf-8")
        ]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
