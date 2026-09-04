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

import re
import unittest
from pathlib import Path

from biofoundation.model_registry import MODEL_REGISTRY

try:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
except ImportError:
    compose = None
    initialize_config_dir = None
    OmegaConf = None


ROOT = Path(__file__).resolve().parents[1]

# Experiments that do not compose today, with the reason. These are recorded rather
# than skipped silently so the breakage stays visible, and the test fails if one of
# them starts composing, which forces the entry to be removed rather than left to rot.
KNOWN_UNCOMPOSABLE_EXPERIMENTS = {
    "FEMBA_quantized": (
        "pre-existing: its defaults list requires scheduler/constant_lr, which does "
        "not exist in config/scheduler/. It also targets "
        "ARES.tests.test_networks.test_24_femba_full_expland2, misspelling the module "
        "test_24_femba_full_expand2."
    ),
}


@unittest.skipIf(compose is None, "hydra-core is not installed")
class HydraCompositionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not OmegaConf.has_resolver("env"):
            OmegaConf.register_new_resolver("env", lambda _: "/tmp/biofoundation")

    def test_every_registered_experiment_composes_and_resolves(self):
        experiments = {
            experiment
            for spec in MODEL_REGISTRY.values()
            for experiment in (spec.pretrain_experiment, spec.finetune_experiment)
        }

        for experiment in sorted(experiments):
            with self.subTest(experiment=experiment):
                with initialize_config_dir(version_base="1.1", config_dir=str(ROOT / "config")):
                    config = compose(
                        config_name="defaults",
                        overrides=[f"+experiment={experiment}"],
                    )
                resolved = OmegaConf.to_container(config, resolve=True)
                self.assertIsInstance(resolved, dict)
                self.assertTrue(resolved["tag"])

    def test_every_experiment_file_composes_or_is_a_known_failure(self):
        """Compose every experiment, including those no registry entry points at.

        The test above only reaches experiments named by a ModelSpec, which leaves
        standalone experiment files unchecked. Composition failures that this catches
        include defaults-list ordering mistakes, which produce a valid-looking YAML
        file that Hydra rejects only at run time.
        """

        for path in sorted((ROOT / "config" / "experiment").glob("*.yaml")):
            name = path.stem
            with self.subTest(experiment=name):
                try:
                    with initialize_config_dir(version_base="1.1", config_dir=str(ROOT / "config")):
                        config = compose(config_name="defaults", overrides=[f"+experiment={name}"])
                    OmegaConf.to_container(config, resolve=True)
                except Exception as error:  # noqa: BLE001 - the failure itself is the assertion
                    if name in KNOWN_UNCOMPOSABLE_EXPERIMENTS:
                        self.skipTest(f"{name}: {KNOWN_UNCOMPOSABLE_EXPERIMENTS[name]}")
                    self.fail(f"{name} failed to compose: {type(error).__name__}: {error}")
                else:
                    self.assertNotIn(
                        name,
                        KNOWN_UNCOMPOSABLE_EXPERIMENTS,
                        f"{name} now composes; remove it from KNOWN_UNCOMPOSABLE_EXPERIMENTS",
                    )


    def test_every_dataset_option_composes_with_a_consistent_head(self):
        """Each config/dataset option must resolve and pair a head its task can drive.

        A dataset file owns its corpus path, sample layout, label kind, channel count,
        prediction head, task and criterion together. Composing them separately is what
        allowed a classification-only key such as num_classes to reach a regression
        head, so this walks every option and checks the combination holds.
        """
        dataset_dir = ROOT / "config" / "dataset"
        options = sorted(path.stem for path in dataset_dir.glob("*.yaml"))
        self.assertTrue(options, "no dataset options found")

        regression_heads = {"MlpRegressionHead"}
        regression_tasks = {"RegressionTask"}

        for option in options:
            with self.subTest(dataset=option):
                with initialize_config_dir(version_base="1.1", config_dir=str(ROOT / "config")):
                    config = compose(
                        config_name="defaults",
                        overrides=["+experiment=SCEReBrO_finetune", f"dataset={option}"],
                    )
                resolved = OmegaConf.to_container(config, resolve=True)

                head = resolved["model_head"]["_target_"].rsplit(".", 1)[-1]
                task = resolved["task"]["_target_"].rsplit(".", 1)[-1]
                self.assertEqual(
                    head in regression_heads,
                    task in regression_tasks,
                    f"{option}: head {head} and task {task} disagree on regression",
                )
                self.assertEqual(
                    resolved["label_mode"] == "regression",
                    task in regression_tasks,
                    f"{option}: label_mode and task disagree on regression",
                )
                # A head only ever receives keys its constructor accepts.
                if head in regression_heads:
                    self.assertNotIn("num_classes", resolved["model_head"], option)
                    self.assertNotIn("num_patches", resolved["model_head"], option)
                else:
                    self.assertGreaterEqual(int(resolved["model_head"]["num_classes"]), 2, option)

    def test_finetuning_experiment_leaves_per_corpus_settings_to_the_dataset_group(self):
        """The experiment must not restate anything config/dataset owns.

        Hydra applies a config's own values after its defaults list, so a key set in
        both places resolves to the experiment's copy and the dataset file is silently
        ignored. Keeping them out of the experiment makes one file the single owner.
        """
        text = (ROOT / "config" / "experiment" / "SCEReBrO_finetune.yaml").read_text(encoding="utf-8")
        # num_channels lives under model:, so leading whitespace is allowed for it.
        owned = {
            "dataset_root": r"^dataset_root:",
            "dataset_kind": r"^dataset_kind:",
            "label_mode": r"^label_mode:",
            "model.num_channels": r"^\s*num_channels:",
        }
        restated = [key for key, pattern in owned.items() if re.search(pattern, text, re.MULTILINE)]
        self.assertEqual(restated, [], f"restated in the experiment: {restated}")

        for group in ("model_head", "task", "criterion"):
            self.assertNotRegex(
                text,
                rf"^\s*-\s*(override\s+)?/?{group}:",
                f"{group} is selected by the experiment; leave it to config/dataset/",
            )


if __name__ == "__main__":
    unittest.main()
