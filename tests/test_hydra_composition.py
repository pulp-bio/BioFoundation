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


if __name__ == "__main__":
    unittest.main()
