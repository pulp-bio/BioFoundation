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
import unittest
from pathlib import Path

from biofoundation.core.batch import BatchRequirements
from biofoundation.model_registry import MODEL_REGISTRY, get_model_spec


ROOT = Path(__file__).resolve().parents[1]


class ModelRegistryTest(unittest.TestCase):
    def test_registry_contains_every_published_model(self):
        self.assertEqual(set(MODEL_REGISTRY), {"femba", "luna", "tinymyo", "lumamba", "panluna"})

    def test_model_targets_and_experiments_exist(self):
        for spec in MODEL_REGISTRY.values():
            module_name, class_name = spec.model_target.rsplit(".", 1)
            module_path = ROOT / f"{module_name.replace('.', '/')}.py"
            self.assertTrue(module_path.is_file(), module_path)

            tree = ast.parse(module_path.read_text(encoding="utf-8"))
            classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
            self.assertIn(class_name, classes, spec.model_target)

            for experiment in (spec.pretrain_experiment, spec.finetune_experiment):
                path = ROOT / "config" / "experiment" / f"{experiment}.yaml"
                self.assertTrue(path.is_file(), path)

            finetune_config = ROOT / "config" / "experiment" / f"{spec.finetune_experiment}.yaml"
            self.assertIn("pretrained_safetensors_path", finetune_config.read_text(encoding="utf-8"))

    def test_batch_requirements_match_model_inputs(self):
        self.assertEqual(MODEL_REGISTRY["femba"].batch_requirements, BatchRequirements())
        self.assertEqual(MODEL_REGISTRY["tinymyo"].batch_requirements, BatchRequirements())
        self.assertEqual(
            MODEL_REGISTRY["luna"].batch_requirements,
            BatchRequirements(channel_locations=True),
        )
        self.assertEqual(
            MODEL_REGISTRY["lumamba"].batch_requirements,
            BatchRequirements(channel_locations=True),
        )
        self.assertEqual(
            MODEL_REGISTRY["panluna"].batch_requirements,
            BatchRequirements(channel_locations=True, sensor_type=True),
        )

    def test_huggingface_and_paper_links_are_unique_and_documented(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        hubs = [spec.huggingface_url for spec in MODEL_REGISTRY.values()]
        self.assertEqual(len(hubs), len(set(hubs)))

        for spec in MODEL_REGISTRY.values():
            self.assertTrue(spec.huggingface_url.startswith("https://huggingface.co/PulpBio/"))
            self.assertTrue(spec.paper_url.startswith("https://arxiv.org/"))
            self.assertIn(spec.huggingface_url, readme)
            self.assertIn(spec.paper_url, readme)

    def test_citation_guide_covers_models_and_publication_venues(self):
        citations = (ROOT / "docs" / "CITATIONS.md").read_text(encoding="utf-8")
        expected_venues = {
            "FEMBA": "EMBC 2025",
            "LUNA": "NeurIPS 2025",
            "TinyMyo": "arXiv preprint",
            "LuMamba": "EUSIPCO 2026",
            "PanLUNA": "AICAS 2026",
        }
        for model, venue in expected_venues.items():
            self.assertIn(f"## {model}", citations)
            self.assertIn(venue, citations)

        panluna_entry = citations.split("## PanLUNA", 1)[1]
        self.assertIn("Benini, Luca", panluna_entry)

    def test_lookup_is_case_insensitive(self):
        self.assertEqual(get_model_spec("PanLUNA"), MODEL_REGISTRY["panluna"])


if __name__ == "__main__":
    unittest.main()
