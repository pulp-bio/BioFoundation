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

"""Discoverable metadata for every model family distributed in this repository."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from biofoundation.core.batch import BatchRequirements


@dataclass(frozen=True)
class ModelSpec:
    """Human- and machine-readable entry for a BioFoundation model family."""

    display_name: str
    modalities: tuple[str, ...]
    architecture: str
    model_target: str
    pretrain_experiment: str
    finetune_experiment: str
    huggingface_url: str
    paper_url: str
    batch_requirements: BatchRequirements = BatchRequirements()


MODEL_REGISTRY: Mapping[str, ModelSpec] = MappingProxyType(
    {
        "femba": ModelSpec(
            display_name="FEMBA",
            modalities=("EEG",),
            architecture="Bidirectional Mamba",
            model_target="models.FEMBA.FEMBA",
            pretrain_experiment="FEMBA_pretrain",
            finetune_experiment="FEMBA_finetune",
            huggingface_url="https://huggingface.co/PulpBio/FEMBA",
            paper_url="https://arxiv.org/abs/2502.06438",
        ),
        "luna": ModelSpec(
            display_name="LUNA",
            modalities=("EEG",),
            architecture="Query-unified Transformer",
            model_target="models.LUNA.LUNA",
            pretrain_experiment="LUNA_pretrain",
            finetune_experiment="LUNA_finetune",
            huggingface_url="https://huggingface.co/PulpBio/LUNA",
            paper_url="https://arxiv.org/abs/2510.22257",
            batch_requirements=BatchRequirements(channel_locations=True),
        ),
        "tinymyo": ModelSpec(
            display_name="TinyMyo",
            modalities=("sEMG",),
            architecture="Rotary Transformer",
            model_target="models.TinyMyo.TinyMyo",
            pretrain_experiment="TinyMyo_pretrain",
            finetune_experiment="TinyMyo_finetune",
            huggingface_url="https://huggingface.co/PulpBio/TinyMyo",
            paper_url="https://arxiv.org/abs/2512.15729",
        ),
        "lumamba": ModelSpec(
            display_name="LuMamba",
            modalities=("EEG",),
            architecture="Query-unified Mamba",
            model_target="models.LuMamba.LuMamba",
            pretrain_experiment="LuMamba_pretrain",
            finetune_experiment="LuMamba_finetune",
            huggingface_url="https://huggingface.co/PulpBio/LuMamba",
            paper_url="https://arxiv.org/abs/2603.19100",
            batch_requirements=BatchRequirements(channel_locations=True),
        ),
        "panluna": ModelSpec(
            display_name="PanLUNA",
            modalities=("EEG", "ECG", "PPG"),
            architecture="Multimodal query-unified Transformer",
            model_target="models.PanLUNA.PanLUNA",
            pretrain_experiment="PanLUNA_pretrain",
            finetune_experiment="PanLUNA_finetune",
            huggingface_url="https://huggingface.co/PulpBio/PanLUNA",
            paper_url="https://arxiv.org/abs/2604.04297",
            batch_requirements=BatchRequirements(
                channel_locations=True,
                sensor_type=True,
            ),
        ),
    }
)


def get_model_spec(name: str) -> ModelSpec:
    """Return a model specification by its case-insensitive registry key."""

    key = name.casefold()
    try:
        return MODEL_REGISTRY[key]
    except KeyError as error:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model '{name}'. Available models: {available}") from error
