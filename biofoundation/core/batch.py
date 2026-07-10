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

"""Canonical batch handling across biosignal datasets and model families."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypedDict, cast


class SignalBatch(TypedDict, total=False):
    """Common batch fields used by BioFoundation tasks.

    Only ``input`` is universally required. Model adapters can require channel
    or sensor metadata without forcing simpler models to manufacture it.
    """

    input: Any
    label: Any
    channel_names: Any
    channel_locations: Any
    sensor_type: Any
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class BatchRequirements:
    """Metadata fields required by a particular model adapter."""

    label: bool = False
    channel_names: bool = False
    channel_locations: bool = False
    sensor_type: bool = False


def as_signal_batch(batch: Any) -> SignalBatch:
    """Normalize existing tensor, tuple, and mapping batches to one contract."""

    if isinstance(batch, Mapping):
        normalized = dict(batch)
    elif isinstance(batch, (tuple, list)):
        if len(batch) == 1:
            normalized = {"input": batch[0]}
        elif len(batch) == 2:
            normalized = {"input": batch[0], "label": batch[1]}
        else:
            raise ValueError(
                "Sequence batches must contain either input or (input, label); "
                f"received {len(batch)} values."
            )
    else:
        normalized = {"input": batch}

    if "input" not in normalized:
        raise ValueError("Signal batches must contain an 'input' field.")

    return cast(SignalBatch, normalized)


def require_batch_fields(batch: SignalBatch, requirements: BatchRequirements) -> SignalBatch:
    """Validate model-specific batch requirements with a useful error message."""

    required_fields = (
        ("label", requirements.label),
        ("channel_names", requirements.channel_names),
        ("channel_locations", requirements.channel_locations),
        ("sensor_type", requirements.sensor_type),
    )
    missing = [name for name, required in required_fields if required and name not in batch]
    if missing:
        raise ValueError(f"Signal batch is missing required fields: {', '.join(missing)}")
    return batch

