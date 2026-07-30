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

    Two independent electrode-geometry representations are supported, and a model
    requires exactly one of them. They are peers rather than alternatives to convert
    between, so a dataset declares which one it produces and the registry records
    which one each model consumes:

    ``channel_locations``
        Shape ``(batch, channels, 3)``. One 3D coordinate per channel. For a bipolar
        derivation this is the midpoint of the two electrodes. Consumed by LUNA,
        LuMamba and PanLUNA.

    ``channel_coords``
        Shape ``(batch, channels, 2, 3)``. Both electrodes of every channel kept
        separate, so a bipolar pair and a scalp-plus-reference channel stay
        distinguishable. Consumed by S-CEReBrO.

    Padding metadata describes how much of a sample is filler, which lets montages of
    different sizes share one batch. It is only meaningful on mapping-shaped batches;
    the tuple form accepted by :func:`as_signal_batch` cannot carry it.
    """

    input: Any
    label: Any
    channel_names: Any
    channel_locations: Any
    channel_coords: Any
    sensor_type: Any
    num_padded_channels: Any
    num_padded_timesteps: Any
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class BatchRequirements:
    """Metadata fields required by a particular model adapter.

    Every field defaults to ``False``, so a model only declares what it actually
    reads. ``channel_locations`` and ``channel_coords`` are the two electrode-geometry
    representations described on :class:`SignalBatch`; a model sets one of them.
    """

    label: bool = False
    channel_names: bool = False
    channel_locations: bool = False
    channel_coords: bool = False
    sensor_type: bool = False
    num_padded_channels: bool = False
    num_padded_timesteps: bool = False


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
        ("channel_coords", requirements.channel_coords),
        ("sensor_type", requirements.sensor_type),
        ("num_padded_channels", requirements.num_padded_channels),
        ("num_padded_timesteps", requirements.num_padded_timesteps),
    )
    missing = [name for name, required in required_fields if required and name not in batch]
    if missing:
        raise ValueError(f"Signal batch is missing required fields: {', '.join(missing)}")
    return batch

