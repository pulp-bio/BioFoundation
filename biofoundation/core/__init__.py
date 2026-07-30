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

"""Stable contracts shared by model-specific implementations."""

from biofoundation.core.batch import BatchRequirements, SignalBatch, as_signal_batch, require_batch_fields
from biofoundation.core.checkpoints import SafetensorsCheckpointMixin, split_state_dict_by_prefix
from biofoundation.core.protocols import PredictionHead, SignalEncoder

__all__ = [
    "BatchRequirements",
    "PredictionHead",
    "SafetensorsCheckpointMixin",
    "SignalBatch",
    "SignalEncoder",
    "as_signal_batch",
    "require_batch_fields",
    "split_state_dict_by_prefix",
]

