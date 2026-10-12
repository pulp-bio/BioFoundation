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

"""Structural contracts for the encoder and prediction-head split.

The five original families bundle their output layer into the model, selecting it from
``num_classes``. Newer families separate an encoder that produces token embeddings from
a prediction head that consumes them, each from its own Hydra group; the protocols
below describe that second shape.

Conformance is structural, so the bundled families stay valid unchanged. The signatures
are concrete on purpose: a ``forward`` accepting ``*args, **kwargs`` would be satisfied
by every :class:`torch.nn.Module` and check nothing.

Tensors are annotated only under :data:`typing.TYPE_CHECKING`, because the fast contract
suite imports this module without PyTorch installed.
"""

from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import Tensor
else:  # pragma: no cover - runtime fallback keeps this module torch-free
    Tensor = Any


@runtime_checkable
class SignalEncoder(Protocol):
    """A model that turns a patched biosignal into contextualised token embeddings.

    The encoder owns tokenisation, positional and channel embeddings, and the
    backbone. It owns no task-specific output layer, so one pre-trained encoder can
    be paired with any :class:`PredictionHead` without being rebuilt.

    Implementations accept ``(batch, channels, patches, patch_size)`` and return
    ``(batch, channels * patches, embed_dim)``. Keyword arguments beyond the input
    carry whatever batch metadata the family requires, declared in its
    :class:`~biofoundation.core.batch.BatchRequirements`.
    """

    def forward(self, x: "Tensor", channel_positions: "Tensor") -> "Tensor":
        """Encode a patched biosignal into token embeddings.

        Args:
            x: ``(batch, channels, patches, patch_size)``.
            channel_positions: Electrode geometry, shaped as the family's
                :class:`~biofoundation.core.batch.BatchRequirements` declares.
        """
        ...


@runtime_checkable
class PredictionHead(Protocol):
    """A module that turns encoder token embeddings into a task prediction.

    Heads accept ``(batch, num_tokens, embed_dim)``. The output shape is the head's
    own concern: class logits, a scalar per window, or one reconstructed patch per
    token. A head never reads the raw waveform and never receives batch metadata, so
    swapping the task means swapping the head alone.
    """

    def forward(self, x: "Tensor") -> "Tensor":
        """Map ``(batch, num_tokens, embed_dim)`` token embeddings to a prediction."""
        ...


__all__ = ["PredictionHead", "SignalEncoder"]
