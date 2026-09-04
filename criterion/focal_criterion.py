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
#* Author:  BioFoundation Contributors                                        *
#*                                                                            *
#* Imported from the S-CEReBrO reference implementation (TimeFM).             *
#*----------------------------------------------------------------------------*

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn


class FocalLossWrapper(nn.Module):
    """Binary focal loss for strongly imbalanced classification.

    Down-weights well-classified examples so that rare positives dominate the
    gradient, which is the regime of seizure detection (CHB-MIT, Neonate). The loss
    is evaluated on the positive-class logit and is computed through
    ``logsigmoid`` so that confident predictions cannot produce infinities.

    Args:
        alpha: Weight of the positive class in ``[0, 1]``.
        gamma: Focusing exponent; larger values suppress easy examples more.
    """

    def __init__(self, alpha: float = 0.8, gamma: float = 0.7):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, pred: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute focal loss against ``batch['label']``.

        Args:
            pred: Logits of shape ``(batch, 2)`` or ``(batch,)``.
            batch: Must contain ``label`` with binary targets of shape ``(batch,)``.

        Returns:
            Scalar loss.
        """
        logits = pred[:, 1] if pred.dim() == 2 and pred.shape[1] == 2 else pred.reshape(-1)
        targets = batch["label"].reshape(-1).to(logits.dtype)

        prob = torch.sigmoid(logits)
        positive = -self.alpha * (1 - prob) ** self.gamma * targets * F.logsigmoid(logits)
        negative = -(1 - self.alpha) * prob ** self.gamma * (1 - targets) * F.logsigmoid(-logits)
        return (positive + negative).mean()
