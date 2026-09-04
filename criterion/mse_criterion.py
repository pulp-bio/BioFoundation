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
from torch import nn


class MSELossWrapper(nn.Module):
    """Mean squared error loss for scalar regression targets."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction=reduction)

    def forward(self, pred: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute MSE against ``batch['label']``.

        Args:
            pred: Predictions broadcastable to the target shape.
            batch: Must contain ``label`` with the regression targets.

        Returns:
            Scalar loss.
        """
        target = batch["label"]
        return self.loss_fn(pred.reshape(target.shape).to(target.dtype), target)
