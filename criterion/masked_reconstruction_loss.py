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

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class MaskedReconstructionLoss(nn.Module):
    """Reconstruction loss for masked EEG pre-training.

    The loss is averaged over masked, non-padded patches. When ``alpha`` is
    non-zero, the loss on visible non-padded patches is added with weight ``alpha``,
    which stabilises early training without letting the visible patches dominate.

    Args:
        loss_type: One of ``l1``, ``l2``, ``smooth_l1``.
        alpha: Weight of the visible-patch term; ``0`` disables it.
    """

    def __init__(self, loss_type: str = "l2", alpha: float = 0.1):
        super().__init__()
        if loss_type not in {"l1", "l2", "smooth_l1"}:
            raise ValueError(f"loss_type must be 'l1', 'l2' or 'smooth_l1', got '{loss_type}'")
        self.loss_type = loss_type
        self.alpha = alpha

    def _elementwise_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "l1":
            return F.l1_loss(pred, target, reduction="none")
        if self.loss_type == "l2":
            return F.mse_loss(pred, target, reduction="none")
        return F.smooth_l1_loss(pred, target, reduction="none")

    def forward(self, pred: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the masked reconstruction loss.

        Args:
            pred: Reconstructed patches of shape ``(batch, num_tokens, patch_size)``.
            batch: Must contain ``target`` with the same shape as ``pred``, ``token_mask``
                of shape ``(batch, num_tokens)`` where 1 marks a masked token, and
                ``attn_mask`` of shape ``(batch, num_tokens)`` where 1 marks a real token.

        Returns:
            Tuple of the scalar loss and a dictionary of values to log.
        """
        target = batch["target"]
        token_mask = batch["token_mask"].bool()
        attn_mask = batch.get("attn_mask")
        attn_mask = torch.ones_like(token_mask) if attn_mask is None else attn_mask.bool()

        per_patch = self._elementwise_loss(pred, target).mean(dim=-1)

        masked = per_patch[token_mask & attn_mask].mean()
        logs = {"masked_loss": masked.item()}

        if self.alpha == 0:
            return masked, logs

        visible = per_patch[(~token_mask) & attn_mask].mean()
        logs["visible_loss"] = visible.item()
        return masked + self.alpha * visible, logs
