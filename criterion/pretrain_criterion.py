#*----------------------------------------------------------------------------*
#* Copyright (C) 2025 ETH Zurich, Switzerland                                 *
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
#* Author:  Anna Tegon                                                        *
#* Author:  Thorir Mar Ingolfsson                                             *
#*----------------------------------------------------------------------------*

import torch
from torch import nn
import torch.nn.functional as F

class PretrainCriterion(nn.Module):
    """
    Criterion module to compute masked reconstruction losses.

    Args:
        loss_type (str): Type of loss to compute. Options are:
                         'l1' - L1 loss (Mean Absolute Error)
                         'l2' - L2 loss (Mean Squared Error)
                         'smooth_l1' - Smooth L1 loss (Huber loss)
    """
    def __init__(self, loss_type):
        super(PretrainCriterion, self).__init__()
        if loss_type not in ['l1', 'l2', 'smooth_l1']:
            raise ValueError("Invalid loss_type. Choose 'l1', 'l2', or 'smooth_l1'.")
        self.loss_type = loss_type

    def forward(self, reconstructed, original, mask):
        """
        Calculate loss between reconstructed and original signals,
        separately for masked and unmasked elements.

        Args:
            reconstructed (torch.Tensor): The reconstructed output from the model.
            original (torch.Tensor): The original input signal.
            mask (torch.BoolTensor): Boolean mask indicating which elements are masked.

        Returns:
            tuple: (masked_loss, unmasked_loss)
        """
        if self.loss_type == 'l1':
            # Mean Absolute Error on masked and unmasked elements
            masked_loss = F.l1_loss(reconstructed[mask], original[mask], reduction='mean')
            unmasked_loss = F.l1_loss(reconstructed[~mask], original[~mask], reduction='mean')
        elif self.loss_type == 'l2':
            # Mean Squared Error on masked and unmasked elements
            masked_loss = F.mse_loss(reconstructed[mask], original[mask], reduction='mean')
            unmasked_loss = F.mse_loss(reconstructed[~mask], original[~mask], reduction='mean')
        elif self.loss_type == 'smooth_l1':
            # Smooth L1 (Huber) loss on masked and unmasked elements
            masked_loss = F.smooth_l1_loss(reconstructed[mask], original[mask], reduction='mean')
            unmasked_loss = F.smooth_l1_loss(reconstructed[~mask], original[~mask], reduction='mean')

        return masked_loss, unmasked_loss
