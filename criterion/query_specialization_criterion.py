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
#* Author:  Berkay Döner                                                      *
#* Author:  Thorir Mar Ingolfsson                                             *
#*----------------------------------------------------------------------------*

import torch
from torch import nn
import torch.nn.functional as F

class QuerySpecializationCriterion(nn.Module):
    def __init__(self, loss_type, loss_coeff=1.0):
        super(QuerySpecializationCriterion, self).__init__()
        if loss_type not in ['l1', 'l2', 'smooth_l1']:
            raise ValueError("Invalid loss_type. Choose 'l1', 'l2', or 'smooth_l1'.")
        self.loss_type = loss_type
        self.loss_coeff = loss_coeff

    def forward(self, attention_scores):
        B, Q, C = attention_scores.size() # B = batch size; Q = num queries; C = num channels
        query_similarity = torch.bmm(attention_scores, attention_scores.permute(0, 2, 1)) # (B, Q, Q)
        # Create a mask to zero out the diagonal elements
        mask = 1.0 - torch.eye(Q, device=attention_scores.device).unsqueeze(0) # Shape (1, Q, Q)
        # Zero out the diagonal elements of the similarity matrix
        off_diagonal_similarity = query_similarity * mask # mask broadcasts to (B, Q, Q)
        if self.loss_type == 'l1':
            loss = torch.mean(torch.abs(off_diagonal_similarity)) 
        elif self.loss_type == 'l2':
            loss = torch.mean(off_diagonal_similarity**2)
        elif self.loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(off_diagonal_similarity, torch.zeros_like(off_diagonal_similarity))
        return loss * self.loss_coeff
