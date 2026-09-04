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

import torch
import torch.nn as nn
from timm.layers import trunc_normal_


class MlpRegressionHead(nn.Module):
    """Scalar regression head over the encoder's token embeddings.

    An optional token-wise pooler is applied, tokens are mean-pooled, and a linear
    layer produces one scalar per input window. When ``bounded_output`` is set the
    prediction passes through a sigmoid, which suits targets defined on ``[0, 1]``
    such as SEED-VIG PERCLOS; disable it for unbounded targets.
    """

    def __init__(
        self,
        embed_dim: int = 200,
        dropout: float = 0.1,
        pooling: bool = True,
        bounded_output: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.pooling = pooling
        self.bounded_output = bounded_output

        if self.pooling:
            self.pooler = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
                nn.Dropout(dropout),
                nn.Tanh(),
            )

        layers = [nn.Linear(embed_dim, 1)]
        if bounded_output:
            layers.append(nn.Sigmoid())
        self.regressor = nn.Sequential(*layers)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict one scalar per input window.

        Args:
            x: Token embeddings of shape ``(batch, num_tokens, embed_dim)``.

        Returns:
            Predictions of shape ``(batch,)``.
        """
        if self.pooling:
            x = self.pooler(x)

        return self.regressor(x.mean(dim=1)).squeeze(-1)
