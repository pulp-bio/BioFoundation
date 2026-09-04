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


class PatchReconstructionHead(nn.Module):
    """Linear decoder mapping each token embedding back to its waveform patch.

    This is the SimMIM-style decoder used for pre-training: every token is projected
    independently, with no decoder transformer and no re-ordering of the sequence.
    """

    def __init__(self, embed_dim: int = 200, patch_size: int = 200):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.decoder_pred = nn.Linear(embed_dim, patch_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct waveform patches.

        Args:
            x: Token embeddings of shape ``(batch, num_tokens, embed_dim)``.

        Returns:
            Reconstructed patches of shape ``(batch, num_tokens, patch_size)``.
        """
        return self.decoder_pred(x)
