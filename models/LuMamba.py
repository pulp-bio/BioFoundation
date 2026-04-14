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
#* Author:  Danaé Broustail                                                        *
#* Author:  Thorir Mar Ingolfsson                                             *
#*----------------------------------------------------------------------------*
from models.FEMBA import MambaWrapper, PatchEmbed, MambaClassifier, Decoder
from models.LUNA import *
import torch.nn as nn
import torch
from typing import Tuple

class BasicLinearClassifier(nn.Module):
    def __init__(self, embed_dim, grid_size, num_classes):
        super(BasicLinearClassifier, self).__init__()
        self.grid_size = grid_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.fc1 = nn.Linear(embed_dim * grid_size[0], num_classes)
        self.activation1 = nn.GELU()

    def forward(self, x):
        # Input x shape: (B, T=grid_size[1], D=embed_dim * grid_size[0])
        x=x.permute(0,2,1) 
        x = x.mean(dim=-1)  # Temporal Pooling -> output: (B, embed_dim * grid_size[0])

        # First linear layer: embed_size * grid_size[0] -> num_classes
        x = self.fc1(x)
        x = self.activation1(x)

        return x

class LuMamba(LUNA):
    def __init__(self, 
                 # ---- LUNA parameters
                 patch_size=40, num_queries=4,
                 embed_dim=64, num_heads=2,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 drop_path=0.0,
                # ----FEMBA parameters
                exp: int = 2, 
                num_blocks: int= 2,
                bidirectional: bool = True,
                bidirectional_strategy: str = "add", # or "ew_multiply"
                # ---- shared parameters
                num_classes: int = 0,
                # ----- classification parameters
                mamba_classifier: bool = False,
                classifier_option: str = None, # None defaults to LUNA classifier, other options: "mamba" or "linear"
                classification_type: str = "bc",
                classification_num_channels: int = 22): # placeholder value, inconsequential for "mcc", "bc" and regression tasks

        # Initialize LUNA with its params
        super().__init__(
            patch_size=patch_size,
            num_queries=num_queries,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            drop_path=drop_path,
            num_classes=num_classes,
        ) 
        # FEMBA parameters
        self.exp = exp
        self.num_blocks = num_blocks

        # FEMBA components
        self.d_model = self.embed_dim * self.num_queries  # d_model = Q * E
        self.mamba_blocks = nn.ModuleList([
            MambaWrapper(d_model=self.d_model, bidirectional=bidirectional,
                          bidirectional_strategy=bidirectional_strategy,
                          expand=self.exp)
            for _ in range(self.num_blocks)
        ])

        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(self.d_model)
            for _ in range(self.num_blocks)
        ])

        # Classifier head
        if mamba_classifier == True and self.num_classes > 0:
            self.classifier = MambaClassifier(embed_dim=self.embed_dim,
                                              grid_size=[self.num_queries, None],
                                              num_classes=self.num_classes,
                                              num_channels=classification_num_channels,
                                              classification_type=classification_type)

        if classifier_option is not None and self.num_classes > 0:
            if classifier_option == "mamba":
                self.classifier = MambaClassifier(embed_dim=self.embed_dim,
                                                  grid_size=[self.num_queries, None],
                                                  num_classes=self.num_classes,
                                                  num_channels=classification_num_channels,
                                                  classification_type=classification_type)
            elif classifier_option == "linear":
                self.classifier = BasicLinearClassifier(embed_dim=self.embed_dim,
                                                        grid_size=[self.num_queries, None],
                                                        num_classes=self.num_classes)

        # deleting unused LUNA components
        del self.blocks
        del self.norm

    def get_attention_maps(self, x_signal, channel_locations, mask=None):
        x, _ = self.prepare_tokens(x_signal, channel_locations, mask=mask)        
        _, attention_scores = self.cross_attn(x) # (B*num_patches, Q, D)
        return attention_scores

    def encode(self, x_signal, channel_locations):
        """
        Compute encoder representations (B, S, d_model)
        """

        B, C, T = x_signal.shape

        # --------------------------------------------------------
        # 1. Embedding
        # --------------------------------------------------------
        x, channel_locations_emb = self.prepare_tokens(
            x_signal,
            channel_locations,
            mask=None
        )
        # --------------------------------------------------------
        # 2. Channel unification (cross attention)
        # --------------------------------------------------------
        x, _ = self.cross_attn(x)

        # reshape
        x = rearrange(x, '(B t) Q D -> B t (Q D)', B=B)

        # --------------------------------------------------------
        # 3. Mamba stack (block-by-block checking)
        # --------------------------------------------------------
        for idx, (mamba_block, norm_layer) in enumerate(zip(self.mamba_blocks, self.norm_layers)):
            res = x
            x = norm_layer(x)
            x = mamba_block(x)
            x = x + res

        return x

    def forward(self, x_signal, mask, channel_locations, channel_names=None):
        x_original = x_signal # (B, C, T)
        B, C, T = x_signal.shape  
        # Embedding: x: (B*S, C, E), channel_locations_emb: (B*S, C, E) 
        x, channel_locations_emb = self.prepare_tokens(x_signal, channel_locations, mask=mask)  

        # Channel unification: (B*S, C, E) -> (B*S, Q, E)
        x, attention_scores = self.cross_attn(x) 
        x = rearrange(x, '(B t) Q D -> B t (Q D)', B=B) # (B, S, Q*E)

        # replace it by Mamba blocks (B, S, Q*E) = (B, L, d_model) where L is length of sequence and d_model = Q*E
        for mamba_block, norm_layer in zip(self.mamba_blocks, self.norm_layers):
            res = x
            x = norm_layer(x)     
            x = mamba_block(x)
            x = res + x           

        # LUNA classifier: from latent representation to classes
        if self.num_classes > 0:
            x_classified = self.classifier(x) # Final: (B, S, Q*E) -> (B, num_classes)
            return x_classified, x_original      
        # LUNA reconstruction decoder
        else:
            # Input: channel_names (B, C) - same indices repeated B times
            # self.channel_emb.embeddings: Embedding table of shape (num_unique_channels, E) ~ (100, E)
            # Each channel gets a learned embedding that encodes channel-specific information
            num_patches = x.shape[1] # S
            channel_emb = self.channel_emb(channel_names) # (B, C) -> (B, C, E)
            channel_emb = channel_emb.repeat(num_patches, 1, 1) # repeated for every patch, (B, C, E) -> (B*S, C, E)
            decoder_queries = channel_locations_emb + channel_emb # (B*S, C, E)
            x_reconstructed = self.decoder_head(x, decoder_queries) # Final: (B, C, T)

            return x_reconstructed, x_original, attention_scores
