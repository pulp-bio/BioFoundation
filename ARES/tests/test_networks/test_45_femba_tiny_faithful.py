# SPDX-License-Identifier: Apache-2.0

"""
Test 45: FEMBA Tiny INT8 -- faithful dataflow

Same dimensions as test_25 (FEMBATinyInt8):
  - Input: (1, 1, 22, 1280), patch (2,16), stride (2,16), embed_dim 35
  - d_model 385, expand 4 (d_inner 1540), d_state 16, d_conv 4
  - 2 Bi-Mamba blocks, global average pool, QuantLinear head (2 classes)

Difference to test_25: ARES's extractor builds the layer order from
``named_modules()`` registration order and only knows about residual adds that
are explicit ``QuantAdd`` modules. test_25 registers its blocks in three
ModuleLists (all Mamba blocks, then all norms, then all post-norm quants) and
uses an implicit ``res + x``, so the extracted graph runs both Mamba blocks
back-to-back and drops the residuals.

Here every module is registered in execution order, one attribute per op, and
the residual is an explicit QuantAdd. ARES's MambaWrapper kernel otherwise adds
the block input as an implicit residual (out = x + fwd + rev), which the Brevitas
QuantMambaWrapper does not do; each wrapper therefore sets
``ares_internal_residual = False`` so the residual is counted exactly once:

  input_quant -> patch_embed -> (+ pos_embed) -> scale_equalizer
  for b in 0..1:
      blk{b}_mamba   : Bi-Mamba (forward + reversed, outputs added)
      blk{b}_add     : QuantAdd(residual, mamba_out)
      blk{b}_norm    : LayerNorm(d_model)
      blk{b}_postq   : QuantIdentity
  global_pool -> pre_classifier_quant -> classifier
"""

import math
import torch
import torch.nn as nn
from brevitas import nn as qnn

from .brevitas_custom_layers import QuantPatchEmbed, QuantMambaWrapper, QuantAdd


def _val(t):
    return t.value if hasattr(t, 'value') else t


class FEMBATinyFaithful(nn.Module):
    """FEMBA-Tiny INT8 with modules registered in execution order and explicit residual adds."""

    def __init__(
        self,
        inp_size=(22, 1280),
        patch_size=(2, 16),
        stride=(2, 16),
        in_chans=1,
        embed_dim=35,
        expand=4,
        d_state=16,
        d_conv=4,
        num_blocks=2,
        num_classes=2,
    ):
        super().__init__()
        bit_width = 8
        self.num_blocks = num_blocks

        H, W = inp_size
        self.grid_h = (H - patch_size[0]) // stride[0] + 1
        self.grid_w = (W - patch_size[1]) // stride[1] + 1
        self.seq_len = self.grid_w
        self.d_model = self.grid_h * embed_dim
        self.d_inner = expand * self.d_model
        self.dt_rank = math.ceil(self.d_model / 16)

        # --- Registration order == execution order (ARES derives layer order from it) ---
        self.input_quant = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True)
        self.patch_embed = QuantPatchEmbed(
            inp_size=inp_size, patch_size=patch_size, stride=stride,
            in_chans=in_chans, embed_dim=embed_dim,
            bit_width=bit_width, return_quant_tensor=True,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, self.d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_quant = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True)
        self.scale_equalizer = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        for b in range(num_blocks):
            setattr(self, f'blk{b}_mamba', QuantMambaWrapper(
                d_model=self.d_model, d_inner=self.d_inner, d_state=d_state,
                conv_kernel=d_conv, bidirectional_strategy="add",
                bit_width=bit_width, return_quant_tensor=True,
            ))
            # The residual is the explicit QuantAdd below, so tell ARES not to add
            # its implicit input residual inside the MambaWrapper kernel.
            getattr(self, f'blk{b}_mamba').ares_internal_residual = False
            setattr(self, f'blk{b}_add', QuantAdd(bit_width=bit_width, return_quant_tensor=True))
            setattr(self, f'blk{b}_norm', nn.LayerNorm(self.d_model))
            setattr(self, f'blk{b}_postq', qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True))

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.pre_classifier_quant = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True)
        self.classifier = qnn.QuantLinear(
            self.d_model, num_classes, bias=True,
            weight_bit_width=bit_width, return_quant_tensor=False,
        )

    def forward(self, x):
        x = _val(self.input_quant(x))
        x = self.patch_embed(x)

        # Positional embedding (same scheme as test_25 so ARES's pos_embed path is reused)
        pos = self.pos_quant(self.pos_embed)
        x = self.scale_equalizer(_val(x))
        pos = self.scale_equalizer(_val(pos))
        x = self.scale_equalizer(_val(x) + _val(pos))

        for b in range(self.num_blocks):
            res = x
            y = getattr(self, f'blk{b}_mamba')(x)
            x = getattr(self, f'blk{b}_add')(res, y)          # residual: x + Mamba(x)
            x = getattr(self, f'blk{b}_norm')(_val(x))        # post-norm (FEMBA)
            x = getattr(self, f'blk{b}_postq')(x)

        x = _val(x).transpose(1, 2)            # [B, D, L]
        x = self.global_pool(x).squeeze(-1)    # [B, D]
        x = self.pre_classifier_quant(x)
        return self.classifier(x)


def get_sample_input(batch_size=1, in_chans=1, inp_size=(22, 1280)):
    return torch.randn(batch_size, in_chans, inp_size[0], inp_size[1])


if __name__ == "__main__":
    m = FEMBATinyFaithful().eval()
    with torch.no_grad():
        print(m(get_sample_input()).shape)
