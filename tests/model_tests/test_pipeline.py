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
#* Author:  Glenn Anta Bucagu                                                 *
#* Author:  BioFoundation Contributors                                        *
#*                                                                            *
#* Imported from the S-CEReBrO reference implementation (TimeFM).             *
#*----------------------------------------------------------------------------*

import torch
from omegaconf import OmegaConf

from models.s_cerebro import SCerebroEncoder
from models.model_heads.mlp_classification_head import MlpClassificationHead
from models.model_heads.sequence_classification_head import SequenceClassificationHead
from models.modules.patching import patchify, unpatchify
from tasks.mae_pretraining import MaskedAutoencoderPretrainingTask


def build_encoder(**overrides):
    """Construct a small encoder for shape and transfer tests."""
    kwargs = dict(
        patch_size=200, num_channels=8, embed_dim=32, depth=4, num_heads=4,
        max_channels=64, max_timesteps=6000, window_size_spatial=5, window_size_temporal=5,
    )
    kwargs.update(overrides)
    return SCerebroEncoder(**kwargs)


def test_patchify_roundtrip():
    """Patching then unpatching returns the original waveform."""
    signal = torch.randn(2, 5, 1200)
    patches = patchify(signal, patch_size=200)
    assert patches.shape == (2, 5 * 6, 200)
    torch.testing.assert_close(unpatchify(patches, num_channels=5), signal)


def test_tokeniser_requires_the_supported_patch_size():
    """The tokeniser accepts 200-sample patches and rejects any other size."""
    encoder = build_encoder(patch_size=200)
    x = torch.randn(1, 8, 4, 200)
    coords = torch.randn(1, 8, 2, 3)
    assert encoder(x, coords).shape == (1, 8 * 4, 32)

    for patch_size in (64, 128):
        try:
            build_encoder(patch_size=patch_size)
        except ValueError as error:
            assert "patch_size" in str(error)
        else:
            raise AssertionError(f"expected ValueError for patch_size={patch_size}")


def test_encoder_accepts_fewer_channels_and_patches_than_capacity():
    """A 6000-timestep position table serves a shorter window without reshaping."""
    encoder = build_encoder(num_channels=3, max_channels=64, max_timesteps=6000)
    x = torch.randn(2, 3, 4, 200)
    coords = torch.randn(2, 3, 2, 3)
    assert encoder(x, coords).shape == (2, 12, 32)


def test_encoder_rejects_more_patches_than_capacity():
    """Exceeding the position table raises instead of silently truncating."""
    encoder = build_encoder(num_channels=2, max_timesteps=800)
    x = torch.randn(1, 2, 8, 200)
    coords = torch.randn(1, 2, 2, 3)
    try:
        encoder(x, coords)
    except ValueError as error:
        assert "positional embedding capacity" in str(error)
    else:
        raise AssertionError("expected ValueError for too many patches")


def test_pretrained_encoder_transfers_to_a_smaller_montage():
    """Every encoder tensor loads into a model built for fewer channels."""
    pretrained = build_encoder(num_channels=64)
    finetuned = build_encoder(num_channels=8)

    source = pretrained.state_dict()
    target = finetuned.state_dict()
    mismatched = [k for k in source if k not in target or source[k].shape != target[k].shape]
    assert mismatched == []


def make_pretraining_task(masking_ratio=0.5, num_channels=8):
    """Build a pre-training task from an in-memory config."""
    cfg = OmegaConf.create({
        "model": {
            "_target_": "models.s_cerebro.SCerebroEncoder",
            "patch_size": 200, "num_channels": num_channels, "embed_dim": 32,
            "depth": 4, "num_heads": 4, "max_channels": 64, "max_timesteps": 6000,
        },
        "model_head": {
            "_target_": "models.model_heads.patch_reconstruction_head.PatchReconstructionHead",
            "embed_dim": 32, "patch_size": 200,
        },
        "criterion": {
            "_target_": "criterion.masked_reconstruction_loss.MaskedReconstructionLoss",
            "loss_type": "l2", "alpha": 0.1,
        },
    })
    return MaskedAutoencoderPretrainingTask(cfg, masking_ratio=masking_ratio)


def test_token_mask_marks_exactly_the_replaced_tokens():
    """The reported mask lines up with the tokens that were actually replaced.

    The reconstruction loss is taken over positions flagged by ``token_mask``, so a
    mask expressed in a different ordering than the tokens would score unmasked
    patches and silently weaken the pre-training objective.
    """
    torch.manual_seed(0)
    task = make_pretraining_task()
    tokens = torch.randn(4, 8 * 30, 32)

    masked, token_mask = task.mask_tokens(tokens.clone(), attn_mask=None)

    replaced = (masked == task.model.mask_token.detach()).all(dim=-1)
    assert torch.equal(replaced, token_mask)
    torch.testing.assert_close(masked[~token_mask], tokens[~token_mask])


def test_masking_ratio_is_respected_and_padding_is_never_masked():
    """Masking hits the requested fraction of real tokens and no padded token."""
    torch.manual_seed(0)
    channels, patches = 8, 30
    task = make_pretraining_task(masking_ratio=0.5, num_channels=channels)
    tokens = torch.randn(2, channels * patches, 32)

    attn_mask = torch.ones(2, channels * patches, dtype=torch.int)
    attn_mask.view(2, channels, patches)[:, 6:] = 0

    _, token_mask = task.mask_tokens(tokens, attn_mask=attn_mask)

    real = attn_mask.sum(dim=1)
    masked = token_mask.sum(dim=1)
    assert torch.equal(masked, (real * 0.5).long())
    assert not token_mask[attn_mask == 0].any()


def test_sequence_head_maps_epochs_to_per_epoch_logits():
    """The ISRUC head restores the sequence axis and returns one logit row per epoch."""
    sequence_length, channels, patches, dim, classes = 20, 6, 30, 32, 5
    head = SequenceClassificationHead(
        sequence_length=sequence_length, num_channels=channels, num_patches=patches,
        embed_dim=dim, num_classes=classes, hidden_dim=64, dim_feedforward=128,
    )
    encoded = torch.randn(2 * sequence_length, channels * patches, dim)
    assert head(encoded).shape == (2 * sequence_length, classes)


def test_classification_head_pooling_modes_agree_on_output_shape():
    """Mean and flatten pooling both produce one logit row per window."""
    channels, patches, dim, classes = 4, 5, 32, 3
    encoded = torch.randn(2, channels * patches, dim)
    for pooling_method in ("mean", "flatten"):
        head = MlpClassificationHead(
            embed_dim=dim, num_classes=classes, pooling_method=pooling_method,
            num_channels=channels, num_patches=patches,
        )
        assert head(encoded).shape == (2, classes)


def test_finetuning_leaves_no_trainable_parameter_without_a_gradient():
    """Every trainable parameter must receive a gradient from a fine-tuning step.

    DistributedDataParallel with find_unused_parameters=False waits for a gradient from
    each parameter it tracks. A trainable parameter the forward pass never touches
    therefore hangs the first training step of a multi-GPU run, with no error and no
    output. The mask and pad tokens are pre-training-only and are frozen for exactly
    this reason; this test fails if any other parameter joins them.
    """
    import torch

    from models.model_heads.mlp_classification_head import MlpClassificationHead
    from tasks.classification_task import freeze_pretraining_only_parameters

    encoder = build_encoder(num_channels=4, embed_dim=40, depth=2, num_heads=4)
    freeze_pretraining_only_parameters(encoder)
    head = MlpClassificationHead(embed_dim=40, num_classes=2, num_channels=4, num_patches=2)

    tokens = encoder(torch.randn(2, 4, 2, 200), channel_positions=torch.randn(2, 4, 2, 3))
    torch.nn.functional.cross_entropy(head(tokens), torch.tensor([0, 1])).backward()

    missing = [
        name
        for module in (encoder, head)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert missing == [], f"trainable parameters with no gradient: {missing}"
