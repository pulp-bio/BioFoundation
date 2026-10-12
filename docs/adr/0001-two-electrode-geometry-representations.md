Copyright (C) 2025-2026 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE at the repository root for details.

# ADR 0001: Two electrode-geometry representations

Status: accepted

## Context

Model families in this repository describe electrode geometry in two incompatible ways.

LUNA, LuMamba and PanLUNA consume `channel_locations` of shape `(batch, channels, 3)`: one 3D coordinate per channel. For a bipolar derivation such as `FP1-F7`, [`models/modules/channel_embeddings.py`](../../models/modules/channel_embeddings.py) resolves both electrodes and returns their midpoint. Channels that are not bipolar contribute their scalp position alone, with no representation of the reference.

S-CEReBrO consumes `channel_coords` of shape `(batch, channels, 2, 3)`: both electrodes of every channel, kept separate. Its channel embedding maps each electrode independently through a shared MLP and concatenates the halves, and it rejects input that does not carry exactly two electrodes per channel. The preprocessing in [`make_datasets/electrode_positions.py`](../../make_datasets/electrode_positions.py) assigns explicit coordinates to references that have no scalp position, so an average-reference or linked-ears channel is representable.

The two are not interchangeable. `channel_coords` is strictly richer: the midpoint is recoverable from it, but the pair is not recoverable from the midpoint, and the reference is not represented in `channel_locations` at all.

## Decision

Both representations are first-class, independent fields on `SignalBatch`. Neither is derived from the other at runtime, and no conversion is applied automatically.

A model declares which one it consumes through `BatchRequirements` in [`biofoundation/model_registry.py`](../../biofoundation/model_registry.py). A dataset produces whichever its target family requires. `require_batch_fields` validates the declared field and does not accept the other in its place.

## Consequences

The five existing families are unaffected. Their datasets, their geometry handling, and their published checkpoints are untouched, and the new field is unreachable from their code paths.

A dataset prepared for S-CEReBrO cannot be fed to LUNA, LuMamba or PanLUNA without an explicit conversion step, and vice versa. This is accepted. Making `channel_coords` canonical with `channel_locations` derived from it would have enabled that reuse, but it would have required changing how the existing families obtain geometry, which is not worth the risk to reproducibility of published results.

If cross-family dataset reuse becomes valuable later, the additive way to get it is a documented reduction helper that a dataset or task calls explicitly. That remains available and is not foreclosed by this decision. What is foreclosed is *implicit* conversion inside `require_batch_fields`, because a model that silently receives midpoints where it expected electrode pairs would train without error and produce quietly wrong geometry.

Adding a third representation is not acceptable without superseding this record. Two exist because two model families genuinely need different information; a third would mean the contract had stopped being designed.
