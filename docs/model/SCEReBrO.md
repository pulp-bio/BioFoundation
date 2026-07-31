Copyright (C) 2025-2026 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE at the repository root for details.

## S-CEReBrO

S-CEReBrO is a compact EEG encoder built on windowed alternating attention. It tokenises a recording at per-channel patch granularity and alternates attention between the channel axis and the time axis, restricting each pass to a dilated, shifted window. Attention cost per block is linear in the token count rather than quadratic, which is what allows a 64-channel, 30-second window to be modelled directly.

It is the first family in this repository to separate the encoder from its output layer. The encoder emits token embeddings and nothing else; a [prediction head](../../config/model_head/) turns those into a reconstruction, a class label, or a scalar. One pre-trained encoder therefore serves every downstream task without being rebuilt.

### Default Input Assumptions

| Property | Value |
| --- | --- |
| Signal | Scalp EEG |
| Sampling rate | 200 Hz |
| Patch size | 200 samples (1 s), fixed by the tokeniser |
| Channels | Up to `max_channels` (64 by default) |
| Window | Up to `max_timesteps` (6000 samples, 30 s) |
| Amplitude | Per-channel min-max scaled to `[-1, 1]` |

The encoder requires `channel_coords` of shape `(batch, channels, 2, 3)`: the 3D coordinates of **both** electrodes forming each channel. This is the second electrode-geometry representation in the repository and is deliberately distinct from the `channel_locations` used by LUNA, LuMamba and PanLUNA, which carry one midpoint per channel. See [ADR 0001](../adr/0001-two-electrode-geometry-representations.md) for why both exist and neither is derived from the other.

Keeping both electrodes separate is what lets a bipolar derivation and a scalp-electrode-plus-reference channel stay distinguishable, and it is why the channel embedding is a function of geometry rather than of a channel index. Montages with different channel counts and orderings share the same parameters.

### Preprocessing

Datasets are prepared into LMDB with [`make_datasets`](../../make_datasets/). Each entry is a pickled dictionary with `eeg`, `channel_coords`, and optionally `label` and `subject_id`. Electrode coordinates come from [`make_datasets/electrode_positions.py`](../../make_datasets/electrode_positions.py), which follows the BESA electrode and surface location tables and assigns fixed coordinates to reference electrodes that have no scalp position of their own.

```bash
python -m make_datasets.make_tuab   --output $DATA_PATH/finetuning/TUAB
python -m make_datasets.make_tueg   --output $DATA_PATH/pretraining/TUEG
```

### Architecture Overview

| Stage | Module |
| --- | --- |
| Tokenisation | `TemporalConvTokenizer`: three strided 1D convolutions per `(channel, patch)` pair, projected to `embed_dim` |
| Position | Learned temporal table, shared across channels, sliced to the patches actually present |
| Channel | Shared MLP over each electrode's 3D coordinate, halves concatenated |
| Backbone | `depth` pre-norm transformer blocks with alternating attention |
| Output | A separate `PredictionHead` |

Attention alternates by block index:

| Block | Attends over | Window |
| --- | --- | --- |
| even | channels, at a fixed patch position (spatial) | `window_size_spatial`, dilated by `dilation_cycle_spatial`, shifted by `shift_cycle_spatial` |
| odd | patch positions, within a fixed channel (temporal) | `window_size_temporal`, dilated by `dilation_cycle_temporal`, shifted by `shift_cycle_temporal` |

Dilation and shift schedules are indexed by spatial/temporal *pair*, so a spatial block and the temporal block after it share a schedule entry. Setting `use_axial_mode: True` runs all spatial blocks before all temporal blocks instead.

`attention_type` selects the mechanism: `windowed-alternating` is the published method; `alternating` (no windowing) and `full` (all tokens at once) are the ablation baselines.

Padded channels are replaced by a learned pad token, excluded from masking, and masked out of attention, so montages of different sizes share one batch safely.

### Self-Supervised Learning (SSL) Objective

SimMIM-style masked reconstruction. Waveforms are patched and embedded, a random subset of real tokens is replaced by a learned mask token, and the encoder sees the full sequence of visible and masked tokens in their original order. A linear decoder reconstructs every patch and the loss is taken over the masked, non-padded positions. `alpha` adds a weighted term over visible patches, which stabilises early training.

### Downstream Tasks

| Layout | Head | Used by |
| --- | --- | --- |
| Window classification | `MlpClassificationHead` | TUAB, CHB-MIT, Neonate, PhysioNet-MI, SHU-MI, STEW, Mumtaz, MentalArithmetic, SEED-V |
| Sequence classification | `SequenceClassificationHead` | ISRUC sleep staging |
| Scalar regression | `MlpRegressionHead` | SEED-VIG vigilance |

Prepared datasets and their shapes:

| Dataset | Task | Channels | Window | Classes |
| --- | --- | --- | --- | --- |
| TUAB | binary classification | 22 | 10 s | 2 |
| CHB-MIT | seizure detection | 16 | 10 s | 2 |
| Neonate | seizure detection | 18 | 5 s | 2 |
| PhysioNet-MI | motor imagery | 64 | 4 s | 4 |
| SHU-MI | motor imagery | 32 | 4 s | 2 |
| STEW | workload | 14 | 4 s | 3 |
| Mumtaz | depression | 20 | 5 s | 2 |
| MentalArithmetic | mental arithmetic | 20 | 5 s | 2 |
| SEED-V | emotion | 62 | 4 s | 5 |
| ISRUC | sleep staging | 6 | 30 s x 20 epochs | 5 |
| SEED-VIG | vigilance regression | 17 | 8 s | continuous |

### Model Variants

| Variant | `embed_dim` | `depth` | `num_heads` | Config |
| --- | --- | --- | --- | --- |
| tiny | 180 | 6 | 5 | [`SCEReBrO_tiny`](../../config/model/SCEReBrO_tiny.yaml) |
| small | 200 | 12 | 10 | [`SCEReBrO_small`](../../config/model/SCEReBrO_small.yaml) |
| base | 400 | 12 | 16 | [`SCEReBrO_base`](../../config/model/SCEReBrO_base.yaml) |

### Training Setup

Pre-train on the union of the prepared corpora:

```bash
python -u run_train.py +experiment=SCEReBrO_pretrain
```

Fine-tune on TUAB, the default corpus:

```bash
python -u run_train.py +experiment=SCEReBrO_finetune \
  pretrained_safetensors_path=/absolute/path/to/SCEReBrO_tiny.safetensors
```

Fine-tune on another prepared dataset by overriding the corpus and its montage:

```bash
python -u run_train.py +experiment=SCEReBrO_finetune \
  dataset_root=$DATA_PATH/finetuning/CHB-MIT \
  model.num_channels=16 model_head.num_classes=2 model_head.num_patches=10
```

Sleep staging and regression swap the head, task and criterion together:

```bash
# ISRUC, one label per 30 s epoch
python -u run_train.py +experiment=SCEReBrO_finetune \
  dataset_root=$DATA_PATH/finetuning/ISRUC dataset_kind=sequence \
  model.num_channels=6 model_head=sequence_classification_head

# SEED-VIG, continuous target
python -u run_train.py +experiment=SCEReBrO_finetune \
  dataset_root=$DATA_PATH/finetuning/SEED-VIG label_mode=regression \
  model.num_channels=17 model_head=mlp_regression_head \
  task=finetune_regression_task_SCEReBrO criterion=mse_criterion
```

A config group is selected on the command line without a leading slash
(`model_head=...`); the leading slash form is only used inside a defaults list in a
config file. Head parameters come from the selected head's own config, so override
them per dataset with `model_head.num_classes=4 model_head.num_patches=4` rather than
expecting the experiment to carry classification-specific keys.

A linear-probe style run freezes the encoder blocks while leaving tokenisation and the embeddings trainable:

```bash
python -u run_train.py +experiment=SCEReBrO_finetune task.freeze_backbone=True
```

Fine-tuning uses layer-wise learning-rate decay: blocks closer to the input receive `lr * decay ** (depth - 1 - block_idx)`. Biases, normalisation weights, and the embedding tables are excluded from weight decay, and the head forms its own parameter group.

### Pretrained Weights

Checkpoints are published at [PulpBio/S-CEReBrO](https://huggingface.co/PulpBio/S-CEReBrO). Encoder weights load independently of the head, and tensors whose shapes do not match the current model are skipped rather than forced, so an encoder pre-trained at 64 channels can seed a 22-channel fine-tune. The loader prints how many tensors were loaded, skipped, and unexpected, so a silent no-op load is visible in the logs.
