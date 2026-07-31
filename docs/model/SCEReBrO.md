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

### Channel Counts

The published encoders are pre-trained at 64 channels, and one checkpoint fine-tunes onto any montage from 1 to `max_channels` without modification. Nothing in the state dict depends on the channel count:

- the temporal position table is sized by `max_timesteps // patch_size` and sliced to the patches present, and is shared across channels;
- the channel embedding is an MLP over 3D electrode coordinates, so it has no per-channel parameters and generalises to montages it never saw;
- the attention blocks take the channel count only as a reshape argument.

Set `model.num_channels` to the montage you are fine-tuning on and leave `max_channels` at the value the checkpoint was pre-trained with:

```bash
python -u run_train.py +experiment=SCEReBrO_finetune model.num_channels=6
```

The encoder validates this rather than guessing: passing input whose channel count differs from `model.num_channels` raises rather than silently reshaping. Set it to match the dataset.

Two channel counts coexist and mean different things. `model.num_channels` is how many channels the encoder is built for and must equal what the dataset yields. `model.max_channels` is the capacity the positional table was sized at, and must stay at the pre-training value or the checkpoint will not match.

For pre-training, corpora with fewer channels are zero-padded up to `max_channels` by `LMDBDataset`, and the padded channels are replaced by a learned pad token, excluded from masking, and masked out of attention. Fine-tuning does not pad: the encoder is simply built at the montage's own size.

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

### Smoke Test With Synthetic Data

To check the pipeline end to end without prepared data, generate synthetic corpora in
the exact on-disk formats the readers expect. The signals are band-limited noise and
the labels are random, so this verifies that a run works, not that it learns anything.

```bash
export DATA_PATH=/absolute/path/to/dummy-data
export CHECKPOINT_DIR=/absolute/path/to/experiments

python -m make_datasets.make_dummy_scerebro_dataset --output $DATA_PATH \
  --pretrain-samples 24 --finetune-samples 32
```

Add `--datasets pretrain tuab isruc seed-vig` to also generate the sequence and
regression corpora.

Pre-train, then fine-tune from the resulting checkpoint:

```bash
python -u run_train.py +experiment=SCEReBrO_pretrain \
  trainer.accelerator=cpu trainer.devices=1 trainer.strategy=auto \
  trainer.max_epochs=1 trainer.accumulate_grad_batches=1 \
  trainer.check_val_every_n_epoch=1 scheduler.warmup_epochs=0 \
  batch_size=4 num_workers=0 final_validate=False

python -u run_train.py +experiment=SCEReBrO_finetune \
  pretrained_checkpoint_path=$CHECKPOINT_DIR/checkpoints/SCEReBrO_pretrain/<run>/last.ckpt \
  trainer.accelerator=cpu trainer.devices=1 trainer.strategy=auto \
  trainer.max_epochs=2 scheduler.warmup_epochs=0 \
  batch_size=4 num_workers=0
```

Drop the `trainer.*` and `num_workers` overrides on a GPU machine; they exist only to
make the run finish quickly on CPU.

On PyTorch 2.6 and newer, reloading a checkpoint for the final validation and test
passes fails with `UnpicklingError: Weights only load failed`, because `torch.load`
now defaults to `weights_only=True` and every task stores its Hydra configuration in
the checkpoint. This affects `run_train.py` for all model families, not only this one.
Either export `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` or pass `final_validate=False
final_test=False`.

### Pretrained Weights

The [PulpBio/S-CEReBrO Hugging Face repository](https://huggingface.co/PulpBio/S-CEReBrO) provides tiny, small and base checkpoints matching the model configs in [`config/model`](../../config/model/). The weights are licensed under CC BY-ND 4.0.

`snapshot_download` writes to whatever `local_dir` you give it, resolved relative to the
current working directory when it is not absolute. The fine-tuning experiment expects
the release under `$CHECKPOINT_DIR/pretrained/S-CEReBrO`, which is the default value of
`pretrained_root`, so download it there:

```python
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="PulpBio/S-CEReBrO",
    local_dir=os.path.join(os.environ["CHECKPOINT_DIR"], "pretrained", "S-CEReBrO"),
)
```

The checkpoint path can then be written as an interpolation instead of an absolute
path. `model_size` is declared by the selected `config/model` group, so it always
matches the encoder being built and the two cannot drift:

```bash
python -u run_train.py +experiment=SCEReBrO_finetune model=SCEReBrO_tiny \
  'pretrained_safetensors_path=${pretrained_root}/SCEReBrO_${model_size}.safetensors'
```

Switching size needs one change, and the checkpoint follows:

```bash
python -u run_train.py +experiment=SCEReBrO_finetune model=SCEReBrO_base \
  'pretrained_safetensors_path=${pretrained_root}/SCEReBrO_${model_size}.safetensors'
```

Single-quote the override so the shell does not expand `${...}` before Hydra sees it.
`pretrained_root` defaults to `${env:CHECKPOINT_DIR}/pretrained/S-CEReBrO` and can be
pointed elsewhere with `pretrained_root=/some/other/dir`. An absolute
`pretrained_safetensors_path` still works exactly as it does for the other families.

The size flag and the checkpoint must agree: loading a `base` checkpoint into a `tiny`
encoder is not an error, because shape-mismatched tensors are skipped rather than
forced. Check the `[load:model] loaded=... shape_mismatch=... unexpected=...` line the
loader prints; on a correct pairing `shape_mismatch` and `unexpected` are both zero and
`loaded` equals `total_target`.

A Lightning `.ckpt` produced by a local pre-training run is passed with
`pretrained_checkpoint_path` instead:

```bash
python -u run_train.py +experiment=SCEReBrO_finetune \
  pretrained_checkpoint_path=$CHECKPOINT_DIR/checkpoints/SCEReBrO_pretrain/<run>/last.ckpt
```

Convert one to safetensors for distribution with the repository's own tool:

```bash
python util/ckpt_to_safetensor.py \
  --ckpt_path $CHECKPOINT_DIR/checkpoints/SCEReBrO_pretrain/<run>/last.ckpt \
  --safetensor_path SCEReBrO_tiny.safetensors
```

Only the encoder needs to transfer. Head weights are loaded from a checkpoint only when
`include_head=True` is requested, so a reconstruction head from pre-training never
overwrites a freshly initialised classification head.
