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

This is why the registry entry declares only `channel_coords` and not `num_padded_channels`. Padding metadata is required by pre-training and irrelevant to fine-tuning, and both stages share one registry entry, so declaring it would force every future S-CEReBrO dataset to emit a field half the pipeline ignores. The pre-training task reads it with `batch.get`, which tolerates its absence.

### Preprocessing

Datasets are prepared into LMDB with [`make_datasets`](../../make_datasets/). Each entry is a pickled dictionary with `eeg`, `channel_coords`, and optionally `label` and `subject_id`. Electrode coordinates come from [`make_datasets/electrode_positions.py`](../../make_datasets/electrode_positions.py), which takes its angles from the BESA electrode and surface location tables and assigns fixed coordinates to reference electrodes that have no scalp position of their own. BESA defines theta (its "azimuth") as the angle from Cz, positive on the right, and phi (its "latitude") as the angle from T7 or T8, for which the Cartesian position is x = sin θ cos φ, y = sin θ sin φ, z = cos θ. The conversion here instead treats theta as a rotation about the vertical axis and phi as elevation, exactly as the reference implementation that produced the released checkpoint does, so the positions are not anatomical: Cz lands on the equator, and Fz, FCz, AFz and Fpz coincide, as do Pz, CPz, POz and Oz. The checkpoint's channel embeddings were learned on these values, so they must not be corrected without pre-training again.

Neonate windows are labelled from the seconds they contain: a window is a seizure if any of its seconds is, and is skipped if any second lacks annotator consensus. The reference implementation gave window *k* the label of second *k* although the window spans seconds 5*k* to 5*k* + 5, so neonate results are not directly comparable with those produced by its preprocessing.

Every script reads the raw corpus from `--input_dir` and writes to `--output_dir`; neither has a default. The configs read the results from `$DATA_PATH` laid out as below, and `dataset_root=` for a fine-tuning corpus or `pretraining_root=` for all pre-training corpora points them anywhere else.

| Corpus | Steps, as `python -m make_datasets.<script>` | Default location |
| --- | --- | --- |
| TUAB | `make_tuab` | `$DATA_PATH/finetuning/TUAB` |
| CHB-MIT | `make_chb_mit_pickles` on the raw EDFs, then `make_chb_mit` on its output | `$DATA_PATH/finetuning/CHB-MIT` |
| Neonate | `make_neonate` | `$DATA_PATH/finetuning/Neonate` |
| PhysioNet-MI | `make_physionet_mi` | `$DATA_PATH/finetuning/PhysioNet-MI` |
| SHU-MI | `make_shu_mi` | `$DATA_PATH/finetuning/SHU-MI` |
| STEW | `make_stew` | `$DATA_PATH/finetuning/STEW` |
| Mumtaz | `make_mumtaz` | `$DATA_PATH/finetuning/Mumtaz` |
| MentalArithmetic | `make_mental_arithmetic` | `$DATA_PATH/finetuning/MentalArithmetic` |
| SEED-V | `make_seed_v` | `$DATA_PATH/finetuning/SEED-V` |
| ISRUC | `unrar_isruc` on the downloaded archives, then `make_isruc` | `$DATA_PATH/finetuning/ISRUC` |
| SEED-VIG | `make_seed_vig`, which also takes `--labels_dir` | `$DATA_PATH/finetuning/SEED-VIG` |
| TUEG | `make_tueg`, which also takes `--tuab_dir` | `$DATA_PATH/pretraining/TUEG` |
| SEED, SEED-IV | `make_seed`, once per corpus | `$DATA_PATH/pretraining/SEED`, `.../SEED-IV` |
| SEED-FRA, SEED-GER | `make_seed_fra_ger`, once per corpus | `$DATA_PATH/pretraining/SEED-FRA`, `.../SEED-GER` |
| BOAS, GWD, SleepEDFx, BCI-NER | `make_boas`, `make_gwd`, `make_sleepedfx`, `make_bci_ner` | `$DATA_PATH/pretraining/<corpus>` |

`make_chb_mit_pickles` reads EDFs with `pyedflib`, as the reference does. `unrar_isruc` needs the `rarfile` package and an extraction tool such as `unrar`; extracting the archives with any such tool gives the same layout.

```bash
python -m make_datasets.make_tuab \
  --input_dir /path/to/raw/tuab --output_dir $DATA_PATH/finetuning/TUAB

python -m make_datasets.make_chb_mit_pickles \
  --input_dir /path/to/raw/chb-mit --output_dir /path/to/chb-mit-pickles
python -m make_datasets.make_chb_mit \
  --input_dir /path/to/chb-mit-pickles --output_dir $DATA_PATH/finetuning/CHB-MIT

python -m make_datasets.unrar_isruc --input_dir /path/to/raw/isruc
python -m make_datasets.make_isruc \
  --input_dir /path/to/raw/isruc --output_dir $DATA_PATH/finetuning/ISRUC

# make_tueg holds out every subject found in the raw TUAB corpus so they cannot leak
# into the TUAB fine-tuning splits. Pass --allow_tuab_overlap only when TUAB is not
# among the downstream datasets.
python -m make_datasets.make_tueg \
  --input_dir /path/to/raw/tueg --output_dir $DATA_PATH/pretraining/TUEG \
  --tuab_dir /path/to/raw/tuab
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

A window does not necessarily include the query itself. `include_self` places an offset of zero in the window, but the shift is applied afterwards, so self-attention survives only when the shift is a multiple of the dilation. With the shipped schedules that holds for the first pair alone, and four of the six blocks do not attend to themselves. The published weights were trained this way; it is the behaviour of the method, not a defect to correct.

Window extents are computed from the channel count the encoder was built for, and positions are clamped at the edges. A montage zero-padded to `max_channels` during pre-training therefore sees different spatial neighbours near the edge than the same montage run unpadded at its own size during fine-tuning.

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
| Mumtaz | depression | 19 | 5 s | 2 |
| MentalArithmetic | mental arithmetic | 20 | 5 s | 2 |
| SEED-V | emotion | 62 | 4 s | 5 |
| ISRUC | sleep staging | 6 | 30 s x 20 epochs | 5 |
| SEED-VIG | vigilance regression | 17 | 8 s | continuous |

### Model Size

S-CEReBrO is released in a single size of 2.4M parameters, built by
[`SCEReBrO_tiny`](../../config/model/SCEReBrO_tiny.yaml): `embed_dim` 180, `depth` 6,
`num_heads` 5.

### Training Setup

Pre-train on the union of the prepared corpora:

```bash
python -u run_train.py +experiment=SCEReBrO_pretrain
```

Fine-tune on TUAB, the default corpus:

```bash
python -u run_train.py +experiment=SCEReBrO_finetune \
  pretrained_safetensors_path=/absolute/path/to/S_CEReBrO.safetensors
```

Fine-tune on another corpus by selecting it. Each file in [`config/dataset`](../../config/dataset/) owns everything that varies per corpus: its path, whether samples are windows or sequences, whether labels are classes or a continuous target, the channel count, and the matching prediction head, task and criterion.

```bash
python -u run_train.py +experiment=SCEReBrO_finetune dataset=chb-mit
python -u run_train.py +experiment=SCEReBrO_finetune dataset=isruc
python -u run_train.py +experiment=SCEReBrO_finetune dataset=seed-vig
```

| `dataset=` | Task | Channels | Window | Classes | Head |
| --- | --- | --- | --- | --- | --- |
| `tuab` | binary classification | 22 | 10 s | 2 | `MlpClassificationHead` |
| `chb-mit` | seizure detection | 16 | 10 s | 2 | `MlpClassificationHead` |
| `neonate` | seizure detection | 18 | 5 s | 2 | `MlpClassificationHead` |
| `physionet-mi` | motor imagery | 64 | 4 s | 4 | `MlpClassificationHead` |
| `shu-mi` | motor imagery | 32 | 4 s | 2 | `MlpClassificationHead` |
| `stew` | workload | 14 | 4 s | 3 | `MlpClassificationHead` |
| `mumtaz` | depression | 19 | 5 s | 2 | `MlpClassificationHead` |
| `mental-arithmetic` | mental arithmetic | 20 | 5 s | 2 | `MlpClassificationHead` |
| `seed-v` | emotion | 62 | 4 s | 5 | `MlpClassificationHead` |
| `isruc` | sleep staging | 6 | 30 s x 20 epochs | 5 | `SequenceClassificationHead` |
| `seed-vig` | vigilance regression | 17 | 8 s | continuous | `MlpRegressionHead` |

Individual values can still be overridden on top of a selection, for a montage that differs from the prepared one:

```bash
python -u run_train.py +experiment=SCEReBrO_finetune dataset=tuab model.num_channels=19
```

**Where each setting lives.** The channel count is `model.num_channels`, and it must equal what the dataset yields; the encoder raises rather than reshaping if they disagree. The window length is not configured at all for window classification with the default mean pooling, because the task derives the patch count from the data and the encoder slices its position table to match; the only limit is `max_timesteps / patch_size`, which is 30 s, and exceeding it raises. `model_head.num_patches` is read only by `SequenceClassificationHead` and by `MlpClassificationHead` when `pooling_method` is `flatten`. The task type is not a separate flag: selecting a dataset selects the head, task and criterion together.

**Adding a corpus.** Copy the closest file in `config/dataset/`, set its path, channel count and label details, and leave the head, task and criterion selections alone unless the task type differs. Do not set these values in the experiment: Hydra applies a config's own values after its defaults list, so a key set in both places resolves to the experiment's copy and the dataset file is silently ignored. A contract test enforces this.

On PyTorch 2.6 and newer, the final validation and test passes fail when reloading a
checkpoint with `UnpicklingError: Weights only load failed`, because `torch.load` now
defaults to `weights_only=True` and every task stores its Hydra configuration in the
checkpoint. This affects `run_train.py` for all model families, not only this one.
Either export `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` or pass `final_validate=False
final_test=False`.

A linear-probe style run freezes the encoder blocks while leaving tokenisation and the embeddings trainable:

```bash
python -u run_train.py +experiment=SCEReBrO_finetune task.freeze_backbone=True
```

Fine-tuning uses layer-wise learning-rate decay: blocks closer to the input receive `lr * decay ** (depth - 1 - block_idx)`. Biases, normalisation weights, and the embedding tables are excluded from weight decay, and the head forms its own parameter group.


### Pretrained Weights

The [PulpBio/S-CEReBrO Hugging Face repository](https://huggingface.co/PulpBio/S-CEReBrO) provides the released checkpoint as `S_CEReBrO.safetensors`, with the Lightning `S_CEReBrO.ckpt` alongside it. It is the 2.4M-parameter encoder built by [`SCEReBrO_tiny`](../../config/model/SCEReBrO_tiny.yaml). The weights are licensed under CC BY-ND 4.0.

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

The checkpoint path can then be written relative to `pretrained_root`:

```bash
python -u run_train.py +experiment=SCEReBrO_finetune \
  'pretrained_safetensors_path=${pretrained_root}/S_CEReBrO.safetensors'
```

Single-quote the override so the shell does not expand `${...}` before Hydra sees it.
`pretrained_root` defaults to `${env:CHECKPOINT_DIR}/pretrained/S-CEReBrO` and can be
pointed elsewhere with `pretrained_root=/some/other/dir`. An absolute
`pretrained_safetensors_path` still works exactly as it does for the other families.

Shape-mismatched tensors are skipped rather than forced, so a checkpoint from a
differently configured encoder loads without error. Check the
`[load:model] loaded=... shape_mismatch=... unexpected=...` line the loader prints: on a
matching checkpoint `shape_mismatch` and `unexpected` are both zero and `loaded` equals
`total_target`.

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
