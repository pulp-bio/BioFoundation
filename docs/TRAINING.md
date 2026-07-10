Copyright (C) 2025-2026 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE at the repository root for details.

# Training Guide

BioFoundation uses Hydra for configuration and PyTorch Lightning for training. Run commands from the repository root.

## Environment

Set the prepared dataset root and experiment-output root before starting Hydra:

```bash
export DATA_PATH=/absolute/path/to/data
export CHECKPOINT_DIR=/absolute/path/to/experiments
```

The CLI validates both variables before Hydra composes a run. Repository modules can still be imported without them.

## Choosing an Experiment

Each published model has matching pre-training and fine-tuning files under `config/experiment/`:

```bash
python -u run_train.py +experiment=FEMBA_pretrain
python -u run_train.py +experiment=LUNA_finetune \
  pretrained_safetensors_path=/absolute/path/to/LUNA_base.safetensors
```

Hydra applies command-line overrides after experiment, module, and default configs. See [`../config/README.md`](../config/README.md) for the composition model. Search the selected experiment and data module for values that must be supplied locally:

```bash
rg -n '#CHANGEME' config/experiment config/data_module
```

## Batch Contract

Tasks normalize batches with `biofoundation.core.batch.as_signal_batch`. The canonical mapping supports:

| Field | Purpose |
| --- | --- |
| `input` | Signal tensor required by every task. |
| `label` | Supervised classification or regression target. |
| `channel_locations` | Required by topology-aware model families. |
| `channel_names` | Optional channel identity metadata. |
| `sensor_type` | Required by PanLUNA for EEG, ECG, and PPG identity. |
| `metadata` | Optional dataset-specific metadata. |

The model registry records the required metadata for each family.

## Distributed Training

Configure Lightning's trainer in the selected experiment:

```yaml
trainer:
  accelerator: gpu
  num_nodes: ${num_nodes}
  devices: -1
  strategy: ddp
```

Set `find_unused_parameters` in the experiment when a fine-tuning strategy leaves branches of the model inactive.

## Checkpoints

Every model page under [`model/`](model/) links to its PulpBio Hugging Face repository and shows the expected checkpoint override. `CHECKPOINT_DIR` remains the output root; pass input weights through `pretrained_safetensors_path` or `pretrained_checkpoint_path`.

## Memory and Runtime

- Reduce `batch_size`, segment duration, or model size first.
- Use mixed precision through Lightning where the target hardware supports it.
- Consider activation checkpointing or sharded training for large models only after establishing a correct single-device run.
- ARES deployment has a separate environment and workflow under [`../ARES`](../ARES/).

## Fast Validation

The fast checks validate repository contracts without model weights or datasets. Install `hydra-core` to include full experiment composition:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python -m compileall -q biofoundation run_train.py models tasks datasets data_module \
  criterion schedulers util make_datasets tests
```

Run representative CPU or GPU batches for any change to numerical model, loss, or data behavior.
