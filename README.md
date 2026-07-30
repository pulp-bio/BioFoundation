# BioFoundation

Copyright (C) 2025-2026 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See [LICENSE](LICENSE) for details.

Authors: Thorir Mar Ingolfsson, Anna Tegon, Berkay Döner, Xiaying Wang, Matteo Fasulo, Danaé Broustail, Marija Zelic, Yawei Li, and Luca Benini.

> **TL;DR:** Choose a model from the table below, install the training dependencies, set `DATA_PATH` and `CHECKPOINT_DIR`, then run `python -u run_train.py +experiment=<MODEL>_pretrain` or the matching fine-tuning experiment. Each model page links its Hugging Face weights and exact checkpoint command. ARES is separate and only needed for embedded deployment.

BioFoundation is a research and onboarding codebase for foundation models across EEG, sEMG, ECG, and PPG. It collects the model implementations, Hydra experiments, preprocessing tools, and pretrained releases behind six model families.

The training stack is built on PyTorch Lightning and Hydra. Embedded deployment through ARES is maintained as a separate toolchain inside the repository.

## Model Zoo

| Model | Signals | Architecture | Resources |
| --- | --- | --- | --- |
| [FEMBA](docs/model/FEMBA.md) | EEG | Bidirectional Mamba | [Paper](https://arxiv.org/abs/2502.06438) / [Hugging Face](https://huggingface.co/PulpBio/FEMBA) |
| [LUNA](docs/model/LUNA.md) | EEG | Query-unified Transformer | [Paper](https://arxiv.org/abs/2510.22257) / [Hugging Face](https://huggingface.co/PulpBio/LUNA) |
| [TinyMyo](docs/model/TinyMyo.md) | sEMG | Rotary Transformer | [Paper](https://arxiv.org/abs/2512.15729) / [Hugging Face](https://huggingface.co/PulpBio/TinyMyo) |
| [LuMamba](docs/model/LuMamba.md) | EEG | Query-unified Mamba | [Paper](https://arxiv.org/abs/2603.19100) / [Hugging Face](https://huggingface.co/PulpBio/LuMamba) |
| [PanLUNA](docs/model/PanLUNA.md) | EEG, ECG, PPG | Multimodal query-unified Transformer | [Paper](https://arxiv.org/abs/2604.04297) / [Hugging Face](https://huggingface.co/PulpBio/PanLUNA) |
| [S-CEReBrO](docs/model/SCEReBrO.md) | EEG | Windowed alternating-attention Transformer | [Paper](https://arxiv.org/abs/2607.03118) / [Hugging Face](https://huggingface.co/PulpBio/S-CEReBrO) |

The machine-readable [`model_registry.py`](biofoundation/model_registry.py) records the experiment names, papers, Hugging Face repositories, modalities, and batch metadata requirements for these families.

## Quick Start

BioFoundation requires Python 3.11 or newer. Create an isolated environment and install the training dependencies:

```bash
git clone https://github.com/pulp-bio/BioFoundation.git
cd BioFoundation
conda create -n biofoundation python=3.11
conda activate biofoundation
pip install -r requirements.txt
```

With `uv`, the equivalent setup is:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r pyproject.toml --torch-backend=auto
```

Set the prepared data and experiment-output roots:

```bash
export DATA_PATH=/absolute/path/to/data
export CHECKPOINT_DIR=/absolute/path/to/experiments
```

Start a pre-training experiment:

```bash
python -u run_train.py +experiment=FEMBA_pretrain
```

Or fine-tune a downloaded checkpoint:

```bash
python -u run_train.py +experiment=LUNA_finetune /model=LUNA_base \
  pretrained_safetensors_path=/absolute/path/to/LUNA_base.safetensors
```

Choose another `+experiment` from the model registry. Before a long run, review the selected file in [`config/experiment`](config/experiment/) and resolve its `#CHANGEME` values.

## Repository Map

| Path | Responsibility |
| --- | --- |
| [`biofoundation`](biofoundation/) | Shared batch, environment, and model metadata contracts. |
| [`models`](models/) | Foundation model implementations. |
| [`tasks`](tasks/) | Lightning pre-training, classification, and regression tasks. |
| [`datasets`](datasets/) | Dataset readers and sample contracts. |
| [`data_module`](data_module/) | Lightning data modules and loader composition. |
| [`config`](config/) | Hydra defaults, modules, and reproducible experiments. |
| [`docs/adr`](docs/adr/) | Architecture decision records for shared contracts. |
| [`make_datasets`](make_datasets/) | Raw-data preprocessing and HDF5 conversion. |
| [`criterion`](criterion/) | Training objectives. |
| [`tests`](tests/) | Fast repository and refactoring contracts. |
| [`ARES`](ARES/) | Independent GAP9 and Siracusa deployment toolchain. |

## Documentation

- [Documentation index](docs/README.md)
- [Training and batch contracts](docs/TRAINING.md)
- [Hydra configuration guide](config/README.md)
- [Dataset preparation](make_datasets/README.md)
- [Contribution guide](CONTRIBUTING.md)
- [Citations](docs/CITATIONS.md)
- [ARES deployment](ARES/README.md)

Each model page linked from the model zoo contains its input assumptions, architecture, results, Hugging Face download, and fine-tuning example.

## Development Checks

The fast suite checks model metadata, Hydra composition and targets, batch adapters, environment handling, documentation links, and Apache headers:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python -m compileall -q biofoundation run_train.py models tasks datasets data_module \
  criterion schedulers util make_datasets tests
```

Numerical changes to models, losses, or datasets should also be tested with representative CPU or GPU batches.

## Licensing and Support

The source code is licensed under Apache 2.0. Pretrained weights in the five PulpBio Hugging Face repositories are licensed under CC BY-ND 4.0; see the model cards for terms and checkpoint-specific details.

For questions and support, open an [issue](https://github.com/pulp-bio/BioFoundation/issues). For changes, start with [CONTRIBUTING.md](CONTRIBUTING.md).
