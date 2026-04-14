# BioFoundation

<p align="center">
  <a href="https://arxiv.org/abs/2502.06438">
    <img src="https://img.shields.io/badge/arXiv-2502.06438-b31b1b.svg" alt="FEMBA Paper">
  </a>
  <a href="https://arxiv.org/abs/2510.22257">
    <img src="https://img.shields.io/badge/arXiv-2510.22257-b31b1b.svg" alt="LUNA Paper">
  </a>
  <a href="https://arxiv.org/abs/2512.15729">
    <img src="https://img.shields.io/badge/arXiv-2512.15729-b31b1b.svg" alt="TinyMyo Paper">
  </a>
  <a href="https://arxiv.org/abs/2603.19100">
    <img src="https://img.shields.io/badge/arXiv-2603.19100-b31b1b.svg" alt="LuMamba Paper">
  </a>
  <a href="https://huggingface.co/PulpBio/FEMBA">
    <img src="https://img.shields.io/badge/HuggingFace-FEMBA-%23ffcc4d?logo=huggingface&logoColor=black" alt="Hugging Face: FEMBA">
  </a>
  <a href="https://huggingface.co/PulpBio/LUNA">
    <img src="https://img.shields.io/badge/HuggingFace-LUNA-%23ffcc4d?logo=huggingface&logoColor=black" alt="Hugging Face: LUNA">
  </a>
  <a href="https://huggingface.co/PulpBio/TinyMyo">
    <img src="https://img.shields.io/badge/HuggingFace-TinyMyo-%23ffcc4d?logo=huggingface&logoColor=black" alt="Hugging Face: TinyMyo">
  </a>
  <a href="https://huggingface.co/PulpBio/LuMamba">
    <img src="https://img.shields.io/badge/HuggingFace-LuMamba-%23ffcc4d?logo=huggingface&logoColor=black" alt="Hugging Face: LuMamba">
  </a>
  <a href="https://github.com/pulp-bio/BioFoundation">
    <img src="https://img.shields.io/github/stars/pulp-bio/BioFoundation?style=social" alt="GitHub Stars">
  </a>
</p>

Copyright (C) 2025 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file for details.

Authors: Thorir Mar Ingolfsson, Anna Tegon, Berkay Döner, Xiaying Wang, Matteo Fasulo, Danaé Broustail, Yawei Li & Luca Benini.

## About

**BioFoundation** is a flexible and extensible codebase for deep learning with biological signals. This repository is designed to support a variety of research projects, and currently hosts the work of multiple papers on EEG analysis.

This repository is built on PyTorch Lightning and Hydra to enable reproducible and scalable research.

## 🤗 Pretrained Weights on Hugging Face

Looking for ready-to-use weights of models? We host them on Hugging Face:

### Currently available

- **FEMBA** ([paper](https://arxiv.org/abs/2502.06438)) [![HF Model Card](https://img.shields.io/badge/Model%20Card-FEMBA-ffcc4d?logo=huggingface&logoColor=black)](https://huggingface.co/PulpBio/FEMBA)
- **LUNA** ([paper](https://arxiv.org/abs/2510.22257)) [![HF Model Card](https://img.shields.io/badge/Model%20Card-LUNA-ffcc4d?logo=huggingface&logoColor=black)](https://huggingface.co/PulpBio/LUNA)
- **TinyMyo** ([paper](https://arxiv.org/abs/2512.15729)) [![HF Model Card](https://img.shields.io/badge/Model%20Card-TinyMyo-ffcc4d?logo=huggingface&logoColor=black)](https://huggingface.co/PulpBio/TinyMyo)
- **LuMamba** ([paper](https://arxiv.org/abs/2603.19100)) [![HF Model Card](https://img.shields.io/badge/Model%20Card-LuMamba-ffcc4d?logo=huggingface&logoColor=black)](https://huggingface.co/PulpBio/LuMamba)

#### Why FEMBA?

- **Scales to long EEG** with linear-time Mamba (no quadratic attention).
- **Strong results** on TUAB/TUAR/TUSL with ready task-specific checkpoints.
- **Simple fine-tune path:** set `CHECKPOINT_DIR`, run `+experiment=FEMBA_finetune`.

**➡️ Model hub:** <https://huggingface.co/PulpBio/FEMBA>
**📄 Model card:** [FEMBA on Hugging Face](https://huggingface.co/PulpBio/FEMBA) — benchmarks, protocols, and efficiency notes.
**📜 Weights license:** CC BY-ND 4.0 (use + redistribute **unmodified** weights with attribution; no redistribution of **modified** weights)
**🧑‍🍳 PR-gated improvements:** If you fine-tune internally and want your variant to become an **official** FEMBA release, open a PR with configs, logs, and evals. We’ll review together; if it looks good, we’ll retrain/validate and publish an **official** FEMBA checkpoint.

**What you’ll find on the hub**

- `TUAB/` → abnormal EEG (base/large)
- `TUAR/` → artifact detection (tiny/base/large)
- `TUSL/` → slowing classification (variants as in the paper)

Quick download with `huggingface_hub`:

```bash
pip install huggingface_hub
```

```python
from huggingface_hub import snapshot_download

# downloads all task folders (TUAB/TUAR/TUSL) and safetensors into ./checkpoints/FEMBA
snapshot_download(repo_id="PulpBio/FEMBA", repo_type="model", local_dir="checkpoints/FEMBA")
```

Use the paths directly in your runs, e.g.:

```bash
export DATA_PATH=/path/to/data
export CHECKPOINT_DIR=checkpoints/FEMBA/TUAR/base.safetensors
python -u run_train.py +experiment=FEMBA_finetune
```

#### Why LUNA?

- **Topology-agnostic** EEG via **query-based channel unification** (consistent latent across arbitrary montages).
- **Linear-in-channels** compute & memory (unifies channels **before** temporal modeling; no quadratic spatio-temporal attention).
- **Pretrained on >21k hours** (TUEG + Siena) with masked-patch reconstruction; strong transfer across datasets/montages.
- **Simple fine-tune path:** pick model size with `LUNA_{base,large,huge}.yaml`, set `pretrained_safetensors_path`, run `+experiment=LUNA_finetune`.

**➡️ Model hub:** <https://huggingface.co/PulpBio/LUNA>
**📄 Model card:** [LUNA on Hugging Face](https://huggingface.co/PulpBio/LUNA) — variants, configs, and fine-tuning walkthrough.
**📜 Weights license:** CC BY-ND 4.0 (use + redistribute **unmodified** weights with attribution; no redistribution of **modified** weights)
**🧑‍🍳 PR-gated improvements:** If you fine-tune internally and want your variant to become an **official** LUNA release, open a PR with configs, logs, and evals. We’ll review; if it looks good, we’ll retrain/validate and publish an **official** LUNA checkpoint.

**What you’ll find on the hub**

- `Base/`, `Large/`, `Huge/` → LUNA size variants (matching `config/model/LUNA_{base,large,huge}.yaml`)
- Task-specific heads/checkpoints for common TUH downstream tasks (TUAB / TUAR / TUSL)

Quick download with `huggingface_hub`:

```bash
pip install huggingface_hub
```

```python
from huggingface_hub import snapshot_download

# downloads LUNA folders and .safetensors into ./checkpoints/LUNA
snapshot_download(repo_id="PulpBio/LUNA", repo_type="model", local_dir="checkpoints/LUNA")
```

Use the paths directly in your runs like here below:

```bash
python -u run_train.py +experiment=LUNA_finetune /model=LUNA_base \
  pretrained_safetensors_path=/absolute/path/to/checkpoints/LUNA/Base/LUNA_base.safetensors

python -u run_train.py +experiment=LUNA_finetune /model=LUNA_large \
  pretrained_safetensors_path=/absolute/path/to/checkpoints/LUNA/Large/LUNA_large.safetensors

python -u run_train.py +experiment=LUNA_finetune /model=LUNA_huge \
  pretrained_safetensors_path=/absolute/path/to/checkpoints/LUNA/Huge/LUNA_huge.safetensors
```

*If your checkpoint path contains spaces, wrap it in quotes.*

Tips:

- TUH datasets (TUAB/TUAR/TUSL): keep `- override /data_module: finetune_data_module` and set `data_module.*.hdf5_file` to your `{train,val,test}.h5`.
- Non-TUH (e.g., SEED-V): use `- override /data_module: subject_independent_data_module` and remove the TUH-specific `data_module` block.
- Match task settings: `classification_type` (`bc`, `mc`, `mmc`, `mcc`) and `model.num_classes` (e.g., TUSL=4, TUAB=2).

#### Why TinyMyo?

- **Ultra-lightweight**: only 3.6M parameters, suitable for microcontroller deployment.
- **Broad generalization**: pretrained on multiple large-scale EMG datasets for versatility across tasks and sensor configurations.
- **Strong results** on surface EMG tasks with ready task-specific checkpoints.

**➡️ Model hub:** <https://huggingface.co/PulpBio/TinyMyo>
**📄 Model card:** [TinyMyo on Hugging Face](https://huggingface.co/PulpBio/TinyMyo) — benchmarks, protocols, and efficiency notes.
**📜 Weights license:** CC BY-ND 4.0 (use + redistribute **unmodified** weights with attribution; no redistribution of **modified** weights)
**🧑‍🍳 PR-gated improvements:** If you fine-tune internally and want your variant to become an **official** TinyMyo release, open a PR with configs, logs, and evals. We’ll review together; if it looks good, we’ll retrain/validate and publish an **official** TinyMyo checkpoint.

**What you’ll find on the hub**

- `DB5/` → gesture classification
- `UCI_EMG/` → gesture classification
- `EPN612/` → gesture classification

> The scripts to download and preprocess the datasets are available at: <https://huggingface.co/PulpBio/TinyMyo/tree/main/scripts>

Quick download with `huggingface_hub`:

```bash
pip install huggingface_hub
```

```python
from huggingface_hub import snapshot_download

# downloads all task folders (DB5/UCI_EMG/EPN612) and safetensors into ./checkpoints/TinyMyo
snapshot_download(repo_id="MatteoFasulo/TinyMyo", repo_type="model", local_dir="checkpoints/TinyMyo")
```

Use the paths directly in your runs, e.g.:

```bash
export DATA_PATH=/path/to/data
export CHECKPOINT_DIR=checkpoints/TinyMyo/UCI_EMG/base.safetensors
python -u run_train.py +experiment=TinyMyo_finetune \
  pretrained_safetensors_path=/path/to/model.safetensors
```

**What you won’t find on the hub**

- **Silent Speech**
  - Codebase: [MatteoFasulo/silent_speech](https://github.com/MatteoFasulo/silent_speech)
- **Generic Neuromotor Interface**
  - Codebase: [MatteoFasulo/generic-neuromotor-interface](https://github.com/MatteoFasulo/generic-neuromotor-interface)

#### Why LuMamba?

- **Merging topology-invariant** EEG representations with **Mamba-based** temporal modeling to jointly achieve **linear-time efficiency** and **channel invariance**.
- **Pretrained on >21k hours** (TUEG) with **LeJEPA** (Balestriero and LeCun, 2025), a recent joint-embedding predictive approach, adapted to biosignals in this repository to enhance cross-montage robustness. 

**➡️ Model hub:** <https://huggingface.co/PulpBio/LuMamba>
**📄 Model card:** [LuMamba on Hugging Face](https://huggingface.co/PulpBio/LuMamba) — variants, configs, and fine-tuning walkthrough.
**📜 Weights license:** CC BY-ND 4.0 (use + redistribute **unmodified** weights with attribution; no redistribution of **modified** weights)
**🧑‍🍳 PR-gated improvements:** If you fine-tune internally and want your variant to become an **official** LuMamba release, open a PR with configs, logs, and evals. We’ll review; if it looks good, we’ll retrain/validate and publish an **official** LuMamba checkpoint.

**What you’ll find on the hub**

- `LeJEPA-only`, `Reconstruction-only`, `Mixed LeJEPA-Reconstruction` pre-trained checkpoints → Pre-training design variants.
- Instructions to get started on fine-tuning experiments.

Quick download with `huggingface_hub`:

```bash
pip install huggingface_hub
```

```python
from huggingface_hub import snapshot_download

# downloads all pre-trained variants and safetensors into ./checkpoints/LuMamba
snapshot_download(repo_id="PulpBio/LuMamba", repo_type="model", local_dir="checkpoints/LuMamba")
```
Include the safetensors checkpoint path as input and run fine-tuning in the commandline:
```bash
python -u run_train.py +experiment=LuMamba_finetune \
  pretrained_safetensors_path=/absolute/path/to/checkpoints/LuMamba/LuMamba.safetensors
```

Tips:
- Data preprocessing scripts are provided in `/make_datasets` for various downstream datasets. See `make_datasets/README.md` for instructions on getting started. 
- Adapt configuration file `config/experiment/LuMamba_finetune.yaml` to your specific task with correct dataset paths, classification type (regression and multi-class classification `mcc`, binary `bc` and change `model.num_classes` accordingly), I/O settings, trainer parameters, etc.
  - Ensure `data_module:train/test/val` are initialized with the compatible dataset class.
  - Configuration file includes sufficient `#CHANGEME` tags and further instructions for a working example.

## Features

- **Modular Design**: The repository is organized into modules for data loading, models, training tasks, and more, making it easy to extend and adapt for new research projects.
- **Flexible Configuration**: We use [Hydra](https://hydra.cc/docs/intro/) to manage experiment configurations, allowing for easy customization of models, data, and training parameters.
- **Reproducibility**: Our use of `Hydra` and PyTorch Lightning helps ensure that our experiments are reproducible.
- **Extensible**: The repository is designed to be easily extended with new datasets, models, and tasks.

## Installation

To use BioFoundation, clone the repository and install the required dependencies.

```bash
git clone https://github.com/pulp-bio/BioFoundation.git
```

We recommend using a virtual environment to manage dependencies. You can use `conda` or `virtualenv` for this purpose. We have provided a `requirements.txt` file that lists the necessary packages. You can install them using pip, and optionally, you can use `conda` to create a new environment.

```bash
conda create -n BioFoundation
conda activate BioFoundation
pip install -r requirements.txt
```

alternatively, you can create a venv using [`uv`](https://docs.astral.sh/uv/) and then install via `pyproject.toml` with:

```bash
uv pip install -r pyproject.toml --torch-backend=auto
```

where `--torch-backend=auto` will query for installed CUDA driver, AMD GPU versions, and Intel GPU presence and then use the most-compatible PyTorch index for all the torch-related packages. Have a look at [uv docs](https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection) for more details.

### Path changes

Throughout the repository, you may find paths that need to be adjusted based on your local setup. For example, the path to the datasets in the configuration files or the scripts that process the datasets. Make sure to update these paths accordingly. They have been named "#CHANGEME" to facilitate finding them.

## Dataset Preparation

The datasets used in this repository should be converted to HDF5 for efficient I/O. Other formats can work, but you’d need to adapt the dataloaders accordingly.
To prepare the TUH EEG datasets (see the [official source](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/index.shtml)), follow these steps:

1. **Download raw data** from the official sources (e.g., TUH EEG corpus).
2. **Preprocess to pickles** (windowing/labels):

   ```bash
   # examples (adjust paths)
   python make_datasets/process_raw_eeg.py tuab --root_dir /eeg_data/TUAB/edf --output_dir /processed_eeg
   python make_datasets/process_raw_eeg.py tusl --root_dir /eeg_data/TUSL/edf --output_dir /processed_eeg
   python make_datasets/process_raw_eeg.py tuar --root_dir /eeg_data/TUAR/edf --output_dir /processed_eeg
   ```

3. **Bundle into HDF5:**: Use the provided script to process the raw data into HDF5 files.

    ```bash
    # all datasets found under /processed_eeg
    python make_datasets/make_hdf5.py --prepath /processed_eeg --dataset All --remove_pkl

    # or a single dataset
    python make_datasets/make_hdf5.py --prepath /processed_eeg --dataset TUSL --remove_pkl
    ```

    You may need to edit the `prepath` variable in the script to point to the directory where you have downloaded the raw data.
4. **Update Configs**: so `data_module.*.hdf5_file` points to your `${DATA_PATH}/<DATASET>_data/{train,val,test}.h5`

## How to Run

### Pre-training

To run a pre-training experiment, you can use the `run_train.py` script with the appropriate configuration file. For example in the case of pre-training FEMBA:

```bash
python -u run_train.py +experiment=FEMBA_pretrain

```

### Fine-tuning

To run a fine-tuning experiment, you can use the `run_train.py` script with the appropriate configuration file. For example in the case of fine-tuning FEMBA:

```bash
python -u run_train.py +experiment=FEMBA_finetune

```

> **Tip:** Pretrained FEMBA weights (TUAB/TUAR/TUSL folders) are available on 🤗 Hugging Face:
> <https://huggingface.co/PulpBio/FEMBA>
> Set `CHECKPOINT_DIR` to the desired `.safetensors` (e.g., `.../TUAR/base.safetensors`) before launching.

Note in both cases one needs to make sure that the dataset that specific experiment is using is downloaded and available in the correct path.

## Repository Structure

```
BioFoundation/
├── config                   # Hydra configuration files
├── criterion                # Loss functions
├── data_module              # PyTorch Lightning DataModules
├── datasets                 # PyTorch Datasets
├── docs                     # Detailed documentation
├── models                   # Model implementations
├── schedulers               # Learning rate schedulers
├── tasks                    # PyTorch Lightning tasks
└── ...
```

## Contributing

We welcome contributions to BioFoundation! If you have a new model, dataset, or task that you would like to add, please follow the guidelines below.

### How to add a new dataset?

1. Add the code of the dataset to [`datasets`](datasets/).
2. Add the configuration file of the dataset to [`./config/dataset`](./config/dataset/).
3. If the dataset is large, consider adding a script to download it in the [`./scripts`](./scripts) directory. Make sure to document how to run the script in the README.

### How to add a new data module?

1. Add the code of the data module to [`./data_module`](./data_module).
2. Add the configuration file of the data module to [`./config/data_module`](./config/data_module).
3. If the data module requires specific datasets, make sure to document how to download and prepare them in the README.

### How to add a new loss function?

1. Add the code of the loss function to [`./criterion`](./criterion).
2. Add the configuration file of the loss function to [`./config/criterion`](./config/criterion).

### How to add a new task?

1. Add the code of the task to [`./tasks`](./tasks).
2. Add the configuration file of the task to [`./config/task`](./config/task).
3. If the task requires specific datasets or models, make sure to document how to download and prepare them in the README.

### How to add a new scheduler?

1. Add the code of the scheduler to [`./schedulers`](./schedulers).
2. Add the configuration file of the scheduler to [`./config/scheduler`](./config/scheduler).
3. If the scheduler requires specific models or tasks, make sure to document how to use it in the README.

### How to add a new model?

1. Add the code of the model to [`./models`](./models).
2. Add the configuration file of the model to [`./config/model`](./config/model).

### How to start a new experiment with the added model?

1. Add experiment configuration file to [`./config/experiment`](./config/experiment).
    If you are interested, you may check the [Hydra document about it](https://hydra.cc/docs/patterns/configuring_experiments/).
2. Override the default configurations in the added experiment configuration file.
3. Run the experiment with the command:

```bash
python -u run_train.py +experiment=your_experiment_name
```

### Contributing improvements to FEMBA weights

We’re excited to see what you build. Because the weights are **CC BY-ND 4.0**, redistribution of **modified** weights (e.g., LoRA/adapters, deltas, pruned or quantized variants) is **not permitted**.
If you fine-tune internally and believe your results should become an **official** FEMBA release, please open a PR with:

- exact **configs**, **seeds**, and **training scripts**,
- **environment** and **hardware** details,
- **evaluation protocol** (TUAB/TUAR/TUSL), **splits**, and full **metrics** (AUROC/AUPR/BA, FLOPs, memory),
- training and validation **logs**.

Maintainers will review; if accepted, we will retrain/validate and publish a new **official** checkpoint on 🤗 under the same license.

## General Tips

### How to use distributed data parallel?

In your experiment configuration file, add the following arguments

```yaml
trainer:
  accelerator: gpu  # Using GPU
  num_nodes: ${num_nodes}  # The number of computing nodes
  devices: -1  # Automatically uses all GPUs available
  strategy: ddp  # distributed data parallel
```

### How to save GPU memory?

1. Try fairscale checkpointing first. Check [here](https://fairscale.readthedocs.io/en/stable/api/nn/checkpoint/checkpoint_activations.html) and [here](https://github.com/ofsoundof/GRL-Image-Restoration/blob/main/models/networks/grl.py#L134)
2. Use sharded training. Check [here](https://lightning.ai/docs/pytorch/stable/advanced/model_parallel.html).

## Contact

For questions and support, please open an issue on the GitHub repository.

## Citing this Work

If you find this work useful, please cite the respective papers:

```bibtex
@misc{tegon2025fembaefficientscalableeeg,
      title={FEMBA: Efficient and Scalable EEG Analysis with a Bidirectional Mamba Foundation Model},
      author={Anna Tegon and Thorir Mar Ingolfsson and Xiaying Wang and Luca Benini and Yawei Li},
      year={2025},
      eprint={2502.06438},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.06438},
}
@inproceedings{doner2025luna,
      title={{LUNA}: Efficient and Topology-Agnostic Foundation Model for {EEG} Signal Analysis},
      author={Berkay D{\"o}ner and Thorir Mar Ingolfsson and Luca Benini and Yawei Li},
      booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
      year={2025},
      url={https://openreview.net/forum?id=uazfjnFL0G}
}

@misc{fasulo2025tinymyotinyfoundationmodel,
      title={TinyMyo: a Tiny Foundation Model for Flexible EMG Signal Processing at the Edge},
      author={Matteo Fasulo and Giusy Spacone and Thorir Mar Ingolfsson and Yawei Li and Luca Benini and Andrea Cossettini},
      year={2025},
      eprint={2512.15729},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2512.15729},
}
@misc{broustail2026lumambalatentunifiedmamba,
      title={LuMamba: Latent Unified Mamba for Electrode Topology-Invariant and Efficient EEG Modeling}, 
      author={Danaé Broustail and Anna Tegon and Thorir Mar Ingolfsson and Yawei Li and Luca Benini},
      year={2026},
      eprint={2603.19100},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2603.19100}, 
}
```

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.

**Note on model weights:** Pretrained weights are hosted at <https://huggingface.co/PulpBio/FEMBA>, <https://huggingface.co/PulpBio/LUNA>, <https://huggingface.co/PulpBio/TinyMyo>, and <https://huggingface.co/PulpBio/LuMamba> and licensed under **CC BY-ND 4.0**. You may use and redistribute the **unmodified** weights with attribution. Redistribution of **modified** weights is not permitted. To upstream improvements, please open a PR; accepted changes will be released as **official** checkpoints.
