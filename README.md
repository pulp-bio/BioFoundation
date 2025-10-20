# BioFoundation

Copyright (C) 2025 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file for details.

Authors: Thorir Mar Ingolfsson, Anna Tegon, Berkay Döner, Xiaying Wang, Yawei Li & Luca Benini.

## About

**BioFoundation** is a flexible and extensible codebase for deep learning with biological signals. This repository is designed to support a variety of research projects, and currently hosts the work of multiple papers on EEG analysis.

This repository is built on PyTorch Lightning and Hydra to enable reproducible and scalable research.

## Features

* **Modular Design**: The repository is organized into modules for data loading, models, training tasks, and more, making it easy to extend and adapt for new research projects.
* **Flexible Configuration**: We use [Hydra](https://hydra.cc/docs/intro/) to manage experiment configurations, allowing for easy customization of models, data, and training parameters.
* **Reproducibility**: Our use of `hydra` and PyTorch Lightning helps ensure that our experiments are reproducible.
* **Extensible**: The repository is designed to be easily extended with new datasets, models, and tasks.

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

### Path changes
Throughout the repository, you may find paths that need to be adjusted based on your local setup. For example, the path to the datasets in the configuration files or the scripts that process the datasets. Make sure to update these paths accordingly. They have been named "#CHANGEME" to faciliate finding them.

## Dataset Preparation

The datasets used in this repository need to be downloaded and processed into the HDF5 format that the dataloaders expect. Other dataformats can be supported, but then the dataloaders need to be modified accordingly. For our experiments we used the HDF5 format. The following steps outline how to prepare the datasets:

1.  **Download Raw Data**: Download the raw TUH EEG datasets (TUEG, TUAB, TUSL, TUAR) from their official sources.
2.  **Process Data**: Use the provided script to process the raw data into HDF5 files.
    ```bash
    python make_datasets/make_hdf5.py
    ```
    You may need to edit the `prepath` variable in the script to point to the directory where you have downloaded the raw data.
3.  **Update Configs**: Make sure the paths to the generated `.h5` files are correctly specified in the relevant data module configuration files (e.g., `config/data_module/pretrain_data_module.yaml`).

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
    If you are interested, you may check the [hydra document about it](https://hydra.cc/docs/patterns/configuring_experiments/).
2. Override the default configurations in the added experiment configuration file.
3. Run the experiment with the command:
```bash
python -u run_train.py +experiment=your_experiment_name
```
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
@inproceedings{döner2025luna,
  title={{LUNA}: Efficient and Topology-Agnostic Foundation Model for {EEG} Signal Analysis},
  author={Berkay Döner and Thorir Mar Ingolfsson and Luca Benini and Yawei Li},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=uazfjnFL0G}
  }
```

## License
This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.
