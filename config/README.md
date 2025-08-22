Copyright (C) 2025 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file at the root of the repository for details.

# Configuration Management with Hydra

This project uses [Hydra](https://hydra.cc/) to manage all experiment configurations. Hydra is a powerful framework that allows for flexible and dynamic configuration from YAML files and the command line. This makes our research highly reproducible and easy to modify.

## Philosophy

Our configuration is built on a principle of **composition**. Instead of having one massive configuration file, we define defaults for smaller, logical units (like the model, dataset, and trainer) in separate files. An "experiment" file then simply composes these pieces and overrides the defaults as needed.

---

## Configuration Types

There are three main types of configuration files in this directory:

### 1. **Default Config (`defaults.yaml`)**
This is the base configuration for the entire project. It acts as the "default experiment" and defines the standard set of modules to be used, such as the default task, data module, and model. It's the foundation upon which all other configurations are built.

### 2. **Module Configs** (`/model`, `/task`, `/data_module`, etc.)
These directories contain YAML files that define the parameters for specific Python classes in our codebase. Each file typically includes:
* `_target_`: A path to the Python class to be instantiated (e.g., `tasks.finetune_task.FinetuneTask`).
* **Default Parameters**: The default arguments and hyperparameters for that class (e.g., learning rate, number of layers).

For example, `config/model/FEMBA_finetune.yaml` defines the default parameters for the `FEMBA` model when used for fine-tuning.

### 3. **Experiment Configs** (`/experiment`)
This is where you define a specific experiment. An experiment file's primary job is to **override the defaults** set in `defaults.yaml` and the module configs. This allows you to mix and match components for different experiments.

For example, the `FEMBA_finetune.yaml` experiment file specifies that for this experiment, we should use the `finetune_data_module`, `FEMBA_finetune` model, `cosine` scheduler, and so on.

The override order is: **Command Line -> Experiment Config -> Module Configs -> `defaults.yaml`**.

---

## How to Run an Experiment

To run an experiment, you use the `run_train.py` script and specify the experiment file you want to use with the `+experiment` flag.

### Basic Command
This command runs the `FEMBA_pretrain` experiment using the settings defined in `config/experiment/FEMBA_pretrain.yaml`:
```bash
python run_train.py +experiment=FEMBA_pretrain
```
### Overriding Parameters from the Command Line
One of the most powerful features of Hydra is the ability to override any configuration parameter from the command line.

For example, to change the learning rate and batch size for a pre-training run, you can do the following:
```bash
python run_train.py +experiment=FEMBA_pretrain optimizer.lr=1e-5 batch_size=128
```
This command will use the `FEMBA_pretrain` experiment configuration but override the learning rate to `1e-5` and batch size to `128`, overriding the values in the YAML files. This is perfect for quick experiments and hyperparameter tuning without needing to create new config files.