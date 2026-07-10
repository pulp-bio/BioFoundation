Copyright (C) 2025-2026 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE for details.

# Contributing to BioFoundation

BioFoundation is both a research repository and an onboarding codebase. Contributions should keep published experiments reproducible while making shared behavior easier to discover.

## Repository Contracts

- Keep changes scoped to one model, dataset, or shared contract where practical.
- Preserve the full Apache 2.0 header in every Python source and Hydra YAML file.
- Normalize task input through `biofoundation.core.batch.as_signal_batch`.
- Keep model-family metadata in `biofoundation/model_registry.py`, including its paper and Hugging Face repository.
- Use Hydra configs for reproducible settings instead of embedding experiment paths or hyperparameters in Python.
- Mark local values that a new user must supply with `#CHANGEME` and document the expected value.

## Adding a Model

1. Add the `nn.Module` implementation under `models/` and its Hydra model config under `config/model/`.
2. Add pre-training and fine-tuning experiments under `config/experiment/`.
3. Use an existing task when its behavior fits. New task steps must use the shared batch adapter.
4. Register the model in `biofoundation/model_registry.py`, including modalities, architecture, experiment names, batch requirements, paper, and Hugging Face URL.
5. Add a model page under `docs/model/` with input assumptions, training details, and checkpoint usage.
6. Extend the contract tests for any new shared behavior.

## Adding Data or Training Components

1. Put dataset implementations in `datasets/` and preprocessing scripts in `make_datasets/`.
2. Compose datasets through a Lightning data module in `data_module/` and a matching `config/data_module/` file.
3. Put reusable losses, tasks, and schedulers in their matching source and config directories.
4. Document data layout, labels, channel or sensor metadata, and a reproducible command.

## Pull Request Checklist

- Explain the research or engineering reason for the change.
- Include the exact configs, seeds, environment, and hardware needed to reproduce reported results.
- Report the evaluation protocol, splits, task metrics, and relevant efficiency metrics.
- Add focused tests and run the fast suite (`hydra-core` enables experiment composition checks):

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python -m compileall -q biofoundation run_train.py models tasks datasets data_module
```

## Official Checkpoint Improvements

The published weights are licensed under CC BY-ND 4.0. Modified weights, including adapters, deltas, pruned variants, and quantized variants, may not be redistributed.

To propose an improvement as an official release, open a PR with the implementation, configs, seeds, logs, environment, hardware, evaluation protocol, and full metrics. Maintainers will review accepted changes, retrain and validate them, and publish the resulting checkpoint to the corresponding PulpBio Hugging Face repository.
