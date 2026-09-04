#*----------------------------------------------------------------------------*
#* Copyright (C) 2026 ETH Zurich, Switzerland                                 *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Author:  BioFoundation Contributors                                       *
#*----------------------------------------------------------------------------*

"""Checkpoint entry points shared by tasks that separate encoder from head.

``run_train.py`` loads pre-trained weights by calling ``load_pretrained_checkpoint``
for a Lightning ``.ckpt`` and ``load_safetensors_checkpoint`` for a ``.safetensors``
file. Tasks that implement Lightning's own ``load_from_checkpoint`` instead can mix
this class in to expose both names without duplicating either loader.

The mixin is deliberately thin: it maps names and formats, and delegates the actual
tensor matching to the task. Loading policy, including which shape mismatches are
tolerated, stays with the task that owns the model.
"""

from typing import Any, Dict, Optional, Protocol


class _CheckpointLoadable(Protocol):  # pragma: no cover - typing only
    """The single method a task must provide for the mixin to delegate to."""

    def load_from_checkpoint(self, checkpoint_path: str, **kwargs: Any) -> Any: ...


class SafetensorsCheckpointMixin:
    """Expose BioFoundation's two checkpoint entry points over ``load_from_checkpoint``.

    Mix in ahead of :class:`pytorch_lightning.LightningModule` on tasks whose loading
    logic already lives in ``load_from_checkpoint``:

    .. code-block:: python

        class MyTask(SafetensorsCheckpointMixin, pl.LightningModule):
            def load_from_checkpoint(self, checkpoint_path, **kwargs): ...

    ``load_safetensors_checkpoint`` converts the flat ``.safetensors`` mapping into the
    ``{"state_dict": ...}`` layout ``load_from_checkpoint`` expects, writes it to a
    temporary file, and delegates. Keys are given a ``model.`` prefix when they lack
    one, matching how the pre-training task saves an encoder.
    """

    def load_pretrained_checkpoint(self, model_ckpt: str, **kwargs: Any) -> Any:
        """Load a Lightning ``.ckpt`` by delegating to ``load_from_checkpoint``."""

        return self.load_from_checkpoint(checkpoint_path=model_ckpt, **kwargs)

    def load_safetensors_checkpoint(self, model_ckpt: str, **kwargs: Any) -> Any:
        """Load a ``.safetensors`` file through the same path as a Lightning checkpoint."""

        import tempfile
        from pathlib import Path

        import torch
        from safetensors.torch import load_file

        state_dict = {
            key if key.startswith(("model.", "model_head.")) else f"model.{key}": value
            for key, value in load_file(model_ckpt).items()
        }

        with tempfile.TemporaryDirectory() as directory:
            converted = Path(directory) / "converted.ckpt"
            torch.save({"state_dict": state_dict}, converted)
            return self.load_from_checkpoint(checkpoint_path=str(converted), **kwargs)


def split_state_dict_by_prefix(
    state_dict: Dict[str, Any],
    prefixes: tuple[str, ...] = ("model_head.", "model."),
) -> Dict[str, Dict[str, Any]]:
    """Group a flat Lightning ``state_dict`` by top-level module prefix.

    Args:
        state_dict: Mapping of parameter name to tensor, as stored by Lightning.
        prefixes: Prefixes to split on, longest first so that ``model_head.`` is
            matched before ``model.``.

    Returns:
        Mapping of prefix-without-the-dot to a prefix-stripped state dict. Keys that
        match no prefix are collected under ``""``.
    """

    grouped: Dict[str, Dict[str, Any]] = {prefix.rstrip("."): {} for prefix in prefixes}
    grouped[""] = {}

    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                grouped[prefix.rstrip(".")][key[len(prefix):]] = value
                break
        else:
            grouped[""][key] = value

    return grouped


__all__ = ["SafetensorsCheckpointMixin", "split_state_dict_by_prefix"]
