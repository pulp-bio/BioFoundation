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

``run_train.py`` calls ``load_pretrained_checkpoint`` for a ``.ckpt`` and
``load_safetensors_checkpoint`` for a ``.safetensors``; this mixin maps both onto a
task's single ``load_weights``, leaving tensor matching to the task.

``load_weights`` is not called ``load_from_checkpoint`` because that name is a
:class:`pytorch_lightning.LightningModule` classmethod, and an instance method of the
same name shadows it so ``MyTask.load_from_checkpoint(path)`` binds the path to ``self``.
"""

from typing import Any


class SafetensorsCheckpointMixin:
    """Expose BioFoundation's two checkpoint entry points over one ``load_weights``.

    Mix in ahead of :class:`pytorch_lightning.LightningModule`. The safetensors path
    wraps the flat mapping in ``{"state_dict": ...}`` and prefixes bare keys with
    ``model.``, matching how the pre-training task saves an encoder.
    """

    def load_pretrained_checkpoint(self, model_ckpt: str, **kwargs: Any) -> Any:
        """Load a Lightning ``.ckpt`` by delegating to ``load_weights``."""

        return self.load_weights(checkpoint_path=model_ckpt, **kwargs)

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
            return self.load_weights(checkpoint_path=str(converted), **kwargs)




__all__ = ["SafetensorsCheckpointMixin"]
