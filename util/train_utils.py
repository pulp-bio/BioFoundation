#*----------------------------------------------------------------------------*
#* Copyright (C) 2025 ETH Zurich, Switzerland                                 *
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
#* Author:  Anna Tegon                                                        *
#* Author:  Thorir Mar Ingolfsson                                             *
#*----------------------------------------------------------------------------*

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from typing import Optional
import os
import os.path as osp
import torch
import torch.nn as nn


def find_last_checkpoint_path(checkpoint_dir: Optional[str]) -> Optional[str]:
    if checkpoint_dir is None:
        return None
    checkpoint_file_name = (
        f"{ModelCheckpoint.CHECKPOINT_NAME_LAST}{ModelCheckpoint.FILE_EXTENSION}"
    )
    last_checkpoint_filepath = os.path.join(checkpoint_dir, checkpoint_file_name)
    if not osp.exists(last_checkpoint_filepath):
        return None

    return last_checkpoint_filepath

class RobustQuartileNormalize:
    def __init__(self, q_lower, q_upper):
        self.q_lower = q_lower
        self.q_upper = q_upper

    def __call__(self, tensor):
        iqr = self.q_upper - self.q_lower
        return (tensor - self.q_lower) / (iqr + 1e-8)