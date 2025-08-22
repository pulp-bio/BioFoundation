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

import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
from omegaconf import DictConfig

class CosineLRSchedulerWrapper(CosineLRScheduler):
    """
    A wrapper for the Cosine Learning Rate Scheduler that provides enhanced functionality 
    and easier configuration for learning rate scheduling during model training.

    This class extends the CosineLRScheduler from the timm library.

    Attributes:
        optimizer (torch.optim.Optimizer): The optimizer being used for training.
        trainer (DictConfig): Configuration dictionary for the training process.
        min_lr (float): The minimum learning rate at the end of the training.
        warmup_lr_init (float): The initial learning rate used during warmup.
        t_in_epochs (bool): Flag to indicate if scheduling is done in epochs or steps.
        num_opt_steps_per_epoch (int): Number of optimization steps per epoch.
        warmup_steps (int): Number of warmup steps.
        total_steps (int): Total number of optimization steps in the training process.
    """
    def __init__(self, optimizer: torch.optim.Optimizer, 
                 total_training_opt_steps: int, 
                 trainer: DictConfig, warmup_epochs: int, 
                 min_lr: float, 
                 warmup_lr_init: float, 
                 t_in_epochs: bool = False):

        """
        Initialize the Cosine Learning Rate Scheduler Wrapper.

        Args:
            optimizer (torch.optim.Optimizer): The torch optimizer used for training.

            total_training_opt_steps (int): The total number of optimization steps 
            across all epochs.

            trainer (DictConfig): PyTorchLightning Trainer object for the training run.

            warmup_epochs (int): Number of epochs to use for learning rate warmup. 

            min_lr (float): The minimum learning rate to be reached at the end of 
                the training process. 

            warmup_lr_init (float): The initial learning rate used during the warmup 
                phase. 

            t_in_epochs (bool, optional): Flag to specify if the scheduler should 
                interpret time in epochs (True) or steps (False). 
                Defaults to False (steps-based scheduling).
        """
        self.optimizer = optimizer
        self.trainer = trainer
        self.min_lr = min_lr
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        self.num_opt_steps_per_epoch = (total_training_opt_steps // self.trainer.max_epochs)
        self.warmup_steps = warmup_epochs * self.num_opt_steps_per_epoch
        self.total_steps = total_training_opt_steps
        
        super().__init__(
            optimizer=self.optimizer,
            t_initial=self.total_steps,
            lr_min=self.min_lr,
            warmup_lr_init=self.warmup_lr_init,
            warmup_t=self.warmup_steps,
            t_in_epochs=self.t_in_epochs
        )