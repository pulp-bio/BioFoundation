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
from torch.optim.lr_scheduler import MultiStepLR

class MultiStepLRWarmup(MultiStepLR):
    """
    Learning rate scheduler that extends PyTorch's MultiStepLR with warmup support.

    During warmup, the learning rate increases linearly from a small initial value 
    (`warmup_init_lr`) to the base learning rate over a specified number of iterations 
    (`warmup_iter`). After the warmup phase, the scheduler follows a step-wise decay 
    based on milestone epochs.

    Attributes:
        warmup_iter (int): Number of warmup iterations.
        warmup_init_lr (float): Initial learning rate during warmup.
    """
    def __init__(
        self,
        optimizer,
        milestones,
        warmup_iter: int = -1,
        warmup_init_lr: float = 0.0,
        gamma: float = 0.1,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        """
        Initializes the MultiStepLRWarmup scheduler.

        Args:
            optimizer (Optimizer): Wrapped optimizer.

            milestones (list of int): List of epoch indices at which to decay the 
                learning rate. Must be increasing.

            warmup_iter (int, optional): Number of iterations for warmup. If -1, 
                no warmup is applied. 

            warmup_init_lr (float, optional): Initial learning rate for warmup phase.
                Learning rate will increase linearly from this value to the base LR. 

            gamma (float, optional): Multiplicative factor of learning rate decay.

            last_epoch (int, optional): The index of last epoch. 

            verbose (bool, optional): If True, prints a message to stdout for each update.
        """
        self.warmup_iter = warmup_iter
        self.warmup_init_lr = warmup_init_lr
        super(MultiStepLRWarmup, self).__init__(
            optimizer, milestones, gamma, last_epoch, verbose
        )

    def get_lr(self):
        """
        Compute learning rate at current step.

        If still in the warmup phase (i.e., current epoch < warmup_iter), 
        apply linear increase from warmup_init_lr to base_lr.
        Otherwise, apply standard MultiStepLR behavior.
        """
        if self.last_epoch < self.warmup_iter:
            return [
                self.warmup_init_lr
                + (base_lr - self.warmup_init_lr) / self.warmup_iter * self.last_epoch
                for base_lr in self.base_lrs
            ]
        else:
            return super(MultiStepLRWarmup, self).get_lr()


def multi_step_lr(
    optimizer, 
    milestones, 
    gamma: float, 
    warmup_iter: int = -1, 
    warmup_init_lr: float = 0.0
):
    """
    Convenience function to instantiate MultiStepLRWarmup.

    Args:
        optimizer (Optimizer): Wrapped optimizer.

        milestones (list[int] or str): List of epoch indices or a '+'-separated string.
            If string, it will be parsed into a list of integers.

        gamma (float): Multiplicative decay factor.

        warmup_iter (int, optional): Number of warmup iterations. 

        warmup_init_lr (float, optional): Initial learning rate for warmup. 

    Returns:
        MultiStepLRWarmup: An instance of the warmup-enhanced scheduler.
    """
    if isinstance(milestones, str):
        milestones = list(map(int, milestones.split("+")))
    lr_scheduler = MultiStepLRWarmup(
        optimizer, milestones, warmup_iter, warmup_init_lr, gamma
    )
    return lr_scheduler
