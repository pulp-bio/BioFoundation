from typing import Dict

import torch
from torch import nn


class MSELossWrapper(nn.Module):
    """Mean squared error loss for scalar regression targets."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction=reduction)

    def forward(self, pred: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute MSE against ``batch['label']``.

        Args:
            pred: Predictions broadcastable to the target shape.
            batch: Must contain ``label`` with the regression targets.

        Returns:
            Scalar loss.
        """
        target = batch["label"]
        return self.loss_fn(pred.reshape(target.shape).to(target.dtype), target)
