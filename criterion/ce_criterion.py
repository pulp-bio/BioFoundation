from typing import Dict, Optional, Sequence

import torch
from torch import nn


class CrossEntropyLossWrapper(nn.Module):
    """Cross-entropy loss with optional label smoothing and class weighting.

    Class weights are held by the wrapped ``nn.CrossEntropyLoss`` as a buffer, so
    they follow the enclosing task onto whichever device it is moved to.

    Args:
        label_smoothing: Smoothing coefficient in ``[0, 1)``.
        weight: Optional per-class weights, ordered by class index.
    """

    def __init__(self, label_smoothing: float = 0.0, weight: Optional[Sequence[float]] = None):
        super().__init__()
        self.label_smoothing = label_smoothing
        class_weight = torch.tensor(weight, dtype=torch.float32) if weight is not None else None
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=class_weight)

    def forward(self, pred: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute cross-entropy against ``batch['label']``.

        Args:
            pred: Logits of shape ``(batch, num_classes)``.
            batch: Must contain ``label`` with integer class indices of shape ``(batch,)``.

        Returns:
            Scalar loss.
        """
        return self.loss_fn(pred, batch["label"])
