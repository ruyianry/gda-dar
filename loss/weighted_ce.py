from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.CrossEntropyLoss):
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']

    def __init__(self, weight: Optional[torch.Tensor] = None, size_average=None,
                 ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0,
                 ) -> None:
        super().__init__(weight, size_average, ignore_index=ignore_index, reduce=reduce,
                         reduction=reduction, label_smoothing=label_smoothing)


    def forward(self, input: torch.Tensor, target: torch.Tensor,
                weight: torch.Tensor=None) -> torch.Tensor:
        base_loss = F.cross_entropy(input, target, weight=self.weight,
                                    ignore_index=self.ignore_index,
                                    reduction='none', # Always use 'none' to apply weights
                                    label_smoothing=self.label_smoothing)

        if weight is not None:
            base_loss = base_loss * weight

            # Apply the reduction method
        if self.reduction == 'mean':
            return base_loss.mean()
        elif self.reduction == 'sum':
            return base_loss.sum()
        else:
            return base_loss  # Return the per-sample loss when reduction is 'none'