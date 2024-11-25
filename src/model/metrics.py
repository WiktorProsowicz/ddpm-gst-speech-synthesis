# -*- coding: utf-8 -*-
"""Contains metrics used to evaluate the model's performance."""
from typing import Optional

import torch


@torch.no_grad()
def mean_absolute_error(y_true: torch.Tensor,
                        y_pred: torch.Tensor,
                        mask: Optional[torch.Tensor] = None,
                        mask_sum: Optional[torch.Tensor] = None):
    """Calculates mean absolute error.

    Args:
        y_true: True values.
        y_pred: Predicted values.
        mask: Loss mask for ignoring some of the computed errors.
        mask_sum: Sum of the mask values.
    """

    if mask is not None:
        return torch.sum(torch.abs(y_true - y_pred) * mask) / mask_sum

    return torch.mean(torch.abs(y_true - y_pred))
