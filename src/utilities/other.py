# -*- coding: utf-8 -*-
"""Contains miscellaneous utilities."""
import json
import logging
import os
import sys
from typing import Any
from typing import Dict
from typing import Tuple

import torch


def create_positional_encoding(time_steps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """Creates the Sinusoidal Positional Embedding for the input time steps."""

    i_steps = torch.arange(0, embedding_dim // 2, device=time_steps.device)
    factor = 10000 ** (i_steps / (embedding_dim // 2))

    t_embedding = time_steps[:, None].repeat(1, embedding_dim // 2)
    t_embedding = t_embedding / factor

    return torch.cat([torch.sin(t_embedding), torch.cos(t_embedding)], dim=-1)


@torch.no_grad()
def create_loss_mask_for_durations(durations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Creates a mask used in the loss calculation for the phoneme durations.

    Args:
        durations: Ground truth phoneme durations.

    Returns:
        A tuple containing the mask and the sum of the mask elements to compute the mean loss.
    """

    mask = (durations > 0.0).to(torch.float)
    return mask, torch.sum(mask)


@torch.no_grad()
def create_loss_mask_for_spectrogram(spectrogram: torch.Tensor,
                                     durations: torch.Tensor,
                                     durations_mask: torch.Tensor
                                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Creates a boolean mask used in the loss calculation for the spectrogram.

    Args:
        spectrogram: The ground truth spectrogram.
        durations: The ground truth phoneme durations in log scale.
        durations_mask: The mask for the phoneme durations.

    Returns:
        A tuple containing the mask and the sum of the mask elements to compute the mean loss.
    """

    pow_durations = (torch.pow(2.0, durations) + 1e-4).to(torch.int32)
    pow_durations = pow_durations * durations_mask

    max_lengths = torch.sum(pow_durations, dim=1).to(torch.int32)

    arange = torch.arange(spectrogram.shape[2]).reshape(1, 1, -1).to(spectrogram.device)
    mask = arange < max_lengths.reshape(-1, 1, 1)
    mask = mask.to(torch.float)

    return mask, torch.sum(mask) * spectrogram.shape[1]


@torch.no_grad()
def create_loss_weight_for_spectrogram(spectrogram: torch.Tensor):
    """Creates a weight matrix used to in the calculation of loss for the spectrogram.

    The weights are intended to guide the spectrogram to focus on lower frequency regions.

    Args:
        spectrogram: Either the  ground truth of the predicted spectrogram.
    """

    return torch.logspace(
        0., 1., steps=spectrogram.shape[1], base=10.0, device=spectrogram.device).reshape(1, -1, 1)
