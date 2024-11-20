# -*- coding: utf-8 -*-
"""Contains utilities for running inference with the trained model."""
from dataclasses import dataclass
from typing import Callable

import torch

from utilities import diffusion as diff_utils


def get_transcript_length(transcript: torch.Tensor) -> int:
    """Returns the actual length of the one-hot encoded transcript.

    Args:
        transcript: The one-hot encoded transcript without the batch_size dimension.
    """

    return torch.sum(transcript, dtype=torch.int).item()


def create_transcript_mask(transcript: torch.Tensor) -> torch.Tensor:
    """Creates a mask for the transcript based on the actual length.

    Args:
        transcript: The one-hot encoded transcript without the batch_size dimension.
    """

    transcript_length = get_transcript_length(transcript)

    return torch.cat((torch.ones(transcript_length),
                     torch.zeros(transcript.shape[1] - transcript_length)))


def sanitize_predicted_durations(log_durations: torch.Tensor,
                                 expected_output_length: int) -> torch.Tensor:
    """Sanitizes the predicted durations so that an alignment matrix can be created.

    Args:
        log_durations: The predicted log durations.
        expected_output_length: The expected length of the tensor stretched by the durations.
    """

    log_durations = torch.clamp(log_durations, min=0.0)
    pow_duration = torch.pow(2.0, log_durations)

    cum_durations = torch.cumsum(pow_duration, dim=1)
    durations_mask = cum_durations <= expected_output_length

    return log_durations * durations_mask


@dataclass
class BackwardDiffusionModelInput:
    """Contains the input data for a single backward diffusion step.

    The input data shape is supposed to contain the batch_size dimension equal to 1.
    """

    noised_data: torch.Tensor
    timestep: torch.Tensor


def run_backward_diffusion(model_callable: Callable[[BackwardDiffusionModelInput], torch.Tensor],
                           diffusion_handler: diff_utils.DiffusionHandler,
                           input_initial_noise: torch.Tensor) -> torch.Tensor:
    """Performs a full backward diffusion process with the given model and data."""

    noised_data = input_initial_noise

    for diff_step in reversed(range(diffusion_handler.num_steps)):

        model_input = BackwardDiffusionModelInput(
            noised_data=noised_data,
            timestep=torch.tensor([diff_step], device=noised_data.device))

        predicted_noise = model_callable(model_input)

        noised_data = diffusion_handler.remove_noise(noised_data, predicted_noise, diff_step)

    return noised_data
