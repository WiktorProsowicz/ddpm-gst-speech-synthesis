# -*- coding: utf-8 -*-
"""Contains utilities for running inference with the trained model."""
from typing import List
from typing import Callable
from typing import Optional
from dataclasses import dataclass

import torch

from utilities import diffusion as diff_utils


def decode_transcript(transcript: torch.Tensor, vocab: List[str]) -> List[str]:
    """Decodes the encoded transcript into a phoneme tokens.

    The encoding may be either one-hot or argmax.
    """

    word_indices = transcript.argmax(dim=1)

    return [vocab[i] for i in word_indices]


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
