# coding=utf-8
"""Contains the length regulator layer for the DDPM-GST-Speech-Gen model."""

from typing import Tuple

import torch


class LengthRegulator(torch.nn.Module):
    """Stretches the encoder's output based on the predicted phoneme durations.

    The regulator is intended to alleviate the problem of the gap between the
    length the phoneme set and the expected spectrogram. Explicit duration
    prediction is an alternative way to guide the decoder to make the generated
    spectrogram frames in line with the phonemes they correspond to. It is a way
    to omit the Soft Attention Collapse problem. Besides, it allows parallelization.
    """

    def __init__(self, input_shape: Tuple[int, int], output_length: int):
        """Initializes the length regulator."""

        super().__init__()

    def forward(self, encoder_output: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
        """Stretches the encoder's output based on the predicted phoneme durations.

        Args:
            encoder_output: The output of the encoder.
            durations: The predicted phoneme durations.

        Returns:
            The stretched encoder's output.
        """
