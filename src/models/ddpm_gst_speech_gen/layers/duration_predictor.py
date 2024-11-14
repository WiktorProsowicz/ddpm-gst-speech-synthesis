# coding=utf-8
"""Contains the duration predictor for the DDPM-GST-Speech-Gen model."""

from typing import Tuple

import torch


class DurationPredictor(torch.nn.Module):
    """Predict the durations of the input phonemes based on their enriched representations.

    The phoneme durations are used by the length regulator layer to stretch the encoder's
    output, so that it matches the expected spectrogram length. The durations predicted are
    in the logarithm scale.
    """

    def __init__(self, input_shape: Tuple[int, int]):
        """Initializes the duration predictor."""

        super().__init__()

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """Predicts the durations of the input phonemes.

        Args:
            encoder_output: The output of the encoder.

        Returns:
            The predicted phoneme durations.
        """
