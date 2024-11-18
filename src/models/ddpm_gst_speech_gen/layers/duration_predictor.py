# coding=utf-8
"""Contains the duration predictor for the DDPM-GST-Speech-Gen model."""

from typing import Tuple

import torch


def _create_conv_blocks(n_blocks: int,
                        input_length: int,
                        input_channels: int) -> torch.nn.Sequential:
    """Creates the convolutional blocks for the duration predictor."""

    def spawn_block():
        return torch.nn.Sequential(
            torch.nn.Conv1d(input_channels, input_channels, kernel_size=3, padding='same'),
            torch.nn.ReLU(),
            torch.nn.LayerNorm((input_channels, input_length)),
            torch.nn.Dropout1d(0.1)
        )

    return torch.nn.Sequential(*[spawn_block() for _ in range(n_blocks)])


class DurationPredictor(torch.nn.Module):
    """Predict the durations of the input phonemes based on their enriched representations.

    The phoneme durations are used by the length regulator layer to stretch the encoder's
    output, so that it matches the expected spectrogram length. The durations predicted are
    in the logarithm scale.
    """

    def __init__(self, input_shape: Tuple[int, int]):
        """Initializes the duration predictor."""

        super().__init__()

        input_length, input_features = input_shape

        n_conv_blocks = 2
        self._conv_blocks = _create_conv_blocks(n_blocks=n_conv_blocks,
                                                input_length=input_length,
                                                input_channels=input_features)

        self._output_layer = torch.nn.Linear(input_features, 1)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """Predicts the durations of the input phonemes.

        Args:
            encoder_output: The output of the encoder.

        Returns:
            The predicted phoneme durations.
        """

        output = encoder_output.transpose(1, 2)
        output = self._conv_blocks(output)
        output = output.transpose(1, 2)
        return self._output_layer(output)
