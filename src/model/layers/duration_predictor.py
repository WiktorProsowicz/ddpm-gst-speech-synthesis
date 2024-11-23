# -*- coding: utf-8 -*-
"""Contains the duration predictor for the DDPM-GST-Speech-Gen model."""
from typing import Tuple

import torch


class _ConvBlock(torch.nn.Module):
    """Convolutional block for the duration predictor."""

    def __init__(self, input_channels: int, input_length: int, dropout_rate: float):
        super().__init__()

        self._layers = torch.nn.Sequential(
            torch.nn.Conv1d(input_channels, input_channels, kernel_size=5, padding='same'),
            torch.nn.ReLU(),
            torch.nn.LayerNorm((input_channels, input_length)),
            torch.nn.Dropout1d(dropout_rate),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self._layers(input_tensor) + input_tensor


def _create_conv_blocks(n_blocks: int,
                        input_length: int,
                        input_channels: int,
                        dropout_rate: float) -> torch.nn.Sequential:
    """Creates the convolutional blocks for the duration predictor."""

    blocks = [_ConvBlock(input_channels=input_channels,
                         input_length=input_length,
                         dropout_rate=dropout_rate)
              for _ in range(n_blocks)]

    return torch.nn.Sequential(*blocks)


class DurationPredictor(torch.nn.Module):
    """Predict the durations of the input phonemes based on their enriched representations.

    The phoneme durations are used by the length regulator layer to stretch the encoder's
    output, so that it matches the expected spectrogram length. The durations predicted are
    in the logarithm scale.
    """

    def __init__(self, input_shape: Tuple[int, int], n_conv_blocks: int, dropout_rate: float):
        """Initializes the duration predictor."""

        super().__init__()

        input_length, input_features = input_shape

        self._conv_blocks = _create_conv_blocks(n_blocks=n_conv_blocks,
                                                input_length=input_length,
                                                input_channels=input_features,
                                                dropout_rate=dropout_rate)

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
