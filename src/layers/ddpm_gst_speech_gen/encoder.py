# -*- coding: utf-8 -*-
"""Contains the encoder layer for the DDPM-GST-Speech-Gen model."""
from typing import List
from typing import Tuple

import torch


class _ResidualBlock(torch.nn.Module):
    """Residual block for the encoder."""

    def __init__(self, input_length: int,
                 conv_kernel_size: int, conv_channels: int, dropout_rate: float):

        super().__init__()

        self._layers = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=conv_channels,
                out_channels=conv_channels,
                kernel_size=conv_kernel_size,
                padding='same'
            ),
            torch.nn.ReLU(),
            torch.nn.LayerNorm((conv_channels, input_length)),
            torch.nn.Dropout(dropout_rate),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:

        return self._layers(input_tensor) + input_tensor


def _create_residual_blocks(input_length: int,
                            conv_channels: int,
                            kernel_sizes: List[int],
                            dropout_rate: float) -> torch.nn.Sequential:

    residual_blocks = [_ResidualBlock(input_length=input_length,
                                      conv_channels=conv_channels,
                                      conv_kernel_size=kernel_size,
                                      dropout_rate=dropout_rate)
                       for kernel_size
                       in kernel_sizes]

    return torch.nn.Sequential(*residual_blocks)


class Encoder(torch.nn.Module):
    """Encodes the input phonemes into enriched representations.

    The encoder is intended to extract the enriched features from the input, so as to
    help the decoder generate proper spectrogram frames. It should be able to capture
    both the low-level relationships between phonemes (helps to generate correct pronunciation,
    intonation, etc.) and the high-level relationships (helps to generate the overall prosody of
    the speech).
    """

    def __init__(self,
                 input_phonemes_shape: Tuple[int, int],
                 n_blocks: int,
                 output_channels: int,
                 embedding_dim: int,
                 dropout_rate: float):
        """Initializes the encoder."""

        super().__init__()

        input_length, input_onehot_dim = input_phonemes_shape

        self._prenet = torch.nn.Sequential(
            torch.nn.Linear(input_onehot_dim, embedding_dim),
            torch.nn.ReLU(),
        )

        self._res_blocks = _create_residual_blocks(
            input_length=input_length,
            conv_channels=embedding_dim,
            kernel_sizes=[7, 5, 3] * n_blocks,
            dropout_rate=dropout_rate
        )

        self._final_lstm = torch.nn.LSTM(
            input_size=embedding_dim,
            hidden_size=output_channels,
            num_layers=1,
            batch_first=True
        )

    def forward(self, phonemes: torch.Tensor) -> torch.Tensor:
        """Encodes the input phonemes.

        Args:
            phonemes: The one-hot encoded input phonemes.
        """

        output = self._prenet(phonemes)
        output = output.permute(0, 2, 1)
        output = self._res_blocks(output)
        output = output.permute(0, 2, 1)
        output, _, = self._final_lstm(output)

        return output
