# -*- coding=utf-8 -*-
"""Contains the definition of the encoder class for the GST predictor."""

from typing import Tuple

import torch


class _ConvBlock(torch.nn.Module):

    def __init__(self,
                 in_channels: int,
                 input_length: int,
                 dropout_rate: float):
        super().__init__()

        self._layers = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, in_channels, kernel_size=3, padding='same'),
            torch.nn.SiLU(),
            torch.nn.LayerNorm((in_channels, input_length)),
            torch.nn.Dropout(dropout_rate)
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self._layers(input_tensor) + input_tensor


class Encoder(torch.nn.Module):
    """Encodes input phonemes into an embedding.

    The created embedding is used to condition the noise generation in the decoder.
    """

    def __init__(self,
                 input_phonemes_shape: Tuple[int, int],
                 embedding_size: int,
                 n_conv_blocks: int,
                 dropout_rate: float):

        super().__init__()

        input_length, input_dim = input_phonemes_shape

        self._prenet = torch.nn.Sequential(
            torch.nn.Linear(input_dim, embedding_size),
            torch.nn.SiLU(),
        )

        self._conv_blocks = torch.nn.Sequential(
            *[_ConvBlock(embedding_size, input_length, dropout_rate) for _ in range(n_conv_blocks)]
        )

        self._postnet = torch.nn.LSTM(embedding_size, embedding_size, batch_first=True)

    def forward(self, input_phonemes: torch.Tensor) -> torch.Tensor:
        """Encodes input phonemes into an embedding.

        Args:
            input_phonemes: Tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Embedding of shape (batch_size, embedding_size).
        """

        prenet_output = self._prenet(input_phonemes)
        prenet_output = prenet_output.transpose(1, 2)

        conv_blocks_output = self._conv_blocks(prenet_output)
        conv_blocks_output = conv_blocks_output.transpose(1, 2)

        _, (_, final_lstm_state) = self._postnet(conv_blocks_output)

        return final_lstm_state.squeeze(0)
