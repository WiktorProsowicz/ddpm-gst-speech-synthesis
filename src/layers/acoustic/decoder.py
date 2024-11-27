# -*- coding: utf-8 -*-
"""Contains the encoder for the acoustic model."""
from typing import Tuple

import torch

from layers.shared import fft_block
from utilities import other as other_utils


class Decoder(torch.nn.Module):
    """Decodes stretched phoneme representations into spectrogram frames.

    The generation is conditioned with help of input style embedding.
    """

    def __init__(self,
                 input_phonemes_shape: Tuple[int, int],
                 output_channels,
                 n_blocks: int,
                 fft_conv_channels: int,
                 n_heads: int,
                 dropout_rate: float):
        """Initializes the decoder.

        Args:
            input_phonemes_shape: The (length, embedding_dim) shape of the input phonemes.
            n_blocks: The number of FFT blocks to use in the decoder.
            n_heads: The number of attention heads to use in the FFT blocks.
            dropout_rate: The dropout rate to use in the FFT blocks.
        """

        super().__init__()

        input_length, input_channels = input_phonemes_shape

        self._positional_encoding = torch.nn.Parameter(
            other_utils.create_positional_encoding(torch.arange(0, input_length),
                                                   input_channels),
            requires_grad=False
        )

        self._fft_blocks = torch.nn.Sequential(
            *[fft_block.FFTBlock(input_shape=(input_length, input_channels),
                                 n_heads=n_heads,
                                 dropout_rate=dropout_rate,
                                 conv_channels=fft_conv_channels)
              for _ in range(n_blocks)]
        )

        self._postnet = torch.nn.Sequential(
            torch.nn.Linear(input_channels, output_channels),
            torch.nn.Sigmoid()
        )

    def forward(self, input_phonemes: torch.Tensor,
                style_embedding: torch.Tensor) -> torch.Tensor:  # pylint: disable=unused-argument
        """Decodes the stretched phoneme representations into spectrogram frames.

        Args:
            input_phonemes: The stretched phoneme representations.
            style_embedding: The style embedding to condition the generation on.

        Returns:
            The generated spectrogram frames.
        """

        output = input_phonemes + self._positional_encoding
        output = self._fft_blocks(output)
        return self._postnet(output).transpose(1, 2)
