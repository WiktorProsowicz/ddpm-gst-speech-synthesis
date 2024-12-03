# -*- coding: utf-8 -*-
"""Contains the encoder for the acoustic model."""
from typing import Optional
from typing import Tuple

import torch

from layers.shared import fft_block
from utilities import other as other_utils


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
                 embedding_dim: int,
                 n_heads: int,
                 dropout_rate: float,
                 fft_conv_channels: int
                 ):
        super().__init__()

        input_length, input_channels = input_phonemes_shape

        self._phoneme_embedding = torch.nn.Sequential(
            torch.nn.Linear(input_channels, embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(embedding_dim, embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
        )

        self._positional_encoding = torch.nn.Parameter(
            other_utils.create_positional_encoding(torch.arange(0, input_length),
                                                   embedding_dim),
            requires_grad=False
        )

        self._cond_layers = torch.nn.ModuleList(
            [torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)
             for _ in range(n_blocks)]
        )

        self._fft_blocks = torch.nn.ModuleList(
            [fft_block.FFTBlock(input_shape=(input_length, embedding_dim),
                                n_heads=n_heads,
                                dropout_rate=dropout_rate,
                                conv_channels=fft_conv_channels)
             for _ in range(n_blocks)]
        )

    def forward(self, input_phonemes: torch.Tensor,
                style_embedding: Optional[torch.Tensor]) -> torch.Tensor:
        """Encodes the input phonemes into enriched representations.

        Args:
            input_phonemes: The input one-hot encoded phonemes.
            style_embedding: The style embedding to condition the generation on.

        Returns:
            The enriched representations of the input phonemes.
        """

        output = self._phoneme_embedding(input_phonemes)
        output += self._positional_encoding

        if style_embedding is not None:
            style_embedding = style_embedding.unsqueeze(1)

            for block, cond_l in zip(self._fft_blocks, self._cond_layers):
                output = block(output)
                output = output + cond_l(style_embedding)

        else:
            for block in self._fft_blocks:
                output = block(output)

        return output
