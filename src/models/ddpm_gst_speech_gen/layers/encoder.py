# coding=utf-8
"""Contains the encoder layer for the DDPM-GST-Speech-Gen model."""

from typing import Tuple

import torch


class Encoder(torch.nn.Module):
    """Encodes the input phonemes into enriched representations.

    The encoder is intended to extract the enriched features from the input, so as to
    help the decoder generate proper spectrogram frames. It should be able to capture
    both the low-level relationships between phonemes (helps to generate correct pronunciation,
    intonation, etc.) and the high-level relationships (helps to generate the overall prosody of
    the speech).
    """

    def __init__(self, input_phonemes_shape: Tuple[int, int], output_channels: int):
        """Initializes the encoder."""

        super().__init__()

    def forward(self, phonemes: torch.Tensor) -> torch.Tensor:
        """Encodes the input phonemes.

        Args:
            phonemes: The one-hot encoded input phonemes.
        """
