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
                 d_model: int,
                 n_heads: int,
                 dropout_rate: float,
                 fft_conv_channels: int
                 ):
        super().__init__()

        input_length, input_channels = input_phonemes_shape

        self._phoneme_embedding = torch.nn.Sequential(
            torch.nn.Linear(input_channels, d_model),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(d_model, d_model),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout_rate),
        )

        self._positional_encoding = torch.nn.Parameter(
            other_utils.create_positional_encoding(torch.arange(0, input_length),
                                                   d_model),
            requires_grad=False
        )

        self._gst_cond_layer = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)

        self._fft_blocks = torch.nn.ModuleList(
            [fft_block.FFTBlock(input_shape=(input_length, d_model),
                                n_heads=n_heads,
                                dropout_rate=dropout_rate,
                                conv_channels=fft_conv_channels)
             for _ in range(n_blocks)]
        )

    def forward(self, input_phonemes: torch.Tensor,
                style_embedding: Optional[torch.Tensor],
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encodes the input phonemes into enriched representations.

        Args:
            input_phonemes: The input one-hot encoded phonemes.
            style_embedding: The style embedding to condition the generation on.
            mask: Indicates which input sequence elements are not padding.

        Returns:
            The enriched representations of the input phonemes.
        """

        if style_embedding is not None:
            return self.apply_gst_conditioning(self.run_basic_blocks(input_phonemes, mask),
                                               style_embedding)

        return self.run_basic_blocks(input_phonemes, mask)

    def run_basic_blocks(self,
                         input_phonemes: torch.Tensor,
                         mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Runs the basic blocks of the encoder.

        This method is intended to be used outside of the basic forward pass of the acoustic model.
        For example, it can be used to obtain enriched phoneme representations for the GST
        Predictor model.

        Args:
            input_phonemes: The input one-hot encoded phonemes.
        """

        reverse_mask = None

        if mask is not None:
            reverse_mask = torch.logical_not(mask).unsqueeze(-1)

        output = self._phoneme_embedding(input_phonemes)
        output += self._positional_encoding

        for block in self._fft_blocks:
            output = block(output, mask, reverse_mask)

        return output

    def apply_gst_conditioning(self, enriched_phonemes: torch.Tensor,
                               style_embedding: torch.Tensor) -> torch.Tensor:
        """Applies the GST conditioning to the basic encoder's output.

        Args:
            enriched_phonemes: The basic output of the encoder.
            style_embedding: The style embedding to condition the generation on.
        """

        style_embedding = style_embedding.unsqueeze(1)
        return enriched_phonemes + self._gst_cond_layer(style_embedding)
