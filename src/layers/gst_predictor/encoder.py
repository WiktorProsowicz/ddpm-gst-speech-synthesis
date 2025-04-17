# -*- coding: utf-8 -*-
"""Contains the definition of the encoder class for the GST predictor."""
from typing import Tuple

import torch

from layers.shared import fft_block


class Encoder(torch.nn.Module):
    """Encodes input phoneme representations into conditioning information.

    The created embedding is used to condition the noise generation in the decoder.
    """

    def __init__(self,
                 input_phonemes_shape: Tuple[int, int],
                 gst_size: int,
                 n_blocks: int,
                 n_heads: int,
                 conv_filters: int,
                 dropout_rate: float):

        super().__init__()

        input_length, input_dim = input_phonemes_shape
        input_dim += 768

        self._fft_blocks = torch.nn.ModuleList(
            [fft_block.FFTBlock((input_length, input_dim),
                                 n_heads,
                                 dropout_rate,
                                 conv_filters)
              for _ in range(n_blocks)]
        )

        self._attention_query = torch.nn.Parameter(torch.rand(gst_size, input_dim),
                                                   requires_grad=True)

        self._attention = torch.nn.MultiheadAttention(input_dim,
                                                      4,
                                                      dropout=dropout_rate,
                                                      batch_first=True)

    def forward(self,
                phoneme_representations: torch.Tensor,
                phonemes_mask: torch.Tensor,
                bert_embeddings: torch.Tensor) -> torch.Tensor:
        """Encodes input phoneme representations into an embedding.

        Args:
            input_phonemes: Tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Embedding of shape (batch_size, embedding_size).
        """

        output = torch.cat((phoneme_representations, bert_embeddings), dim=-1)

        for block in self._fft_blocks:
            reversed_mask = torch.logical_not(phonemes_mask).unsqueeze(-1)
            output = block(output, phonemes_mask, reversed_mask)

        attention_query = self._attention_query.unsqueeze(0).expand(
            output.size(0), -1, -1)
        attention_output, _ = self._attention(attention_query,
                                              output,
                                              output)

        return attention_output
