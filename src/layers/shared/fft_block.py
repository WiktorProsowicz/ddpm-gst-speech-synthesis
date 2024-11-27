# -*- coding=utf-8 -*-
"""Contains definition of Feed Forward Transformer Block."""

from typing import Tuple

import torch


class FFTBlock(torch.nn.Module):
    """Runs the input sequence of representations through attention and normalization.

    The block applies the self-attention to the input and than runs it through two 1D
    convolution layers which expand and shrink the input. Those two phases use normalization
    and residual connections.
    """

    def __init__(self, input_shape: Tuple[int, int],
                 n_heads: int, dropout_rate: float,
                 conv_channels: int):
        """Initializes the FFTBlock.

        Args:
            input_shape: The (length, embedding_dim) shape of the input tensor.
            n_heads: The number of attention heads to use.
            dropout_rate: The dropout rate to use in the attention layer.
            conv_channels: The number of channels the input is extended to in
                the convolution block.
        """

        super().__init__()

        _, input_embedding_dim = input_shape

        self._attention = torch.nn.MultiheadAttention(
            embed_dim=input_embedding_dim,
            num_heads=n_heads,
            dropout=dropout_rate,
            batch_first=True)

        self._layer_norm1 = torch.nn.LayerNorm(input_embedding_dim)

        self._conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=input_embedding_dim,
                out_channels=conv_channels,
                kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=conv_channels,
                out_channels=input_embedding_dim,
                kernel_size=3),
            torch.nn.ReLU()
        )

        self._layer_norm2 = torch.nn.LayerNorm(input_embedding_dim)

    def forward(self, input_sequence: torch.Tensor):
        """Runs the input sequence through the block.

        Args:
            input_sequence: The input sequence of representations.

        Returns:
            The output sequence of representations.
        """

        attention_output, _ = self._attention(
            query=input_sequence,
            key=input_sequence,
            value=input_sequence)

        attention_output = self._layer_norm1(attention_output + input_sequence)

        conv_output = self._conv(attention_output.transpose(1, 2)).transpose(1, 2)

        return self._layer_norm2(conv_output + attention_output)
