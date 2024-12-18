# -*- coding: utf-8 -*-
"""Contains the decoder layer for the DDPM-GST-Speech-Gen model."""
from typing import Optional
from typing import Tuple

import torch

from utilities import other as other_utils


class _ResidualBlock(torch.nn.Module):

    def __init__(self, skip_channels: int,
                 block_input_channels: int,
                 decoder_input_channels: int,
                 conv_kernel_size: int,
                 dropout_rate: float):

        super().__init__()

        self._norm = torch.nn.Sequential(torch.nn.BatchNorm1d(block_input_channels),
                                         torch.nn.Conv1d(block_input_channels,
                                                         block_input_channels,
                                                         kernel_size=conv_kernel_size,
                                                         padding='same'),
                                         torch.nn.Dropout(dropout_rate))

        self._skip_connection = torch.nn.Sequential(torch.nn.Conv1d(block_input_channels,
                                                                    skip_channels,
                                                                    kernel_size=conv_kernel_size,
                                                                    padding='same'),
                                                    torch.nn.Dropout(dropout_rate))

        self._cond_timestep_conv = torch.nn.Sequential(torch.nn.Conv1d(decoder_input_channels,
                                                                       block_input_channels,
                                                                       kernel_size=1),
                                                       torch.nn.Dropout(dropout_rate))

        self._cond_phonemes_conv = torch.nn.Sequential(torch.nn.Conv1d(decoder_input_channels,
                                                                       block_input_channels,
                                                                       kernel_size=conv_kernel_size,
                                                                       padding='same'),
                                                       torch.nn.Dropout(dropout_rate))

        self._output_conv = torch.nn.Sequential(torch.nn.Conv1d(block_input_channels,
                                                                block_input_channels,
                                                                kernel_size=conv_kernel_size,
                                                                padding='same'),
                                                torch.nn.Dropout(dropout_rate))

    def forward(self, input_noise: torch.Tensor, timestep_embedding: torch.Tensor,
                phoneme_representations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        output = self._norm(input_noise)
        output = self._cond_timestep_conv(timestep_embedding) + output
        output = self._cond_phonemes_conv(phoneme_representations) + output
        output = torch.nn.functional.tanh(output) * torch.nn.functional.sigmoid(output)

        skip_connection = self._skip_connection(output)
        output = self._output_conv(output) + input_noise

        return output, skip_connection


def _create_residual_blocks(n_blocks: int,
                            skip_connections_channels: int,
                            block_input_channels: int,
                            decoder_input_channels: int,
                            conv_kernel_size: int,
                            dropout_rate) -> torch.nn.ModuleList:

    blocks = [
        _ResidualBlock(
            skip_channels=skip_connections_channels,
            block_input_channels=block_input_channels,
            decoder_input_channels=decoder_input_channels,
            conv_kernel_size=conv_kernel_size,
            dropout_rate=dropout_rate)
        for _ in range(n_blocks)
    ]

    return torch.nn.ModuleList(blocks)


class Decoder(torch.nn.Module):
    """Predicts the noise added to the spectrogram within the diffusion process.

    The noise is predicted based on the:
    - diffusion step t
    - noised spectrogram at step t
    - enriched phoneme representations from the encoder (stretched by the length regulator)
    - style embedding from the GST module

    The decoder's output can be interpreted as the amount of noise that should be added to
    the 'clean' spectrogram to create the 'noised' spectrogram at the diffusion step t.
    """

    def __init__(self,
                 input_noise_shape: Tuple[int, int],
                 timestep_embedding_dim: int,
                 n_res_blocks: int,
                 internal_channels: int,
                 skip_connections_channels: int,
                 conv_kernel_size: int,
                 dropout_rate: float):
        """Initializes the decoder."""

        super().__init__()

        input_channels, _ = input_noise_shape

        self._timestep_embedding_dim = timestep_embedding_dim

        self._timestep_encoder = torch.nn.Sequential(
            torch.nn.Linear(timestep_embedding_dim, input_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(input_channels, input_channels),
            torch.nn.SiLU()
        )

        self._residual_blocks = _create_residual_blocks(
            n_res_blocks,
            skip_connections_channels,
            internal_channels,
            input_channels,
            conv_kernel_size,
            dropout_rate)

        self._prenet = torch.nn.Sequential(
            torch.nn.Conv1d(input_channels,
                            internal_channels,
                            kernel_size=conv_kernel_size,
                            padding='same'),
        )

        self._postnet = torch.nn.Sequential(
            torch.nn.Conv1d(
                skip_connections_channels,
                input_channels,
                kernel_size=conv_kernel_size,
                padding='same'),
        )

    def forward(self,
                diffusion_step: torch.Tensor,
                noised_spectrogram: torch.Tensor,
                phoneme_representations: torch.Tensor,
                style_embedding: Optional[torch.Tensor]) -> torch.Tensor:  # pylint: disable=unused-argument
        """Predicts the noise added to the spectrogram.

        Args:
            diffusion_step: The diffusion step t.
            noised_spectrogram: The spectrogram at the diffusion step t.
            phoneme_representations: The enriched phoneme representations from the encoder.
            style_embedding: The style embedding from the GST module.
        """

        time_embedding = other_utils.create_positional_encoding(
            diffusion_step, self._timestep_embedding_dim)
        time_embedding = self._timestep_encoder(time_embedding)
        time_embedding = time_embedding.unsqueeze(-1)
        phoneme_representations = phoneme_representations.transpose(1, 2)

        output = self._prenet(noised_spectrogram)
        skip_output = None

        for block in self._residual_blocks:
            output, block_skip_output = block(output, time_embedding, phoneme_representations)

            if skip_output is not None:
                skip_output = skip_output + block_skip_output

            else:
                skip_output = block_skip_output

        output = self._postnet(skip_output)

        return output
