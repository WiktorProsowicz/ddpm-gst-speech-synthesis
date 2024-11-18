# -*- coding: utf-8 -*-
"""Contains the decoder layer for the DDPM-GST-Speech-Gen model."""
from typing import Optional
from typing import Tuple

import torch


class _ResidualBlock(torch.nn.Module):

    def __init__(self, skip_channels: int, input_channels: int):

        super().__init__()

        self._norm = torch.nn.BatchNorm1d(input_channels)
        self._conv1d = torch.nn.Conv1d(
            input_channels,
            input_channels,
            kernel_size=3,
            padding='same')
        self._skip_connection = torch.nn.Conv1d(input_channels, skip_channels, kernel_size=1)
        self._cond_timestep_conv = torch.nn.Conv1d(input_channels, input_channels, kernel_size=1)
        self._cond_phonemes_conv = torch.nn.Conv1d(input_channels, input_channels, kernel_size=1)
        self._output_conv = torch.nn.Conv1d(
            input_channels, input_channels, kernel_size=1, padding='same')

    def forward(self, input_noise: torch.Tensor, timestep_embedding: torch.Tensor,
                phoneme_representations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        output = self._norm(input_noise)
        output = self._cond_timestep_conv(timestep_embedding) + output
        output = self._cond_phonemes_conv(phoneme_representations) + output
        output = torch.nn.functional.tanh(output) * torch.nn.functional.sigmoid(output)

        skip_connection = self._skip_connection(output)
        output = self._output_conv(output) + input_noise

        return output, skip_connection


def _create_residual_blocks(n_blocks: int, skip_connections_channel: int,
                            input_channels: int) -> torch.nn.ModuleList:

    blocks = [
        _ResidualBlock(
            skip_channels=skip_connections_channel,
            input_channels=input_channels)
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
                 ):
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

        skip_connections_channel = 512
        self._residual_blocks = _create_residual_blocks(
            12, skip_connections_channel, input_channels)

        self._postnet = torch.nn.Sequential(
            torch.nn.Conv1d(
                skip_connections_channel,
                input_channels,
                kernel_size=3,
                padding='same'),
        )

    def forward(self,
                diffusion_step: torch.Tensor,
                noised_spectrogram: torch.Tensor,
                phoneme_representations: torch.Tensor,
                style_embedding: Optional[torch.Tensor]) -> torch.Tensor:
        """Predicts the noise added to the spectrogram.

        Args:
            diffusion_step: The diffusion step t.
            noised_spectrogram: The spectrogram at the diffusion step t.
            phoneme_representations: The enriched phoneme representations from the encoder.
            style_embedding: The style embedding from the GST module.
        """

        time_embedding = self._create_time_embedding(diffusion_step)
        time_embedding = self._timestep_encoder(time_embedding)
        time_embedding = time_embedding.unsqueeze(-1)
        phoneme_representations = phoneme_representations.transpose(1, 2)

        output = noised_spectrogram
        skip_output = None

        for block in self._residual_blocks:
            output, block_skip_output = block(output, time_embedding, phoneme_representations)

            if skip_output is not None:
                skip_output = skip_output + block_skip_output

            else:
                skip_output = block_skip_output

        output = self._postnet(skip_output)

        return output

    def _create_time_embedding(self, diffusion_step: torch.Tensor) -> torch.Tensor:
        """Creates the Sinusoidal Positional Embedding for the input time steps."""

        i_steps = torch.arange(0, self._timestep_embedding_dim // 2, device=diffusion_step.device)
        factor = 10000 ** (i_steps / (self._timestep_embedding_dim // 2))

        t_embedding = diffusion_step[:, None].repeat(1, self._timestep_embedding_dim // 2)
        t_embedding = t_embedding / factor

        return torch.cat([torch.sin(t_embedding), torch.cos(t_embedding)], dim=-1)
