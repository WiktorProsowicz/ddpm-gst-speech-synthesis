# -*- coding: utf-8 -*-
"""Contains the decoder layer for the DDPM-GST-Speech-Gen model."""
from typing import Optional
from typing import Tuple

import torch


class _ResidualBlock(torch.nn.Module):

    def __init__(self,
                 block_input_channels: int,
                 decoder_input_channels: int,
                 dropout_rate: float):

        super().__init__()

        self._cond_time = torch.nn.Sequential(
            torch.nn.Linear(decoder_input_channels, block_input_channels),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout_rate)
        )

        self._cond_phonemes = torch.nn.Sequential(torch.nn.Conv1d(
            decoder_input_channels, decoder_input_channels, kernel_size=3, padding='same'),
            torch.nn.SiLU(),
            torch.nn.Dropout1d(dropout_rate)
        )

        self._conv1 = torch.nn.Sequential(
            torch.nn.GroupNorm(8, block_input_channels),
            torch.nn.SiLU(),
            torch.nn.Conv2d(
                block_input_channels,
                block_input_channels,
                kernel_size=3,
                padding='same'),
            torch.nn.Dropout2d(dropout_rate)
        )

        self._conv2 = torch.nn.Sequential(
            torch.nn.GroupNorm(8, block_input_channels),
            torch.nn.SiLU(),
            torch.nn.Conv2d(
                block_input_channels,
                block_input_channels,
                kernel_size=3,
                padding='same'),
            torch.nn.Dropout2d(dropout_rate)
        )

    def forward(self, input_noise: torch.Tensor, timestep_embedding: torch.Tensor,
                phoneme_representations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        time_cond = self._cond_time(timestep_embedding).unsqueeze(-1).unsqueeze(-1)
        phoneme_cond = self._cond_phonemes(phoneme_representations).unsqueeze(1)

        output = self._conv1(input_noise)
        output = output + time_cond + phoneme_cond
        output = self._conv2(output)
        output = output + input_noise

        return output


def _create_residual_blocks(n_blocks: int,
                            block_input_channels: int,
                            decoder_input_channels: int,
                            dropout_rate) -> torch.nn.ModuleList:

    blocks = [
        _ResidualBlock(
            block_input_channels=block_input_channels,
            decoder_input_channels=decoder_input_channels,
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
                 skip_connections_channels: int,  # pylint: disable=unused-argument
                 dropout_rate: float):
        """Initializes the decoder."""

        super().__init__()

        input_height, _ = input_noise_shape

        self._timestep_embedding_dim = timestep_embedding_dim

        self._timestep_encoder = torch.nn.Sequential(
            torch.nn.Linear(timestep_embedding_dim, input_height),
            torch.nn.SiLU(),
            torch.nn.Linear(input_height, input_height),
            torch.nn.SiLU()
        )

        self._prenet = torch.nn.Sequential(
            torch.nn.Conv2d(1, internal_channels, kernel_size=3, padding='same'),
            torch.nn.Tanh(),
        )

        self._residual_blocks = _create_residual_blocks(
            n_res_blocks, internal_channels, input_height, dropout_rate)

        self._postnet = torch.nn.Sequential(
            torch.nn.Conv2d(
                internal_channels,
                1,
                kernel_size=3,
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

        time_embedding = self._create_time_embedding(diffusion_step)
        time_embedding = self._timestep_encoder(time_embedding)

        phoneme_representations = phoneme_representations.transpose(1, 2)
        noised_spectrogram = noised_spectrogram.unsqueeze(1)

        output = self._prenet(noised_spectrogram)

        for block in self._residual_blocks:
            output = block(output, time_embedding, phoneme_representations)

        return self._postnet(output).squeeze(1)

    def _create_time_embedding(self, diffusion_step: torch.Tensor) -> torch.Tensor:
        """Creates the Sinusoidal Positional Embedding for the input time steps."""

        i_steps = torch.arange(0, self._timestep_embedding_dim // 2, device=diffusion_step.device)
        factor = 10000 ** (i_steps / (self._timestep_embedding_dim // 2))

        t_embedding = diffusion_step[:, None].repeat(1, self._timestep_embedding_dim // 2)
        t_embedding = t_embedding / factor

        return torch.cat([torch.sin(t_embedding), torch.cos(t_embedding)], dim=-1)
