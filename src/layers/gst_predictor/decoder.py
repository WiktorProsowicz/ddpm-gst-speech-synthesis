# -*- coding: utf-8 -*-
"""Contains the definition of the decoder layer for the GST predictor."""
from typing import Optional
from typing import Tuple

import torch

from utilities import other as other_utils


class _ConvBlock(torch.nn.Module):
    """Convolutional block used in the decoder."""

    def __init__(self,
                 internal_channels: int,
                 dropout_rate: float,):
        super().__init__()

        self._conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=internal_channels,
                out_channels=internal_channels,
                kernel_size=1,
                padding='same'
            ),
            torch.nn.GroupNorm(8, internal_channels),
            torch.nn.SiLU(),
            torch.nn.Dropout1d(dropout_rate),
        )

        self._conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=internal_channels,
                out_channels=internal_channels,
                kernel_size=1,
                padding='same'
            ),
            torch.nn.GroupNorm(8, internal_channels),
            torch.nn.SiLU(),
            torch.nn.Dropout1d(dropout_rate),
        )

        self._skip_conv = torch.nn.Conv1d(
            in_channels=internal_channels,
            out_channels=internal_channels,
            kernel_size=1,
            padding='same'
        )

    def forward(self, input_tensor: torch.Tensor, timestep_embedding: torch.Tensor,
                phoneme_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        output = self._conv1(input_tensor)
        output = output + phoneme_embedding
        output = output + timestep_embedding
        output = self._conv2(output)

        return output + input_tensor, self._skip_conv(output)


class Decoder(torch.nn.Module):
    """Predicts the diffusion noise based on the encoded phonemes and diffusion timestep."""

    def __init__(self,
                 input_gst_size: int,
                 timestep_embedding_size: int,
                 internal_channels: int,
                 n_conv_blocks: int,
                 dropout_rate: float):

        super().__init__()

        self._timestep_embedding_dim = timestep_embedding_size

        self._timestep_encoder = torch.nn.Sequential(
            torch.nn.Linear(timestep_embedding_size, input_gst_size),
            torch.nn.SiLU(),
        )

        self._prenet = torch.nn.Sequential(
            torch.nn.Conv1d(1, internal_channels, kernel_size=1, padding='same'),
            torch.nn.SiLU()
        )

        self._phoneme_cond = torch.nn.Sequential(
            torch.nn.Linear(input_gst_size, input_gst_size),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout_rate),
        )

        self._timestep_cond = torch.nn.Sequential(
            torch.nn.Linear(input_gst_size, input_gst_size),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout_rate),
        )

        self._conv_blocks = torch.nn.ModuleList(
            [_ConvBlock(internal_channels,
                        dropout_rate)
             for _ in range(n_conv_blocks)]
        )

        self._postnet = torch.nn.Sequential(
            torch.nn.Conv1d(n_conv_blocks * internal_channels, 1, kernel_size=1, padding='same'),
        )

    def forward(self, input_gst: torch.Tensor,
                diffusion_step: torch.Tensor,
                phoneme_embedding: Optional[torch.Tensor]) -> torch.Tensor:
        """Predicts the diffusion noise based on the encoded phonemes and diffusion timestep.

        Args:
            input_gst: Noised gst at timestep t.
            diffusion_step: The diffusion step t.
            phoneme_embedding: Output of the phoneme encoder.

        Returns:
            Predicted diffusion noise.
        """

        input_gst = input_gst.unsqueeze(1)

        time_embedding = other_utils.create_positional_encoding(
            diffusion_step, self._timestep_embedding_dim)
        time_embedding = self._timestep_encoder(time_embedding)

        time_embedding = self._timestep_cond(time_embedding)
        phoneme_embedding = self._phoneme_cond(phoneme_embedding)

        time_embedding = time_embedding.unsqueeze(1)
        phoneme_embedding = phoneme_embedding.unsqueeze(1)

        output = self._prenet(input_gst)
        total_skip_output = None

        for conv_block in self._conv_blocks:
            output, skip_output = conv_block(output,
                                             time_embedding,
                                             phoneme_embedding)

            if total_skip_output is None:
                total_skip_output = skip_output

            else:
                total_skip_output = torch.cat((total_skip_output, skip_output), dim=1)

        return self._postnet(total_skip_output).squeeze(1)
