# -*- coding: utf-8 -*-
"""Contains the definition of the decoder layer for the GST predictor."""
from typing import Optional

import torch

from utilities import other as other_utils


class _ResBlock(torch.nn.Module):
    """Residual block used in the decoder."""

    def __init__(self,
                 n_channels: int,
                 phoneme_embedding_dim: int,
                 dropout_rate: float,):

        super().__init__()

        self._layer1 = torch.nn.Sequential(
            torch.nn.GroupNorm(8, n_channels),
            torch.nn.Conv1d(n_channels, n_channels, kernel_size=3, padding='same'),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout_rate),
        )

        self._cond_proj = torch.nn.Sequential(
            torch.nn.Linear(phoneme_embedding_dim, n_channels),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout_rate),
        )

        self._layer2 = torch.nn.Sequential(
            torch.nn.GroupNorm(8, n_channels),
            torch.nn.Conv1d(n_channels, n_channels, kernel_size=3, padding='same'),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout_rate),
        )

    def forward(self,
                input_tensor: torch.Tensor,
                timestep_embedding: torch.Tensor,
                phoneme_embedding: Optional[torch.Tensor]) -> torch.Tensor:

        output = self._layer1(input_tensor)

        if phoneme_embedding is not None:
            output = self._cond_proj(phoneme_embedding).transpose(1, 2) + output

        output = output + timestep_embedding.unsqueeze(-1)

        output = self._layer2(output)

        return output + input_tensor


class Decoder(torch.nn.Module):
    """Predicts the diffusion noise based on the encoded phonemes and diffusion timestep."""

    def __init__(self,
                 input_gst_size: int,
                 phoneme_embedding_dim: int,
                 timestep_embedding_size: int,
                 internal_channels: int,
                 n_blocks: int,
                 dropout_rate: float):

        super().__init__()

        self._timestep_embedding_dim = timestep_embedding_size

        self._timestep_encoder = torch.nn.Sequential(
            torch.nn.Linear(timestep_embedding_size, timestep_embedding_size),
            torch.nn.SiLU(),
        )

        self._pre_net = torch.nn.Conv1d(
            1, internal_channels, kernel_size=1)

        self._res_blocks = torch.nn.ModuleList(
            [_ResBlock(internal_channels,
                       phoneme_embedding_dim,
                       dropout_rate) for _ in range(n_blocks)]
        )

        self._postnet = torch.nn.Sequential(torch.nn.Conv1d(internal_channels, 1, kernel_size=1),
                                            torch.nn.Linear(input_gst_size, input_gst_size))

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

        output = self._pre_net(input_gst.unsqueeze(1))

        time_embedding = other_utils.create_positional_encoding(
            diffusion_step, self._timestep_embedding_dim)
        time_embedding = self._timestep_encoder(time_embedding)

        for res_block in self._res_blocks:
            output = res_block(output, time_embedding, phoneme_embedding)

        return self._postnet(output).squeeze(1)
