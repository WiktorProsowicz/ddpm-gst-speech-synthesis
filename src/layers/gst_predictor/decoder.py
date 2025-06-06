# -*- coding: utf-8 -*-
"""Contains the definition of the decoder layer for the GST predictor."""
from typing import Optional

import torch

from utilities import other as other_utils


class _ResBlock(torch.nn.Module):
    """Residual block used in the decoder."""

    def __init__(self,
                 n_channels: int,
                 input_gst_size: int,
                 dropout_rate: float,):

        super().__init__()

        self._conv_layers = torch.nn.ModuleList(
            [torch.nn.Sequential(
                torch.nn.GroupNorm(8, n_channels),
                torch.nn.Conv1d(n_channels, n_channels, kernel_size=5, padding='same'),
                torch.nn.SiLU(),
                torch.nn.Dropout(dropout_rate),
            ) for _ in range(3)]
        )

        self._attention = torch.nn.MultiheadAttention(
            input_gst_size,
            4,
            dropout_rate,
            batch_first=True)

    def forward(self,
                input_tensor: torch.Tensor,
                timestep_embedding: torch.Tensor,
                phoneme_cond: Optional[torch.Tensor],
                phoneme_cond_mask: Optional[torch.Tensor]) -> torch.Tensor:

        output = self._conv_layers[0](input_tensor)

        if phoneme_cond is not None:
            att_output, _ = self._attention(output,
                                            phoneme_cond,
                                            phoneme_cond,
                                            key_padding_mask=phoneme_cond_mask)
            output = self._conv_layers[1](att_output + output)

        output = output + timestep_embedding.unsqueeze(1)
        output = self._conv_layers[2](output)

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

        self._gst_size = input_gst_size
        self._timestep_embedding_dim = timestep_embedding_size

        self._timestep_encoder = torch.nn.Sequential(
            torch.nn.Linear(timestep_embedding_size, timestep_embedding_size),
            torch.nn.SiLU(),
        )

        self._pre_net = torch.nn.Conv1d(
            1, internal_channels, kernel_size=1)

        self._res_blocks = torch.nn.ModuleList(
            [_ResBlock(internal_channels,
                       input_gst_size,
                       dropout_rate) for _ in range(n_blocks)]
        )

        self._postnet_query = torch.nn.Parameter(torch.empty(1, 1, input_gst_size),
                                                 requires_grad=True)
        torch.nn.init.xavier_uniform_(self._postnet_query)
        self._postnet = torch.nn.MultiheadAttention(input_gst_size,
                                                    1,
                                                    dropout_rate,
                                                    batch_first=True)

    def forward(self, input_gst: torch.Tensor,
                diffusion_step: torch.Tensor,
                phoneme_cond: Optional[torch.Tensor],
                phoneme_cond_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Predicts the diffusion noise based on the encoded phonemes and diffusion timestep.

        Args:
            input_gst: Noised gst at timestep t.
            diffusion_step: The diffusion step t.
            phoneme_cond: Output of the phoneme encoder.

        Returns:
            Predicted diffusion noise.
        """

        output = self._pre_net(input_gst.unsqueeze(1))

        time_embedding = other_utils.create_positional_encoding(
            diffusion_step, self._timestep_embedding_dim)
        time_embedding = self._timestep_encoder(time_embedding)

        for res_block in self._res_blocks:
            output = res_block(output, time_embedding, phoneme_cond, phoneme_cond_mask)

        postnet_query = self._postnet_query.expand(input_gst.size(0), -1, -1)

        att_out, _ = self._postnet(postnet_query, output, output)

        return att_out.squeeze(1)

    @property
    def gst_size(self) -> int:
        """Returns the size of the GST input and output."""

        return self._gst_size
