# -*- coding: utf-8 -*-
"""Contains the module creating embedding from the reference audio."""
from typing import Tuple
from typing import Optional

import torch

from layers.shared import fft_block
from utilities import other as other_utils


def create_gst(n, dim):
    """Generates random Global Style Tokens.

    The created tokens are orthogonal to ensure there's no overlap between them.
    """

    # base_matrix = torch.randn(n, dim)
    # gst, _ = torch.linalg.qr(base_matrix) # pylint: disable=not-callable

    gst = torch.zeros(n, dim)
    torch.nn.init.normal_(gst, mean=0, std=1)

    return gst


class ReferenceEmbedder(torch.nn.Module):
    """Converts the reference audio into GST-based style embedding.

    The reference embedder encodes the reference audio and converts it into the style embedding
    with use of the Global Style Tokens. This way the one-to-many mapping between the
    input phonemes and the expected audio is mitigated.
    """

    def __init__(self,
                 reference_spectrogram_shape: Tuple[int, int],
                 gst_shape: Tuple[int, int],
                 n_ref_encoder_blocks: int,
                 dropout_rate: float,
                 fft_conv_channels: int):
        """Initializes the reference embedder."""

        super().__init__()

        spec_channels, spec_length = reference_spectrogram_shape
        gst_count, gst_size = gst_shape

        self._gst = torch.nn.Parameter(
            create_gst(gst_count, gst_size),
            requires_grad=True)

        self._positional_encoding = torch.nn.Parameter(
            other_utils.create_positional_encoding(torch.arange(0, spec_length),
                                                   gst_size),
            requires_grad=False
        )

        self._pre_enc = torch.nn.Sequential(
            torch.nn.Linear(spec_channels, gst_size),
            torch.nn.SiLU())

        self._fft_blocks = torch.nn.ModuleList([
            fft_block.FFTBlock((spec_length, gst_size),
                               4,
                               dropout_rate,
                               fft_conv_channels)
            for _ in range(n_ref_encoder_blocks)
        ])

        self._recurr_pool_query = torch.nn.Parameter(
            torch.randn(gst_size),
            requires_grad=True)
        self._recurr_pool = torch.nn.MultiheadAttention(
            embed_dim=gst_size,
            num_heads=16,
            batch_first=True,
            dropout=dropout_rate)

        self._gst_att = torch.nn.MultiheadAttention(
            embed_dim=gst_size,
            num_heads=16,
            batch_first=True,)

    def forward(self,
                reference_spectrogram: torch.Tensor,
                spectrogram_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Converts the reference audio into the style embedding."""

        batch_size = reference_spectrogram.size(0)
        reverse_mask = None

        if spectrogram_mask is not None:
            reverse_mask = torch.logical_not(spectrogram_mask).unsqueeze(-1)

        output = reference_spectrogram.transpose(1, 2)
        output = self._pre_enc(output)
        output = output + self._positional_encoding

        for fft_b in self._fft_blocks:
            output = fft_b(output, spectrogram_mask, reverse_mask)

        recurr_pool_query = self._recurr_pool_query.unsqueeze(
            0).unsqueeze(0).expand(batch_size, -1, -1)
        output, _ = self._recurr_pool(recurr_pool_query,
                                      output,
                                      output,
                                      key_padding_mask=spectrogram_mask)

        gst = self._gst.unsqueeze(0).expand(batch_size, -1, -1)
        output, _ = self._gst_att(output, gst, gst)

        return output.squeeze(1)
