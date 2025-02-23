# -*- coding: utf-8 -*-
"""Contains the module creating embedding from the reference audio."""
from typing import Tuple
from typing import Optional

import torch

from layers.shared import fft_block


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
                 dropout_rate: float):
        """Initializes the reference embedder."""

        super().__init__()

        spec_channels, spec_length = reference_spectrogram_shape
        gst_count, gst_size = gst_shape

        self._gst = torch.nn.Parameter(
            torch.randn((gst_count, gst_size)),
            requires_grad=False)

        self._fft_blocks = torch.nn.ModuleList([
            fft_block.FFTBlock((spec_length, spec_channels),
                               4,
                               dropout_rate,
                               gst_size)
            for _ in range(n_ref_encoder_blocks)
        ])

        self._recurr_pool_key = torch.nn.Parameter(
            torch.randn(gst_size),
            requires_grad=True)
        self._recurr_pool = torch.nn.MultiheadAttention(
            embed_dim=gst_size,
            num_heads=4,
            batch_first=True,
            dropout=dropout_rate)

        self._post_enc = torch.nn.Sequential(
            torch.nn.Linear(spec_channels, gst_size),
            torch.nn.ReLU())

        self._gst_att = torch.nn.MultiheadAttention(
            embed_dim=gst_size,
            num_heads=4,
            batch_first=True)

    def forward(self,
                reference_spectrogram: torch.Tensor,
                spectrogram_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Converts the reference audio into the style embedding."""

        batch_size = reference_spectrogram.size(0)
        output = reference_spectrogram.transpose(1, 2)

        for fft_b in self._fft_blocks:
            output = fft_b(output, spectrogram_mask)

        output = self._post_enc(output)

        recurr_pool_key = self._recurr_pool_key.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1)
        output, _ = self._recurr_pool(recurr_pool_key,
                                      output,
                                      output,
                                      key_padding_mask=spectrogram_mask)

        gst = self._gst.unsqueeze(0).expand(batch_size, -1, -1)
        output, _ = self._gst_att(output, gst, gst)

        return output.squeeze(1)
