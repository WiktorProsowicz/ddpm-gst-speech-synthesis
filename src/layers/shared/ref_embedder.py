# -*- coding: utf-8 -*-
"""Contains the module creating embedding from the reference audio."""
from typing import Tuple
from typing import Iterator
import itertools

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
                 dropout_rate: float,
                 use_gst_att: bool):
        """Initializes the reference embedder."""

        super().__init__()

        _, spec_length = reference_spectrogram_shape
        gst_count, gst_size = gst_shape

        self._use_gst_att = use_gst_att
        self._chosen_spec_bins = 80

        self._gst = torch.nn.Parameter(
            torch.randn((gst_count, gst_size)),
            requires_grad=False)

        self._pre_enc = torch.nn.Sequential(
            torch.nn.Linear(self._chosen_spec_bins, gst_size),
            torch.nn.ReLU())

        self._fft_blocks = torch.nn.ModuleList([
            fft_block.FFTBlock((spec_length, gst_size),
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

        self._gst_att = torch.nn.MultiheadAttention(
            embed_dim=gst_size,
            num_heads=1,
            batch_first=True)

    def forward(self,
                reference_audio: torch.Tensor,
                spectrogram_mask: torch.Tensor) -> torch.Tensor:
        """Converts the reference audio into the style embedding.

        Args:
            reference_audio: The reference spectrogram.

        Returns:
            The style embedding.
        """

        encoded_ref = self._obtain_reference_embedding(reference_audio,
                                                       spectrogram_mask)

        if not self._use_gst_att:
            return encoded_ref

        att_out, _ = self._obtain_gst_output(encoded_ref)

        return att_out.squeeze(1)

    def gst_att_params(self) -> Iterator[torch.nn.Parameter]:
        return self._gst_att.parameters()

    def obtain_gst_weights(self,
                           reference_audio: torch.Tensor,
                           spectrogram_mask: torch.Tensor):

        encoded_ref = self._obtain_reference_embedding(reference_audio,
                                                       spectrogram_mask)

        _, att_weights = self._obtain_gst_output(encoded_ref)

        return att_weights.squeeze(1)

    def get_style_embedding_from_weights(self, weights: torch.Tensor):

        _, _, W_v = self._gst_att.in_proj_weight.chunk(3)  # pylint: disable=all
        _, _, b_v = self._gst_att.in_proj_bias.chunk(3)  # pylint: disable=all

        gst = self._gst.unsqueeze(0).expand(weights.size(0), -1, -1)

        V = gst @ W_v.T + b_v  # pylint: disable=all

        context = weights.unsqueeze(1) @ V

        embedding = context @ self._gst_att.out_proj.weight.T + self._gst_att.out_proj.bias
        return embedding.squeeze(1)

    def _obtain_reference_embedding(self,
                                    reference_audio: torch.Tensor,
                                    spectrogram_mask: torch.Tensor):

        batch_size = reference_audio.size(0)
        reference_audio = reference_audio[:, :self._chosen_spec_bins, :]
        output = reference_audio.transpose(1, 2)

        output = self._pre_enc(output)

        for fft_b in self._fft_blocks:
            output = fft_b(output, spectrogram_mask)

        recurr_pool_key = self._recurr_pool_key.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1)
        output, _ = self._recurr_pool(recurr_pool_key,
                                      output,
                                      output,
                                      key_padding_mask=spectrogram_mask)

        output = output.squeeze(1)
        encoded_ref = output

        return encoded_ref

    def _obtain_gst_output(self, reference_embedding: torch.Tensor):

        reference_embedding = reference_embedding.unsqueeze(1)

        gst = self._gst.unsqueeze(0).expand(reference_embedding.shape[0], -1, -1)
        att_output, att_weights = self._gst_att(reference_embedding, gst, gst)

        return att_output, att_weights
