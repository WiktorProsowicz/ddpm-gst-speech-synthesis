# -*- coding: utf-8 -*-
"""Contains the module creating embedding from the reference audio."""
from typing import Tuple

import torch


class _DownsamplingBlock(torch.nn.Module):
    """Downsamples and encodes the input spectrogram."""

    def __init__(self, input_channels: int, output_channels: int, dropout_rate: float):

        super().__init__()

        self._convs = torch.nn.Sequential(
            torch.nn.BatchNorm2d(input_channels),
            torch.nn.Dropout2d(dropout_rate),
            torch.nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=5,
                padding='same'),
            torch.nn.MaxPool2d(3, padding=(1, 1), stride=(1, 2)),
            torch.nn.SiLU(),
        )

    def forward(self, input_spec: torch.Tensor) -> torch.Tensor:
        """Downsamples and encodes the input spectrogram."""

        output = self._convs(input_spec)
        return output


def _create_downsampling_blocks(dropout_rate: float,
                                num_blocks: int) -> torch.nn.Module:
    """Creates the downsampling blocks."""

    blocks = [_DownsamplingBlock(1, 4 ** (num_blocks - 1), dropout_rate)]

    for i in range(num_blocks - 1, 0, -1):
        blocks.append(_DownsamplingBlock(4**(i), 4**(i - 1), dropout_rate))

    return torch.nn.Sequential(*blocks)


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

        self._spec_channels = 20  # How many first frequency bins to take
        gst_count, gst_size = gst_shape

        assert self._spec_channels <= reference_spectrogram_shape[1]

        self._gst = torch.nn.Parameter(
            torch.randn((gst_count, gst_size)),
            requires_grad=True)

        self._down_blocks = _create_downsampling_blocks(
            dropout_rate, n_ref_encoder_blocks)

        self._recurr_pool = torch.nn.LSTM(
            input_size=self._spec_channels,
            hidden_size=gst_size,
            num_layers=1,
            batch_first=True
        )

        self._gst_att = torch.nn.MultiheadAttention(
            embed_dim=gst_size,
            num_heads=1,
            batch_first=True)

        self._use_gst_att = use_gst_att

    def forward(self, reference_audio: torch.Tensor) -> torch.Tensor:
        """Converts the reference audio into the style embedding.

        Args:
            reference_audio: The reference spectrogram.

        Returns:
            The style embedding.
        """

        encoded_ref = self._obtain_reference_embedding(reference_audio)

        if not self._use_gst_att:
            return encoded_ref

        att_out, _ = self._obtain_gst_output(encoded_ref)

        return att_out.squeeze(1)

    def obtain_gst_weights(self, reference_audio: torch.Tensor):

        encoded_ref = self._obtain_reference_embedding(reference_audio)

        _, att_weights = self._obtain_gst_output(encoded_ref)

        return att_weights.squeeze(1)
    
    def get_style_embedding_from_weights(self, weights: torch.Tensor):

        _, _, W_v = self._gst_att.in_proj_weight.chunk(3) # pylint: disable=unpacking-non-sequence
        _, _, b_v = self._gst_att.in_proj_bias.chunk(3) # pylint: disable=unpacking-non-sequence

        gst = self._gst.unsqueeze(0).expand(weights.size(0), -1, -1)

        V = gst @ W_v.T + b_v

        context = weights.unsqueeze(1) @ V

        embedding = context @ self._gst_att.out_proj.weight.T + self._gst_att.out_proj.bias
        return embedding.squeeze(1)

    def _obtain_reference_embedding(self, reference_audio: torch.Tensor):

        reference_audio = reference_audio.unsqueeze(1)

        output = self._down_blocks(reference_audio[:, :, :self._spec_channels])

        output = output.squeeze(1).transpose(1, 2)
        
        _, (_, final_state) = self._recurr_pool(output)

        encoded_ref = final_state.squeeze(0)
        encoded_ref = torch.nn.functional.tanh(encoded_ref)

        return encoded_ref
    
    def _obtain_gst_output(self, reference_embedding: torch.Tensor):

        reference_embedding = reference_embedding.unsqueeze(1)

        gst = self._gst.unsqueeze(0).expand(reference_embedding.shape[0], -1, -1)
        gst = torch.nn.functional.tanh(gst)
        att_output, att_weights = self._gst_att(reference_embedding, gst, gst)

        return att_output, att_weights