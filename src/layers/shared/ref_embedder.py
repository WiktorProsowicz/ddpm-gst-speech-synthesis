# -*- coding=utf-8 -*-
"""Contains the module creating embedding from the reference audio."""

from typing import Tuple

import torch


class _DownsamplingBlock(torch.nn.Module):
    """Downsamples and encodes the input spectrogram."""

    def __init__(self, input_channels: int, output_channels: int, dropout_rate: float):

        super().__init__()

        self._convs = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels,
                            output_channels,
                            kernel_size=3,
                            padding='same'),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(output_channels),
            torch.nn.Dropout2d(dropout_rate),
            torch.nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                stride=2,
                padding='same'),
            torch.nn.ReLU(),
        )

    def forward(self, input_spec: torch.Tensor) -> torch.Tensor:
        """Downsamples and encodes the input spectrogram."""

        return self._convs(input_spec)


def _create_downsampling_blocks(input_channels: int, output_channels: int, dropout_rate: float,
                                num_blocks: int) -> torch.nn.Module:
    """Creates the downsampling blocks."""

    blocks = [_DownsamplingBlock(input_channels, output_channels, dropout_rate)]

    for _ in range(num_blocks - 1):
        blocks.append(_DownsamplingBlock(output_channels, output_channels, dropout_rate))

    return torch.nn.Sequential(*blocks)


class ReferenceEmbedder(torch.nn.Module):
    """Converts the reference audio into GST-based style embedding.

    The reference embedder encodes the reference audio and converts it into the style embedding
    with use of the provided Global Style Tokens. This way the one-to-many mapping between the
    input phonemes and the expected audio is mitigated.
    """

    def __init__(self,
                 reference_spectrogram_shape: Tuple[int, int],
                 gst_shape: Tuple[int, int],
                 n_ref_encoder_blocks: int,
                 n_attention_heads: int,
                 dropout_rate: float):
        """Initializes the reference embedder."""

        super().__init__()

        spec_channels, _ = reference_spectrogram_shape
        _, gst_size = gst_shape

        # Calculate the most suitable number of channels for the downsampling
        # blocks so that the out_channels * downsampled_height is close to the
        # gst_size.
        downsampled_height = (spec_channels // 2**n_ref_encoder_blocks)
        blocks_out_ch = gst_size // downsampled_height

        self._down_blocks = _create_downsampling_blocks(
            spec_channels, blocks_out_ch, dropout_rate, n_ref_encoder_blocks)

        self._recurr_pool = torch.nn.LSTM(
            input_size=blocks_out_ch * downsampled_height,
            hidden_size=gst_size,
            num_layers=1,
            batch_first=True
        )

        self._post_enc = torch.nn.Sequential(
            torch.nn.Linear(gst_size, gst_size),
            torch.nn.ReLU())

        self._gst_att = torch.nn.MultiheadAttention(
            embed_dim=gst_size,
            num_heads=n_attention_heads,
            dropout=dropout_rate)

    def forward(self, reference_audio: torch.Tensor, gst: torch.Tensor) -> torch.Tensor:
        """Converts the reference audio into the style embedding.

        Args:
            reference_audio: The reference spectrogram.
            gst: The Global Style Tokens.

        Returns:
            The style embedding.
        """

        reference_audio = reference_audio.unsqueeze(1)
        gst = gst.unsqueeze(0)

        output = self._down_blocks(reference_audio).transpose(1, 3)

        batch_size, channels, height, width = output.shape
        output = output.reshape(batch_size, width, channels * height)

        _, (_, final_state) = self._recurr_pool(output)

        encoded_ref = self._post_enc(final_state.squeeze(0))
        encoded_ref = encoded_ref.unsqueeze(1)

        att_out, _ = self._gst_att(encoded_ref, gst, gst)

        return att_out.squeeze(1)
