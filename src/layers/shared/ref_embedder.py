# -*- coding=utf-8 -*-
"""Contains the module creating embedding from the reference audio."""

from typing import Tuple

import torch


class ReferenceEmbedder(torch.nn.Module):
    """Converts the reference audio into GST-based style embedding.

    The reference embedder encodes the reference audio and converts it into the style embedding
    with use of the provided Global Style Tokens. This way the one-to-many mapping between the
    input phonemes and the expected audio is mitigated.
    """

    def __init__(self, reference_spectrogram_shape: Tuple[int, int], gst_shape: Tuple[int, int]):  # pylint: disable=unused-argument
        """Initializes the reference embedder."""

        super().__init__()

    def forward(self, reference_audio: torch.Tensor, gst: torch.Tensor) -> torch.Tensor:
        """Converts the reference audio into the style embedding.

        Args:
            reference_audio: The reference spectrogram.
            gst: The Global Style Tokens.

        Returns:
            The style embedding.
        """
