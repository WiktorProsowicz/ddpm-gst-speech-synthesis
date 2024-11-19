# -*- coding: utf-8 -*-
"""Contains the GST module for the DDPM-GST-Speech-Gen model."""
from typing import Tuple

import torch


class GSTProvider(torch.nn.Module):
    """Provides the Global Style Token for the DDPM-GST-Speech-Gen model.

    GST stands for Global Style Token. This concept has been introduced in:
    https://arxiv.org/abs/1803.09017

    The module contains fixed random-initialized vectors that are used to provide the
    conditioning information for spectrogram generation. The weighed sum of the tokens may
    serve as a style embedding. The interpretation of the tokens depends on the task and the
    model architecture.
    """

    def __init__(self, gst_embedding_dim: int, gst_token_count: int):  # pylint: disable=unused-argument
        """Initializes the GST provider."""

        super().__init__()

    def forward(self) -> torch.Tensor:
        """Provides the Global Style Tokens.

        Returns:
            The set of Global Style Token.
        """


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
