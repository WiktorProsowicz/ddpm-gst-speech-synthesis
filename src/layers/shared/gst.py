# -*- coding: utf-8 -*-
"""Contains the GST module for the DDPM-GST-Speech-Gen model."""
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

    def __init__(self, gst_embedding_dim: int, gst_token_count: int):
        """Initializes the GST provider."""

        super().__init__()

        self._gst = torch.nn.Parameter(
            torch.randn(
                gst_token_count,
                gst_embedding_dim),
            requires_grad=False)

    def forward(self) -> torch.Tensor:
        """Provides the Global Style Tokens.

        Returns:
            The set of Global Style Token.
        """

        return self._gst
