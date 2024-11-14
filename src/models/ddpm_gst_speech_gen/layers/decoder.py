# coding=utf-8
"""Contains the decoder layer for the DDPM-GST-Speech-Gen model."""

from typing import Tuple, Optional

import torch


class Decoder(torch.nn.Module):
    """Predicts the noise added to the spectrogram within the diffusion process.

    The noise is predicted based on the:
    - diffusion step t
    - noised spectrogram at step t
    - enriched phoneme representations from the encoder (stretched by the length regulator)
    - style embedding from the GST module

    The decoder's output can be interpreted as the amount of noise that should be added to
    the 'clean' spectrogram to create the 'noised' spectrogram at the diffusion step t.
    """

    def __init__(self, input_noise_shape: Tuple[int, int]):
        """Initializes the decoder."""

        super().__init__()

    def forward(self,
                diffusion_step: torch.Tensor,
                noised_spectrogram: torch.Tensor,
                phoneme_representations: torch.Tensor,
                style_embedding: Optional[torch.Tensor]) -> torch.Tensor:
        """Predicts the noise added to the spectrogram.

        Args:
            diffusion_step: The diffusion step t.
            noised_spectrogram: The spectrogram at step t.
            phoneme_representations: The enriched phoneme representations from the encoder.
                The shape of the representations is expected to be equal to the shape of the
                input noise.
            style_embedding: The style embedding from the GST module.
        """
