# -*- coding=utf-8 -*-
"""Contains utilities specific to the mel-to-linear spectrogram converter model."""

from typing import Tuple
from typing import Dict
from typing import Any
from dataclasses import dataclass
import os
import sys
import logging

import torch

from utilities import other as other_utils
from models import utils as shared_m_utils
from layers.shared import fft_block


class MelToLinConverter(torch.nn.Module):
    """Converts mel-spectrograms to linear-spectrograms."""

    def __init__(self, input_shape: Tuple[int, int],
                 output_dim: int,
                 n_blocks: int,
                 n_heads: int,
                 d_model: int,
                 fft_conv_channels: int,
                 dropout_rate: float):
        """Initializes the converter.

        Args:
            input_shape: The shape of the input mel-spectrogram.
            output_dim: The number of frequency bins in the output linear-spectrogram.
            n_blocks: The number of FFT blocks to use in the converter.
            n_heads: The number of attention heads to use in the FFT blocks.
            d_model: The size of the hidden dimension in the converter.
            fft_conv_channels: The number of convolutional channels to use in the FFT blocks.
            dropout_rate: The dropout rate to use in the FFT blocks.
        """
        super().__init__()

        input_dim, input_length = input_shape

        self._positional_encoding = torch.nn.Parameter(
            other_utils.create_positional_encoding(torch.arange(0, input_length),
                                                   input_dim),
            requires_grad=False
        )

        self._prenet = torch.nn.Sequential(
            torch.nn.Linear(input_dim, d_model),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate)
        )

        self._fft_blocks = torch.nn.Sequential(
            *[fft_block.FFTBlock(input_shape=(input_length, d_model),
                                 n_heads=n_heads,
                                 dropout_rate=dropout_rate,
                                 conv_channels=fft_conv_channels)
              for _ in range(n_blocks)]
        )

        self._postnet = torch.nn.Sequential(
            torch.nn.Linear(d_model, output_dim),
            torch.nn.Sigmoid()
        )

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """Converts the input mel-spectrogram to a linear-spectrogram.

        Args:
            mel_spectrogram: The input mel-spectrogram.

        Returns:
            The output linear-spectrogram.
        """
        mel_spectrogram = mel_spectrogram.transpose(1, 2)
        output = mel_spectrogram + self._positional_encoding
        output = self._prenet(output)
        output = self._fft_blocks(output)
        return self._postnet(output).transpose(1, 2)


@dataclass
class ModelComponents(shared_m_utils.BaseModelComponents):
    """Contains the components of the mel-to-linear spectrogram converter model."""

    converter: MelToLinConverter

    def train(self):
        self.converter.train()

    def eval(self):
        self.converter.eval()

    def parameters(self):
        return self.converter.parameters()


def create_model_components(input_spectrogram_shape: Tuple[int, int],
                            output_dim: int,
                            cfg: Dict[str, Any],
                            device: torch.device) -> ModelComponents:
    """Creates the model components based on the configuration.

    Args:
        input_spectrogram_shape: The shape of the input mel-spectrogram.
        output_dim: The number of frequency bins in the output linear-spectrogram.
        cfg: Contains the model hyperparameters. It contains the following keys:
            - n_blocks: The number of FFT blocks to use in the converter.
            - n_heads: The number of attention heads to use in the FFT blocks.
            - fft_conv_channels: The number of convolutional channels to use in the FFT blocks.
            - dropout_rate: The dropout rate to use in the FFT blocks.
            - d_model: The size of the hidden dimension in the converter.
    """

    return ModelComponents(
        converter=MelToLinConverter(input_shape=input_spectrogram_shape,
                                    output_dim=output_dim,
                                    n_blocks=cfg['n_blocks'],
                                    n_heads=cfg['n_heads'],
                                    d_model=cfg['d_model'],
                                    fft_conv_channels=cfg['fft_conv_channels'],
                                    dropout_rate=cfg['dropout_rate']).to(device)
    )


def load_model_components(components: ModelComponents, path: str):
    """Loads the model components from the specified path"""

    if not os.path.exists(path):
        logging.critical("Model components not found at '%s'.", path)
        sys.exit(1)

    shared_m_utils.try_load_state_dict(components.converter, os.path.join(path, 'converter.pth'))


def save_model_components(components: ModelComponents, path: str):
    """Saves the model components to the specified path"""

    os.makedirs(path, exist_ok=True)

    torch.save(components.converter.state_dict(), os.path.join(path, 'converter.pth'))
