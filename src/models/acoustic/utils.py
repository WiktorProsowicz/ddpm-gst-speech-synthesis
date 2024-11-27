# -*- coding: utf-8 -*-
"""Contains utilities specific fot the acoustic model."""
import itertools
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import torch

from layers.acoustic import decoder as m_decoder
from layers.acoustic import encoder as m_encoder
from layers.shared import duration_predictor as m_dp
from layers.shared import gst as m_gst
from layers.shared import length_regulator as m_lr
from utilities import other as other_utils


@dataclass
class ModelComponents:
    """Contains the components of the acoustic model."""
    encoder: m_encoder.Encoder
    decoder: m_decoder.Decoder
    length_regulator: m_lr.LengthRegulator
    duration_predictor: m_dp.DurationPredictor
    gst: Optional[m_gst.GSTProvider]
    embedder: Optional[m_gst.ReferenceEmbedder]

    def parameters(self):
        """Returns the parameters of the model."""

        if self.gst and self.embedder:
            return itertools.chain(
                self.encoder.parameters(),
                self.decoder.parameters(),
                self.length_regulator.parameters(),
                self.duration_predictor.parameters(),
                self.gst.parameters(),
                self.embedder.parameters()
            )

        return itertools.chain(
            self.encoder.parameters(),
            self.decoder.parameters(),
            self.length_regulator.parameters(),
            self.duration_predictor.parameters()
        )

    def eval(self):
        """Sets the model to evaluation mode."""

        self.encoder.eval()
        self.decoder.eval()
        self.length_regulator.eval()
        self.duration_predictor.eval()

        if self.gst and self.embedder:
            self.gst.eval()
            self.embedder.eval()

    def train(self):
        """Sets the model to training mode."""

        self.encoder.train()
        self.decoder.train()
        self.length_regulator.train()
        self.duration_predictor.train()

        if self.gst and self.embedder:
            self.gst.train()
            self.embedder.train()


def create_model_components(output_spectrogram_shape: Tuple[int, int],
                            input_phonemes_shape: Tuple[int, int],
                            cfg: Dict[str, Any],
                            device: torch.device) -> ModelComponents:
    """Creates the components of the acoustic model.

    Args:
        output_spectrogram_shape: The shape of the output spectrogram.
        input_phonemes_shape: The shape of the input one-hot encoded phonemes.
        cfg: The internal configuration of the model. It contains the following keys:
            - n_heads: The number of attention heads to use in the FFT blocks.
            - dropout_rate: The dropout rate to use in the FFT blocks.
            - encoder::n_blocks: The number of residual blocks to use in the encoder.
            - encoder::embedding_dim: The dimension of the embeddings in the encoder.
            - encoder::fft_conv_channels: The number of convolutional channels in the FFT blocks.
            - decoder::n_blocks: The number of FFT blocks to use in the decoder.
            - decoder::fft_conv_channels: The number of convolutional channels in the FFT blocks.
            - decoder::output_channels: The number of output channels in the decoder.
            - duration_predictor::n_blocks: The number of convolutional blocks to use in
                the duration predictor.
            - gst::use_gst: Whether to use the global style tokens.
    """

    encoder = m_encoder.Encoder(
        input_phonemes_shape=input_phonemes_shape,
        n_blocks=cfg['encoder']['n_blocks'],
        embedding_dim=cfg['encoder']['embedding_dim'],
        dropout_rate=cfg['dropout_rate'],
        fft_conv_channels=cfg['encoder']['fft_conv_channels']
    ).to(device)

    decoder = m_decoder.Decoder(
        input_phonemes_shape=(output_spectrogram_shape[1], cfg['encoder']['embedding_dim']),
        output_channels=cfg['decoder']['output_channels'],
        n_blocks=cfg['decoder']['n_blocks'],
        fft_conv_channels=cfg['decoder']['fft_conv_channels'],
        n_heads=cfg['n_heads'],
        dropout_rate=cfg['dropout_rate']
    ).to(device)

    length_regulator = m_lr.LengthRegulator(
        output_length=output_spectrogram_shape[1]
    ).to(device)

    duration_predictor = m_dp.DurationPredictor(
        input_shape=(input_phonemes_shape[0], cfg['encoder']['embedding_dim']),
        n_conv_blocks=cfg['duration_predictor']['n_blocks'],
        dropout_rate=cfg['dropout_rate']
    ).to(device)

    gst = None
    embedder = None

    return ModelComponents(
        encoder=encoder,
        decoder=decoder,
        length_regulator=length_regulator,
        duration_predictor=duration_predictor,
        gst=gst,
        embedder=embedder
    )


def load_model_components(components: ModelComponents, path: str) -> ModelComponents:
    """Loads the model components from the specified path.

    Args:
        components: The freshly initialized components of the model.
        path: The path to the saved model.
    """

    if not os.path.exists(path):
        logging.critical("Model components not found at '%s'.", path)
        sys.exit(1)

    other_utils.try_load_state_dict(components.encoder, os.path.join(path, 'encoder.pth'))
    other_utils.try_load_state_dict(components.decoder, os.path.join(path, 'decoder.pth'))
    other_utils.try_load_state_dict(
        components.duration_predictor, os.path.join(
            path, 'duration_predictor.pth'))
    other_utils.try_load_state_dict(
        components.length_regulator, os.path.join(
            path, 'length_regulator.pth'))

    if components.gst and components.embedder:
        other_utils.try_load_state_dict(
            components.gst, os.path.join(path, 'gst.pth'))
        other_utils.try_load_state_dict(
            components.embedder, os.path.join(path, 'embedder.pth'))

    return components


def save_model_components(components: ModelComponents, path: str):
    """Saves the model components to the specified path.

    Args:
        components: The components of the model.
        path: The path to save the model.
    """

    os.makedirs(path, exist_ok=True)

    torch.save(components.encoder.state_dict(), os.path.join(path, 'encoder.pth'))
    torch.save(components.decoder.state_dict(), os.path.join(path, 'decoder.pth'))
    torch.save(components.duration_predictor.state_dict(),
               os.path.join(path, 'duration_predictor.pth'))
    torch.save(components.length_regulator.state_dict(), os.path.join(path, 'length_regulator.pth'))

    if components.gst and components.embedder:
        torch.save(components.gst.state_dict(), os.path.join(path, 'gst.pth'))
        torch.save(components.embedder.state_dict(), os.path.join(path, 'embedder.pth'))
