# -*- coding: utf-8 -*-
"""Contains utility functions for the DDPM-GST-Speech-Gen model."""
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

from layers.ddpm_gst_speech_gen import decoder as m_dec
from layers.ddpm_gst_speech_gen import encoder as m_enc
from layers.shared import duration_predictor as m_dp
from layers.shared import gst as m_gst
from layers.shared import length_regulator as m_lr
from utilities import other as other_utils


@dataclass
class ModelComponents:
    """Contains the components of the DDPM-GST-Speech-Gen model."""

    encoder: m_enc.Encoder
    decoder: m_dec.Decoder
    duration_predictor: m_dp.DurationPredictor
    length_regulator: m_lr.LengthRegulator
    gst_provider: Optional[m_gst.GSTProvider]
    reference_embedder: Optional[m_gst.ReferenceEmbedder]

    def parameters(self):
        """Returns the model's parameters."""

        if self.gst_provider and self.reference_embedder:

            return itertools.chain(
                self.encoder.parameters(),
                self.decoder.parameters(),
                self.duration_predictor.parameters(),
                self.length_regulator.parameters(),
                self.gst_provider.parameters(),
                self.reference_embedder.parameters()
            )

        return itertools.chain(
            self.encoder.parameters(),
            self.decoder.parameters(),
            self.duration_predictor.parameters(),
            self.length_regulator.parameters()
        )

    def eval(self):
        """Sets the model to evaluation mode."""

        self.encoder.eval()
        self.decoder.eval()
        self.duration_predictor.eval()
        self.length_regulator.eval()

        if self.gst_provider and self.reference_embedder:
            self.gst_provider.eval()
            self.reference_embedder.eval()

    def train(self):
        """Sets the model to training mode."""

        self.encoder.train()
        self.decoder.train()
        self.duration_predictor.train()
        self.length_regulator.train()

        if self.gst_provider and self.reference_embedder:
            self.gst_provider.train()
            self.reference_embedder.train()


def create_model_components(input_spectrogram_shape: Tuple[int, int],
                            input_phonemes_shape: Tuple[int, int],
                            cfg: Dict[str, Any],
                            device: torch.device) -> ModelComponents:
    """Creates the components of the DDPM-GST-Speech-Gen model.

    Args:
        input_spectrogram_shape: The shape of the input spectrogram.
        input_phonemes_shape: The shape of the input one-hot encoded phonemes.
        cfg: The internal configuration of the model. It contains the following keys:
            - gst::use_gst: Whether to use the global style token (GST) module.
            - gst::embedding_dim: The dimension of the GST embeddings.
            - gst::token_count: The number of tokens in the GST embeddings.
            - decoder::timestep_embedding_dim: The dimension of the time embeddings in the decoder.
            - duration_predictor::n_blocks: The number of convolutional blocks in the
                duration predictor.
            - decoder::n_res_blocks: The number of residual blocks in the decoder.
            - decoder::internal_channels: The number of internal channels in the decoder.
            - decoder::skip_connections_channels: The number of channels in the skip connections.
            - encoder::n_blocks: The number of convolutional in the encoder.
            - encoder::embedding_dim: The dimension of the embeddings in the encoder.
            - dropout_rate: The dropout rate to use in the whole model.
    """

    decoder_input_channels, decoder_input_length = input_spectrogram_shape
    input_phonemes_length, _ = input_phonemes_shape

    decoder = m_dec.Decoder(
        input_spectrogram_shape,
        cfg['decoder']['timestep_embedding_dim'],
        cfg['decoder']['n_res_blocks'],
        cfg['decoder']['internal_channels'],
        cfg['decoder']['skip_connections_channels'],
        cfg['dropout_rate'])
    decoder.to(device)

    encoder = m_enc.Encoder(input_phonemes_shape,
                            cfg['encoder']['n_blocks'],
                            decoder_input_channels,
                            cfg['encoder']['embedding_dim'],
                            cfg['dropout_rate'])
    encoder.to(device)

    duration_predictor = m_dp.DurationPredictor((input_phonemes_length, decoder_input_channels),
                                                cfg['duration_predictor']['n_blocks'],
                                                cfg['dropout_rate'])
    duration_predictor.to(device)

    length_regulator = m_lr.LengthRegulator(decoder_input_length)
    length_regulator.to(device)

    if cfg['gst']['use_gst']:

        gst_provider = m_gst.GSTProvider(
            cfg['gst']['embedding_dim'],
            cfg['gst']['token_count'])
        gst_provider.to(device)

        reference_embedder = m_gst.ReferenceEmbedder(
            input_spectrogram_shape,
            (cfg['gst']['token_count'],
             cfg['gst']['embedding_dim']))
        reference_embedder.to(device)

    else:

        gst_provider = None
        reference_embedder = None

    return ModelComponents(
        encoder, decoder, duration_predictor, length_regulator, gst_provider, reference_embedder
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

    if components.gst_provider and components.reference_embedder:
        other_utils.try_load_state_dict(
            components.gst_provider, os.path.join(
                path, 'gst_provider.pth'))
        other_utils.try_load_state_dict(
            components.reference_embedder, os.path.join(
                path, 'reference_embedder.pth'))

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

    if components.gst_provider and components.reference_embedder:
        torch.save(components.gst_provider.state_dict(), os.path.join(path, 'gst_provider.pth'))
        torch.save(components.reference_embedder.state_dict(),
                   os.path.join(path, 'reference_embedder.pth'))
