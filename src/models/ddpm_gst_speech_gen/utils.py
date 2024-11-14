# coding=utf-8
"""Contains utility functions for the DDPM-GST-Speech-Gen model."""

from dataclasses import dataclass
from typing import Tuple, Optional, Any, Dict
import itertools
import logging
import os
import sys

import torch

from models.ddpm_gst_speech_gen.layers import decoder as m_dec
from models.ddpm_gst_speech_gen.layers import encoder as m_enc
from models.ddpm_gst_speech_gen.layers import duration_predictor as m_dp
from models.ddpm_gst_speech_gen.layers import length_regulator as m_lr
from models.ddpm_gst_speech_gen.layers import gst as m_gst


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


def create_model_components(input_spectrogram_shape: Tuple[int, int],
                            input_phonemes_shape: Tuple[int, int],
                            gst_embedding_dim: Optional[int],
                            gst_token_count: Optional[int]) -> ModelComponents:
    """Creates the components of the DDPM-GST-Speech-Gen model.

    Args:
        input_spectrogram_shape: The shape of the input spectrogram.
        input_phonemes_shape: The shape of the input one-hot encoded phonemes.
        gst_embedding_dim: The length of each Global Style Token.
        gst_token_count: The number of Global Style Tokens.
    """

    decoder_input_channels, decoder_input_length = input_spectrogram_shape
    _, input_phonemes_length = input_phonemes_shape

    decoder = m_dec.Decoder(input_spectrogram_shape)
    encoder = m_enc.Encoder(input_phonemes_shape, decoder_input_channels)
    duration_predictor = m_dp.DurationPredictor((input_phonemes_length, decoder_input_channels))
    length_regulator = m_lr.LengthRegulator(
        (decoder_input_channels, input_phonemes_length), decoder_input_length)

    if gst_embedding_dim and gst_token_count:
        gst_provider = m_gst.GSTProvider(gst_embedding_dim, gst_token_count)
        reference_embedder = m_gst.ReferenceEmbedder(
            input_spectrogram_shape, (gst_token_count, gst_embedding_dim))
    else:
        gst_provider = None

    return ModelComponents(
        encoder, decoder, duration_predictor, length_regulator, gst_provider, reference_embedder
    )


def _try_load_state_dict(module: torch.nn.Module, saved_module_path: str):
    """Attempts to load the state dict of the module from the specified path."""

    if not os.path.exists(saved_module_path):
        logging.critical("Module state dict not found at '%s'.", saved_module_path)
        sys.exit(1)

    module.load_state_dict(torch.load(saved_module_path, weights_only=True))


def load_model_components(components: ModelComponents, path: str) -> ModelComponents:
    """Loads the model components from the specified path.

    Args:
        components: The freshly initialized components of the model.
        path: The path to the saved model.
    """

    if not os.path.exists(path):
        logging.critical("Model components not found at '%s'.", path)
        sys.exit(1)

    _try_load_state_dict(components.encoder, os.path.join(path, "encoder.pth"))
    _try_load_state_dict(components.decoder, os.path.join(path, "decoder.pth"))
    _try_load_state_dict(
        components.duration_predictor, os.path.join(
            path, "duration_predictor.pth"))
    _try_load_state_dict(components.length_regulator, os.path.join(path, "length_regulator.pth"))

    if components.gst_provider and components.reference_embedder:
        _try_load_state_dict(components.gst_provider, os.path.join(path, "gst_provider.pth"))
        _try_load_state_dict(
            components.reference_embedder, os.path.join(
                path, "reference_embedder.pth"))


def save_model_components(components: ModelComponents, path: str):
    """Saves the model components to the specified path.

    Args:
        components: The components of the model.
        path: The path to save the model.
    """

    os.makedirs(path, exist_ok=True)

    torch.save(components.encoder.state_dict(), os.path.join(path, "encoder.pth"))
    torch.save(components.decoder.state_dict(), os.path.join(path, "decoder.pth"))
    torch.save(components.duration_predictor.state_dict(),
               os.path.join(path, "duration_predictor.pth"))
    torch.save(components.length_regulator.state_dict(), os.path.join(path, "length_regulator.pth"))

    if components.gst_provider and components.reference_embedder:
        torch.save(components.gst_provider.state_dict(), os.path.join(path, "gst_provider.pth"))
        torch.save(components.reference_embedder.state_dict(),
                   os.path.join(path, "reference_embedder.pth"))
