# -*- coding: utf-8 -*-
"""Contains utility functions for the DDPM-GST-Speech-Gen model."""
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
from layers.shared import ref_embedder
from models import utils as shared_m_utils


@dataclass
class ModelComponents(shared_m_utils.BaseModelComponents):
    """Contains the components of the DDPM-GST-Speech-Gen model."""

    encoder: m_enc.Encoder
    decoder: m_dec.Decoder
    duration_predictor: m_dp.DurationPredictor
    length_regulator: m_lr.LengthRegulator
    gst_provider: Optional[m_gst.GSTProvider]
    reference_embedder: Optional[ref_embedder.ReferenceEmbedder]

    def get_components(self) -> Dict[str, Optional[torch.nn.Module]]:

        return {
            'encoder': self.encoder,
            'decoder': self.decoder,
            'duration_predictor': self.duration_predictor,
            'length_regulator': self.length_regulator,
            'gst_provider': self.gst_provider,
            'reference_embedder': self.reference_embedder
        }


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
            - decoder::conv_kernel_size: The kernel size of the convolutional layers in the decoder.
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
        cfg['decoder']['conv_kernel_size'],
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

        reference_embedder = ref_embedder.ReferenceEmbedder(
            input_spectrogram_shape,
            (cfg['gst']['token_count'],
             cfg['gst']['embedding_dim']),
            cfg['gst']['n_ref_encoder_blocks'],
            cfg['gst']['n_attention_heads'],
            cfg['dropout_rate'])
        reference_embedder.to(device)

    else:

        gst_provider = None
        reference_embedder = None

    return ModelComponents(
        encoder, decoder, duration_predictor, length_regulator, gst_provider, reference_embedder
    )
