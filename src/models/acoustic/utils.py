# -*- coding: utf-8 -*-
"""Contains utilities specific fot the acoustic model."""
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
from layers.shared import ref_embedder
from models import utils as shared_m_utils


@dataclass
class ModelComponents(shared_m_utils.BaseModelComponents):
    """Contains the components of the acoustic model."""
    encoder: m_encoder.Encoder
    decoder: m_decoder.Decoder
    length_regulator: m_lr.LengthRegulator
    duration_predictor: m_dp.DurationPredictor
    gst: Optional[m_gst.GSTProvider]
    embedder: Optional[ref_embedder.ReferenceEmbedder]

    def get_components(self) -> Dict[str, Optional[torch.nn.Module]]:
        return {
            'encoder': self.encoder,
            'decoder': self.decoder,
            'length_regulator': self.length_regulator,
            'duration_predictor': self.duration_predictor,
            'gst': self.gst,
            'embedder': self.embedder
        }


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
        n_heads=cfg['n_heads'],
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

    if cfg['gst']['use_gst']:
        gst = m_gst.GSTProvider(
            gst_embedding_dim=cfg['gst']['token_dim'],
            gst_token_count=cfg['gst']['n_tokens']
        ).to(device)

        embedder = ref_embedder.ReferenceEmbedder(
            reference_spectrogram_shape=output_spectrogram_shape,
            gst_shape=(cfg['gst']['n_tokens'], cfg['gst']['token_dim']),
            n_ref_encoder_blocks=cfg['gst']['n_ref_encoder_blocks'],
            n_attention_heads=cfg['gst']['n_attention_heads'],
            dropout_rate=cfg['dropout_rate']
        ).to(device)

    else:
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
