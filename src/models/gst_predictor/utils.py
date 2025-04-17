# -*- coding: utf-8 -*-
"""Contains utilities for the GST predictor model."""
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import torch

from layers.gst_predictor import decoder as m_decoder
from layers.gst_predictor import encoder as m_encoder
from models import utils as shared_m_utils


@dataclass
class ModelComponents(shared_m_utils.BaseModelComponents):
    """Contains components of the GST predictor model."""

    encoder: m_encoder.Encoder
    decoder: m_decoder.Decoder

    def get_components(self) -> Dict[str, Optional[torch.nn.Module]]:

        return {
            'encoder': self.encoder,
            'decoder': self.decoder,
        }


def create_model_components(input_phonemes_shape: Tuple[int, int],
                            input_gst_shape: Tuple[int],
                            cfg: Dict[str, Any], device: torch.device) -> ModelComponents:
    """Creates the components of the GST predictor model.

    Args:
        input_phonemes_shape: The shape of the input phonemes.
        cfg: The model's configuration dictionary. The dictionary should contain the following keys:
            - encoder::n_conv_blocks: The number of convolutional blocks in the encoder.
            - decoder::timestep_embedding_size: The size of the embedding created by the
                timestep encoder.
            - decoder::internal_channels: The number of internal channels in the decoder blocks.
            - decoder::n_conv_blocks: The number of convolutional blocks in the decoder.
            - dropout_rate: The dropout rate to use in the encoder and decoder.
        device: The device to use for the model.
    """

    BERT_EMBEDDING_SIZE = 768

    return ModelComponents(
        encoder=m_encoder.Encoder(
            input_phonemes_shape=input_phonemes_shape,
            gst_size=input_gst_shape[0],
            n_blocks=cfg['encoder']['n_blocks'],
            n_heads=cfg['encoder']['n_heads'],
            conv_filters=cfg['encoder']['conv_filters'],
            dropout_rate=cfg['dropout_rate']).to(device),
        decoder=m_decoder.Decoder(
            input_gst_size=input_gst_shape[0],
            timestep_embedding_size=cfg['decoder']['timestep_embedding_size'],
            internal_channels=cfg['decoder']['internal_channels'],
            n_blocks=cfg['decoder']['n_blocks'],
            phoneme_embedding_dim=input_phonemes_shape[1] + BERT_EMBEDDING_SIZE,
            dropout_rate=cfg['dropout_rate']).to(device)
    )
