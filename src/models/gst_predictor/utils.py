# -*- coding=utf-8 -*-
"""Contains utilities for the GST predictor model."""

from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Any
from dataclasses import dataclass

import torch

from models import utils as shared_m_utils
from layers.gst_predictor import encoder as m_encoder
from layers.gst_predictor import decoder as m_decoder


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
                            cfg: Dict[str, Any], device: torch.device) -> ModelComponents:
    """Creates the components of the GST predictor model.

    Args:
        input_phonemes_shape: The shape of the input phonemes.
        cfg: The model's configuration dictionary. The dictionary should contain the following keys:
            - encoder::embedding_size: The size of the embedding created by the encoder.
            - encoder::n_conv_blocks: The number of convolutional blocks in the encoder.
            - decoder::timestep_embedding_size: The size of the embedding created by the timestep encoder.
            - decoder::internal_channels: The number of internal channels in the decoder blocks.
            - decoder::n_conv_blocks: The number of convolutional blocks in the decoder.
            - dropout_rate: The dropout rate to use in the encoder and decoder.
        device: The device to use for the model.
    """

    return ModelComponents(
        encoder=m_encoder.Encoder(
            input_phonemes_shape=input_phonemes_shape,
            embedding_size=cfg['encoder']['embedding_size'],
            n_conv_blocks=cfg['encoder']['n_conv_blocks'],
            dropout_rate=cfg['dropout_rate']).to(device),
        decoder=m_decoder.Decoder(
            timestep_embedding_size=cfg['decoder']['timestep_embedding_size'],
            phoneme_embedding_size=cfg['encoder']['embedding_size'],
            internal_channels=cfg['decoder']['internal_channels'],
            n_conv_blocks=cfg['decoder']['n_conv_blocks'],
            dropout_rate=cfg['dropout_rate']).to(device)
    )
