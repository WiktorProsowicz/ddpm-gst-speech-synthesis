# -*- coding: utf-8 -*-
"""Contains utilities specific fot the acoustic model."""
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
import torch_dev_utils as tdu

from layers.acoustic import decoder as m_decoder
from layers.acoustic import encoder as m_encoder
from layers.shared import duration_predictor as m_dp
from layers.shared import length_regulator as m_lr
from layers.shared import ref_embedder


@dataclass
class ModelComponents(tdu.model.BaseModelComponents):
    """Contains the components of the acoustic model."""
    encoder: m_encoder.Encoder
    decoder: m_decoder.Decoder
    length_regulator: m_lr.LengthRegulator
    duration_predictor: m_dp.DurationPredictor
    embedder: Optional[ref_embedder.ReferenceEmbedder]

    def get_components(self) -> Dict[str, Optional[torch.nn.Module]]:
        return {
            'encoder': self.encoder,
            'decoder': self.decoder,
            'length_regulator': self.length_regulator,
            'duration_predictor': self.duration_predictor,
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
        cfg: The internal configuration of the model. See scripts/training/train_acoustic_model.py
    """

    encoder = m_encoder.Encoder(
        input_phonemes_shape=input_phonemes_shape,
        n_blocks=cfg['encoder']['n_blocks'],
        d_model=cfg['d_model'],
        n_heads=cfg['n_heads'],
        dropout_rate=cfg['dropout_rate'],
        fft_conv_channels=cfg['fft_conv_channels']
    ).to(device)

    decoder = m_decoder.Decoder(
        input_phonemes_shape=(output_spectrogram_shape[1], cfg['d_model']),
        output_channels=cfg['decoder']['output_channels'],
        n_blocks=cfg['decoder']['n_blocks'],
        fft_conv_channels=cfg['fft_conv_channels'],
        n_heads=cfg['n_heads'],
        dropout_rate=cfg['dropout_rate']
    ).to(device)

    length_regulator = m_lr.LengthRegulator(
        output_length=output_spectrogram_shape[1]
    ).to(device)

    duration_predictor = m_dp.DurationPredictor(
        input_shape=(input_phonemes_shape[0], cfg['d_model']),
        n_conv_blocks=cfg['duration_predictor']['n_blocks'],
        dropout_rate=cfg['dropout_rate']
    ).to(device)

    if cfg['use_reference_encoder']:

        embedder = ref_embedder.ReferenceEmbedder(
            reference_spectrogram_shape=output_spectrogram_shape,
            gst_shape=(cfg['gst']['n_tokens'], cfg['d_model']),
            n_ref_encoder_blocks=cfg['gst']['n_ref_encoder_blocks'],
            dropout_rate=cfg['dropout_rate'],
            use_gst_att=cfg['gst']['use_gst_att']
        ).to(device)

    else:
        embedder = None

    return ModelComponents(
        encoder=encoder,
        decoder=decoder,
        length_regulator=length_regulator,
        duration_predictor=duration_predictor,
        embedder=embedder
    )
