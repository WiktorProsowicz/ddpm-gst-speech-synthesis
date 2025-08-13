# -*- coding: utf-8 -*-
"""Contains utilities for the GST predictor model."""
import itertools
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import Iterator
from typing import Optional
from typing import Tuple

import torch
import torch_dev_utils as tdu

from layers.gst_predictor import decoder as m_decoder
from layers.gst_predictor import deterministic_weights_pred as m_deterministic_pred
from layers.gst_predictor import encoder as m_encoder


BERT_EMBEDDING_SIZE = 768


@dataclass
class ModelComponents(tdu.model.BaseModelComponents):
    """Contains components of the GST predictor model."""

    encoder: m_encoder.Encoder
    decoder: m_decoder.Decoder
    deterministic_pred: m_deterministic_pred.DeterministicWeightsPred

    def get_components(self) -> Dict[str, Optional[torch.nn.Module]]:

        return {
            'encoder': self.encoder,
            'decoder': self.decoder,
            'deterministic_pred': self.deterministic_pred
        }

    def weights_pred_params(self) -> Iterator[torch.nn.Parameter]:
        """Returns the parameters of the weights predictor."""

        return self.deterministic_pred.parameters()

    def diffusion_params(self) -> Iterator[torch.nn.Parameter]:
        """Returns the parameters of the diffusion model."""

        return itertools.chain(
            self.encoder.parameters(),
            self.decoder.parameters()
        )


def create_model_components(input_phonemes_shape: Tuple[int, int],
                            gst_emb_size: int,
                            gst_weights_size: int,
                            cfg: Dict[str, Any],
                            device: torch.device) -> ModelComponents:
    """Creates the components of the GST predictor model.

    Args:
        input_phonemes_shape: The shape of the input phonemes.
        cfg: The model's configuration dictionary.
        device: The device to use for the model.
    """

    return ModelComponents(
        encoder=m_encoder.Encoder(
            input_phonemes_shape=input_phonemes_shape,
            gst_size=gst_emb_size,
            n_blocks=cfg['encoder']['n_blocks'],
            n_heads=cfg['encoder']['n_heads'],
            conv_filters=cfg['encoder']['conv_filters'],
            dropout_rate=cfg['encoder']['dropout_rate']).to(device),
        decoder=m_decoder.Decoder(
            input_gst_size=gst_emb_size,
            timestep_embedding_size=cfg['decoder']['timestep_embedding_size'],
            internal_channels=cfg['decoder']['internal_channels'],
            n_blocks=cfg['decoder']['n_blocks'],
            dropout_rate=cfg['decoder']['dropout_rate']).to(device),
        deterministic_pred=m_deterministic_pred.DeterministicWeightsPred(
                input_phonemes_shape=input_phonemes_shape,
                output_weights_size=gst_weights_size,
                n_blocks=cfg['deterministic_pred']['n_blocks'],
                internal_dim=cfg['deterministic_pred']['internal_dim'],
                fft_conv_channels=cfg['deterministic_pred']['fft_conv_channels'],
                dropout_rate=cfg['deterministic_pred']['dropout_rate']
        ).to(device)
    )
