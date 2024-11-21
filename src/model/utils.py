# -*- coding: utf-8 -*-
"""Contains utility functions for the DDPM-GST-Speech-Gen model."""
import itertools
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import torch

from model.layers import decoder as m_dec
from model.layers import duration_predictor as m_dp
from model.layers import encoder as m_enc
from model.layers import gst as m_gst
from model.layers import length_regulator as m_lr


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
    """

    decoder_input_channels, decoder_input_length = input_spectrogram_shape
    input_phonemes_length, _ = input_phonemes_shape

    decoder = m_dec.Decoder(
        input_spectrogram_shape,
        cfg['decoder']['timestep_embedding_dim'])
    decoder.to(device)

    encoder = m_enc.Encoder(input_phonemes_shape, decoder_input_channels)
    encoder.to(device)

    duration_predictor = m_dp.DurationPredictor((input_phonemes_length, decoder_input_channels))
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

    _try_load_state_dict(components.encoder, os.path.join(path, 'encoder.pth'))
    _try_load_state_dict(components.decoder, os.path.join(path, 'decoder.pth'))
    _try_load_state_dict(
        components.duration_predictor, os.path.join(
            path, 'duration_predictor.pth'))
    _try_load_state_dict(components.length_regulator, os.path.join(path, 'length_regulator.pth'))

    if components.gst_provider and components.reference_embedder:
        _try_load_state_dict(components.gst_provider, os.path.join(path, 'gst_provider.pth'))
        _try_load_state_dict(
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


class ModelCheckpointHandler:
    """Handles the saving and loading of model checkpoints.

    The class is responsible for managing files inside the checkpoints directory. It shall
    store both the metadata of the saved checkpoints as well as the model components.
    """

    def __init__(self, checkpoint_dir: str,
                 checkpoint_basename: str):
        """Initializes the ModelCheckpointHandler.

        Args:
            checkpoint_dir: The directory to store the model checkpoints.
            checkpoint_basename: The base name of the checkpoints.
        """

        self._checkpoint_dir = checkpoint_dir
        self._metadata_path = os.path.join(checkpoint_dir, 'metadata.json')
        self._checkpoint_basename = checkpoint_basename

    def num_checkpoints(self) -> int:
        """Returns the number of saved checkpoints."""

        return len(self._get_metadata()['checkpoints'])

    def get_newest_checkpoint(
            self, model_components: ModelComponents) -> Tuple[ModelComponents, Dict[str, Any]]:
        """Loads the newest checkpoint from the checkpoint directory.

        Args:
            model_components: The components of the model to load the checkpoint into.

        Returns:
            A tuple containing the model components and the metadata of the newest checkpoint.
        """

        metadata = self._get_metadata()

        if not metadata['checkpoints']:
            logging.critical('No checkpoints found.')
            sys.exit(1)

        newest_checkpoint = metadata['checkpoints'][-1]

        checkpoint_path = os.path.join(self._checkpoint_dir, newest_checkpoint['directory_name'])
        load_model_components(model_components, checkpoint_path)

        return model_components, newest_checkpoint['metadata']

    def save_checkpoint(self, model_components: ModelComponents,
                        checkpoint_metadata: Dict[str, Any]):
        """Saves the model components as a checkpoint.

        Args:
            model_components: The components of the model to save.
            checkpoint_metadata: The metadata of the checkpoint.
        """

        checkpoint_dir = os.path.join(self._checkpoint_dir,
                                      f'{self._checkpoint_basename}_{self.num_checkpoints()}')

        save_model_components(model_components, checkpoint_dir)

        metadata = self._get_metadata()

        metadata['checkpoints'].append({
            'directory_name': os.path.basename(checkpoint_dir),
            'metadata': checkpoint_metadata
        })

        self._save_metadata(metadata)

    def _get_metadata(self) -> Dict[str, Any]:
        """Returns the metadata of the saved checkpoints."""

        if not os.path.exists(self._metadata_path):
            return {
                'checkpoints': []
            }

        with open(self._metadata_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def _save_metadata(self, metadata: Dict[str, Any]):
        """Saves the metadata of the saved checkpoints."""

        with open(self._metadata_path, 'w', encoding='utf-8') as file:
            json.dump(metadata, file)
