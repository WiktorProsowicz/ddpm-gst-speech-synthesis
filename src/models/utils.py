# -*- coding: utf-8 -*-
"""Contains shared utils for the models."""
import json
import logging
import os
import sys
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import Iterator
from typing import Tuple

import torch


class BaseModelComponents(ABC):
    """Base class for the model components."""

    @abstractmethod
    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """Returns the parameters of the model."""

    @abstractmethod
    def eval(self):
        """Sets the model to evaluation mode."""

    @abstractmethod
    def train(self):
        """Sets the model to training mode."""


def try_load_state_dict(module: torch.nn.Module, saved_module_path: str):
    """Attempts to load the state dict of the module from the specified path."""

    if not os.path.exists(saved_module_path):
        logging.critical("Module state dict not found at '%s'.", saved_module_path)
        sys.exit(1)

    module.load_state_dict(torch.load(saved_module_path, weights_only=True))


class ModelCheckpointHandler:
    """Handles the saving and loading of model checkpoints.

    The class is responsible for managing files inside the checkpoints directory. It shall
    store both the metadata of the saved checkpoints as well as the model components.
    """

    def __init__(self, checkpoint_dir: str,
                 checkpoint_basename: str,
                 loading_func: Any,
                 saving_func: Any):
        """Initializes the ModelCheckpointHandler.

        Args:
            checkpoint_dir: The directory to store the model checkpoints.
            checkpoint_basename: The base name of the checkpoints.
            loading_func: The function to load the model components.
            saving_func: The function to save the model components.
        """

        self._checkpoint_dir = checkpoint_dir
        self._metadata_path = os.path.join(checkpoint_dir, 'metadata.json')
        self._checkpoint_basename = checkpoint_basename
        self._loading_func = loading_func
        self._saving_func = saving_func

    def num_checkpoints(self) -> int:
        """Returns the number of saved checkpoints."""

        return len(self._get_metadata()['checkpoints'])

    def get_newest_checkpoint(
            self, model_components: Any) -> Tuple[Any, Dict[str, Any]]:
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
        self._loading_func(model_components, checkpoint_path)

        return model_components, newest_checkpoint['metadata']

    def save_checkpoint(self, model_components: Any,
                        checkpoint_metadata: Dict[str, Any]):
        """Saves the model components as a checkpoint.

        Args:
            model_components: The components of the model to save.
            checkpoint_metadata: The metadata of the checkpoint.
        """

        checkpoint_dir = os.path.join(self._checkpoint_dir,
                                      f'{self._checkpoint_basename}_{self.num_checkpoints()}')

        self._saving_func(model_components, checkpoint_dir)

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
