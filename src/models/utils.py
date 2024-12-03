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

import torch.optim.optimizer


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


class IOptimizerWrapper(ABC):
    """Interface for custom classes wrapping torch.Optimizer objects."""

    @abstractmethod
    def step(self):
        """Performs a single optimization step."""

    @abstractmethod
    def zero_grad(self):
        """Zeroes the gradients of the optimizer."""

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the optimizer as a dictionary."""

    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Loads the optimizer state from the specified state dictionary."""


class TransformerScheduledOptim(IOptimizerWrapper):
    """Modifies the learning rate of the wrapped optimizer according to a schedule.

    The schedule is described in https://arxiv.org/abs/1706.03762

    The schedule is intended to linearly increase the learning rate for the first warmup_steps
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 d_model: int,
                 warmup_steps: int):
        """Initializes the TransformerScheduledOptim.

        Args:
            optimizer: The optimizer to wrap.
            d_model: The dimensionality of the model.
            warmup_steps: The number of warmup steps.
        """

        self._optimizer = optimizer
        self._d_model = d_model
        self._warmup_steps = warmup_steps
        self._step_num = 0

    def step(self):
        """Performs a single optimization step."""

        self._step_num += 1
        lr = (self._d_model ** -0.5) * min(self._step_num ** -
                                           0.5, self._step_num * (self._warmup_steps ** -1.5))

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

        self._optimizer.step()

    def zero_grad(self):
        """Zeroes the gradients of the optimizer."""

        self._optimizer.zero_grad()

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the optimizer as a dictionary."""

        return self._optimizer.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Loads the optimizer state from the specified state dictionary."""

        self._optimizer.load_state_dict(state_dict)
