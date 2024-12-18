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
from typing import Optional
from typing import Tuple
from typing import Union

import torch


class BaseModelComponents(ABC):
    """Base class for the model components."""

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """Returns the parameters of the model."""

        for component in self.get_components().values():
            if component is not None:
                yield from component.parameters()

    def eval(self):
        """Sets the model to evaluation mode."""

        for component in self.get_components().values():
            if component is not None:
                component.eval()

    def train(self):
        """Sets the model to training mode."""

        for component in self.get_components().values():
            if component is not None:
                component.train()

    def load_from_path(self, path: str):
        """Loads the model components from the specified directory."""

        if not os.path.exists(path):
            logging.critical("Model components not found at '%s'.", path)
            sys.exit(1)

        for component_name, component in self.get_components().items():
            if component is not None:
                try_load_state_dict(component, os.path.join(path, f'{component_name}.pth'))

    def save_to_path(self, path: str):
        """Saves the model components to the specified directory."""

        os.makedirs(path, exist_ok=True)

        for component_name, component in self.get_components().items():
            if component is not None:
                torch.save(component.state_dict(), os.path.join(path, f'{component_name}.pth'))

    @abstractmethod
    def get_components(self) -> Dict[str, Optional[torch.nn.Module]]:
        """Returns all named components possessed by the concrete class' instance."""


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
                 checkpoint_basename: str):
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

    def num_checkpoints(self) -> int:
        """Returns the number of saved checkpoints."""

        return len(self._get_metadata()['checkpoints'])

    def get_newest_checkpoint(
        self,
        model_components: BaseModelComponents,
        optimizer: Union[torch.optim.Optimizer, 'IOptimizerWrapper']
    ) -> Tuple[BaseModelComponents,
               Union[torch.optim.Optimizer, 'IOptimizerWrapper'],
               Dict[str, Any]]:
        """Loads the newest checkpoint from the checkpoint directory.

        Args:
            model_components: The components of the model to load the checkpoint into.

        Returns:
            A tuple containing the model components, optimizer and the metadata of the
            newest checkpoint.
        """

        metadata = self._get_metadata()

        if not metadata['checkpoints']:
            logging.critical('No checkpoints found.')
            sys.exit(1)

        newest_checkpoint = metadata['checkpoints'][-1]

        checkpoint_path = os.path.join(self._checkpoint_dir, newest_checkpoint['directory_name'])
        optim_state_path = os.path.join(checkpoint_path, 'optimizer_state.pth')

        model_components.load_from_path(checkpoint_path)
        optimizer.load_state_dict(torch.load(optim_state_path, weights_only=True))

        return model_components, optimizer, newest_checkpoint['metadata']

    def save_checkpoint(self,
                        model_components: BaseModelComponents,
                        optimizer: Union[torch.optim.Optimizer, 'IOptimizerWrapper'],
                        checkpoint_metadata: Dict[str, Any]):
        """Saves the model components as a checkpoint.

        Args:
            model_components: The components of the model to save.
            optimizer: The optimizer at a current training stage to be saved.
            checkpoint_metadata: The metadata of the checkpoint.
        """

        checkpoint_dir = os.path.join(self._checkpoint_dir,
                                      f'{self._checkpoint_basename}_{self.num_checkpoints()}')

        model_components.save_to_path(checkpoint_dir)
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, 'optimizer_state.pth'))

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

        state_dict = self._optimizer.state_dict()
        state_dict['wrapper_state'] = {
            'step_num': self._step_num
        }
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Loads the optimizer state from the specified state dictionary."""

        wrapper_state = state_dict.pop('wrapper_state')
        self._step_num = wrapper_state['step_num']

        self._optimizer.load_state_dict(state_dict)
