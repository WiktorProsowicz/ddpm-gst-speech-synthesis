"""Contains shared utils for the models."""

from abc import ABC
from abc import abstractmethod
from typing import Iterator

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
