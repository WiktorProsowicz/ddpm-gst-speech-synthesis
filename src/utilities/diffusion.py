# -*- coding=utf-8 -*-
"""Contains utilities for handling Diffusion denoising."""

from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch


class ParametrizationScheduler(ABC):
    """Abstract class for noise scheduling."""

    @abstractmethod
    def get_beta_params(self) -> torch.Tensor:
        """Returns the noising parameters"""


class LinearScheduler(ParametrizationScheduler):
    """Linear scheduler for noise scheduling."""

    def __init__(self, start: float, end: float, n_steps: int):
        """Initializes the scheduler.

        Args:
            start: The initial value of the parameter.
            end: The final value of the parameter.
            n_steps: The number of diffusion steps.
        """

        self._betas = torch.linspace(start, end, n_steps)

    def get_beta_params(self) -> torch.Tensor:
        """Returns the noising parameters"""

        return self._betas


class DiffusionHandler:
    """Performs the diffusion actions based on the diffusion process parameters."""

    def __init__(self, scheduler: ParametrizationScheduler):
        """Initializes the diffusion handler.

        Args:
            scheduler: Provides parameters for the diffusion process.
        """

        self._betas = scheduler.get_beta_params()
        self._alphas = 1 - self._betas
        self._alpha_cumprod = torch.cumprod(self._alphas, dim=0)
        self._sqrt_alpha_cumprod = torch.sqrt(self._alpha_cumprod)
        self._sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self._alpha_cumprod)

    @property
    def num_steps(self) -> int:
        """Returns the number of diffusion steps."""

        return len(self._betas)

    def add_noise(self, clean_data: torch.Tensor, noise: torch.Tensor,
                  t: torch.Tensor) -> torch.Tensor:
        """Adds noise to the clean data.

        Args:
            clean_data: The original data at timestep 0.
            noise: The noise sampled fromN(0, 1) to be added to the data.
            t: The diffusion step.

        Returns:
            The noised data.
        """

        data_shape = clean_data.shape[1:]
        batch_size = clean_data.shape[0]

        sqrt_alpha_cumprod = self._sqrt_alpha_cumprod[t].reshape(
            batch_size, *([1] * len(data_shape)))
        sqrt_one_minus_alpha_cumprod = self._sqrt_one_minus_alpha_cumprod[t].reshape(
            batch_size, *([1] * len(data_shape)))

        return sqrt_alpha_cumprod * clean_data + sqrt_one_minus_alpha_cumprod * noise

    def remove_noise(self, noised_data: torch.Tensor, noise: torch.Tensor,
                     t: int) -> torch.Tensor:
        """Denoises the given data.

        Args:
            noised_data: The noised data at timestep t.
            noise: The noise that has been added to the original data to create the noised data.
            t: The diffusion step.

        Returns:
            The denoised data.
        """

        mean_t_1 = noised_data - ((self._betas[t] * noise) / self._sqrt_one_minus_alpha_cumprod[t])
        mean_t_1 = mean_t_1 / torch.sqrt(self._alphas[t])

        variance_t_1 = (1. - self._alpha_cumprod[t - 1]) / (1. - self._alpha_cumprod[t])
        stddev_t_1 = torch.sqrt(variance_t_1 * self._betas[t])

        return mean_t_1 + stddev_t_1 * torch.randn_like(noised_data)
