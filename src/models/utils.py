# -*- coding: utf-8 -*-
"""Contains shared utils for the models."""
from typing import Any
from typing import Dict
from typing import List

import torch
import torch_dev_utils as tdu


class TransformerScheduledOptim(tdu.misc.IOptimizerWrapper):
    """Modifies the learning rate of the wrapped optimizer according to a schedule.

    The schedule is described in https://arxiv.org/abs/1706.03762

    The schedule is intended to linearly increase the learning rate for the first warmup_steps
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 d_model: int,
                 warmup_steps: int,
                 tracked_param_groups: List[str]):
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
        self._tracked_param_groups = tracked_param_groups

    def step(self):
        """Performs a single optimization step."""

        self._step_num += 1
        lr = (self._d_model ** -0.5) * min(self._step_num ** -0.5,
                                           self._step_num * (self._warmup_steps ** -1.5))

        for param_group in self._optimizer.param_groups:

            if param_group['name'] not in self._tracked_param_groups:
                continue

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
