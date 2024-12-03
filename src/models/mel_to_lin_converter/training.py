# -*- coding: utf-8 -*-
"""Contains raining/validation/profiling pipeline for the mel-to-linear spectrogram converter."""
import logging
from typing import Callable
from typing import Dict
from typing import Tuple

import torch
import torch.utils.tensorboard as pt_tensorboard

from data import visualization
from models import base_trainer
from models import utils as shared_m_utils
from models.mel_to_lin_converter import utils as m_utils
from utilities import metrics


class ModelTrainer(base_trainer.BaseTrainer):
    """Trains and validated the mel-to-linear spectrogram converter model.

    The trainer performs the following operations:
    - Sampling batches from the dataset
    - Forward pass through the model
    - Updating the parameters w.r.t. the loss
    - Logging the training/validation metrics
    """

    def __init__(self,
                 model_provider: Callable[[], m_utils.ModelComponents],
                 train_data_loader: torch.utils.data.DataLoader,
                 val_data_loader: torch.utils.data.DataLoader,
                 tb_logger: pt_tensorboard.SummaryWriter,
                 device: torch.device,
                 checkpoints_handler: shared_m_utils.ModelCheckpointHandler,
                 checkpoints_interval: int,
                 validation_interval: int,
                 d_model: int,
                 warmup_steps: int,
                 ):
        """Initializes the model trainer.

        See the arguments of the BaseTrainer constructor.

        Args:
            learning_rate: The learning rate for the optimizer.

        """

        model_components = model_provider()
        base_optimizer = torch.optim.Adam(model_components.parameters(),
                                          lr=2e-4,
                                          betas=(0.9, 0.98))

        super().__init__(
            model_comps=model_components,
            train_data_loader=train_data_loader,
            val_data_loader=val_data_loader,
            tb_logger=tb_logger,
            device=device,
            checkpoints_handler=checkpoints_handler,
            checkpoints_interval=checkpoints_interval,
            validation_interval=validation_interval,
            optimizer=shared_m_utils.TransformerScheduledOptim(base_optimizer,
                                                               d_model,
                                                               warmup_steps))

        self._visualization_interval = self._validation_interval * 5
        self._loss = torch.nn.MSELoss()

    @property
    def model_comps(self) -> m_utils.ModelComponents:
        """Returns the model components."""

        assert isinstance(self._model_comps, m_utils.ModelComponents)
        return self._model_comps

    def _compute_losses(self, input_batch: Tuple[torch.Tensor, ...]
                        ) -> Dict[str, torch.Tensor]:
        """Overrides BaseTrainer::_compute_losses."""

        mel_spec, lin_spec = input_batch

        pred_lin_spec = self.model_comps.converter(mel_spec)

        return {
            'total_loss': self._loss(pred_lin_spec, lin_spec),
            'spec_pred_mae': metrics.mean_absolute_error(pred_lin_spec, lin_spec)
        }

    def _on_step_end(self, step_idx):

        if (step_idx + 1) % self._visualization_interval == 0:

            logging.info('Visualizing model output after %d steps.', step_idx + 1)

            self.model_comps.eval()

            lin_spec, pred_lin_spec = self._perform_visualization_for_loader(self._val_data_loader)

            self._tb_logger.add_image('Validation/Visualization/Original_Spectrogram',
                                      visualization.colorize_spectrogram(lin_spec, 'viridis'),
                                      global_step=step_idx)

            self._tb_logger.add_image('Validation/Visualization/Predicted_Spectrogram',
                                      visualization.colorize_spectrogram(pred_lin_spec, 'viridis'),
                                      global_step=step_idx)

            lin_spec, pred_lin_spec = self._perform_visualization_for_loader(
                self._train_data_loader)

            self._tb_logger.add_image('Training/Visualization/Original_Spectrogram',
                                      visualization.colorize_spectrogram(lin_spec, 'viridis'),
                                      global_step=step_idx)

            self._tb_logger.add_image('Training/Visualization/Predicted_Spectrogram',
                                      visualization.colorize_spectrogram(pred_lin_spec, 'viridis'),
                                      global_step=step_idx)

    def _perform_visualization_for_loader(
            self, data_loader: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:

        with torch.no_grad():
            mel_spec, lin_spec = next(iter(data_loader))

            mel_spec = mel_spec[:1]
            lin_spec = lin_spec[:1]

            mel_spec = mel_spec.to(self._device)
            lin_spec = lin_spec.to(self._device)

            return lin_spec[0], self.model_comps.converter(mel_spec)[0]
