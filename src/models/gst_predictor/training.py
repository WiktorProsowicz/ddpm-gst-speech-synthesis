# -*- coding: utf-8 -*-
"""Contains the training/validation/profiling pipeline for the GST predictor model."""
import logging
from typing import Dict
from typing import Tuple

import torch
from torch.utils import tensorboard as pt_tensorboard

from data import visualization
from models import base_trainer
from models import utils as shared_m_utils
from models.gst_predictor import utils as m_utils
from utilities import diffusion as diff_utils
from utilities import metrics


class ModelTrainer(base_trainer.BaseTrainer):
    """Trains and validates the GST predictor model.

    The trainer performs the following operations:
    - Sampling batches from the dataset
    - Forward pass through the model
    - Updating the parameters w.r.t. the loss
    - Logging the training/validation metrics
    - Visualizing the model's predictions.

    The GST weights used as the input to the model are normalized to (mean=0, stddev=1)
    before being fed to the model so that it can learn the noise distribution more accurately.
    """

    def __init__(self,
                 model_components: m_utils.ModelComponents,
                 train_data_loader: torch.utils.data.DataLoader,
                 val_data_loader: torch.utils.data.DataLoader,
                 global_ds_stats: Tuple[torch.Tensor, torch.Tensor],
                 tb_logger: pt_tensorboard.SummaryWriter,
                 device: torch.device,
                 checkpoints_handler: shared_m_utils.ModelCheckpointHandler,
                 checkpoints_interval: int,
                 validation_interval: int,
                 learning_rate: float,
                 diff_params_scheduler: diff_utils.ParametrizationScheduler):
        """Initializes the model trainer.

        See the arguments of the BaseTrainer constructor.

        Args:
            learning_rate: The learning rate to use in the optimizer.
            diff_params_scheduler: The scheduler for the diffusion parameters.
            global_ds_stats: The global mean and stddev of the dataset.
        """

        super().__init__(
            model_comps=model_components,
            train_data_loader=train_data_loader,
            val_data_loader=val_data_loader,
            tb_logger=tb_logger,
            device=device,
            checkpoints_handler=checkpoints_handler,
            checkpoints_interval=checkpoints_interval,
            validation_interval=validation_interval,
            optimizer=torch.optim.Adam(model_components.parameters(), lr=learning_rate),
        )

        self._diffusion_handler = diff_utils.DiffusionHandler(diff_params_scheduler,
                                                              self._device)
        self._backward_diff_interval = self._validation_interval * 5
        self._loss = torch.nn.MSELoss()

        self._global_mean, self._global_stddev = global_ds_stats

    @property
    def model_comps(self) -> m_utils.ModelComponents:
        """Returns the model components."""

        assert isinstance(self._model_comps, m_utils.ModelComponents)
        return self._model_comps

    def _compute_losses(self, input_batch: Tuple[torch.Tensor, ...]
                        ) -> Dict[str, torch.Tensor]:
        """Overrides BaseTrainer::_compute_losses."""

        phonemes, gst_targets = input_batch
        gst_targets = (gst_targets - self._global_mean) / self._global_stddev
        batch_size = phonemes.size(0)

        noise = torch.randn_like(gst_targets)
        diff_timestep = torch.randint(
            0, self._diffusion_handler.num_steps, (batch_size,), device=self._device)

        noised_gst = self._diffusion_handler.add_noise(gst_targets, noise, diff_timestep)

        encoder_output = self.model_comps.encoder(phonemes)
        pred_noise = self.model_comps.decoder(noised_gst, diff_timestep, encoder_output)

        return {
            'total_loss': self._loss(pred_noise, noise),
            'noise_pred_mae': metrics.mean_absolute_error(noise, pred_noise),
        }

    def _on_step_end(self, step_idx):

        if (step_idx + 1) % self._backward_diff_interval == 0:
            logging.debug('Running full backward diffusion.')
            self._run_backward_diff(step_idx)

    def _run_backward_diff(self, step_idx: int):
        """Runs the backward diffusion step and visualizes the output."""

        original_gst, denoised_gst = self._run_backward_diff_for_loader(self._val_data_loader)

        self._tb_logger.add_figure(
            'Validation/Visualization/gst_prediction',
            visualization.plot_pred_and_gt_gst_weights(original_gst, denoised_gst),
            step_idx
        )

        original_gst, denoised_gst = self._run_backward_diff_for_loader(self._train_data_loader)

        self._tb_logger.add_figure(
            'Training/Visualization/gst_prediction',
            visualization.plot_pred_and_gt_gst_weights(original_gst, denoised_gst),
            step_idx
        )

    def _run_backward_diff_for_loader(self,
                                      data_loader: torch.utils.data.DataLoader
                                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs the backward diffusion step for the given data loader."""

        self.model_comps.eval()

        with torch.no_grad():

            batch = next(iter(data_loader))
            batch = tuple(t.to(self._device) for t in batch)

            phonemes, gst_targets = batch

            phonemes = phonemes[:1]
            gst_targets = gst_targets[:1]

            phoneme_embedding = self.model_comps.encoder(phonemes)

            noised_gst = torch.randn_like(gst_targets)

            for diff_step in reversed(range(self._diffusion_handler.num_steps)):

                predicted_noise = self.model_comps.decoder(
                    noised_gst,
                    torch.tensor([diff_step], device=self._device),
                    phoneme_embedding)

                noised_gst = self._diffusion_handler.remove_noise(
                    noised_gst, predicted_noise, diff_step)

        return (
            gst_targets[0],
            (noised_gst[0] * self._global_stddev) + self._global_mean
        )
