# -*- coding: utf-8 -*-
"""Contains the training/validation/profiling pipeline for the acoustic model."""
import logging
from typing import Dict
from typing import Tuple

import torch
from torch.utils import tensorboard as pt_tensorboard

from data import visualization
from models import base_trainer
from models import utils as shared_m_utils
from models.acoustic import utils as model_utils
from utilities import inference as inf_utils
from utilities import metrics
from utilities import other as other_utils


class ModelTrainer(base_trainer.BaseTrainer):
    """Runs the training pipeline for the acoustic model.

    The trainer does the following:
    - iterates over the training data for a specified number of steps
    - computes the loss and gradients for all model's components
    - updates the model's parameters
    - logs the training progress
    - logs statistics for profiling purposes
    """

    def __init__(self,
                 model_components: model_utils.ModelComponents,
                 train_data_loader: torch.utils.data.DataLoader,
                 val_data_loader: torch.utils.data.DataLoader,
                 tb_logger: pt_tensorboard.SummaryWriter,
                 device: torch.device,
                 checkpoints_handler: shared_m_utils.ModelCheckpointHandler,
                 checkpoints_interval: int,
                 validation_interval: int,
                 d_model: int,
                 warmup_steps: int,
                 use_gt_durations_for_visualization: bool,
                 use_loss_weights: bool):
        """Initializes the model trainer.

        See the arguments of the BaseTrainer constructor.

        Args:
            d_model: Dimensionality of the transformer architecture. It is the size of the
                embedding every input sequence's element is projected to.
            warmup_steps: The number of warmup steps for the learning rate scheduler.
            use_gt_durations_for_visualization: Tells whether to use ground truth durations
                instead of the predicted ones while performing visualization.
            use_loss_weights: Tells whether to use loss weights for the loss computation.
        """

        base_optimizer = torch.optim.Adam(model_components.parameters(),
                                          lr=2e-4,
                                          betas=(0.9, 0.98))
        optimizer = shared_m_utils.TransformerScheduledOptim(base_optimizer,
                                                             d_model,
                                                             warmup_steps)

        super().__init__(
            model_comps=model_components,
            train_data_loader=train_data_loader,
            val_data_loader=val_data_loader,
            tb_logger=tb_logger,
            device=device,
            checkpoints_handler=checkpoints_handler,
            checkpoints_interval=checkpoints_interval,
            validation_interval=validation_interval,
            optimizer=optimizer)

        self._use_gt_durations_for_visualization = use_gt_durations_for_visualization
        self._use_loss_weights = use_loss_weights
        self._visualization_interval = validation_interval * 5

        self._spec_prediction_loss = torch.nn.MSELoss(reduction='none')
        self._duration_loss = torch.nn.MSELoss(reduction='none')

    @property
    def model_comps(self) -> model_utils.ModelComponents:
        """Returns the model components."""

        assert isinstance(self._model_comps, model_utils.ModelComponents)
        return self._model_comps

    def _compute_losses(self, input_batch: Tuple[torch.Tensor, ...]  # pylint: disable=too-many-locals
                        ) -> Dict[str, torch.Tensor]:
        """Overrides BaseTrainer::_compute_losses."""

        spectrogram, phonemes, durations = input_batch
        durations = torch.unsqueeze(durations, -1)

        if self.model_comps.gst and self.model_comps.embedder:

            style_embedding: torch.Tensor = self.model_comps.embedder(
                spectrogram, self.model_comps.gst())

        else:
            style_embedding = None

        encoder_output: torch.Tensor = self.model_comps.encoder(phonemes, style_embedding)

        predicted_durations: torch.Tensor = self.model_comps.duration_predictor(
            encoder_output.detach())

        stretched_encoder_output: torch.Tensor = self.model_comps.length_regulator(
            encoder_output, durations)

        decoder_output: torch.Tensor = self.model_comps.decoder(stretched_encoder_output)

        spec_prediction_loss = self._spec_prediction_loss(decoder_output, spectrogram)
        duration_loss = self._duration_loss(predicted_durations, durations)

        dur_mask, dur_mask_sum = other_utils.create_loss_mask_for_durations(durations)
        duration_loss = torch.sum(duration_loss * dur_mask) / dur_mask_sum

        spec_mask, spec_mask_sum = other_utils.create_loss_mask_for_spectrogram(spectrogram,
                                                                                durations,
                                                                                dur_mask)
        if self._use_loss_weights:
            spec_weights = other_utils.create_loss_weight_for_spectrogram(spectrogram)
            spec_prediction_loss = torch.sum(spec_prediction_loss * spec_mask * spec_weights)

        else:
            spec_prediction_loss = torch.sum(spec_prediction_loss * spec_mask)

        spec_prediction_loss /= spec_mask_sum

        return {
            'spec_pred_loss': spec_prediction_loss,
            'duration_loss': duration_loss,
            'duration_pred_mae': metrics.mean_absolute_error(
                predicted_durations, durations, dur_mask, dur_mask_sum),
            'spec_pred_mae': metrics.mean_absolute_error(
                decoder_output, spectrogram, spec_mask, spec_mask_sum),
            'total_loss': spec_prediction_loss + duration_loss
        }

    def _on_step_end(self, step_idx):

        if (step_idx + 1) % self._visualization_interval == 0:
            logging.info('Visualizing model output after %d steps.', step_idx + 1)
            self._perform_visualization(step_idx)

    def _perform_visualization(self, step_idx: int):
        """Performs visualization of the model's predictions."""

        self.model_comps.eval()

        spectrogram, decoder_output = self._perform_visualization_for_loader(self._val_data_loader)

        self._tb_logger.add_image(
            'Validation/Visualization/Original',
            visualization.colorize_spectrogram(spectrogram[0], 'viridis'),
            step_idx)

        self._tb_logger.add_image(
            'Validation/Visualization/Predicted',
            visualization.colorize_spectrogram(decoder_output[0], 'viridis'),
            step_idx)

        spectrogram, decoder_output = self._perform_visualization_for_loader(
            self._train_data_loader)

        self._tb_logger.add_image(
            'Training/Visualization/Original',
            visualization.colorize_spectrogram(spectrogram[0], 'viridis'),
            step_idx)

        self._tb_logger.add_image(
            'Training/Visualization/Predicted',
            visualization.colorize_spectrogram(decoder_output[0], 'viridis'),
            step_idx)

    def _perform_visualization_for_loader(self,
                                          data_loader: torch.utils.data.DataLoader
                                          ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs visualization for the given data loader."""

        with torch.no_grad():

            batch = next(iter(data_loader))
            spectrogram, phonemes, durations = batch
            spectrogram = spectrogram[0:1]
            phonemes = phonemes[0:1]

            spectrogram = spectrogram.to(self._device)
            phonemes = phonemes.to(self._device)

            if self.model_comps.gst and self.model_comps.embedder:

                style_embedding: torch.Tensor = self.model_comps.embedder(
                    spectrogram, self.model_comps.gst())

            else:
                style_embedding = None

            phoneme_representations = self.model_comps.encoder(phonemes, style_embedding)

            durations_mask = inf_utils.create_transcript_mask(phonemes).to(self._device)
            durations_mask = torch.reshape(durations_mask, (1, -1, 1))

            if self._use_gt_durations_for_visualization:
                phoneme_durations = self.model_comps.duration_predictor(phoneme_representations)
                phoneme_durations = inf_utils.sanitize_predicted_durations(phoneme_durations,
                                                                           spectrogram.shape[2])
                phoneme_durations = phoneme_durations * durations_mask

            else:
                phoneme_durations = durations

            stretched_phoneme_representations = self.model_comps.length_regulator(
                phoneme_representations, phoneme_durations)

            decoder_output = self.model_comps.decoder(
                stretched_phoneme_representations)

            return spectrogram, decoder_output
