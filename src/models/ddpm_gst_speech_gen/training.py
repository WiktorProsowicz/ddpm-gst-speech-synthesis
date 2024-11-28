# -*- coding: utf-8 -*-
"""Contains the training/validation/profiling pipeline for the DDPM-GST-Speech-Gen model."""
import logging
from typing import Callable
from typing import Dict
from typing import Tuple

import torch
from torch.utils import tensorboard as pt_tensorboard

from data import visualization
from models import base_trainer
from models import utils as shared_m_utils
from models.ddpm_gst_speech_gen import utils as model_utils
from utilities import diffusion as diff_utils
from utilities import inference as inf_utils
from utilities import metrics
from utilities import other as other_utils


class ModelTrainer(base_trainer.BaseTrainer):
    """Runs the training pipeline for the DDPM-GST-Speech-Gen model.

    The trainer does the following:
    - iterates over the training data for a specified number of steps
    - samples noising steps and the noised data
    - computes the loss and gradients for all model's components
    - updates the model's parameters
    - logs the training progress
    - logs statistics for profiling purposes
    """

    def __init__(self,
                 model_provider: Callable[[], model_utils.ModelComponents],
                 train_data_loader: torch.utils.data.DataLoader,
                 val_data_loader: torch.utils.data.DataLoader,
                 tb_logger: pt_tensorboard.SummaryWriter,
                 device: torch.device,
                 diff_params_scheduler: diff_utils.ParametrizationScheduler,
                 checkpoints_handler: shared_m_utils.ModelCheckpointHandler,
                 checkpoints_interval: int,
                 validation_interval: int,
                 learning_rate: float,
                 use_gt_durations_for_back_diff: bool,
                 use_loss_weights: bool):
        """Initializes the model trainer.

        See the arguments of the BaseTrainer constructor.

        Args:
            diff_params_scheduler: Provides parameters for the diffusion process.
            use_gt_durations_for_back_diff: Tells whether to use ground truth durations
                instead of the predicted ones while performing backward diffusion for validation
                purposes.
            use_loss_weights: Tells whether to use loss weights for the loss computation. The
                weights are intended to guide the model to focus on the lower frequencies in
                the generated spectrogram.
        """

        model_components = model_provider()

        super().__init__(
            model_comps=model_components,
            train_data_loader=train_data_loader,
            val_data_loader=val_data_loader,
            tb_logger=tb_logger,
            device=device,
            checkpoints_handler=checkpoints_handler,
            checkpoints_interval=checkpoints_interval,
            validation_interval=validation_interval,
            optimizer=torch.optim.Adam(model_components.parameters(), lr=learning_rate))

        self._diffusion_handler = diff_utils.DiffusionHandler(diff_params_scheduler, self._device)
        self._backward_diff_interval = validation_interval * 5
        self._use_gt_durations_for_back_diff = use_gt_durations_for_back_diff
        self._use_loss_weights = use_loss_weights

        self._noise_prediction_loss = torch.nn.MSELoss(reduction='none')
        self._duration_loss = torch.nn.MSELoss(reduction='none')

    @property
    def model_comps(self) -> model_utils.ModelComponents:
        """Returns the model components."""

        assert isinstance(self._model_comps, model_utils.ModelComponents)
        return self._model_comps

    def _compute_losses(self,  # pylint: disable=too-many-locals
                        input_batch: Tuple[torch.Tensor, ...]
                        ) -> Dict[str, torch.Tensor]:
        """Calls the model with the given input data and computes the losses.

        Returns:
            A tuple containing the noise prediction loss, duration prediction loss and metrics.
        """

        spectrogram, phonemes, durations = input_batch
        durations = torch.unsqueeze(durations, -1)
        batch_size = spectrogram.shape[0]

        noise = torch.randn_like(spectrogram)
        diff_timestep = torch.randint(
            0, self._diffusion_handler.num_steps, (batch_size,), device=self._device)

        noised_data = self._diffusion_handler.add_noise(spectrogram, noise, diff_timestep)

        if self.model_comps.gst_provider and self.model_comps.reference_embedder:

            style_embedding: torch.Tensor = self.model_comps.reference_embedder(
                spectrogram, self.model_comps.gst_provider())

        else:
            style_embedding = None

        encoder_output: torch.Tensor = self.model_comps.encoder(phonemes)

        predicted_durations: torch.Tensor = self.model_comps.duration_predictor(
            encoder_output.detach())

        stretched_encoder_output: torch.Tensor = self.model_comps.length_regulator(
            encoder_output, durations)

        decoder_output: torch.Tensor = self.model_comps.decoder(
            diff_timestep, noised_data, stretched_encoder_output, style_embedding)

        noise_prediction_loss = self._noise_prediction_loss(decoder_output, noise)
        duration_loss = self._duration_loss(predicted_durations, durations)

        dur_mask, dur_mask_sum = other_utils.create_loss_mask_for_durations(durations)
        duration_loss = torch.sum(duration_loss * dur_mask) / dur_mask_sum

        spec_mask, spec_mask_sum = other_utils.create_loss_mask_for_spectrogram(spectrogram,
                                                                                durations,
                                                                                dur_mask)
        if self._use_loss_weights:
            spec_weights = other_utils.create_loss_weight_for_spectrogram(spectrogram)
            noise_prediction_loss = torch.sum(noise_prediction_loss * spec_mask * spec_weights)

        else:
            noise_prediction_loss = torch.sum(noise_prediction_loss * spec_mask)

        noise_prediction_loss /= spec_mask_sum

        return {
            'noise_pred_loss': noise_prediction_loss,
            'duration_loss': duration_loss,
            'duration_pred_mae': metrics.mean_absolute_error(
                predicted_durations, durations, dur_mask, dur_mask_sum),
            'noise_pred_mae': metrics.mean_absolute_error(
                decoder_output, noise, spec_mask, spec_mask_sum),
            'total_loss': noise_prediction_loss + duration_loss
        }

    def _on_step_end(self, step_idx: int):

        if (step_idx + 1) % self._backward_diff_interval == 0:
            logging.info('Running backward diffusion after %d steps.', step_idx + 1)
            self._perform_backward_diffusion(step_idx)

    def _perform_backward_diffusion(self, step_idx: int):
        """Tries to run the backward diffusion and logs the results."""

        self.model_comps.eval()

        spectrogram, denoised_spectrogram = self._perform_backward_diff_for_loader(
            self._val_data_loader)

        self._tb_logger.add_image(
            'Validation/BackwardDiffusion/Original',
            visualization.colorize_spectrogram(spectrogram[0], 'viridis'),
            step_idx)

        self._tb_logger.add_image(
            'Validation/BackwardDiffusion/Denoised',
            visualization.colorize_spectrogram(denoised_spectrogram[0], 'viridis'),
            step_idx)

        spectrogram, denoised_spectrogram = self._perform_backward_diff_for_loader(
            self._train_data_loader)

        self._tb_logger.add_image(
            'Training/BackwardDiffusion/Original',
            visualization.colorize_spectrogram(spectrogram[0], 'viridis'),
            step_idx)

        self._tb_logger.add_image(
            'Training/BackwardDiffusion/Denoised',
            visualization.colorize_spectrogram(denoised_spectrogram[0], 'viridis'),
            step_idx)

    def _perform_backward_diff_for_loader(self,
                                          loader: torch.utils.data.DataLoader
                                          ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs backward diffusion using the specified data loader."""

        with torch.no_grad():

            batch = next(iter(loader))
            batch = tuple(tensor.to(self._device) for tensor in batch)

            spectrogram, phonemes, durations = batch
            spectrogram = spectrogram[0:1]
            phonemes = phonemes[0:1]

            spectrogram = spectrogram.to(self._device)
            phonemes = phonemes.to(self._device)

            initial_noise = torch.randn_like(spectrogram)

            phoneme_representations = self.model_comps.encoder(phonemes)

            durations_mask = inf_utils.create_transcript_mask(phonemes).to(self._device)
            durations_mask = torch.reshape(durations_mask, (1, -1, 1))

            if self._use_gt_durations_for_back_diff:
                phoneme_durations = self.model_comps.duration_predictor(phoneme_representations)
                phoneme_durations = inf_utils.sanitize_predicted_durations(phoneme_durations,
                                                                           spectrogram.shape[2])
                phoneme_durations = phoneme_durations * durations_mask

            else:
                phoneme_durations = durations

            stretched_phoneme_representations = self.model_comps.length_regulator(
                phoneme_representations, phoneme_durations)

            def model_callable(model_inputs: inf_utils.BackwardDiffusionModelInput) -> torch.Tensor:

                return self.model_comps.decoder(
                    model_inputs.timestep,
                    model_inputs.noised_data,
                    stretched_phoneme_representations,
                    None)

            denoised_spectrogram = inf_utils.run_backward_diffusion(model_callable,
                                                                    self._diffusion_handler,
                                                                    initial_noise)

        return spectrogram, denoised_spectrogram
