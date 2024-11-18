# -*- coding=utf-8 -*-
"""Contains the training/validation/profiling pipeline for the DDPM-GST-Speech-Gen model."""

from typing import Callable, Optional, Tuple

import torch
from torch.utils import tensorboard as pt_tensorboard

from models.ddpm_gst_speech_gen import utils as model_utils
from utilities import diffusion as diff_utils


class ModelTrainer:
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
                 checkpoints_handler: model_utils.ModelCheckpointHandler,
                 checkpoints_interval: int,
                 validation_interval: int,
                 learning_rate: float):
        """Initializes the model trainer.

        Args:
            model_provider: A callable that returns the model components.
            train_data_loader: The data loader for the training data.
            val_data_loader: The data loader for the validation data.
            tb_logger: The tensorboard logger.
            optimizer: The optimizer to use for training.
            device: The device to run the computations on.
            checkpoints_handler: The handler for saving/loading model checkpoints.
            checkpoints_interval: The number of steps between saving checkpoints.
            diff_params_scheduler: The scheduler for the diffusion process parameters.
            validation_interval: The number of steps between validation runs.
        """

        self._model_comps = model_provider()
        self._train_data_loader = train_data_loader
        self._val_data_loader = val_data_loader
        self._tb_logger = tb_logger
        self._device = device
        self._diffusion_handler = diff_utils.DiffusionHandler(diff_params_scheduler)
        self._checkpoints_handler = checkpoints_handler
        self._checkpoints_interval = checkpoints_interval
        self._validation_interval = validation_interval

        self._optimizer = torch.optim.Adam(self._model_comps.parameters(), lr=learning_rate)
        self._noise_prediction_loss = torch.nn.MSELoss()
        self._duration_loss = torch.nn.MSELoss()

    def run_training(self, num_steps: int, start_step: int = 0, use_profiler: bool = False):
        """Runs the training pipeline.

        Args:
            num_steps: The number of training steps to run.
            start_step: The step to start training from.
            use_profiler: Whether to use the code profiling while training.
        """

        if use_profiler:

            with torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=5, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    self._tb_logger.get_logdir()),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as profiler:

                self._run_training_pipeline(num_steps, start_step, profiler)

        else:

            self._run_training_pipeline(num_steps, start_step)

    def _run_training_pipeline(self, num_steps: int,
                               start_step: int = 0,
                               profiler: Optional[torch.profiler.profile] = None):
        """Runs the training pipeline.

        Args:
            num_steps: The number of training steps to run.
            profiler: The profiler to use for profiling the code.
        """

        data_loader_enum = enumerate(self._train_data_loader)

        for step_idx in range(start_step, num_steps):

            try:
                _, batch = next(data_loader_enum)

            except StopIteration:
                data_loader_enum = enumerate(self._train_data_loader)
                _, batch = next(self._train_data_loader)

            if profiler:
                profiler.step()

            self._run_training_step(step_idx, batch)

            if (step_idx + 1) % self._validation_interval == 0:
                self._run_validation(step_idx)

            if (step_idx + 1) % self._checkpoints_interval == 0:
                self._checkpoints_handler.save_checkpoint(self._model_comps, {
                    "n_training_steps": step_idx + 1,
                })

    def _run_training_step(self, step_idx: int, batch):
        """Runs a single training step.

        Args:
            step_idx: The index of the current step.
        """

        spectrogram, phonemes, durations = batch
        durations = torch.unsqueeze(durations, -1)

        spectrogram = spectrogram.to(self._device)
        phonemes = phonemes.to(self._device)
        durations = durations.to(self._device)

        self._optimizer.zero_grad()

        noise_prediction_loss, duration_loss = self._compute_losses(
            spectrogram, phonemes, durations)

        total_loss = noise_prediction_loss + duration_loss

        self._tb_logger.add_scalar(
            'Training/Loss/NoisePrediction',
            noise_prediction_loss.item(),
            step_idx)
        self._tb_logger.add_scalar('Training/Loss/Duration', duration_loss.item(), step_idx)
        self._tb_logger.add_scalar('Training/Loss/Total', total_loss.item(), step_idx)

        total_loss.backward()
        self._optimizer.step()

    def _run_validation(self, step_idx: int):
        """Runs validation on the validation data.

        Args:
            step_idx: The index of the current training step.
        """

        with torch.no_grad():

            avg_noise_prediction_loss = torch.tensor(0, device=self._device)
            avg_duration_loss = torch.tensor(0, device=self._device)
            avg_total_loss = torch.tensor(0, device=self._device)

            for batch in self._val_data_loader:

                spectrogram, phonemes, durations = batch

                spectrogram = spectrogram.to(self._device)
                phonemes = phonemes.to(self._device)
                durations = durations.to(self._device)

                noise_prediction_loss, duration_loss = self._compute_losses(
                    spectrogram, phonemes, durations)

                avg_noise_prediction_loss += noise_prediction_loss
                avg_duration_loss += duration_loss
                avg_total_loss += noise_prediction_loss + duration_loss

            avg_total_loss /= len(self._val_data_loader)
            avg_noise_prediction_loss /= len(self._val_data_loader)
            avg_duration_loss /= len(self._val_data_loader)

            self._tb_logger.add_scalar('Validation/Loss/Total', avg_total_loss[0], step_idx)

            self._tb_logger.add_scalar(
                'Validation/Loss/NoisePrediction',
                avg_noise_prediction_loss[0],
                step_idx)

            self._tb_logger.add_scalar('Validation/Loss/Duration', avg_duration_loss[0], step_idx)

    def _compute_losses(self, spectrogram, phonemes,
                        durations) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calls the model with the given input data and computes the losses.

        Returns:
            A tuple containing the noise prediction and duration prediction losses.
        """

        batch_size = spectrogram.shape[0]

        noise = torch.randn_like(spectrogram)
        diff_timestep = torch.randint(
            0, self._diffusion_handler.num_steps, (batch_size,), device=self._device)

        noised_data = self._diffusion_handler.add_noise(spectrogram, noise, diff_timestep)

        if self._model_comps.gst_provider and self._model_comps.reference_embedder:

            style_embedding: torch.Tensor = self._model_comps.reference_embedder(
                spectrogram, self._model_comps.gst_provider())

        else:
            style_embedding = None

        encoder_output: torch.Tensor = self._model_comps.encoder(phonemes)

        predicted_durations: torch.Tensor = self._model_comps.duration_predictor(
            encoder_output.detach())

        stretched_encoder_output: torch.Tensor = self._model_comps.length_regulator(
            encoder_output, durations)

        decoder_output: torch.Tensor = self._model_comps.decoder(
            diff_timestep, noised_data, stretched_encoder_output, style_embedding)

        noise_prediction_noise = self._noise_prediction_loss(decoder_output, spectrogram)
        duration_loss = self._duration_loss(predicted_durations, durations)

        return noise_prediction_noise, duration_loss
