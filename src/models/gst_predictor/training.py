# -*- coding: utf-8 -*-
"""Contains the training/validation/profiling pipeline for the GST predictor model."""
import logging
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
import torch_dev_utils as tdu
from torch.utils import tensorboard as pt_tensorboard

from data import visualization
from models.gst_predictor import utils as m_utils
from utilities import diffusion as diff_utils
from utilities import inference
from utilities import metrics


class ModelTrainer(tdu.training.BaseTrainer):
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
                 checkpoints_handler: tdu.serialization.ModelCheckpointHandler,
                 checkpoints_interval: int,
                 validation_interval: int,
                 learning_rate: float,
                 diff_params_scheduler: diff_utils.ParametrizationScheduler,
                 global_ds_stats: Tuple[torch.Tensor, torch.Tensor],
                 guidance_scale: Optional[float]):
        """Initializes the model trainer.

        See the arguments of the BaseTrainer constructor.

        Args:
            learning_rate: The learning rate to use in the optimizer.
            diff_params_scheduler: The scheduler for the diffusion parameters.
            global_ds_stats: The (factor, shift) used to scale the input samples.
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
            optimizer=torch.optim.Adam(model_components.parameters(),
                                       lr=learning_rate,
                                       weight_decay=0.01),
        )

        self._diffusion_handler = diff_utils.DiffusionHandler(diff_params_scheduler,
                                                              self._device)
        self._backward_diff_interval = self._validation_interval
        self._noise_loss = torch.nn.MSELoss()
        self._noise_mae = torch.nn.L1Loss()
        self._weights_loss = torch.nn.MSELoss()
        self._weights_mae = torch.nn.L1Loss()
        self._guidance_scale = guidance_scale

        (self._emb_scale_factor, self._emb_scale_shift,
         self._w_scale_factor, self._w_scale_shift) = global_ds_stats

    @property
    def model_comps(self) -> m_utils.ModelComponents:
        """Returns the model components."""

        assert isinstance(self._model_comps, m_utils.ModelComponents)
        return self._model_comps

    def _compute_losses_and_metrics(self, input_batch: Tuple[torch.Tensor, ...]
                                    ) -> Tuple[Dict[str, torch.Tensor], ...]:
        """Overrides BaseTrainer::_compute_losses."""

        phonemes, phoneme_mask, bert_embeddings, gst_embedding, gst_weights = input_batch
        gst_embedding = (gst_embedding + self._emb_scale_shift) * self._emb_scale_factor
        gst_weights = (gst_weights + self._w_scale_shift) * self._w_scale_factor
        batch_size = phonemes.size(0)

        pred_weights = self.model_comps.deterministic_pred(
            phonemes, bert_embeddings, phoneme_mask)

        noise = torch.randn_like(gst_embedding)
        diff_timestep = torch.randint(
            0, self._diffusion_handler.num_steps, (batch_size,), device=self._device)

        noised_gst = self._diffusion_handler.add_noise(gst_embedding, noise, diff_timestep)

        encoder_output = self.model_comps.encoder(phonemes, phoneme_mask, bert_embeddings)

        pred_noise = self.model_comps.decoder(noised_gst,
                                              diff_timestep,
                                              encoder_output,
                                              phoneme_mask)

        if self._guidance_scale is not None:

            pred_noise_uncond = self.model_comps.decoder(
                noised_gst, diff_timestep, None, None)

            pred_noise = (1 + self._guidance_scale) * pred_noise
            pred_noise -= self._guidance_scale * pred_noise_uncond

        noise_loss = self._noise_loss(pred_noise, noise)
        weights_loss = self._weights_loss(pred_weights, gst_weights)

        return (
            {
                'weights_pred_loss': weights_loss,
                'noise_pred_loss': noise_loss},
            {
                'weights_pred_mae': self._weights_mae(pred_weights, gst_weights),
                'noise_pred_mae': self._noise_mae(pred_noise, noise)}
        )

    def _on_step_end(self, step_idx):

        if (step_idx + 1) % self._backward_diff_interval == 0:
            # if step_idx == 200    00:
            logging.debug('Running full backward diffusion.')
            self._run_backward_diff(step_idx)

    def _on_step_start(self, step_idx: int):
        pass

    def _run_backward_diff(self, step_idx: int):
        """Runs the backward diffusion step and visualizes the output."""

        for label, loader in (('validation', self._val_data_loader),
                              ('training', self._train_data_loader)):

            n_visualized_files = min(5, loader.batch_size)

            ((original_emb, denoised_emb),
             (original_w, pred_w)) = self._run_backward_diff_for_loader(loader,
                                                                        n_visualized_files)

            for i in range(n_visualized_files):

                self._tb_logger.add_figure(
                    f'{label}/{i}/gst_emb_prediction',
                    visualization.plot_pred_and_gt_gst_weights(original_emb[i], denoised_emb[i]),
                    step_idx
                )

                self._tb_logger.add_figure(
                    f'{label}/{i}/gst_weights_prediction',
                    visualization.plot_pred_and_gt_gst_weights(original_w[i], pred_w[i]),
                    step_idx
                )

    def _run_backward_diff_for_loader(self,
                                      data_loader: torch.utils.data.DataLoader,
                                      n_runs: int
                                      ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """Runs the backward diffusion step for the given data loader."""

        self.model_comps.eval()

        inference_model = inference.InferenceGSTPredictor(self.model_comps,
                                                          self._diffusion_handler,
                                                          (self._emb_scale_factor,
                                                           self._emb_scale_shift,
                                                           self._w_scale_factor,
                                                           self._w_scale_shift),
                                                          self._guidance_scale)

        batch = next(iter(data_loader))
        batch = tuple(t.to(self._device) for t in batch)

        phonemes, phoneme_mask, bert_embeddings, gst_emb, gst_weights = batch

        emb_results = ([], [])
        w_results = ([], [])

        for i in range(n_runs):

            phonemes_i = phonemes[i:i + 1]
            gst_weights_i = gst_weights[i:i + 1]
            phoneme_mask_i = phoneme_mask[i:i + 1]
            bert_embeddings_i = bert_embeddings[i:i + 1]
            gst_emb_i = gst_emb[i:i + 1]

            with torch.no_grad():

                pred_emb, pred_weights = inference_model(
                    phonemes_i, bert_embeddings_i, phoneme_mask_i)

            emb_results[0].append(gst_emb_i[0])
            emb_results[1].append(pred_emb[0])
            w_results[0].append(gst_weights_i[0])
            w_results[1].append(pred_weights[0])

        return emb_results, w_results
