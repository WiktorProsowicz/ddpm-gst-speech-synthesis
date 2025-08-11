# -*- coding: utf-8 -*-
"""Contains the training/validation/profiling pipeline for the acoustic model."""
import logging
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import torch
import torch_dev_utils as tdu
from torchaudio.prototype.pipelines import HIFIGAN_VOCODER_V3_LJSPEECH as hifigan_bundle

from data import visualization
from models.acoustic import utils as model_utils
from utilities import inference as inf_utils
from utilities import metrics
from utilities import other as other_utils


class ModelTrainer(tdu.training.BaseTrainer):
    """Runs the training pipeline for the acoustic model.

    The trainer does the following:
    - iterates over the training data for a specified number of steps
    - computes the loss and gradients for all model's components
    - updates the model's parameters
    - logs the training progress
    - logs statistics for profiling purposes
    """

    def __init__(self, params: tdu.training.BaseTrainerParams):
        """Initializes the model trainer.
        """

        super().__init__(params)

        self._visualization_interval = params.validation_interval * 5
        self._metrics_interval = params.validation_interval * 20
        self._max_samples_for_metrics = 50
        self._n_samples_for_visualization = 5

        self._spec_prediction_loss = torch.nn.MSELoss(reduction='none')
        self._duration_loss = torch.nn.MSELoss(reduction='none')

    @property
    def model_comps(self) -> model_utils.ModelComponents:
        """Returns the model components."""

        assert isinstance(self._model_comps, model_utils.ModelComponents)
        return self._model_comps

    def _compute_model_outputs(self,
                               input_batch: Tuple[torch.Tensor, ...]
                               ) -> Tuple[torch.Tensor, torch.Tensor]:

        spectrogram, phonemes, durations, p_mask, s_mask = input_batch

        if self.model_comps.embedder:
            style_embedding = self.model_comps.embedder(spectrogram, s_mask)

        else:
            style_embedding = None

        encoder_output: torch.Tensor = self.model_comps.encoder(phonemes, style_embedding, p_mask)

        predicted_durations: torch.Tensor = self.model_comps.duration_predictor(
            encoder_output.detach())

        stretched_encoder_output: torch.Tensor = self.model_comps.length_regulator(
            encoder_output, durations)

        decoder_output: torch.Tensor = self.model_comps.decoder(stretched_encoder_output, s_mask)

        return decoder_output, predicted_durations

    def _compute_losses_and_metrics(self,  # pylint: disable=too-many-locals
                                    input_batch: Tuple[torch.Tensor, ...]
                                    ) -> Tuple[Dict[str, torch.Tensor], ...]:
        """Overrides BaseTrainer::_compute_losses."""

        gt_spectrogram, _, gt_durations, _, _ = input_batch

        pred_spectrogram, pred_durations = self._compute_model_outputs(input_batch)

        spec_prediction_loss = self._spec_prediction_loss(pred_spectrogram, gt_spectrogram)
        duration_loss = self._duration_loss(pred_durations, gt_durations)

        l_dur_mask, l_dur_mask_sum = other_utils.create_loss_mask_for_durations(gt_durations)
        duration_loss = torch.sum(duration_loss * l_dur_mask) / l_dur_mask_sum

        l_spec_mask, l_spec_mask_sum = other_utils.create_loss_mask_for_spectrogram(gt_spectrogram,
                                                                                    gt_durations,
                                                                                    l_dur_mask)
        spec_prediction_loss = torch.sum(spec_prediction_loss * l_spec_mask)
        spec_prediction_loss /= l_spec_mask_sum

        return (
            {'spec_pred_loss': spec_prediction_loss,
             'duration_loss': duration_loss},
            {'duration_pred_mae': metrics.mean_absolute_error(
                pred_durations, gt_durations, l_dur_mask, l_dur_mask_sum),
             'spec_pred_mae': metrics.mean_absolute_error(
                pred_spectrogram, gt_spectrogram, l_spec_mask, l_spec_mask_sum)}
        )

    def _on_step_end(self, step_idx):

        if (step_idx + 1) % self._visualization_interval == 0:

            logging.info('Visualizing model output after %d steps.', step_idx + 1)

            self._perform_visualization(step_idx)

        if (step_idx + 1) % self._metrics_interval == 0:

            logging.info('Calculating metrics after %d steps.', step_idx + 1)

            self._calculate_and_plot_metrics(step_idx)

    def _on_step_start(self, step_idx: int):
        pass

    def _perform_visualization(self, step_idx: int):
        """Performs visualization of the model's predictions."""

        self.model_comps.eval()

        for label, data_loader in [('validation', self._val_data_loader),
                                   ('training', self._train_data_loader)]:

            n_visualized_files = min(self._n_samples_for_visualization, data_loader.batch_size)

            gt_output, pred_output = self._run_inference_for_loader(data_loader,
                                                                    n_visualized_files)

            gt_wav, gt_dur, gt_spec = gt_output
            pred_wav, pred_dur, pred_spec = pred_output

            for i in range(n_visualized_files):

                self._tb_logger.add_image(
                    f'{label}/spectrogram/{i}/original',
                    visualization.colorize_spectrogram(gt_spec[i], 'viridis'),
                    step_idx)

                self._tb_logger.add_image(
                    f'{label}/spectrogram/{i}/predicted',
                    visualization.colorize_spectrogram(pred_spec[i], 'viridis'),
                    step_idx)

                self._tb_logger.add_audio(
                    f'{label}/waveform/{i}/original',
                    gt_wav[i].cpu(),
                    step_idx,
                    22050)

                self._tb_logger.add_audio(
                    f'{label}/waveform/{i}/predicted',
                    pred_wav[i].cpu(),
                    step_idx,
                    22050)

                self._tb_logger.add_figure(
                    f'{label}/durations/{i}',
                    visualization.plot_pred_and_gt_durations(gt_dur[i], pred_dur[i]),
                    step_idx)

    def _calculate_and_plot_metrics(self, step_idx: int):
        """Runs inference, calculates metrics and plots them."""

        self.model_comps.eval()

        for label, data_loader in [('validation', self._val_data_loader),
                                   ('training', self._train_data_loader)]:

            n_chosen_files = min(self._max_samples_for_metrics, data_loader.batch_size)

            gt_output, pred_output = self._run_inference_for_loader(data_loader,
                                                                    n_chosen_files)

            gt_wav, _, _ = gt_output
            pred_wav, _, _ = pred_output

            f0_rmse_sum = np.float32(0.0)
            f0_corr_sum = np.float32(0.0)

            for i in range(n_chosen_files):

                f0_corr, f0_rmse = metrics.f0_pearson_corr_and_rmse(gt_wav[i].cpu().numpy(),
                                                                    pred_wav[i].cpu().numpy())

                f0_rmse_sum += f0_rmse
                f0_corr_sum += f0_corr

            self._tb_logger.add_scalars('f0_rmse',
                                        {label: f0_rmse_sum / n_chosen_files},
                                        step_idx)

            self._tb_logger.add_scalars('f0_corr',
                                        {label: f0_corr_sum / n_chosen_files},
                                        step_idx)

    def _run_inference_for_loader(self,  # pylint: disable=too-many-locals
                                  data_loader: torch.utils.data.DataLoader,
                                  n_runs: int) -> Tuple[Tuple[List[torch.Tensor], ...],
                                                        Tuple[List[torch.Tensor], ...]]:
        """Runs inference on `n_runs` chosen files and returns the results."""

        self.model_comps.eval()

        vocoder = hifigan_bundle.get_vocoder().to(self._device)
        inference_model = inf_utils.InferenceAcousticModel(self.model_comps,
                                                           vocoder,
                                                           0.5)

        batch = next(iter(data_loader))
        batch = [elem.to(self._device) for elem in batch]

        spectrogram, phonemes, durations, p_mask, s_mask = batch

        gt_results: Tuple[List[torch.Tensor], ...] = ([], [], [])
        pred_results: Tuple[List[torch.Tensor], ...] = ([], [], [])

        for i in range(n_runs):

            i_spectrogram = spectrogram[i:i + 1]
            i_phonemes = phonemes[i:i + 1]
            i_durations = durations[i:i + 1]
            i_p_mask = p_mask[i:i + 1]
            i_s_mask = s_mask[i:i + 1]

            with torch.no_grad():

                if self.model_comps.embedder:
                    gst_weights = self.model_comps.embedder.obtain_gst_weights(i_spectrogram,
                                                                               i_s_mask)
                    gst_embedding = self.model_comps.embedder(i_spectrogram, i_s_mask)

                else:
                    gst_weights = None
                    gst_embedding = None

                pred_wave, pred_dur, pred_spec = inference_model(i_phonemes,
                                                                 i_p_mask,
                                                                 gst_weights,
                                                                 gst_embedding,
                                                                 return_intermediate_results=True)

                gt_wave = vocoder(i_spectrogram)  # pylint: disable=not-callable

                gt_results[0].append(gt_wave[0])
                gt_results[1].append(i_durations[0])
                gt_results[2].append(i_spectrogram[0])

                pred_results[0].append(pred_wave[0])
                pred_results[1].append(pred_dur[0])
                pred_results[2].append(pred_spec[0])

        return gt_results, pred_results
