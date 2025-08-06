# -*- coding: utf-8 -*-
"""Contains utilities for running inference with the trained model."""
from typing import Optional
from typing import Tuple

import numpy as np
import pytorch_pretrained_bert as bert_lib
import torch
from torchvision import transforms

from models.acoustic import utils as acoustic_utils
from models.gst_predictor import utils as gst_utils
from utilities import diffusion as diff_utils


def _split_transcript_into_tokens(transcript: str):

    special_chars = ['.', ',', '!', '?']

    for char in special_chars:
        transcript = transcript.replace(char, f' {char} ')

    return transcript.split()


def obtain_gst_predictor_inputs(transcript: str,
                                text_transforms: transforms.Compose,
                                tokenizer: bert_lib.BertTokenizer,
                                bert_model: bert_lib.BertModel,
                                device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:

    init_tokens = _split_transcript_into_tokens(transcript)

    phonemes_counts = []
    input_phonemes = []

    for token in init_tokens:
        phonemes = text_transforms.transforms[0](token)

        input_phonemes += phonemes
        phonemes_counts.append(len(phonemes))

    tokens_counts = []
    bert_tokens = []

    for token in init_tokens:
        tokens = tokenizer.tokenize(token)

        bert_tokens += tokens
        tokens_counts.append(len(tokens))

    bert_tokens = ['[CLS]'] + bert_tokens + ['[SEP]']
    bert_tokens = tokenizer.convert_tokens_to_ids(bert_tokens)
    segments_ids = [0] * len(bert_tokens)

    tokens_tensor = torch.tensor([bert_tokens]).to(device)
    segments_tensors = torch.tensor([segments_ids]).to(device)

    with torch.no_grad():
        encoded_layers, _ = bert_model(tokens_tensor, segments_tensors)
        bert_embeddings = encoded_layers[-1].squeeze(0)[1:-1]

    tokens_counts = np.array(tokens_counts)
    phonemes_counts = torch.tensor(phonemes_counts, dtype=torch.int64).to(device)
    tokens_upper_bounds = np.cumsum(tokens_counts)

    averaged_bert_embeddings = torch.zeros(
        (len(tokens_counts), bert_embeddings.shape[1]), device=device)

    for i, token_count in enumerate(tokens_counts):
        upper_bound = tokens_upper_bounds[i]
        lower_bound = upper_bound - token_count

        averaged_bert_embeddings[i] = torch.mean(bert_embeddings[lower_bound:upper_bound], dim=0)

    input_phonemes_t = text_transforms.transforms[1](input_phonemes).to(device)

    averaged_bert_embeddings = torch.repeat_interleave(averaged_bert_embeddings,
                                                       phonemes_counts,
                                                       dim=0)

    assert input_phonemes_t.shape[0] == averaged_bert_embeddings.shape[0]

    return averaged_bert_embeddings, input_phonemes_t


def get_transcript_length(transcript: torch.Tensor) -> torch.Tensor:
    """Returns the actual length of the one-hot encoded transcript.

    Args:
        transcript: The one-hot encoded transcript.
    """

    if len(transcript.shape) == 2:
        return torch.sum(transcript, dtype=torch.int)

    output = torch.sum(transcript, dim=1)
    return torch.sum(output, dtype=torch.int, dim=1)


def create_transcript_mask(transcript: torch.Tensor) -> torch.Tensor:
    """Creates a mask for the transcript based on the actual length.

    Args:
        transcript: The one-hot encoded transcript.
    """

    return (torch.sum(transcript, dim=-1) > 0).to(torch.float32)


def create_spectrogram_mask(spectrogram: torch.Tensor) -> torch.Tensor:
    """Creates a mask for the spectrogram based on the actual length."""

    if len(spectrogram.shape) == 2:
        return torch.sum(spectrogram == torch.min(spectrogram), dim=0) != spectrogram.shape[0]

    return torch.sum(spectrogram == torch.min(spectrogram), dim=1) != spectrogram.shape[1]


def create_mask_from_durations(log_durations: torch.Tensor,
                               expected_output_length: int) -> torch.Tensor:
    """Creates a mask for the stretched phoneme representations based on the predicted durations."""

    durations_mask = (log_durations > 0).to(torch.int64)
    durations = (torch.pow(2.0, log_durations)).to(torch.int64) * durations_mask
    cum_length = torch.sum(durations, dim=1).to(torch.int64)
    mask = torch.zeros((durations.shape[0], expected_output_length), dtype=torch.bool)

    for i in range(durations.shape[0]):
        mask[i, :cum_length[i]] = 1

    return mask.to(torch.bool).to(durations.device)


def sanitize_predicted_durations(log_durations: torch.Tensor,
                                 expected_output_length: int) -> torch.Tensor:
    """Sanitizes the predicted durations so that an alignment matrix can be created.

    Args:
        log_durations: The predicted log durations.
        expected_output_length: The expected length of the tensor stretched by the durations.
    """

    log_durations = torch.clamp(log_durations, min=0.0)
    pow_duration = torch.pow(2.0, log_durations)

    cum_durations = torch.cumsum(pow_duration, dim=1)
    durations_mask = cum_durations <= expected_output_length

    return log_durations * durations_mask


class InferenceGSTPredictor(torch.nn.Module):
    """Runs the whole GST prediction pipeline."""

    def __init__(self,
                 gst_components: gst_utils.ModelComponents,
                 diffusion_handler: diff_utils.DiffusionHandler,
                 scaling_values: Tuple[torch.Tensor, torch.Tensor],
                 guidance_scale: Optional[float] = None):
        """Initializes the GST predictor.

        Args:
            gst_components: The components of the GST predictor.
            diffusion_handler: The diffusion handler used during the training.
            scaling_values: The (factor, shift) values used to scale the output weights.
        """

        super().__init__()

        self._gst_comps = gst_components
        self._diffusion_handler = diffusion_handler
        self._emb_factor, self._emb_shift, self._w_factor, self._w_shift = scaling_values
        self._guidance_scale = guidance_scale

    def forward(self,
                phoneme_representations: torch.Tensor,
                bert_embeddings: torch.Tensor,
                phoneme_mask: torch.Tensor):
        """Runs the inference model.

        Args:
            phoneme_representations: The acoustic encoder's output.
            bert_embeddings: The BERT embeddings.
            phoneme_mask: The mask for the phoneme representations.
        """

        pred_weights = self._gst_comps.deterministic_pred(
            phoneme_representations, bert_embeddings, phoneme_mask)

        noised_gst = torch.randn(1, self._gst_comps.decoder.gst_size,
                                 device=phoneme_representations.device)

        phoneme_embedding = self._gst_comps.encoder(phoneme_representations,
                                                    phoneme_mask,
                                                    bert_embeddings)

        for diff_step in reversed(range(self._diffusion_handler.num_steps)):

            timestep = torch.tensor([diff_step], device=phoneme_representations.device)

            predicted_noise = self._gst_comps.decoder(
                noised_gst,
                timestep,
                phoneme_embedding,
                phoneme_mask
            )

            if self._guidance_scale is not None:

                pred_noise_uncond = self._gst_comps.decoder(
                    noised_gst,
                    timestep,
                    None)

                predicted_noise = (self._guidance_scale + 1) * predicted_noise
                predicted_noise -= self._guidance_scale * pred_noise_uncond

            noised_gst = self._diffusion_handler.remove_noise(noised_gst,
                                                              predicted_noise,
                                                              diff_step)

        return ((noised_gst / self._emb_factor) - self._emb_shift,
                (pred_weights / self._w_factor) - self._w_shift)


class InferenceAcousticModel(torch.nn.Module):
    """Runs the whole acoustic pipeline with optional style embedding prediction."""

    def __init__(self,
                 acoustic_components: acoustic_utils.ModelComponents,
                 vocoder: torch.nn.Module,
                 det_embedding_weight: float):

        super().__init__()

        self._ac_comps = acoustic_components
        self._vocoder = vocoder
        self._use_style_embedding = acoustic_components.embedder is not None
        self._det_emb_weight = det_embedding_weight

    def forward(self,
                input_phonemes: torch.Tensor,
                phoneme_mask: torch.Tensor,
                gst_weights: Optional[torch.Tensor] = None,
                gst_embedding: Optional[torch.Tensor] = None,
                return_intermediate_results: bool = False
                ):
        """Runs the inference model.

        Args:
            input_phonemes: Input one-hot encoded phonemes.
            phoneme_mask: Binary mask indicating non-padding values.
            gst_weights: The GST weights to create the style embedding, if supported.

        Returns:
            The generated waveform.
        """

        phoneme_representations = self._ac_comps.encoder.run_basic_blocks(input_phonemes,
                                                                          phoneme_mask)

        if self._use_style_embedding:
            assert gst_weights is not None
            assert gst_embedding is not None
            assert self._ac_comps.embedder is not None

            embedding_from_w = self._ac_comps.embedder.get_style_embedding_from_weights(gst_weights)

            st_embedding = self._det_emb_weight * embedding_from_w
            st_embedding += (1 - self._det_emb_weight) * gst_embedding

            phoneme_representations = self._ac_comps.encoder.apply_gst_conditioning(
                phoneme_representations,
                st_embedding
            )

        log_durations = self._ac_comps.duration_predictor(phoneme_representations)

        log_durations = sanitize_predicted_durations(
            log_durations,
            self._ac_comps.length_regulator.output_length
        )
        durations_mask = torch.logical_not(phoneme_mask)
        log_durations = log_durations * torch.reshape(durations_mask, (1, -1, 1))

        stretched_phoneme_repr = self._ac_comps.length_regulator(phoneme_representations,
                                                                 log_durations)

        decoder_mask = create_mask_from_durations(
            log_durations.reshape(1, -1),
            self._ac_comps.length_regulator.output_length
        )

        total_dur = torch.sum(decoder_mask).to(torch.int64)
        mel_spec = self._ac_comps.decoder(stretched_phoneme_repr,
                                          torch.logical_not(decoder_mask))

        mel_spec = mel_spec[:, :, :total_dur]

        if not return_intermediate_results:
            return self._vocoder(mel_spec)

        return self._vocoder(mel_spec), log_durations, mel_spec
