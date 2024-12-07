# -*- coding: utf-8 -*-
"""Contains utilities for running inference with the trained model."""
from typing import Optional
from typing import Tuple
from dataclasses import dataclass
from typing import Callable

import torch

from utilities import diffusion as diff_utils


def get_transcript_length(transcript: torch.Tensor) -> torch.Tensor:
    """Returns the actual length of the one-hot encoded transcript.

    Args:
        transcript: The one-hot encoded transcript without the batch_size dimension.
    """

    return torch.sum(transcript, dtype=torch.int)


def create_transcript_mask(transcript: torch.Tensor) -> torch.Tensor:
    """Creates a mask for the transcript based on the actual length.

    Args:
        transcript: The one-hot encoded transcript without the batch_size dimension.
    """

    transcript_length = get_transcript_length(transcript)

    return torch.cat((torch.ones(transcript_length),
                     torch.zeros(transcript.shape[1] - transcript_length)))


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


@dataclass
class BackwardDiffusionModelInput:
    """Contains the input data for a single backward diffusion step.

    The input data shape is supposed to contain the batch_size dimension equal to 1.
    """

    noised_data: torch.Tensor
    timestep: torch.Tensor


def run_backward_diffusion(model_callable: Callable[[BackwardDiffusionModelInput], torch.Tensor],
                           diffusion_handler: diff_utils.DiffusionHandler,
                           input_initial_noise: torch.Tensor) -> torch.Tensor:
    """Performs a full backward diffusion process with the given model and data."""

    noised_data = input_initial_noise

    for diff_step in reversed(range(diffusion_handler.num_steps)):

        model_input = BackwardDiffusionModelInput(
            noised_data=noised_data,
            timestep=torch.tensor([diff_step], device=noised_data.device))

        predicted_noise = model_callable(model_input)

        noised_data = diffusion_handler.remove_noise(noised_data, predicted_noise, diff_step)

    return noised_data


def style_embedding_from_weights(gts_tokens: torch.Tensor,
                                 gst_weights: torch.Tensor) -> torch.Tensor:
    """Creates the style embedding from the GST weights and tokens.

    Args:
        gts_tokens: The global style tokens.
        gst_weights: The weights for the global style tokens.
    """

    gst_weights = torch.unsqueeze(gst_weights, dim=-1)
    return torch.sum(gst_weights * gts_tokens, dim=0)


class InferenceModel(torch.nn.Module):
    """Contains all the components of the model required for inference.

    The model is convertible to a TorchScript.
    """

    def __init__(self,
                 ac_encoder: torch.nn.Module,
                 ac_decoder: torch.nn.Module,
                 duration_predictor: torch.nn.Module,
                 length_regulator: torch.nn.Module,
                 output_spec_length: int,
                 mel_to_lin_converter: torch.nn.Module,
                 gst_provider: Optional[torch.nn.Module],
                 reference_embedder: Optional[torch.nn.Module]):

        super().__init__()

        self._ac_encoder = torch.jit.script(ac_encoder)
        self._ac_decoder = torch.jit.script(ac_decoder)
        self._duration_predictor = duration_predictor
        self._length_regulator = length_regulator
        self._expected_output_length = output_spec_length
        self._gst_provider = gst_provider
        self._reference_embedder = reference_embedder
        self._mel_to_lin_converter = torch.jit.script(mel_to_lin_converter)

    def forward(self, inputs: Tuple[torch.Tensor, ...]):
        """Runs the full inference pass.

        The data flow depends on the internal module configuration, yet the control-flow allows to
        create a traced TorchScript from the model.

        Args:
            inputs: The input data for the inference. The expected number of elements depends
            on the model's configuration.
        """

        input_phonemes = inputs[0]

        if self._gst_provider is None and self._reference_embedder is None:
            phoneme_representations = self._ac_encoder(input_phonemes, None)

        elif self._reference_embedder is None:
            gst_weights = inputs[1]
            style_embedding = style_embedding_from_weights(self._gst_provider(), gst_weights)
            phoneme_representations = self._ac_encoder(input_phonemes,
                                                       style_embedding)

        else:
            reference_spec = inputs[1]
            style_embedding = self._reference_embedder(reference_spec, self._gst_provider())
            phoneme_representations = self._ac_encoder(input_phonemes,
                                                       style_embedding)

        phoneme_durations = self._duration_predictor(phoneme_representations)

        durations_mask = create_transcript_mask(input_phonemes)
        durations_mask = torch.reshape(durations_mask, (1, -1, 1))

        phoneme_durations = sanitize_predicted_durations(phoneme_durations,
                                                         self._expected_output_length)
        phoneme_durations = phoneme_durations * durations_mask

        stretched_phoneme_repr = self._length_regulator(phoneme_representations,
                                                        phoneme_durations)

        mel_spec = self._ac_decoder(stretched_phoneme_repr)

        return self._mel_to_lin_converter(mel_spec), phoneme_durations
