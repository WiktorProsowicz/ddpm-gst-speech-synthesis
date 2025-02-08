# -*- coding: utf-8 -*-
"""Contains utilities for running inference with the trained model."""
from typing import Optional
from typing import Tuple

import torch


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
    """Creates a mask for the spectrogram based on the actual length.

    Args:
        spectrogram: The spectrogram without the batch_size dimension.
    """

    if len(spectrogram.shape) == 2:
        return torch.sum(spectrogram == torch.min(spectrogram), dim=0) != spectrogram.shape[0]

    return torch.sum(spectrogram == torch.min(spectrogram), dim=1) != spectrogram.shape[1]


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


class InferenceModel(torch.nn.Module):
    """Contains all the components of the model required for inference.

    The model is convertible to a TorchScript.
    """

    def __init__(self,
                 ac_encoder: torch.nn.Module,
                 ac_decoder: torch.nn.Module,
                 duration_predictor: torch.nn.Module,
                 length_regulator: torch.nn.Module,
                 output_spec_length: int):

        super().__init__()

        self._ac_encoder = torch.jit.script(ac_encoder)
        self._ac_decoder = torch.jit.script(ac_decoder)
        self._duration_predictor = duration_predictor
        self._length_regulator = length_regulator
        self._expected_output_length = output_spec_length

    def forward(self, inputs: Tuple[torch.Tensor, ...]):
        """Runs the full inference pass.

        The data flow depends on the internal module configuration, yet the control-flow allows to
        create a traced TorchScript from the model.

        Args:
            inputs: The input data for the inference. The expected number of elements depends
            on the model's configuration.
        """

        input_phonemes = inputs[0]
        style_embedding = None if len(inputs) == 1 else inputs[1]

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

        return mel_spec, phoneme_durations
