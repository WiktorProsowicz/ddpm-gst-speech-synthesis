# -*- coding: utf-8 -*-
"""Contains utilities for running inference with the trained model."""
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pytorch_pretrained_bert as bert_lib
import torch
from torchvision import transforms

import layers.acoustic.decoder
import layers.acoustic.encoder
import layers.shared.duration_predictor
import layers.shared.length_regulator


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

    input_phonemes = text_transforms.transforms[1](input_phonemes).to(device)

    averaged_bert_embeddings = torch.repeat_interleave(averaged_bert_embeddings,
                                                       phonemes_counts,
                                                       dim=0)

    assert input_phonemes.shape[0] == averaged_bert_embeddings.shape[0]

    return averaged_bert_embeddings, input_phonemes


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
    durations = (torch.pow(2.0, log_durations) + 1e-4).to(torch.int64) * durations_mask
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


class InferenceAcousticEnc(torch.nn.Module):
    """Contains the acoustic encoder required for inference.

    The module performs the encoding of the input phonemes and returns their enriched
    representations. The module does not use any style information.

    The model is convertible to a TorchScript.
    """

    def __init__(self, acoustic_encoder: layers.acoustic.encoder.Encoder):
        super().__init__()

        self._acoustic_encoder = acoustic_encoder

    def forward(self, input_phonemes):
        """Runs the acoustic encoder.

        Args:
            input_phonemes: The one-hot encoded phonemes.
        """

        mask = torch.logical_not(create_transcript_mask(input_phonemes))

        return self._acoustic_encoder.run_basic_blocks(input_phonemes, mask)


class InferenceVarianceReg(torch.nn.Module):
    """Contains the variance regulator required for inference."""

    def __init__(self, acoustic_encoder: layers.acoustic.encoder.Encoder):
        super().__init__()

        self._acoustic_encoder = acoustic_encoder

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]):
        """Runs the variance regulator.

        Args:
            inputs: The phoneme representations and the style embedding.
        """

        phoneme_representations, style_embedding = inputs

        return self._acoustic_encoder.apply_gst_conditioning(phoneme_representations,
                                                             style_embedding)


class InferenceAcousticDec(torch.nn.Module):
    """Contains the acoustic decoder required fot the inference.

    The module performs explicit duration prediction, stretches the input and runs the acoustic
    decoder.

    The model is convertible to a TorchScript.
    """

    def __init__(self,
                 ac_decoder: layers.acoustic.decoder.Decoder,
                 duration_predictor: layers.shared.duration_predictor.DurationPredictor,
                 length_regulator: layers.shared.length_regulator.LengthRegulator,
                 output_spec_length: int):

        super().__init__()

        self._ac_decoder = ac_decoder
        self._duration_predictor = duration_predictor
        self._length_regulator = length_regulator
        self._expected_output_length = output_spec_length

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]):
        """Runs the acoustic decoder.

        Args:
            inputs: The input phoneme representations, transcript mask and the style embedding,
            if supported.
        """

        phoneme_representations, transcript_mask = inputs
        transcript_mask = torch.reshape(transcript_mask, (1, -1, 1))

        log_durations = self._duration_predictor(phoneme_representations)

        phoneme_durations = sanitize_predicted_durations(log_durations,
                                                         self._expected_output_length)
        phoneme_durations = phoneme_durations * transcript_mask

        stretched_phoneme_repr = self._length_regulator(phoneme_representations,
                                                        phoneme_durations)

        decoder_mask = create_mask_from_durations(phoneme_durations.reshape(1, -1),
                                                  self._expected_output_length)
        decoder_mask = torch.logical_not(decoder_mask)

        mel_spec = self._ac_decoder(stretched_phoneme_repr, decoder_mask)

        return mel_spec, phoneme_durations
