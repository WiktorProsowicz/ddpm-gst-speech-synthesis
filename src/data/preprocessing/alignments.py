# -*- coding: utf-8 -*-
"""Contains functions for preprocessing forced phoneme alignments.

The alignments come from the Montreal Forced Aligner tool:
https://montreal-forced-aligner.readthedocs.io/en/latest/index.html

It is expected the tool's output is a directory containing files in
TextGrid format. The names should match the audio files in the dataset.
"""

from typing import List, Dict
import os
import math

import textgrid
import torch


def load_alignments(alignments_path: str) -> Dict[str, List[textgrid.Interval]]:
    """Loads the alignments from the specified path.

    Args:
        alignments_path: Path to the directory containing the alignments.

    Returns:
        Dictionary where keys are the audio file names and values are lists of
        individual phoneme alignments.
    """

    textgrids_path = os.path.join(alignments_path, 'textgrids')
    alignments_dict = {}

    phone_mark_replacing = {
        'spn': '<unk>',
        '': '<sil>'
    }

    for alignment_file_name in os.listdir(textgrids_path):

        file_path = os.path.join(textgrids_path, alignment_file_name)
        text_grid = textgrid.TextGrid.fromFile(file_path)

        dict_key = alignment_file_name.split('.')[0]
        phones = text_grid.tiers[1]

        for phone in phones:
            if phone.mark in phone_mark_replacing:
                phone.mark = phone_mark_replacing[phone.mark]

        alignments_dict[dict_key] = phones

    return alignments_dict


class PhonemeDurationsExtractingTransform(torch.nn.Module):
    """Extracts phoneme durations from forced alignments.

    The output durations are in logarithmic scale.
    """

    def __init__(self, output_length: int, output_spectrogram_length: int,
                 output_audio_length: float):
        """Initializes the transform.

        Args:
            output_length: Expected length of the output sequence.
            output_spectrogram_length: Length of the spectrogram in frames. This value should be
                equal to the maximal sum of the durations of the phonemes.
            output_audio_length: Duration of the audio in seconds. This value accounts for the
                number of frames in the output spectrogram with the given sample rate.
        """

        super().__init__()

        self._output_length = output_length
        self._output_frames_per_second = output_spectrogram_length / output_audio_length
        self._min_log_duration = .01

    def forward(self, alignments: List[textgrid.Interval]) -> torch.Tensor:
        """Transforms the forced phoneme alignments into phoneme durations.

        Args:
            alignments: List of individual phoneme alignments.

        Returns:
            List of of phoneme durations. The output sequence is padded to
            the configured expected length with zeros.
        """

        durations = []

        for alignment in alignments:

            length = alignment.maxTime - alignment.minTime
            duration = round(length * self._output_frames_per_second)

            if duration > 1:
                durations.append(math.log2(duration))

            else:
                durations.append(self._min_log_duration)

        if len(durations) < self._output_length:
            durations += ([0] * (self._output_length - len(durations)))

        return torch.Tensor(durations[:self._output_length])