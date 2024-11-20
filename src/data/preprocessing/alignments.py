# -*- coding: utf-8 -*-
"""Contains functions for downloading and preprocessing forced phoneme alignments.

The alignments come from the Montreal Forced Aligner tool:
https://montreal-forced-aligner.readthedocs.io/en/latest/index.html

It is expected the tool's output is a directory containing files in
TextGrid format. The names should match the audio files in the dataset.
"""
import os
from typing import Dict
from typing import List
import subprocess
import logging

import numpy as np
import textgrid
import torch

import gdown


GDRIVE_ALIGNMENTS_URL = 'https://drive.google.com/uc?id=1d9A6K1qgwUCR4shci_RvTe2eeDHqq6au'


def download_phoneme_alignments(destination_path: str):
    """Downloads the phoneme alignments for the LJSpeech dataset.

    See data.preprocessing.alignments for more details.

    Args:
        destination_path: Path to the directory where the alignments should be saved.
    """

    arch_path = os.path.join(destination_path, 'ljspeech_1.1_alignments.tar.bz2')

    gdown.download(GDRIVE_ALIGNMENTS_URL, arch_path, quiet=False)

    try:
        subprocess.run(['bzip2', '-d', arch_path], check=True)

        tar_path = arch_path[:-4]

        subprocess.run(['tar', '-xf', tar_path, '-C', destination_path], check=True)

        subprocess.run(['rm', tar_path], check=True)

    except subprocess.CalledProcessError as proc_err:
        logging.critical('Failed to download the phoneme alignments: %s', proc_err)


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

    def forward(self, alignments: List[textgrid.Interval]) -> torch.Tensor:
        """Transforms the forced phoneme alignments into phoneme durations.

        Args:
            alignments: List of individual phoneme alignments.

        Returns:
            List of of phoneme durations. The output sequence is padded to
            the configured expected length with zeros.
        """

        durations = np.zeros(self._output_length)

        for alignment_idx, alignment in enumerate(alignments):

            frames_up_now = np.floor(alignment.maxTime * self._output_frames_per_second)

            duration = frames_up_now - np.sum(durations)

            durations[alignment_idx] = duration if duration > 1 else 1.

        durations[:len(alignments)] = np.log2(durations[:len(alignments)])
        durations[len(alignments):] = 0.

        return torch.Tensor(durations)
