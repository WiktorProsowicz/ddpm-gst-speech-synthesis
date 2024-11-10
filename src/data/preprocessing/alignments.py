# -*- coding: utf-8 -*-
"""Contains functions for preprocessing forced phoneme alignments.

The alignments come from the Montreal Forced Aligner tool:
https://montreal-forced-aligner.readthedocs.io/en/latest/index.html

It is expected the tool's output is a directory containing files in
TextGrid format. The names should match the audio files in the dataset.
"""

from dataclasses import dataclass
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

    def __init__(self, output_length: int,
                 sample_rate: int,
                 window_size: int,
                 hop_size: int):

        super().__init__()

        self._output_length = output_length
        self._bin_size = ((sample_rate - window_size) // hop_size) + 1

    def forward(self, alignments: List[textgrid.Interval]) -> List[float]:
        """Transforms the forced phoneme alignments into phoneme durations.

        Args:
            alignments: List of individual phoneme alignments.

        Returns:
            List of of phoneme durations. The output sequence is padded to
            the configured expected length with zeros.
        """

        durations = []

        for alignment in alignments:

            length = alignment.minTime
