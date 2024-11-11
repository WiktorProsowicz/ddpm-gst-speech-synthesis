# -*- coding: utf-8 -*-
"""Contains functions performing textual data preprocessing."""

from typing import Dict, List

import torch
import g2p_en
import nltk
import textgrid

ENHANCED_MFA_ARP_VOCAB = ['<pad>', '<unk>', '<sil>'] + ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
                                                        'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH',
                                                        'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
                                                        'EY2', 'F', 'G', 'HH',
                                                        'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L',
                                                        'M', 'N', 'NG', 'OW0', 'OW1',
                                                        'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH',
                                                        'UH0', 'UH1', 'UH2', 'UW',
                                                        'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']


def get_max_phonemes_for_audio_length(
        phoneme_alignments: Dict[str, textgrid.IntervalTier], audio_length: float):
    """Calculate the maximal number of phonemes accounting for the audio length.

    Args:
        phoneme_alignments: Phoneme alignments with their file ids.
        audio_length: Length of the audio in seconds.
    """

    max_phonemes = 0

    for alignment in phoneme_alignments.values():

        n_phonemes = len([interval for interval in alignment if interval.maxTime < audio_length])
        max_phonemes = max(max_phonemes, n_phonemes)

    return max_phonemes


def get_phonemes_from_alignments(phoneme_alignments: textgrid.IntervalTier):
    """Extracts the phonemes from the forced alignments.

    Args:
        phoneme_alignments: Forced phoneme alignments.

    Returns:
        List of phonemes.
    """

    return [interval.mark for interval in phoneme_alignments]


class PadSequenceTransform(torch.nn.Module):
    """Pads the input sequence to the desired length."""

    def __init__(self, output_length: int):

        super().__init__()

        self._output_length = output_length

    def forward(self, sequence: List[str]):

        if len(sequence) < self._output_length:
            return sequence + (['<pad>'] * (self._output_length - len(sequence)))

        return sequence


class OneHotEncodeTransform(torch.nn.Module):
    """Converts a sequence of tokens to a one-hot encoded tensor."""

    def __init__(self, vocabulary: List[str], padding_token: str = '<pad>'):

        super().__init__()

        self._vocabulary = vocabulary
        self._padding_token = padding_token

    def forward(self, sequence: List[str]) -> torch.Tensor:

        one_hot = torch.zeros(len(sequence), len(self._vocabulary))

        for i, token in enumerate(sequence):
            if token != self._padding_token:
                one_hot[i, self._vocabulary.index(token)] = 1

        return one_hot
