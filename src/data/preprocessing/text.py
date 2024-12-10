# -*- coding: utf-8 -*-
"""Contains functions performing textual data preprocessing."""
from typing import List

import g2p_en
import nltk
import textgrid
import torch

_ENHANCED_MFA_ARP_PHO = ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2',
                         'AH0', 'AH1', 'AH2', 'AO0', 'AO1', 'AO2',
                         'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2',
                         'B', 'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2',
                         'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2',
                         'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0',
                         'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG',
                         'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2', 'P',
                         'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2',
                         'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']

ENHANCED_MFA_ARP_VOCAB = ['<pad>', '<unk>', '<sil>'] + _ENHANCED_MFA_ARP_PHO


def get_n_phonemes_for_audio_length(
        phoneme_alignments: textgrid.IntervalTier, audio_length: float):
    """Calculate the number of phonemes accounting for the audio length.

    Args:
        phoneme_alignments: MFA phoneme alignments.
        audio_length: Length of the audio in seconds.
    """

    return len(
        [interval for interval in phoneme_alignments if interval.maxTime < audio_length])


def get_phonemes_from_alignments(phoneme_alignments: textgrid.IntervalTier):
    """Extracts the phonemes from the forced alignments.

    Args:
        phoneme_alignments: Forced phoneme alignments.

    Returns:
        List of phonemes.
    """

    return [interval.mark for interval in phoneme_alignments]


class G2PTransform(torch.nn.Module):
    """Converts the input text into a list of phonemes."""

    def __init__(self):
        super().__init__()

        nltk.download('averaged_perceptron_tagger_eng', quiet=True)

        self._conv = g2p_en.G2p()

        self._tokens_to_remove = [' ']
        self._tokens_to_replace = {
            '.': '<sil>',
            ',': '<sil>',
            '?': '<sil>',
            '!': '<sil>',
        }

    def forward(self, text: str) -> List[str]:

        phonemes = self._conv(text)

        phonemes = [phoneme for phoneme in phonemes if phoneme not in self._tokens_to_remove]
        phonemes = list(map(self._replace_token_if_unhandled, phonemes))

        return phonemes

    def _replace_token_if_unhandled(self, token: str) -> str:

        if token in self._tokens_to_replace:
            return self._tokens_to_replace[token]

        return token


class PadSequenceTransform(torch.nn.Module):
    """Pads the input sequence to the desired length."""

    def __init__(self, output_length: int):

        super().__init__()

        self._output_length = output_length

    def forward(self, sequence: List[str]):

        if len(sequence) < self._output_length:
            return sequence + (['<pad>'] * (self._output_length - len(sequence)))

        return sequence[:self._output_length]


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
