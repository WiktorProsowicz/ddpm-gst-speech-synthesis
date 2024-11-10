# -*- coding: utf-8 -*-
"""Contains functions performing textual data preprocessing."""

import torch
import g2p_en
import nltk


class GraphemeToPhonemeTransform(torch.nn.Module):
    """Converts a grapheme sequence to a tokenized phoneme sequence."""

    def __init__(self):

        super().__init__()

        # Download the required NLTK resources
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger_eng.zip')

        except LookupError:
            nltk.download('averaged_perceptron_tagger_eng')

        self._g2p = g2p_en.G2p()

    def forward(self, text: str):
        return self._g2p(text)
