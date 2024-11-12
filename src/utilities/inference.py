# -*- coding: utf-8 -*-
"""Contains utilities for running inference with the trained model."""
from typing import List

import torch


def decode_transcript(transcript: torch.Tensor, vocab: List[str]) -> List[str]:
    """Decodes the encoded transcript into a phoneme tokens.

    The encoding may be either one-hot or argmax.
    """

    word_indices = transcript.argmax(dim=1)

    return [vocab[i] for i in word_indices]
