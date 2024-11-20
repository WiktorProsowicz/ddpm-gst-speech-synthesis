# -*- coding: utf-8 -*-
"""Contains utilities used for visualization purposes."""
from typing import List

import matplotlib as mlp
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import torch

from utilities import inference


def colorize_spectrogram(spectrogram: torch.Tensor, colormap: str) -> torch.Tensor:
    """Converts a greyscale spectrogram into an RGB image using a colormap.

    Args:
        spectrogram: A 2D tensor of shape (n_freq_bins, n_time_bins) containing the spectrogram.
        colormap: The name of the colormap to use. Must be a valid matplotlib colormap.

    Returns:
        A 3D tensor of shape (3, n_freq_bins, n_time_bins) containing the coolorized spectrogram.
    """

    spectrogram = spectrogram.cpu().numpy()
    spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())

    cmap = mlp.colormaps[colormap]

    colorized_spectrogram = cmap(spectrogram)[:, :, :3]

    return torch.from_numpy(colorized_spectrogram).permute(2, 0, 1)


def decode_transcript(transcript: torch.Tensor, vocab: List[str]) -> List[str]:
    """Decodes the encoded transcript into a phoneme tokens.

    The encoding may be either one-hot or argmax.
    """

    word_indices = transcript.argmax(dim=1)

    return [vocab[i] for i in word_indices][:inference.get_transcript_length(transcript)]


def annotate_spectrogram_with_phoneme_durations(spectrogram: np.ndarray,
                                                durations: np.ndarray) -> matplotlib.figure.Figure:
    """Annotates a spectrogram with phoneme boundaries.

    Args:
        spectrogram: A 2D tensor of shape (n_freq_bins, n_time_bins) containing the spectrogram.
        durations: A list of log-scale durations of each input phoneme.
    """

    frame_boundaries = np.cumsum(durations)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')

    for i, _ in enumerate(durations):

        if i > 0:
            ax.axvline(frame_boundaries[i - 1], color='red', linestyle='--')

    ax.set_title('Spectrogram with Phoneme Boundaries')
    ax.set_xlabel('Time bin index')
    ax.set_ylabel('Frequency bin index')

    return fig
