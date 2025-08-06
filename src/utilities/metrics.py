# -*- coding: utf-8 -*-
"""Contains metrics used to evaluate the model's performance."""
from typing import Optional

import fastdtw
import librosa
import numpy as np
import scipy
import torch
from scipy.spatial.distance import euclidean


@torch.no_grad()
def mean_absolute_error(y_true: torch.Tensor,
                        y_pred: torch.Tensor,
                        mask: Optional[torch.Tensor] = None,
                        mask_sum: Optional[torch.Tensor] = None):
    """Calculates mean absolute error.

    Args:
        y_true: True values.
        y_pred: Predicted values.
        mask: Loss mask for ignoring some of the computed errors.
        mask_sum: Sum of the mask values.
    """

    if mask is not None:
        return torch.sum(torch.abs(y_true - y_pred) * mask) / mask_sum

    return torch.mean(torch.abs(y_true - y_pred))


def _get_aligned_f0_contours(waveform_true: np.ndarray,
                             waveform_pred: np.ndarray):

    f0_true, _, _ = librosa.pyin(waveform_true,
                                 fmin=librosa.note_to_hz('C1'),
                                 fmax=librosa.note_to_hz('C8'))
    f0_gen, _, _ = librosa.pyin(waveform_pred,
                                fmin=librosa.note_to_hz('C1'),
                                fmax=librosa.note_to_hz('C8'))

    f0_true[np.isnan(f0_true)] = 0.0
    f0_gen[np.isnan(f0_gen)] = 0.0

    f0_gen = f0_gen.reshape(-1, 1)
    f0_true = f0_true.reshape(-1, 1)

    _, path = fastdtw.fastdtw(f0_gen, f0_true, dist=euclidean)

    path = np.array(path)
    index1 = path[:, 0]
    index2 = path[:, 1]

    aligned_true = f0_true[index2]
    aligned_gen = f0_gen[index1]

    return aligned_true, aligned_gen


def f0_rmse(waveform_true: np.ndarray,
            waveform_pred: np.ndarray):
    """Calculates RMSE between F0 contours of two waveforms."""

    aligned_true, aligned_gen = _get_aligned_f0_contours(waveform_true, waveform_pred)

    return np.sqrt(np.mean((aligned_true - aligned_gen) ** 2))


def f0_pearson_corr(waveform_true: np.ndarray,
                    waveform_pred: np.ndarray):
    """Calculates Pearson correlation between the F0 contours of two waveforms."""

    aligned_true, aligned_gen = _get_aligned_f0_contours(waveform_true, waveform_pred)

    return scipy.stats.pearsonr(aligned_true, aligned_gen).statistic


def f0_pearson_corr_and_rmse(waveform_true: np.ndarray,
                             waveform_pred: np.ndarray):
    """Calculates Pearson correlation and RMSE of F0 in one run to speed up the computation."""

    aligned_true, aligned_gen = _get_aligned_f0_contours(waveform_true, waveform_pred)

    return (scipy.stats.pearsonr(aligned_true, aligned_gen).statistic,
            np.sqrt(np.mean((aligned_true - aligned_gen) ** 2)))
