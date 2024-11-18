# -*- coding: utf-8 -*-
"""Contains the length regulator layer for the DDPM-GST-Speech-Gen model."""
import numpy as np
import torch


def _create_alignment_matrix(log_durations: torch.Tensor, max_length: int) -> torch.Tensor:
    """Creates a matrix for stretching the input based on the predicted phoneme durations."""

    log_durations = log_durations.numpy()
    batch_size, n_phonemes, _ = log_durations.shape

    durations_mask = (log_durations > 0).astype(np.uint8)
    durations = (np.power(2.0, log_durations) + 1e-4).astype(np.uint8) * durations_mask
    durations = durations.reshape(batch_size, n_phonemes)

    indexes_space = np.arange(n_phonemes).astype(np.uint16)
    alignment_matrix = np.zeros((batch_size, max_length, n_phonemes), dtype=np.uint8)

    for i in range(batch_size):
        repeated_indexes = np.repeat(indexes_space, durations[i])
        alignment_matrix[i, np.arange(len(repeated_indexes)), repeated_indexes] = 1

    return torch.from_numpy(alignment_matrix)


class LengthRegulator(torch.nn.Module):
    """Stretches the encoder's output based on the predicted phoneme durations.

    The regulator is intended to alleviate the problem of the gap between the
    length the phoneme set and the expected spectrogram. Explicit duration
    prediction is an alternative way to guide the decoder to make the generated
    spectrogram frames in line with the phonemes they correspond to. It is a way
    to omit the Soft Attention Collapse problem. Besides, it allows parallelization.
    """

    def __init__(self, output_length: int):
        """Initializes the length regulator."""

        super().__init__()

        self._output_length = output_length

    def forward(self, encoder_output: torch.Tensor, log_durations: torch.Tensor) -> torch.Tensor:
        """Stretches the encoder's output based on the predicted phoneme durations.

        Args:
            encoder_output: The output of the encoder. The tensor is expected to have the shape
                (batch_size, n_phonemes, encoder_output_channels).
            log_durations: The predicted phoneme durations. The tensor is expected to have the shape
                (batch_size, n_phonemes, 1).

        Returns:
            The stretched encoder's output.
        """

        alignment_matrix = _create_alignment_matrix(log_durations, self._output_length)
        return torch.matmul(alignment_matrix.float(), encoder_output)
