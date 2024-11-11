# -*- coding: utf-8 -*-
"""Contains functions performing audio data preprocessing."""

import torch


class AudioClippingTransform(torch.nn.Module):
    """Clips the audio waveform to the desired duration."""

    def __init__(self, max_duration, sample_rate):

        super().__init__()
        self.max_samples = int(max_duration * sample_rate)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:

        waveform = waveform[:, :self.max_samples]

        if waveform.size(1) < self.max_samples:
            waveform = torch.cat(
                (waveform, torch.zeros(1, self.max_samples - waveform.size(1))),
                dim=1)

        return waveform
