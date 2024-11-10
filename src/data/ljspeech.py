# -*- coding: utf-8 -*-
"""Contains definition of a dataset class for the LJSpeech dataset.

The dataset's details is available at https://keithito.com/LJ-Speech-Dataset/.
"""

from typing import Optional, Callable
import math
import csv
import os

from torch.utils import data as torch_data
from torchaudio import datasets  # type: ignore

from data.preprocessing import alignments


class LJSpeechDataset(torch_data.Dataset):
    """Decodes and preprocesses the LJSpeech dataset."""

    def __init__(self,
                 ds_path: str,
                 alignments_path: str,
                 text_transform: Optional[Callable],
                 audio_transform: Optional[Callable],
                 sample_rate: int = 22050,
                 fft_window_size: int = 1024,
                 fft_hop_size: int = 256,
                 audio_max_length: float = 6.0) -> None:
        """Initializes the dataset.

        Args:
            ds_path: Path to the dataset. This path should be organized according to the
                torchaudio.datasets.LJSPEECH generated structure.
            alignments_path: Path to a directory containing output of the Montreal Forced
                Aligner tool. See data.preprocessing.alignments.
            text_transform: A pipeline of transformations to apply to the textual data.
            audio_transform: A pipeline of transformations to apply to the audio data.
        """
        super().__init__()

        self._dataset = datasets.LJSPEECH(root=ds_path, download=True)

        with open(os.path.join(ds_path, "metadata.csv"), "r") as file:
            self._metadata = list(csv.reader(file))

        self._sample_rate = sample_rate
        self._audio_max_length = audio_max_length
        self._fft_window_size = fft_window_size
        self._fft_hop_size = fft_hop_size
        self._alignments = alignments.load_alignments(alignments_path)

        waveform_length = int(audio_max_length * sample_rate)
        output_spectrogram_length = math.floor(
            (waveform_length - fft_window_size) / fft_hop_size) + 1

        self._alignments_transform = alignments.PhonemeDurationsExtractingTransform(
            output_spectrogram_length,
            sample_rate,
            fft_window_size,
            fft_hop_size
        )

        self._text_transform = text_transform
        self._audio_transform = audio_transform

    def __getitem__(self, idx: int):
        """Returns a single item from the dataset.

        Args:
            idx: Index of the item to return.

        Returns:
            A tuple containing the waveform and the sample rate.
        """

        audio, sample_rate, _, transcript = self._dataset[idx]
        audio_file_id, _, _ = self._metadata[idx]

        if self._audio_transform:
            audio = self._audio_transform(audio)

        if self._text_transform:
            transcript = self._text_transform(transcript)

        phoneme_durations = self._alignments_transform(
            self._alignments[audio_file_id])

        return audio, transcript, phoneme_durations

    def __len__(self) -> int:
        """Returns the number of items in the dataset."""
        return len(self._dataset)


def serialize_ds(ds: LJSpeechDataset, path: str) -> None:
    """Serializes the dataset to a file.

    Args:
        ds: The dataset to serialize.
        path: Path to the file to serialize the dataset to.
    """

    for sample_idx in range(len(ds)):
        audio, transcript = ds[sample_idx]

        with open(f"{path}/sample_{sample_idx}.txt", "w") as file:
            file.write(transcript)

        # Save the audio to a file
        pass
