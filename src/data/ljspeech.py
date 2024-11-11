# -*- coding: utf-8 -*-
"""Contains definition of a dataset class for the LJSpeech dataset.

The dataset's details is available at https://keithito.com/LJ-Speech-Dataset/.
"""

from typing import Callable, Tuple
import math
import csv
import os
import json

import torch
from torch.utils import data as torch_data
from torchaudio import datasets  # type: ignore
from torchvision.transforms import transforms
from torchaudio import transforms as audio_transforms

from data.preprocessing import alignments
from data.preprocessing import text
from data.preprocessing import audio as audio_prep


class LJSpeechDataset(torch_data.Dataset):
    """Decodes and preprocesses the LJSpeech dataset."""

    def __init__(self,
                 ds_path: str,
                 alignments_path: str,
                 sample_rate: int,
                 fft_window_size: int,
                 fft_hop_size: int,
                 audio_max_length: float) -> None:
        """Initializes the dataset.

        Args:
            ds_path: Path to the dataset. This path should be organized according to the
                torchaudio.datasets.LJSPEECH generated structure.
            alignments_path: Path to a directory containing output of the Montreal Forced
                Aligner tool. See data.preprocessing.alignments.
        """
        super().__init__()

        self._dataset = datasets.LJSPEECH(root=ds_path, download=True)

        with open(os.path.join(ds_path, "LJSpeech-1.1", "metadata.csv"), "r") as file:
            self._metadata = list(csv.reader(file, delimiter="|", quoting=csv.QUOTE_NONE))

        self._sample_rate = sample_rate
        self._audio_max_length = audio_max_length
        self._fft_window_size = fft_window_size
        self._fft_hop_size = fft_hop_size
        self._alignments = alignments.load_alignments(alignments_path)

        self._max_phonemes = text.get_max_phonemes_for_audio_length(
            self._alignments, audio_max_length)

        self._alignments = {file_id: alignment[:self._max_phonemes]
                            for file_id, alignment in self._alignments.items()}

        waveform_length = int(audio_max_length * sample_rate)
        output_spectrogram_length = math.ceil(waveform_length / fft_hop_size)

        self._alignments_transform = alignments.PhonemeDurationsExtractingTransform(
            self._max_phonemes,
            output_spectrogram_length,
            audio_max_length
        )

        self._text_transform = transforms.Compose([
            text.PadSequenceTransform(self._max_phonemes),
            text.OneHotEncodeTransform(text.ENHANCED_MFA_ARP_VOCAB)
        ])

        self._global_spec_mean, self._global_spec_std = self._get_global_spec_stats()

        self._audio_transform = transforms.Compose([
            self._create_base_audio_transform(),
            transforms.Normalize(self._global_spec_mean, self._global_spec_std)
        ])

    def __getitem__(self, idx: int):
        """Returns a single item from the dataset.

        Args:
            idx: Index of the item to return.

        Returns:
            A tuple containing the waveform and the sample rate.
        """

        audio, _, _, _ = self._dataset[idx]
        audio_file_id, _, _ = self._metadata[idx]

        transcript = text.get_phonemes_from_alignments(
            self._alignments[audio_file_id])

        audio = self._audio_transform(audio)
        transcript = self._text_transform(transcript)

        phoneme_durations = self._alignments_transform(
            self._alignments[audio_file_id])

        return audio[0, :], transcript, phoneme_durations

    def __len__(self) -> int:
        """Returns the number of items in the dataset."""
        return len(self._dataset)

    def get_spectrogram_stats(self) -> Tuple[float, float]:
        """Returns the global mean and standard deviation of the preprocessed spectrograms."""
        return self._global_spec_mean, self._global_spec_std

    def _create_base_audio_transform(self) -> Callable:
        """Composes the basic transform for the audio data.

        The transform does not contain the normalization step.
        """

        return transforms.Compose([
            audio_prep.AudioClippingTransform(self._audio_max_length, self._sample_rate),
            audio_transforms.MelSpectrogram(sample_rate=self._sample_rate,
                                            n_fft=self._fft_window_size,
                                            n_mels=80,
                                            hop_length=self._fft_hop_size),
            audio_transforms.AmplitudeToDB()
        ])

    def _get_global_spec_stats(self) -> Tuple[float, float]:
        """Calculates the global mean and standard deviation of the spectrograms.

        The stats are created with respect to the output spectrograms, i.e. the
        outputs of the audio transforms.
        """

        spec_mean = 0.0
        spec_std = 0.0

        transform = self._create_base_audio_transform()

        for idx in range(len(self._dataset)):
            audio, _, _, _ = self._dataset[idx]

            audio = transform(audio)

            spec_mean += audio.mean()
            spec_std += audio.std()

        spec_mean = spec_mean.item()
        spec_std = spec_std.item()

        spec_mean /= len(self._dataset)
        spec_std /= len(self._dataset)

        return spec_mean, spec_std


def serialize_ds(ds: LJSpeechDataset, path: str) -> None:
    """Serializes the dataset to a file.

    Args:
        ds: The dataset to serialize.
        path: Path to the file to serialize the dataset to.
    """

    global_spec_mean, global_spec_std = ds.get_spectrogram_stats()

    metadata = {
        "global_spec_mean": global_spec_mean,
        "global_spec_std": global_spec_std
    }

    for sample_idx in range(len(ds)):
        sample_path = os.path.join(path, f"sample_{sample_idx:06d}.pt")
        torch.save(ds[sample_idx], sample_path)

    with open(os.path.join(path, "metadata.json"), "w") as file:
        json.dump(metadata, file)
