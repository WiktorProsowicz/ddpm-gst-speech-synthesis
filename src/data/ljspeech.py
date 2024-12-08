# -*- coding: utf-8 -*-
"""Contains definition of a dataset class for the LJSpeech dataset.

The dataset's details is available at https://keithito.com/LJ-Speech-Dataset/.
"""
import csv
import json
import logging
import math
import os
from typing import Callable
from typing import Optional
from typing import Tuple

import torch
from torch.utils import data as torch_data
from torchaudio import datasets  # type: ignore
from torchaudio import transforms as audio_transforms
from torchvision.transforms import transforms

from data.preprocessing import alignments
from data.preprocessing import audio as audio_prep
from data.preprocessing import text


class LJSpeechDataset(torch_data.Dataset):
    """Decodes and preprocesses the LJSpeech dataset."""

    def __init__(self,
                 ds_path: str,
                 alignments_path: str,
                 sample_rate: int,
                 fft_window_size: int,
                 fft_hop_size: int,
                 audio_max_length: float,
                 normalize_spectrograms: bool) -> None:
        """Initializes the dataset.

        Args:
            ds_path: Path to the dataset. This path should be organized according to the
                torchaudio.datasets.LJSPEECH generated structure.
            alignments_path: Path to a directory containing output of the Montreal Forced
                Aligner tool. See data.preprocessing.alignments.
        """
        super().__init__()

        self._dataset = datasets.LJSPEECH(root=ds_path, download=True)

        metadata_path = os.path.join(ds_path, 'LJSpeech-1.1', 'metadata.csv')
        with open(metadata_path, 'r', encoding='utf-8') as file:
            self._metadata = list(csv.reader(file, delimiter='|', quoting=csv.QUOTE_NONE))

        self._sample_rate = sample_rate
        self._audio_max_length = audio_max_length
        self._fft_window_size = fft_window_size
        self._fft_hop_size = fft_hop_size

        logging.debug('Loading alignments...')
        self._alignments = alignments.load_alignments(alignments_path)

        logging.debug('Selecting phonemes up to the given audio length...')
        n_phonemes_for_alignments = {
            file_id: text.get_n_phonemes_for_audio_length(alignment, audio_max_length)
            for file_id, alignment
            in self._alignments.items()}

        self._phonemes_sequence_length = max(n_phonemes_for_alignments.values())

        self._alignments = {file_id: alignment[:n_phonemes_for_alignments[file_id]]
                            for file_id, alignment in self._alignments.items()}

        waveform_length = int(audio_max_length * sample_rate)
        output_spectrogram_length = math.ceil(waveform_length / fft_hop_size)

        self._alignments_transform = alignments.PhonemeDurationsExtractingTransform(
            self._phonemes_sequence_length,
            output_spectrogram_length,
            audio_max_length
        )

        self._text_transform = transforms.Compose([
            text.PadSequenceTransform(self._phonemes_sequence_length),
            text.OneHotEncodeTransform(text.ENHANCED_MFA_ARP_VOCAB)
        ])

        self._global_spec_mean, self._global_spec_std = None, None

        if normalize_spectrograms:
            logging.debug('Calculating global spectrogram mean and stddev...')
            self._global_spec_mean, self._global_spec_std = self._get_global_spec_stats()

            self._audio_transform = transforms.Compose([
                self._create_base_audio_transform(),
                transforms.Normalize(self._global_spec_mean, self._global_spec_std)
            ])
        else:
            self._audio_transform = self._create_base_audio_transform()

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

    def get_sample_id(self, idx: int) -> str:
        """Returns the sample ID for the given index."""
        return self._metadata[idx][0]

    def get_spectrogram_stats(self) -> Optional[Tuple[float, float]]:
        """Returns the global mean and standard deviation of the preprocessed spectrograms."""

        if self._global_spec_mean is None or self._global_spec_std is None:
            return None

        return self._global_spec_mean, self._global_spec_std

    def _create_base_audio_transform(self) -> Callable:
        """Composes the basic transform for the audio data.

        The transform does not contain the normalization step.
        """

        return transforms.Compose([
            audio_transforms.Resample(orig_freq=22050, new_freq=self._sample_rate),
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

        spec_mean = torch.tensor(0.0)
        spec_std = torch.tensor(0.0)

        transform = self._create_base_audio_transform()

        for audio, _, _, _ in self._dataset:

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

    debug_log_interval = 1000

    global_spec = ds.get_spectrogram_stats()

    if global_spec is not None:

        global_spec_mean, global_spec_std = global_spec

        metadata = {
            'global_spec_mean': global_spec_mean,
            'global_spec_std': global_spec_std
        }

    else:
        metadata = {}

    for sample_idx, sample in enumerate(ds):
        sample_path = os.path.join(path, f'{ds.get_sample_id(sample_idx)}.pt')
        torch.save(sample, sample_path)

        if (sample_idx + 1) % debug_log_interval == 0:
            logging.debug('Serialized %d samples.', sample_idx + 1)

    with open(os.path.join(path, 'metadata.json'), 'w', encoding='utf-8') as file:
        json.dump(metadata, file)
