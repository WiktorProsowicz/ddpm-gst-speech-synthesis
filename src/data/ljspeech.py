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
                 normalize_spectrograms: bool,
                 scale_spectrograms: bool) -> None:
        """Initializes the dataset.

        Args:
            ds_path: Path to the dataset. This path should be organized according to the
                torchaudio.datasets.LJSPEECH generated structure.
            alignments_path: Path to a directory containing output of the Montreal Forced
                Aligner tool. See data.preprocessing.alignments.
            normalize_spectrograms: If True, the spectrograms are normalized to have zero
                mean and unit variance.
            scale_spectrograms: If True, the spectrograms are scaled to [0, 1].
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
        self._scale_spectrograms = scale_spectrograms

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
                transforms.Lambda(lambda x: (x - self._global_spec_mean) / self._global_spec_std)
            ])
        else:
            self._audio_transform = self._create_base_audio_transform()

    def __getitem__(self, idx: int):
        """Returns a single item from the dataset.

        Args:
            idx: Index of the item to return.

        Returns:
            A tuple containing the preprocessed spectrogram, the transcript and the
                phoneme durations. The spectrogram's values are normalized to [0, 1].
                Each individual spectrogram's element has mean 0 and std 1 across the dataset.
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

    def get_spectrogram_stats(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Returns the global mean and standard deviation of the preprocessed spectrograms."""

        if self._global_spec_mean is None or self._global_spec_std is None:
            return None

        return self._global_spec_mean, self._global_spec_std

    def _create_base_audio_transform(self) -> Callable:
        """Composes the basic transform for the audio data.

        The transform does not contain the normalization step.
        """

        if self._scale_spectrograms:
            def scaling_lambda(x):
                return (x - x.min()) / (x.max() - x.min())
        else:
            def scaling_lambda(x):
                return x

        return transforms.Compose([
            audio_transforms.Resample(orig_freq=22050, new_freq=self._sample_rate),
            audio_prep.AudioClippingTransform(self._audio_max_length, self._sample_rate),
            audio_transforms.MelSpectrogram(sample_rate=self._sample_rate,
                                            n_fft=self._fft_window_size,
                                            n_mels=80,
                                            hop_length=self._fft_hop_size),
            audio_transforms.AmplitudeToDB(),
            transforms.Lambda(scaling_lambda)
        ])

    def _get_global_spec_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates the global mean and standard deviation of the spectrograms.

        The stats are created with respect to the output spectrograms, i.e. the
        outputs of the audio transforms.
        """

        spec_mean = None

        transform = self._create_base_audio_transform()

        for audio, _, _, _ in self._dataset:

            audio = transform(audio)

            if spec_mean is None:
                spec_mean = torch.zeros_like(audio)

            spec_mean += audio

        spec_mean /= len(self._dataset)  # type: ignore

        spec_std = None

        for audio, _, _, _ in self._dataset:

            audio = transform(audio)

            if spec_std is None:
                spec_std = torch.zeros_like(audio)

            spec_std += (audio - spec_mean) ** 2

        spec_std /= len(self._dataset)  # type: ignore

        return spec_mean, spec_std


class LJSpeechSpectrogramsDs(torch_data.Dataset):
    """Contains (mel-spectrogram, linear-spectrogram) pairs from the LJSpeech dataset."""

    def __init__(self,
                 raw_ds_path: str,
                 sample_rate: int,
                 fft_window_size: int,
                 fft_hop_size: int,
                 scale_spectrograms: bool):

        super().__init__()

        self._dataset = datasets.LJSPEECH(root=raw_ds_path, download=True)

        metadata_path = os.path.join(raw_ds_path, 'LJSpeech-1.1', 'metadata.csv')
        with open(metadata_path, 'r', encoding='utf-8') as file:
            self._metadata = list(csv.reader(file, delimiter='|', quoting=csv.QUOTE_NONE))

        self._sample_rate = sample_rate
        self._fft_window_size = fft_window_size
        self._fft_hop_size = fft_hop_size
        self._scale_spectrograms = scale_spectrograms

        if self._scale_spectrograms:
            def scaling_lambda(x):
                return (x - x.min()) / (x.max() - x.min())
        else:
            def scaling_lambda(x):
                return x

        self._mel_spec_transform = transforms.Compose([
            audio_transforms.Resample(orig_freq=22050, new_freq=sample_rate),
            audio_transforms.MelSpectrogram(sample_rate=sample_rate,
                                            n_fft=fft_window_size,
                                            n_mels=80,
                                            hop_length=fft_hop_size),
            audio_transforms.AmplitudeToDB(),
            transforms.Lambda(scaling_lambda)
        ])

        self._linear_spec_transform = transforms.Compose([
            audio_transforms.Resample(orig_freq=22050, new_freq=sample_rate),
            audio_transforms.Spectrogram(n_fft=fft_window_size,
                                         hop_length=fft_hop_size),
            audio_transforms.AmplitudeToDB(),
            transforms.Lambda(scaling_lambda)
        ])

    def __getitem__(self, idx: int):
        """Returns a single item from the dataset.

        Args:
            idx: Index of the item to return.

        Returns:
            A tuple containing the (mel-spectrogram, linear-spectrogram) pair.
        """

        audio, _, _, _ = self._dataset[idx]

        return self._mel_spec_transform(audio)[0], self._linear_spec_transform(audio)[0]

    def __len__(self) -> int:
        """Returns the number of items in the dataset."""
        return len(self._dataset)

    def get_sample_id(self, idx: int) -> str:
        """Returns the sample ID for the given index."""
        return self._metadata[idx][0]


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
            'global_spec_mean': global_spec_mean.tolist(),
            'global_spec_std': global_spec_std.tolist()
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


def _split_spectrograms(specs: Tuple[torch.Tensor, torch.Tensor], max_frames: int):
    """Splits the given spectrograms pair into chunks of the given length."""

    mel_spec, linear_spec = specs

    n_chunks = mel_spec.size(1) // max_frames

    for i in range(n_chunks):
        start_idx = i * max_frames
        end_idx = (i + 1) * max_frames

        yield mel_spec[:, start_idx:end_idx], linear_spec[:, start_idx:end_idx]


def serialize_spectrograms_ds(ds: LJSpeechSpectrogramsDs,
                              path: str,
                              max_frames_in_split_spectrogram: Optional[int]) -> None:
    """Serializes the dataset to a file.

    Args:
        ds: The dataset to serialize.
        path: The directory the files shall be saved to.
        max_frames_in_split_spectrogram: If not None, the spectrograms are split into chunks
            of the given length.
    """

    debug_log_interval = 1000

    for sample_idx, sample in enumerate(ds):

        if max_frames_in_split_spectrogram is not None:
            for split_idx, split_sample in enumerate(_split_spectrograms(
                    sample, max_frames_in_split_spectrogram)):

                sample_path = os.path.join(path, f'{ds.get_sample_id(sample_idx)}_{split_idx}.pt')
                torch.save(split_sample, sample_path)
        else:
            sample_path = os.path.join(path, f'{ds.get_sample_id(sample_idx)}.pt')
            torch.save(sample, sample_path)

        if (sample_idx + 1) % debug_log_interval == 0:
            logging.debug('Serialized %d samples.', sample_idx + 1)
