# -*- coding: utf-8 -*-
"""Contains definition of a dataset class for the LJSpeech dataset.

The dataset's details is available at https://keithito.com/LJ-Speech-Dataset/.
"""
import csv
import json
import logging
import os
from typing import Dict

import torch
from torch.utils import data as torch_data
from torchaudio import datasets  # type: ignore
from torchaudio.prototype.pipelines import HIFIGAN_VOCODER_V3_LJSPEECH as hifigan_bundle
from torchvision.transforms import transforms

from data.preprocessing import alignments
from data.preprocessing import audio as audio_prep
from data.preprocessing import text
from utilities import inference


class LJSpeechDataset(torch_data.Dataset):
    """Decodes and preprocesses the LJSpeech dataset."""

    def __init__(self,
                 ds_path: str,
                 alignments_path: str,
                 audio_max_length: float) -> None:
        """Initializes the dataset.

        Args:
            ds_path: Path to the dataset. This path should be organized according to the
                torchaudio.datasets.LJSPEECH generated structure.
            alignments_path: Path to a directory containing output of the Montreal Forced
                Aligner tool. See data.preprocessing.alignments.
            scale_spectrograms: If True, the spectrograms are scaled to [0, 1].
        """
        super().__init__()

        self._dataset = datasets.LJSPEECH(root=ds_path, download=True)

        metadata_path = os.path.join(ds_path, 'LJSpeech-1.1', 'metadata.csv')
        with open(metadata_path, 'r', encoding='utf-8') as file:
            self._metadata = list(csv.reader(
                file, delimiter='|', quoting=csv.QUOTE_NONE))

        logging.debug('Loading alignments...')
        self._alignments = alignments.load_alignments(alignments_path)

        logging.debug('Selecting phonemes up to the given audio length...')
        n_phonemes_for_alignments = {
            file_id: text.get_n_phonemes_for_audio_length(
                alignment, audio_max_length)
            for file_id, alignment
            in self._alignments.items()}

        self._phonemes_sequence_length = max(
            n_phonemes_for_alignments.values())

        self._alignments = {file_id: alignment[:n_phonemes_for_alignments[file_id]]
                            for file_id, alignment in self._alignments.items()}

        fft_hop_size = 256
        sample_rate = 22050

        waveform_length = int(audio_max_length * sample_rate)
        self._output_spectrogram_length = int(waveform_length / fft_hop_size)

        self._alignments_transform = alignments.PhonemeDurationsExtractingTransform(
            self._phonemes_sequence_length,
            self._output_spectrogram_length,
            audio_max_length
        )

        self._text_transform = transforms.Compose([
            text.PadSequenceTransform(self._phonemes_sequence_length),
            text.OneHotEncodeTransform(text.ENHANCED_MFA_ARP_VOCAB)
        ])

        self._audio_transform = transforms.Compose([
            audio_prep.AudioClippingTransform(audio_max_length, sample_rate),
            hifigan_bundle.get_mel_transform()
        ])

    def __getitem__(self, idx: int):
        """Returns a single item from the dataset.

        Args:
            idx: Index of the item to return.

        Returns:
            A tuple containing the preprocessed spectrogram, transcript, phoneme durations and
            attention masks for the phonemes and spectrogram.
        """

        audio, _, _, _ = self._dataset[idx]
        audio_file_id, _, _ = self._metadata[idx]

        transcript = text.get_phonemes_from_alignments(
            self._alignments[audio_file_id])

        audio = self._audio_transform(audio)[0, :]
        transcript = self._text_transform(transcript)

        phoneme_durations = self._alignments_transform(
            self._alignments[audio_file_id])

        phoneme_mask = inference.create_transcript_mask(transcript).to(torch.bool)
        spectrogram_mask = inference.create_spectrogram_mask(audio).to(torch.bool)

        phoneme_mask = torch.logical_not(phoneme_mask)
        spectrogram_mask = torch.logical_not(spectrogram_mask)

        return audio, transcript, phoneme_durations, phoneme_mask, spectrogram_mask

    def __len__(self) -> int:
        """Returns the number of items in the dataset."""
        return len(self._dataset)

    def get_sample_id(self, idx: int) -> str:
        """Returns the sample ID for the given index."""
        return self._metadata[idx][0]

    def get_dataset_metadata(self) -> Dict:
        """Returns metadata of the dataset."""

        return {
            'phonemes_sequence_length': self._phonemes_sequence_length,
            'output_spectrogram_length': self._output_spectrogram_length
        }


def serialize_ds(ds: LJSpeechDataset, path: str) -> None:
    """Serializes the dataset to a file.

    Args:
        ds: The dataset to serialize.
        path: Path to the file to serialize the dataset to.
    """

    debug_log_interval = 1000

    metadata = ds.get_dataset_metadata()
    metadata_path = os.path.join(path, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=4)

    for sample_idx, sample in enumerate(ds):
        sample_path = os.path.join(path, f'{ds.get_sample_id(sample_idx)}.pt')
        torch.save(sample, sample_path)

        if (sample_idx + 1) % debug_log_interval == 0:
            logging.debug('Serialized %d samples.', sample_idx + 1)
