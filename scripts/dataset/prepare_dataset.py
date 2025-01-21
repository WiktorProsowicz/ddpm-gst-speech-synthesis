# -*- coding: utf-8 -*-
"""Prepares a ready-to-use dataset with desired features.

The script downloads the LJSpeech dataset and phoneme alignments. Then it converts the raw
waveform files into spectrograms according to the provided parameters. The transcripts are
encoded into phoneme tokens. The phoneme alignments are converted into log-scale durations
of the particular phonemes. The dataset is serialized into a single folder for further use.

For expected configuration parameters, see the DEFAULT_CONFIG constant.
"""
import argparse
import logging
import os
import pathlib

import yaml  # type: ignore

from data import ljspeech
from data.preprocessing import alignments as align_prep
from utilities import logging_utils
from utilities import scripts_utils

SCRIPT_PATH = pathlib.Path(__file__).absolute().parent.as_posix()
HOME_PATH = pathlib.Path(__file__).absolute().parent.parent.parent.as_posix()

DEFAULT_CONFIG = {
    # The path where the dataset will be (or is already) stored
    'dataset_path': scripts_utils.CfgRequired(),
    # Length (in seconds) of the output audio clips
    'audio_max_length': 6.0
}


def main(config):
    """Runs the dataset preparation pipeline based on the configuration."""

    logging.info('Running dataset preparation pipeline...')
    logging.info('Configuration:\n%s', yaml.dump(config))

    os.makedirs(config['dataset_path'], exist_ok=True)

    raw_dataset_path = os.path.join(config['dataset_path'], 'raw')
    phoneme_alignments_path = os.path.join(config['dataset_path'], 'phoneme_alignments')
    processed_dataset_path = os.path.join(config['dataset_path'], 'processed')

    if not os.path.exists(phoneme_alignments_path):

        logging.info('Downloading phoneme alignments...')

        os.makedirs(phoneme_alignments_path, exist_ok=True)
        align_prep.download_phoneme_alignments(phoneme_alignments_path)

    logging.info('Preparing the preprocessed dataset...')

    ds = ljspeech.LJSpeechDataset(raw_dataset_path,
                                  phoneme_alignments_path,
                                  config['audio_max_length'])

    logging.info('Serializing the dataset...')

    os.makedirs(processed_dataset_path, exist_ok=True)
    ljspeech.serialize_ds(ds, processed_dataset_path)

    logging.info('Dataset preparation completed.')


if __name__ == '__main__':

    logging_utils.setup_logging()

    configuration = scripts_utils.try_obtain_cfg_from_cl(
        'Prepares a ready-to-use dataset with desired features.',
        DEFAULT_CONFIG
    )

    main(configuration)
