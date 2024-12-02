# -*- coding: utf-8 -*-
"""Prepares a ready-to-use dataset with desired features.

The script downloads the LJSpeech dataset and converts each audio sample into a pair containing
a mel-spectrogram and a linear-spectrogram.

For expected configuration parameters, see the DEFAULT_CONFIG constant.
"""

import argparse
import logging
import os
import pathlib

import yaml  # type: ignore

from data import ljspeech
from utilities import logging_utils
from utilities import scripts_utils

SCRIPT_PATH = pathlib.Path(__file__).absolute().parent.as_posix()
HOME_PATH = pathlib.Path(__file__).absolute().parent.parent.parent.as_posix()

DEFAULT_CONFIG = {
    # The path where the raw dataset will be (or is already) stored
    'raw_dataset_path': scripts_utils.CfgRequired(),
    # The path where the preprocessed dataset will be stored
    'processed_dataset_path': scripts_utils.CfgRequired(),
    'sample_rate': 22050,
    'fft_window_size': 1024,
    'fft_hop_size': 256,
    # If not None, the audio samples will be divided into chunks of this length
    'max_frames_in_split_spectrogram': None,
    'scale_spectrograms': False
}


def main(config):
    """Runs the dataset preparation pipeline based on the configuration."""

    logging.info('Running dataset preparation pipeline...')
    logging.info('Configuration:\n%s', yaml.dump(config))

    os.makedirs(config['processed_dataset_path'], exist_ok=True)
    os.makedirs(config['raw_dataset_path'], exist_ok=True)

    logging.info('Preparing the preprocessed dataset...')

    ds = ljspeech.LJSpeechSpectrogramsDs(
        config['raw_dataset_path'],
        config['sample_rate'],
        config['fft_window_size'],
        config['fft_hop_size'],
        config['scale_spectrograms']
    )

    logging.info('Serializing the dataset...')

    ljspeech.serialize_spectrograms_ds(ds,
                                       config['processed_dataset_path'],
                                       config['max_frames_in_split_spectrogram'])

    logging.info('Dataset preparation completed.')


def _get_cl_args() -> argparse.Namespace:

    arg_parser = argparse.ArgumentParser(
        description="Prepares a preprocessed dataset used to train the Mel-to-Linear converter.")

    arg_parser.add_argument(
        '--config_path',
        type=str,
        help='Path to the folder containing configuration files.'
    )

    return arg_parser.parse_args()


if __name__ == '__main__':

    logging_utils.setup_logging()

    args = _get_cl_args()

    configuration = scripts_utils.try_load_user_config(args.config_path, DEFAULT_CONFIG)
    main(configuration)
