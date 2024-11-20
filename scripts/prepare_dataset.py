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
from utilities import logging_utils
from utilities import scripts_utils

SCRIPT_PATH = pathlib.Path(__file__).absolute().parent.as_posix()
HOME_PATH = pathlib.Path(__file__).absolute().parent.parent.parent.as_posix()

DEFAULT_CONFIG = {
    # The path where the raw dataset will be (or is already) stored
    'raw_dataset_path': scripts_utils.CfgRequired(),
    # The path where the preprocessed dataset will be stored
    'processed_dataset_path': scripts_utils.CfgRequired(),
    # The path where the phoneme alignments will be (or have been) downloaded into
    'phoneme_alignments_path': scripts_utils.CfgRequired(),
    'sample_rate': 22050,
    'fft_window_size': 1024,
    'fft_hop_size': 256,
    'audio_max_length': 6.0
}


def main(config):
    """Runs the dataset preparation pipeline based on the configuration."""

    logging.info('Running dataset preparation pipeline...')
    logging.info('Configuration:\n%s', yaml.dump(config))

    os.makedirs(config['processed_dataset_path'], exist_ok=True)
    os.makedirs(config['raw_dataset_path'], exist_ok=True)

    if not os.path.exists(config['phoneme_alignments_path']):

        logging.info('Downloading phoneme alignments...')

        os.makedirs(config['phoneme_alignments_path'])
        scripts_utils.download_phoneme_alignments(config['phoneme_alignments_path'])

    logging.info('Preparing the preprocessed dataset...')

    ds = ljspeech.LJSpeechDataset(config['raw_dataset_path'],
                                  config['phoneme_alignments_path'],
                                  config['sample_rate'],
                                  config['fft_window_size'],
                                  config['fft_hop_size'],
                                  config['audio_max_length'])

    logging.info('Serializing the dataset...')

    ljspeech.serialize_ds(ds, config['processed_dataset_path'])

    logging.info('Dataset preparation completed.')


def _get_cl_args() -> argparse.Namespace:

    arg_parser = argparse.ArgumentParser(
        description="Performs the model's training pipeline based on the configuration.")

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
