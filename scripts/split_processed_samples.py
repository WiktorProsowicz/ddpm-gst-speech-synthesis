# -*- coding: utf-8 -*-
"""Splits training samples from the processed dataset into smaller chunks.

A given number of phonemes and their corresponding spectrogram frames are
extracted from each sample and saved into a specified directory.

For expected configuration parameters, see the DEFAULT_CONFIG constant.
"""

import argparse
import yaml  # type: ignore
import logging
import os

from utilities import scripts_utils
from utilities import logging_utils
from data import data_loading

import torch

DEFAULT_CONFIG = {
    'dataset_path': scripts_utils.CfgRequired(),
    'output_path': scripts_utils.CfgRequired(),
    'n_phonemes': 20,
    'max_accepted_spec_length': 200
}


def main(config):
    """Loads, splits and saves the samples from the processed dataset."""

    logging.info('Splitting the data samples...')
    logging.info('Configuration:\n%s', yaml.dump(config))

    os.makedirs(config['output_path'], exist_ok=True)

    logging.debug('Determining the maximum spectrogram length...')

    max_spec_length = 0

    for _, (spec, _, _) in data_loading.split_processed_samples_into_chunks(
            config['dataset_path'], config['n_phonemes']):

        if spec.shape[1] > config['max_accepted_spec_length']:
            continue

        max_spec_length = max(max_spec_length, spec.shape[1])

    logging.debug('Maximum spectrogram length: %d', max_spec_length)

    n_chunks = 0
    n_skipped = 0

    for sample_id, (spec, phonemes, durations) in data_loading.split_processed_samples_into_chunks(
            config['dataset_path'], config['n_phonemes']):

        spec_length = spec.shape[1]

        if spec_length > max_spec_length:
            n_skipped += 1
            continue

        padding_size = max_spec_length - spec_length
        padding = torch.full((spec.shape[0], padding_size), spec.min())
        spec = torch.cat([spec, padding], dim=1)

        torch.save((spec, phonemes, durations), os.path.join(
            config['output_path'], f'{sample_id}.pt'))

        n_chunks += 1

        if n_chunks % 1000 == 0:
            logging.debug('Saved %d chunks.', n_chunks)

    logging.info('Split samples into %d chunks.', n_chunks)
    logging.info('Skipped %d samples due to too long spectrogram.', n_skipped)


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
