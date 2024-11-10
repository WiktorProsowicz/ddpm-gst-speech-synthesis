# -*- coding: utf-8 -*-
"""Prepares a ready-to-use dataset with desired features."""

from torchvision.transforms import transforms
import argparse
import yaml  # Type: ignore
import pathlib
import sys
import os
import logging


SCRIPT_PATH = pathlib.Path(__file__).absolute().parent.as_posix()
HOME_PATH = pathlib.Path(__file__).absolute().parent.parent.parent.as_posix()

DEFAULT_CONFIG = {
    # Destination paths
    "raw_dataset_path": f"{SCRIPT_PATH}/.dataset/raw",
    "processed_dataset_path": f"{SCRIPT_PATH}/.dataset/processed",
    "phoneme_alignments_path": f"{SCRIPT_PATH}/.dataset/alignments",
}


def main(config):
    """Runs the dataset preparation pipeline based on the configuration."""

    from data import ljspeech
    from data.preprocessing import text as text_prep
    from utilities import other as other_utils

    logging.info("Running dataset preparation pipeline...")
    logging.info("Configuration:\n%s", yaml.dump(config))

    os.makedirs(config['processed_dataset_path'], exist_ok=True)
    os.makedirs(config['raw_dataset_path'], exist_ok=True)

    if not os.path.exists(config['phoneme_alignments_path']):

        logging.info("Downloading phoneme alignments...")

        os.makedirs(config['phoneme_alignments_path'])
        other_utils.download_phoneme_alignments(config['phoneme_alignments_path'])

    ds_text_transform = transforms.Compose([
        text_prep.GraphemeToPhonemeTransform()
    ])
    ds_audio_transform = None

    logging.info("Preparing the preprocessed dataset...")

    ds = ljspeech.LJSpeechDataset(config['raw_dataset_path'],
                                  config['phoneme_alignments_path'],
                                  ds_text_transform,
                                  ds_audio_transform)
    # ljspeech.serialize_ds(ds, config['processed_dataset_path'])


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

    args = _get_cl_args()

    configuration = DEFAULT_CONFIG

    if args.config_path:
        with open(args.config_path, 'r', encoding='utf-8') as file:
            configuration = yaml.safe_load(file)

    sys.path.append(os.path.join(HOME_PATH, 'src'))

    from utilities import logging_utils

    logging_utils.setup_logging()
    logging.getLogger().setLevel(logging.INFO)

    main(configuration)
