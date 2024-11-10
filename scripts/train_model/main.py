# -*- coding: utf-8 -*-
"""Contains training pipeline for the model."""

import os
import pathlib
import sys
import argparse
from typing import List
from dataclasses import dataclass
import yaml  # Type: ignore

HOME_PATH = pathlib.Path(__file__).absolute().parent.parent.as_posix()

DEFAULT_CONFIG = {
    "training": {
        "batch_size": 32,
        "epochs": 32
    },
    "model": {
        "encoder_input_shape": [256, 80],
        "decoder_input_shape": [1024, 80],
    }
}


def main(config):
    """Runs the training pipeline based on the configuration."""

    print(config)


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

    config = DEFAULT_CONFIG

    if args.config_path:
        with open(args.config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)

    main(config)
