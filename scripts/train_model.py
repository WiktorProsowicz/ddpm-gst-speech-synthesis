# -*- coding: utf-8 -*-
"""Contains training pipeline for the model."""

import os
import pathlib
import argparse
import sys
import logging
import yaml  # Type: ignore

from torch.utils import data as torch_data
from torch.utils import tensorboard as torch_tb
import torchvision
import torch
import numpy as np

from data import data_loading
from utilities import logging_utils
from utilities import scripts_utils

HOME_PATH = pathlib.Path(__file__).absolute().parent.parent.parent.as_posix()
SCRIPT_PATH = os.path.join(HOME_PATH, 'scripts', 'train_model')

DEFAULT_CONFIG = {
    "data": {
        "dataset_path": scripts_utils.CfgRequired(),
        "train_split_ratio": 0.98,
        "n_test_files": 100
    },
    "training": {
        "batch_size": 64,
        "epochs": 32,
    },
    "model": {
        "encoder_input_shape": [256, 80],
        "decoder_input_shape": [1024, 80],
    }
}


def _log_example_data(train_ds: torch_data.Dataset, tb_writer: torch_tb.SummaryWriter):

    example_data = train_ds[np.random.randint(0, len(train_ds))]

    spec, transcript, durations = example_data

    tb_writer.add_image('example_mel_spectrograms', np.expand_dims(spec, axis=0))


def main(config):
    """Runs the training pipeline based on the configuration."""

    logging.info("Starting training pipeline.")
    logging.info("Configuration:\n%s", yaml.dump(config))

    tb_writer = torch_tb.SummaryWriter()

    train_ds, val_ds, _ = data_loading.get_datasets(
        config['data']['dataset_path'],
        config['data']['train_split_ratio'],
        config['data']['n_test_files']
    )

    logging.info("Datasets loaded.")

    _log_example_data(train_ds, tb_writer)

    train_loader = torch_data.DataLoader(
        train_ds,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = torch_data.DataLoader(
        val_ds,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )

    logging.info("Data loaders created.")

    tb_writer.close()


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

    config = scripts_utils.try_load_user_config(args.config_path, DEFAULT_CONFIG)
    main(config)
