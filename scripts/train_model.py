# -*- coding: utf-8 -*-
"""Contains training pipeline for the model."""

from typing import Any, Dict, Tuple
import argparse
import logging
import os
import pathlib
import sys

import numpy as np
import torch
import torchvision
import yaml  # type: ignore
from torch.utils import data as torch_data
from torch.utils import tensorboard as torch_tb

from data import data_loading
from utilities import logging_utils
from utilities import scripts_utils
from models.ddpm_gst_speech_gen import utils as m_utils
from models.ddpm_gst_speech_gen import training
from utilities import diffusion as diff_utils

HOME_PATH = pathlib.Path(__file__).absolute().parent.parent.parent.as_posix()
SCRIPT_PATH = os.path.join(HOME_PATH, 'scripts', 'train_model')

DEFAULT_CONFIG = {
    'data': {
        'dataset_path': scripts_utils.CfgRequired(),
        'train_split_ratio': 0.98,
        'n_test_files': 100
    },
    'training': {
        'batch_size': 64,
        "lr": 2e-4,
        'validation_interval': 100,
        'steps': 1000,
        'start_step': 0,
        'checkpoint_interval': 200,
        'checkpoints_path': scripts_utils.CfgRequired(),

        'diffusion': {
            'n_steps': 400,
            'beta_min': 0.0001,
            'beta_max': 0.02,
        }
    },
    'model': {
        'decoder': {
            'timestep_embedding_dim': 128,
        },
        'gst': {
            'use_gst': True,
            'embedding_dim': 256,
            'token_count': 32,
        }
    }
}


def _get_model_trainer(
        input_spectrogram_shape,
        input_phonemes_shape,
        train_loader: torch_data.DataLoader,
        val_loader: torch_data.DataLoader,
        config: Dict[str, Any],
        tb_writer: torch_tb.SummaryWriter
) -> training.ModelTrainer:

    checkpoints_handler = m_utils.ModelCheckpointHandler(
        config['training']['checkpoints_path'], 'ddpm_gst_speech_gen_ckpt')

    model_components = m_utils.create_model_components(
        input_spectrogram_shape, input_phonemes_shape, config['model'])

    def model_provider():

        if checkpoints_handler.num_checkpoints() > 0:
            checkpoint, _ = checkpoints_handler.get_newest_checkpoint(model_components)
            return checkpoint

        return model_components

    return training.ModelTrainer(
        model_provider,
        train_loader,
        val_loader,
        tb_writer,
        torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        diff_utils.LinearScheduler(
            config['training']['diffusion']['beta_min'],
            config['training']['diffusion']['beta_max'],
            config['training']['diffusion']['n_steps']
        ),
        checkpoints_handler,
        config['training']['checkpoint_interval'],
        config['training']['validation_interval'],
        config['training']['lr']
    )


def _log_example_data(train_ds: torch_data.Dataset, tb_writer: torch_tb.SummaryWriter):

    example_data = train_ds[np.random.randint(0, len(train_ds))]

    spec, _, _ = example_data

    tb_writer.add_image('example_mel_spectrograms', np.expand_dims(spec, axis=0))


def main(config):
    """Runs the training pipeline based on the configuration."""

    logging.info('Starting training pipeline.')
    logging.info('Configuration:\n%s', yaml.dump(config))

    tb_writer = torch_tb.SummaryWriter()

    train_ds, val_ds, _ = data_loading.get_datasets(
        config['data']['dataset_path'],
        config['data']['train_split_ratio'],
        config['data']['n_test_files']
    )

    logging.info('Datasets loaded.')

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

    logging.info('Data loaders created.')

    input_spectrogram_shape = train_ds[0][0].shape
    input_phonemes_shape = train_ds[0][1].shape

    model_trainer = _get_model_trainer(
        input_spectrogram_shape,
        input_phonemes_shape,
        train_loader,
        val_loader,
        config,
        tb_writer)

    logging.info('Running training for %d steps starting from %d...',
                 config['training']['steps'],
                 config['training']['start_step'])

    model_trainer.run_training(config['training']['steps'],
                               config['training']['start_step'],
                               use_profiler=True)

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

    configuration = scripts_utils.try_load_user_config(args.config_path, DEFAULT_CONFIG)
    main(configuration)
