# -*- coding: utf-8 -*-
"""Contains training pipeline for the model.

The training pipeline is responsible for running the training process for the DDPM-GST-Speech-Gen
model. The user is supposed to provide the script with the directory containing the preprocessed
dataset, destination directory for the checkpoints and various hyperparameters for the model,
training and diffusion process.

For the expected configuration parameters, see the DEFAULT_CONFIG constant.
"""
import argparse
import logging
import os
import pathlib
import sys
from typing import Any
from typing import Dict

import numpy as np
import torch
import yaml  # type: ignore
from torch.utils import data as torch_data
from torch.utils import tensorboard as torch_tb

from data import data_loading
from data import visualisation
from data.preprocessing import text
from model import training
from model import utils as m_utils
from utilities import diffusion as diff_utils
from utilities import logging_utils
from utilities import scripts_utils

HOME_PATH = pathlib.Path(__file__).absolute().parent.parent.parent.as_posix()
SCRIPT_PATH = os.path.join(HOME_PATH, 'scripts', 'train_model')

DEFAULT_CONFIG = {
    'data': {
        # The path to the preprocessed dataset
        'dataset_path': scripts_utils.CfgRequired(),
        # The split ratio of the dataset after removing the test files
        'train_split_ratio': 0.98,
        'n_test_files': 100
    },
    'training': {
        'batch_size': 64,
        'lr': 2e-4,
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
            'n_res_blocks': 12,
            'internal_channels': 128,
            'skip_connections_channels': 512
        },
        'gst': {
            'use_gst': False,
            'embedding_dim': 256,
            'token_count': 32,
        },
        'duration_predictor': {
            'n_blocks': 4
        },
    },
    # The name of the script run. Shall be used for the TensorBoard logging
    'run_label': None
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_components = m_utils.create_model_components(
        input_spectrogram_shape, input_phonemes_shape, config['model'], device)

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
        device,
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

    spec, transcript, durations = example_data

    tb_writer.add_image(
        'Example/InputMelSpectrogram',
        visualisation.colorize_spectrogram(spec, 'viridis'))

    tb_writer.add_text(
        'Example/Transcript',
        ' '.join(visualisation.decode_transcript(transcript, text.ENHANCED_MFA_ARP_VOCAB)))

    durations_mask = (durations.numpy() > 0).astype(np.uint16)
    pow_durations = (np.power(2, durations.numpy()) +
                     1e-4).astype(np.uint16)[:np.sum(durations_mask).item()]

    tb_writer.add_figure('Example/InputSpectrogramWithPhonemeBoundaries',
                         visualisation.annotate_spectrogram_with_phoneme_durations(
                             spec.numpy(), pow_durations))


def main(config):
    """Runs the training pipeline based on the configuration."""

    if config['model']['gst']['use_gst']:
        logging.critical('GST support is not implemented yet!')
        sys.exit(1)

    logging.info('Starting training pipeline.')
    logging.info('Configuration:\n%s', yaml.dump(config))

    tb_writer = torch_tb.SummaryWriter(
        log_dir=f"runs/{config['run_label']}" if config['run_label'] is not None else None)

    tb_writer.add_text('Configuration', yaml.dump(config))

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

    logging.info('Running training for %d steps starting from the step %d...',
                 config['training']['steps'],
                 config['training']['start_step'])

    model_trainer.run_training(config['training']['steps'],
                               config['training']['start_step'],
                               use_profiler=False)

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
