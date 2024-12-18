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

import torch
import yaml  # type: ignore
from torch.utils import data as torch_data
from torch.utils import tensorboard as torch_tb

from data import data_loading
from data import visualization
from models import utils as shared_m_utils
from models.ddpm_gst_speech_gen import training
from models.ddpm_gst_speech_gen import utils as m_utils
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
        },

        'use_gt_durations_for_backward_diff': True,
        'use_loss_weights': True
    },
    'model': {
        'decoder': {
            'timestep_embedding_dim': 128,
            'n_res_blocks': 12,
            'internal_channels': 128,
            'skip_connections_channels': 512,
            'conv_kernel_size': 3,
        },
        'gst': {
            'use_gst': False,
            'embedding_dim': 256,
            'token_count': 32
        },
        'duration_predictor': {
            'n_blocks': 4
        },
        'encoder': {
            'n_blocks': 2,
            'embedding_dim': 128
        },
        'dropout_rate': 0.1
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

    checkpoints_handler = shared_m_utils.ModelCheckpointHandler(
        config['training']['checkpoints_path'], 'ddpm_gst_speech_gen_ckpt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_components = m_utils.create_model_components(
        input_spectrogram_shape, input_phonemes_shape, config['model'], device)

    return training.ModelTrainer(
        model_components,
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
        config['training']['lr'],
        config['training']['use_gt_durations_for_backward_diff'],
        config['training']['use_loss_weights']
    )


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

    visualization.log_example_ljspeech_data(train_ds, tb_writer)

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

    WARN_MSG = """This script is related to the full acoustic model that incorporates GST and
    DDPM in to generation process. The model is currently under development and, though it
    theoretically can be trained etc, it is not yet ready for the production use. Additionally
    the script is likely to be moved to the `training` directory in the future major version."""

    logging.warning(WARN_MSG)

    args = _get_cl_args()

    configuration = scripts_utils.try_load_user_config(args.config_path, DEFAULT_CONFIG)
    main(configuration)
