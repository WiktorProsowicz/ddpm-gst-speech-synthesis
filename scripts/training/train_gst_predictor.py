# -*- coding=utf-8 -*-
"""Contains training pipeline for the GST predictor model.

The training pipeline is responsible for running the training process for the GST predictor model.
The user is supposed to provide the script with the directory containing the preprocessed
dataset, destination directory for checkpoints and hyperparameters for the training and
diffusion process.

For the expected configuration parameters, see the DEFAULT_CONFIG constant.
"""

import logging
import argparse
import yaml  # type: ignore
from typing import Tuple
from typing import Dict
from typing import Any

from torch.utils import tensorboard as torch_tb
from torch.utils import data as torch_data
import torch
import matplotlib


from utilities import scripts_utils
from utilities import logging_utils
from utilities import diffusion as diff_utils
from data import data_loading
from models.gst_predictor import training
from models.gst_predictor import utils as m_utils
from models import utils as shared_m_utils


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
    },
    'model': {
        'decoder': {
            'timestep_embedding_size': 128,
            'internal_channels': 64,
            'n_conv_blocks': 6,
        },
        'encoder': {
            'n_conv_blocks': 6,
            'embedding_size': 128,
        },
        'dropout_rate': 0.1
    },
    # The name of the script run. Shall be used for the TensorBoard logging
    'run_label': None,
    'use_profiler': False
}


def _get_model_trainer(input_phonemes_shape: Tuple[int, int],
                       config: Dict[str, Any],
                       train_loader: torch_data.DataLoader,
                       val_loader: torch_data.DataLoader,
                       tb_writer: torch_tb.SummaryWriter) -> training.ModelTrainer:

    checkpoints_handler = shared_m_utils.ModelCheckpointHandler(
        config['training']['checkpoints_path'],
        'gst_predictor_ckpt',
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_components = m_utils.create_model_components(
        input_phonemes_shape,
        config['model'],
        device
    )

    return training.ModelTrainer(
        model_components,
        train_loader,
        val_loader,
        tb_writer,
        device,
        checkpoints_handler,
        config['training']['checkpoint_interval'],
        config['training']['validation_interval'],
        config['training']['lr'],
        diff_utils.LinearScheduler(
            config['training']['diffusion']['beta_min'],
            config['training']['diffusion']['beta_max'],
            config['training']['diffusion']['n_steps']
        )
    )


def main(config):
    """Runs the training pipeline for the GST predictor model."""

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

    logging.info('Dataset loaded.')

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

    input_phonemes_shape = train_ds[0][0].shape

    model_trainer = _get_model_trainer(
        input_phonemes_shape,
        config,
        train_loader,
        val_loader,
        tb_writer
    )

    logging.info('Running training for %d steps starting from the step %d...',
                 config['training']['steps'],
                 config['training']['start_step'])

    model_trainer.run_training(config['training']['steps'],
                               config['training']['start_step'],
                               use_profiler=config['use_profiler'])

    tb_writer.close()


def _get_cl_args() -> argparse.Namespace:

    arg_parser = argparse.ArgumentParser(
        description="Performs the gST predictor's training pipeline based on the configuration.")

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
