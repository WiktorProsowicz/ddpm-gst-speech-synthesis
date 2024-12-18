# -*- coding: utf-8 -*-
"""Contains training pipeline for the mel-to-linear spectrogram converter model.

The script trains the mel-to-linear spectrogram converter model on the LJSpeech dataset.
The user is supposed to provide the script with the dataset directory, destination directory
for the checkpoints and various hyperparameters for the model and training.

For the expected configuration parameters, see the DEFAULT_CONFIG constant.
"""
import argparse
import logging
from typing import Any
from typing import Dict

import torch
import yaml  # type: ignore
from torch.utils import data as torch_data
from torch.utils import tensorboard as torch_tb

from data import data_loading
from models import utils as shared_m_utils
from models.mel_to_lin_converter import training
from models.mel_to_lin_converter import utils as m_utils
from utilities import logging_utils
from utilities import scripts_utils


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
        'warmup_steps': 4000,
        'validation_interval': 100,
        'steps': 1000,
        'start_step': 0,
        'checkpoint_interval': 200,
        'checkpoints_path': scripts_utils.CfgRequired(),
    },
    'model': {
        'n_heads': 4,
        'dropout_rate': 0.1,
        'fft_conv_channels': 1536,
        'n_blocks': 6,
        'd_model': 384
    },
    # The name of the script run. Shall be used for the TensorBoard logging
    'run_label': None,
    'use_profiler': False
}


def _get_model_trainer(
        input_spectrogram_shape,
        output_dim: int,
        train_loader: torch_data.DataLoader,
        val_loader: torch_data.DataLoader,
        config: Dict[str, Any],
        tb_writer: torch_tb.SummaryWriter
) -> training.ModelTrainer:

    checkpoints_handler = shared_m_utils.ModelCheckpointHandler(
        config['training']['checkpoints_path'], 'mel_to_lin_converter')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_components = m_utils.create_model_components(
        input_spectrogram_shape, output_dim, config['model'], device)

    return training.ModelTrainer(
        model_components,
        train_loader,
        val_loader,
        tb_writer,
        device,
        checkpoints_handler,
        config['training']['checkpoint_interval'],
        config['training']['validation_interval'],
        config['model']['d_model'],
        config['training']['warmup_steps'])


def main(config):
    """Runs the training pipeline for the mel-to-linear spectrogram converter model."""

    logging.info('Running the training pipeline...')
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
    output_dim, _ = train_ds[0][1].shape

    model_trainer = _get_model_trainer(
        input_spectrogram_shape,
        output_dim,
        train_loader,
        val_loader,
        config,
        tb_writer)

    logging.info('Running training for %d steps starting from the step %d...',
                 config['training']['steps'],
                 config['training']['start_step'])

    model_trainer.run_training(config['training']['steps'],
                               config['training']['start_step'],
                               use_profiler=config['use_profiler'])

    tb_writer.close()


def _get_cl_args() -> argparse.Namespace:

    arg_parser = argparse.ArgumentParser(
        description="Performs the mel-to-linear converter's training pipeline.")

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
