# -*- coding: utf-8 -*-
"""Contains training pipeline for the GST predictor model.

The training pipeline is responsible for running the training process for the GST predictor model.
The user is supposed to provide the script with the directory containing the preprocessed
dataset, destination directory for checkpoints and hyperparameters for the training and
diffusion process.

For the expected configuration parameters, see the DEFAULT_CONFIG constant.
"""
import logging
import os
from typing import Any
from typing import Dict
from typing import Tuple

import torch
import torch_dev_utils as tdu
import yaml  # type: ignore
from torch.utils import data as torch_data
from torch.utils import tensorboard as torch_tb

from models.gst_predictor import training
from models.gst_predictor import utils as m_utils
from utilities import diffusion as diff_utils
from utilities import logging_utils
from utilities import scripts_utils
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
            'guidance_scale': None,
        },
    },
    'model': {
        'decoder': {
            'input_dimension': 10,
            'timestep_embedding_size': 128,
            'internal_channels': 128,
            'n_blocks': 10,
            'dropout_rate': 0.0
        },
        'encoder': {
            'n_blocks': 6,
            'n_heads': 4,
            'conv_filters': 1536,
            'dropout_rate': 0.1
        },
        'deterministic_pred': {
            'n_blocks': 1,
            'fft_conv_channels': 1536,
            'internal_dim': 384,
            'dropout_rate': 0.1
        }
    },
    # The name of the script run. Shall be used for the TensorBoard logging
    'run_label': None,
    'use_profiler': False
}


def _get_model_trainer(input_phonemes_shape: Tuple[int, int],
                       gst_emb_size: int,
                       gst_weights_size: int,
                       config: Dict[str, Any],
                       train_loader: torch_data.DataLoader,
                       val_loader: torch_data.DataLoader,
                       global_ds_stats: Tuple[torch.Tensor, torch.Tensor],
                       tb_writer: torch_tb.SummaryWriter) -> training.ModelTrainer:

    torch.multiprocessing.set_start_method('spawn')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoints_handler = tdu.serialization.ModelCheckpointHandler(
        config['training']['checkpoints_path'],
        device,
        False)

    model_components = m_utils.create_model_components(
        input_phonemes_shape,
        gst_emb_size,
        gst_weights_size,
        config['model'],
        device
    )

    global_ds_stats = tuple(stat.to(device) for stat in global_ds_stats)

    optimizer = torch.optim.Adam([{'name': 'weight_pred_params',
                                  'params': model_components.weights_pred_params(),
                                 'lr': config['training']['lr'],
                                   'betas': (0.9, 0.98),
                                   'weight_decay': 2e-6},
                                  {'name': 'emb_pred_params',
                                   'params': model_components.diffusion_params(),
                                   'lr': config['training']['lr'],
                                   'betas': (0.9, 0.98),
                                   'weight_decay': 2e-6}])

    optimizer = shared_m_utils.TransformerScheduledOptim(
        optimizer,
        config['model']['deterministic_pred']['internal_dim'],
        config['training']['warmup_steps'],
        ['weight_pred_params'])

    params = tdu.training.BaseTrainerParams(
        model_comps=model_components,
        optimizer=optimizer,
        checkpoints_handler=checkpoints_handler,
        train_data_loader=train_loader,
        val_data_loader=val_loader,
        tb_logger=tb_writer,
        device=device,
        validation_interval=config['training']['validation_interval'],
        checkpoints_interval=config['training']['checkpoint_interval'],
        log_interval=100
    )

    return training.ModelTrainer(params,
                                 diff_utils.LinearScheduler(
                                     config['training']['diffusion']['beta_min'],
                                     config['training']['diffusion']['beta_max'],
                                     config['training']['diffusion']['n_steps']
                                 ),
                                 global_ds_stats,
                                 config['training']['diffusion']['guidance_scale']
                                 )


def main(config):
    """Runs the training pipeline for the GST predictor model."""

    logging.info('Starting training pipeline.')
    logging.info('Configuration:\n%s', yaml.dump(config))

    tb_writer = torch_tb.SummaryWriter(
        log_dir=f"runs/{config['run_label']}" if config['run_label'] is not None else None)

    tb_writer.add_text('Configuration', yaml.dump(config))

    train_ds, val_ds, _ = tdu.data_loading.get_datasets(
        config['data']['dataset_path'],
        config['data']['train_split_ratio'],
        config['data']['n_test_files']
    )

    ds_stats_path = os.path.join(config['data']['dataset_path'], 'stats', 'gst_embedding_stats.pt')
    global_ds_stats = torch.load(ds_stats_path, weights_only=True)

    logging.info('Dataset loaded.')

    train_loader = torch_data.DataLoader(
        train_ds,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=False
    )

    val_loader = torch_data.DataLoader(
        val_ds,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )

    logging.info('Data loaders created.')

    input_phonemes_shape = train_ds[0][0].shape
    input_gst_emb_size = train_ds[0][3].shape[0]
    input_gst_weights_size = train_ds[0][4].shape[0]

    model_trainer = _get_model_trainer(
        input_phonemes_shape,
        input_gst_emb_size,
        input_gst_weights_size,
        config,
        train_loader,
        val_loader,
        global_ds_stats,
        tb_writer
    )

    logging.info('Running training for %d steps starting from the step %d...',
                 config['training']['steps'],
                 config['training']['start_step'])

    model_trainer.run_training(config['training']['steps'],
                               config['training']['start_step'],
                               use_profiler=config['use_profiler'])

    tb_writer.close()


if __name__ == '__main__':

    logging_utils.setup_logging()

    configuration = scripts_utils.try_obtain_cfg_from_cl(
        'Performs the GST Predictor\'s training pipeline.',
        DEFAULT_CONFIG
    )

    main(configuration)
