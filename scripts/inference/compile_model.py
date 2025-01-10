# -*- coding: utf-8 -*-
"""Creates components of the whole model and compiles a runnable version for inference.

For script's configuration, see `DEFAULT_CONFIG` constant.
"""
import argparse
import os
import logging
import json
from typing import Tuple

import torch

from models.acoustic import utils as acoustic_utils
from models.gst_predictor import utils as gst_predictor_utils
from utilities import inference
from utilities import logging_utils
from utilities import scripts_utils

DEFAULT_CONFIG = {
    'acoustic_model_checkpoint': scripts_utils.CfgRequired(),
    # Should be the configuration of the acoustic model used during the training
    'acoustic_model_cfg': scripts_utils.CfgRequired(),
    'phonemes_encoding_size': 73,
    'phonemes_length': 20,
    'mel_spec_freq_bins': 80,
    'mel_spec_time_frames': 200,
    # Specifies the way the output model should use the Global Style Tokens.
    # Should be one of ('none', 'weights', 'reference', 'predicted')
    'gst_mode': 'weights',
    # If 'gst_mode' is 'predicted', this should contain the gst predictor's configuration.
    'gst_predictor_cfg': {
        'checkpoint_path': None,
        'model_cfg': None,
        'diff_beta_min': 0.0001,
        'diff_beta_max': 0.02,
        'diff_timesteps': 200,
    },
    # Directory where the compiled model's components will be saved.
    'output_path': scripts_utils.CfgRequired()
}


def _create_inference_model(acoustic_components: acoustic_utils.ModelComponents,
                            mel_spec_frames: int,
                            gst_mode: str):

    if gst_mode == 'none':

        gst = None
        embedder = None

    elif gst_mode == 'weights':

        gst = acoustic_components.gst
        embedder = None

    elif gst_mode == 'reference':

        gst = acoustic_components.gst
        embedder = acoustic_components.embedder

    return inference.InferenceModel(
        acoustic_components.encoder,
        acoustic_components.decoder,
        acoustic_components.duration_predictor,
        acoustic_components.length_regulator,
        mel_spec_frames,
        gst,
        embedder)


def _compile_acoustic_model(config) -> torch.jit.ScriptModule:

    device = torch.device('cpu')

    logging.info('Loading acoustic model components...')

    acoustic_components = acoustic_utils.create_model_components(
        (config['mel_spec_freq_bins'], config['mel_spec_time_frames']),
        (config['phonemes_length'], config['phonemes_encoding_size']),
        config['acoustic_model_cfg'],
        device)

    acoustic_components.load_from_path(config['acoustic_model_checkpoint'], device)
    acoustic_components.eval()

    logging.info('Preparing the acoustic inference model...')

    inference_model = _create_inference_model(acoustic_components,
                                              config['mel_spec_time_frames'],
                                              config['gst_mode'])
    inference_model.eval()

    example_phonemes = torch.randint(0, config['phonemes_encoding_size'],
                                     (1, config['phonemes_length']))
    example_phonemes = torch.nn.functional.one_hot(  # pylint: disable=not-callable
        example_phonemes,
        config['phonemes_encoding_size'])

    if config['gst_mode'] == 'none':
        example_input = (
            example_phonemes,
        )

    elif config['gst_mode'] in ('weights', 'predicted'):
        example_input = (
            example_phonemes,
            torch.randn(1, config['acoustic_model_cfg']['gst']['n_tokens'])
        )

    elif config['gst_mode'] == 'reference':
        example_input = (
            example_phonemes,
            torch.randn(1, config['mel_spec_freq_bins'], config['mel_spec_time_frames'])
        )

    with torch.no_grad():
        inference_model(example_input)

    logging.info('Tracing the inference model...')

    return torch.jit.trace_module(inference_model, {'forward': (example_input,)})


def _compile_gst_predictor(config) -> Tuple[torch.jit.ScriptModule, torch.jit.ScriptModule]:

    device = torch.device('cpu')

    logging.info("Loading the GST predictor components...")

    gst_predictor = gst_predictor_utils.create_model_components(
        (config['phonemes_length'], config['phonemes_encoding_size']),
        config['gst_predictor_cfg']['model_cfg'],
        device)

    gst_predictor.load_from_path(config['gst_predictor_cfg']['checkpoint_path'], device)
    gst_predictor.eval()

    logging.info("Tracing the GST predictor...")

    example_phonemes = torch.randint(0, config['phonemes_encoding_size'],
                                     (1, config['phonemes_length']))
    example_phonemes = torch.nn.functional.one_hot(  # pylint: disable=not-callable
        example_phonemes,
        config['phonemes_encoding_size'])

    example_noise = torch.randn(1, config['gst_predictor_cfg']['model_cfg']['n_tokens'])

    example_phoneme_embedding = torch.randn(
        1, config['gst_predictor_cfg']['model_cfg']['embedding_size'])

    example_diff_timestep = torch.randint(0, config['gst_predictor_cfg']['diff_timesteps'], (1,))

    encoder = torch.jit.trace_module(gst_predictor.encoder, {'forward': (example_phonemes,)})
    decoder = torch.jit.trace_module(gst_predictor.decoder,
                                     {'forward': (example_noise,
                                                  example_diff_timestep,
                                                  example_phoneme_embedding)})

    return encoder, decoder


def main(config):
    """Loads the model and runs inference."""

    os.makedirs(config['output_path'], exist_ok=True)

    gst_predictor_metadata = None

    if config['gst_mode'] == 'predicted':
        gst_predictor_metadata = {
            'diff_beta_min': config['gst_predictor_cfg']['diff_beta_min'],
            'diff_beta_max': config['gst_predictor_cfg']['diff_beta_max'],
            'diff_timesteps': config['gst_predictor_cfg']['diff_timesteps']
        }

    compiled_model_metadata = {
        'input_phonemes_shape': (config['phonemes_length'], config['phonemes_encoding_size']),
        'output_spec_shape': (config['mel_spec_freq_bins'], config['mel_spec_time_frames']),
        'gst_mode': config['gst_mode'],
        'gst_predictor_cfg': gst_predictor_metadata
    }

    acoustic_torchscript = _compile_acoustic_model(config)
    torch.jit.save(acoustic_torchscript, os.path.join(config['output_path'], 'inference_model.pt'))

    gst_pred_enc, gst_pred_dec = _compile_gst_predictor(config)
    torch.jit.save(gst_pred_enc, os.path.join(config['output_path'], 'gst_predictor_encoder.pt'))
    torch.jit.save(gst_pred_dec, os.path.join(config['output_path'], 'gst_predictor_decoder.pt'))

    with open(os.path.join(config['output_path'], 'metadata.json'),
              'w',
              encoding='utf-8') as metadata_file:
        json.dump(compiled_model_metadata, metadata_file, indent=4)

    logging.info("Inference model has been saved to '%s'.", config['output_path'])


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
