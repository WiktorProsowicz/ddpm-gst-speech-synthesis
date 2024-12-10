# -*- coding: utf-8 -*-
"""Creates components of the whole model and compiles a runnable version for inference.

For script's configuration, see `DEFAULT_CONFIG` constant.
"""
import argparse
import logging

import torch

from models.acoustic import utils as acoustic_utils
from models.mel_to_lin_converter import utils as mel2lin_utils
from utilities import inference
from utilities import logging_utils
from utilities import scripts_utils

DEFAULT_CONFIG = {

    'mel_to_lin_model_checkpoint': scripts_utils.CfgRequired(),
    # Should be the configuration of the mel2linear converter used during the training
    'mel_to_lin_model_cfg': scripts_utils.CfgRequired(),
    'acoustic_model_checkpoint': scripts_utils.CfgRequired(),
    # Should be the configuration of the acoustic model used during the training
    'acoustic_model_cfg': scripts_utils.CfgRequired(),
    'phonemes_encoding_size': 73,
    'phonemes_length': 20,
    'mel_spec_freq_bins': 80,
    'mel_spec_time_frames': 200,
    'linear_spec_freq_bins': 513,
    # Specifies the way the output model should use the Global Style Tokens.
    # Should be one of ('none', 'weights', 'reference')
    'gst_mode': 'weights',
    'output_path': scripts_utils.CfgRequired()
}


def _create_inference_model(acoustic_components: acoustic_utils.ModelComponents,
                            mel2lin_components: mel2lin_utils.ModelComponents,
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
        mel2lin_components.converter,
        gst,
        embedder
    )


def main(config):
    """Loads the model and runs inference."""

    device = torch.device('cpu')

    logging.info('Loading model components...')

    acoustic_components = acoustic_utils.create_model_components(
        (config['mel_spec_freq_bins'], config['mel_spec_time_frames']),
        (config['phonemes_length'], config['phonemes_encoding_size']),
        config['acoustic_model_cfg'],
        device
    )

    mel2lin_components = mel2lin_utils.create_model_components(
        (config['mel_spec_freq_bins'], config['mel_spec_time_frames']),
        config['linear_spec_freq_bins'],
        config['mel_to_lin_model_cfg'],
        device
    )

    acoustic_components.load_from_path(config['acoustic_model_checkpoint'])
    mel2lin_components.load_from_path(config['mel_to_lin_model_checkpoint'])

    logging.info('Preparing the inference model...')

    inference_model = _create_inference_model(
        acoustic_components,
        mel2lin_components,
        config['mel_spec_time_frames'],
        config['gst_mode']
    )

    acoustic_components.eval()
    mel2lin_components.eval()
    inference_model.eval()

    example_phonemes = torch.ones(1, config['phonemes_length'], config['phonemes_encoding_size'])
    example_phonemes /= config['phonemes_encoding_size']

    if config['gst_mode'] == 'none':
        example_input = (
            example_phonemes,
        )

    elif config['gst_mode'] == 'weights':
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

    torchscript = torch.jit.trace_module(inference_model, {'forward': (example_input,)})

    torch.jit.save(torchscript, config['output_path'])

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
