# -*- coding: utf-8 -*-
"""Creates components of the whole model and compiles a runnable version for inference.

For script's configuration, see `DEFAULT_CONFIG` constant.
"""
import json
import logging
import os
from typing import Tuple

import torch

from data.preprocessing import text as text_prep
from models.acoustic import utils as acoustic_utils
from models.gst_predictor import utils as gst_predictor_utils
from utilities import inference
from utilities import logging_utils
from utilities import scripts_utils

DEFAULT_CONFIG = {
    'acoustic_model_checkpoint': scripts_utils.CfgRequired(),
    # Should be the configuration of the acoustic model used during the training
    'acoustic_model_cfg': scripts_utils.CfgRequired(),
    'phonemes_length': 20,
    'mel_spec_time_frames': 200,
    # Specifies the way the output model should use the Global Style Tokens.
    'use_gst': True,
    # If use_gst is true this should contain the gst predictor's configuration.
    'gst_predictor_cfg': {
        'checkpoint_path': scripts_utils.CfgRequired(),
        'model_cfg': scripts_utils.CfgRequired(),
        'diff_beta_min': 0.0001,
        'diff_beta_max': 0.02,
        'diff_timesteps': 200,
    },
    # Directory where the compiled model's components will be saved.
    'output_path': scripts_utils.CfgRequired()
}


def _compile_acoustic_model(config) -> Tuple[torch.jit.ScriptModule,
                                             torch.jit.ScriptModule,
                                             torch.jit.ScriptModule]:

    device = torch.device('cpu')

    logging.info('Loading acoustic model components...')

    acoustic_components = acoustic_utils.create_model_components(
        (80, config['mel_spec_time_frames']),
        (config['phonemes_length'], len(text_prep.ENHANCED_MFA_ARP_VOCAB)),
        config['acoustic_model_cfg'],
        device)

    acoustic_components.load_from_path(config['acoustic_model_checkpoint'], device)
    acoustic_components.eval()

    logging.info('Preparing the acoustic inference model...')

    inference_encoder = inference.InferenceAcousticEnc(
        acoustic_components.encoder
    )

    inference_var_reg = inference.InferenceVarianceReg(
        acoustic_components.encoder
    )

    inference_decoder = inference.InferenceAcousticDec(
        acoustic_components.decoder,
        acoustic_components.duration_predictor,
        acoustic_components.length_regulator,
        config['mel_spec_time_frames'])

    inference_encoder.eval()
    inference_var_reg.eval()
    inference_decoder.eval()

    example_phonemes = torch.randint(0, len(text_prep.ENHANCED_MFA_ARP_VOCAB),
                                     (1, config['phonemes_length']))
    example_phonemes = torch.nn.functional.one_hot(  # pylint: disable=not-callable
        example_phonemes,
        len(text_prep.ENHANCED_MFA_ARP_VOCAB)).to(torch.float)

    transcript_mask = inference.create_transcript_mask(example_phonemes)

    logging.info('Tracing the inference model...')

    traced_encoder = torch.jit.script(inference_encoder,
                                      example_inputs={'forward': (example_phonemes,)})

    with torch.no_grad():
        phoneme_representations = inference_encoder(example_phonemes)

    traced_decoder = torch.jit.script(inference_decoder,
                                      example_inputs={'forward': (
                                          phoneme_representations,
                                          transcript_mask)})

    example_style_embedding = torch.randn(
        1, config['acoustic_model_cfg']['d_model'])

    traced_var_reg = torch.jit.script(inference_var_reg,
                                      example_inputs={'forward': (phoneme_representations,
                                                                  example_style_embedding)})

    return traced_encoder, traced_var_reg, traced_decoder


def _compile_gst_predictor(config) -> Tuple[torch.jit.ScriptModule, torch.jit.ScriptModule]:

    device = torch.device('cpu')

    logging.info('Loading the GST predictor components...')

    gst_predictor = gst_predictor_utils.create_model_components(
        (config['phonemes_length'], config['acoustic_model_cfg']['d_model']),
        config['gst_predictor_cfg']['model_cfg'],
        device)

    gst_predictor.load_from_path(config['gst_predictor_cfg']['checkpoint_path'], device)
    gst_predictor.eval()

    logging.info('Tracing the GST predictor...')

    example_phonemes = torch.randn(1,
                                   config['phonemes_length'],
                                   config['acoustic_model_cfg']['d_model'])

    encoder = torch.jit.script(gst_predictor.encoder,
                               example_inputs={'forward': (example_phonemes,)})

    example_noise = torch.randn(1, config['acoustic_model_cfg']['d_model'])

    example_phoneme_embedding = torch.randn(
        1, config['acoustic_model_cfg']['d_model'])

    example_diff_timestep = torch.randint(0, config['gst_predictor_cfg']['diff_timesteps'], (1,))

    decoder = torch.jit.script(gst_predictor.decoder,
                               example_inputs={'forward': (example_noise,
                                                           example_diff_timestep,
                                                           example_phoneme_embedding)})

    return encoder, decoder


def main(config):
    """Loads the model and runs inference."""

    os.makedirs(config['output_path'], exist_ok=True)

    gst_predictor_metadata = None

    if config['use_gst']:
        gst_predictor_metadata = {
            'diff_beta_min': config['gst_predictor_cfg']['diff_beta_min'],
            'diff_beta_max': config['gst_predictor_cfg']['diff_beta_max'],
            'diff_timesteps': config['gst_predictor_cfg']['diff_timesteps'],
            'style_embedding_size': config['acoustic_model_cfg']['d_model']
        }

    compiled_model_metadata = {
        'input_phonemes_length': config['phonemes_length'],
        'gst_predictor_cfg': gst_predictor_metadata
    }

    acoustic_enc, acoustic_var_reg, acoustic_dec = _compile_acoustic_model(config)
    torch.jit.save(acoustic_enc, os.path.join(config['output_path'], 'acoustic_encoder.pt'))
    torch.jit.save(acoustic_dec, os.path.join(config['output_path'], 'acoustic_decoder.pt'))

    if config['use_gst']:
        gst_pred_enc, gst_pred_dec = _compile_gst_predictor(config)
        torch.jit.save(gst_pred_enc,
                       os.path.join(config['output_path'], 'gst_predictor_encoder.pt'))
        torch.jit.save(gst_pred_dec,
                       os.path.join(config['output_path'], 'gst_predictor_decoder.pt'))

        torch.jit.save(acoustic_var_reg, os.path.join(config['output_path'], 'acoustic_var_reg.pt'))

    with open(os.path.join(config['output_path'], 'metadata.json'),
              'w',
              encoding='utf-8') as metadata_file:
        json.dump(compiled_model_metadata, metadata_file, indent=4)

    logging.info("Inference model has been saved to '%s'.", config['output_path'])


if __name__ == '__main__':

    logging_utils.setup_logging()

    configuration = scripts_utils.try_obtain_cfg_from_cl(
        'Compiles the system models and metadata for inference.',
        DEFAULT_CONFIG)

    main(configuration)
