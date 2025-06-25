# -*- coding: utf-8 -*-
"""Loads trained model components and runs inference on a given input.

For script's configuration, see `DEFAULT_CONFIG` constant.
"""
import json
import logging
import os
from typing import Tuple

import torch
import torch_dev_utils as tdu
import torchaudio
from torchaudio.prototype.pipelines import HIFIGAN_VOCODER_V3_LJSPEECH as hifigan_bundle

from models.acoustic import utils as acoustic_utils
from models.gst_predictor import utils as gst_utils
from utilities import diffusion as diff_utils
from utilities import inference
from utilities import logging_utils
from utilities import scripts_utils

DEFAULT_CONFIG = {
    # Path to the json configuration of the acoustic model training script.
    'acoustic_training_cfg': scripts_utils.CfgRequired(),
    # The ID of the acoustic checkpoint to load
    'acoustic_ckpt': scripts_utils.CfgRequired(),
    # Path to a chosen preprocessed sample as expected by the acoustic model trainer.
    'acoustic_input_sample': scripts_utils.CfgRequired(),
    # Path to the json configuration of the GST Predictor training script.
    'gst_pred_training_cfg': scripts_utils.CfgOptional(None),
    # The ID of the GST Predictoe checkpoint to load
    'gst_pred_ckpt': scripts_utils.CfgOptional(None),
    # Path to a chosen preprocessed sample as expected by the GST Predictor trainer.
    'gst_pred_input_sample': scripts_utils.CfgOptional(None),
    # Tells how much from the deterministically calculated GST embedding to use in the final one
    'deterministic_gst_weight': scripts_utils.CfgOptional(None),

    'output_path': scripts_utils.CfgRequired()
}


def _load_acoustic_model_components(acoustic_config,
                                    ckpt: str,
                                    output_spec_shape: Tuple[int, int],
                                    input_phonemes_shape: Tuple[int, int],
                                    device: torch.device):

    ckpt_handler = tdu.serialization.ModelCheckpointHandler(
        acoustic_config['training']['checkpoints_path'],
        device,
        missing_modules_strict=False
    )

    model_comps = acoustic_utils.create_model_components(
        output_spec_shape,
        input_phonemes_shape,
        acoustic_config['model'],
        device
    )

    acoustic_model_comps, _, _ = ckpt_handler.get_checkpoint(ckpt, model_comps)

    return acoustic_model_comps


def _load_gst_pred_components(gst_pred_config,
                              ckpt: str,
                              input_phonemes_shape: Tuple[int, int],
                              gst_embedding_size: int,
                              gst_weights_size: int,
                              device: torch.device):

    ckpt_handler = tdu.serialization.ModelCheckpointHandler(
        gst_pred_config['training']['checkpoints_path'],
        device,
        missing_modules_strict=False)

    model_comps = gst_utils.create_model_components(
        input_phonemes_shape,
        gst_embedding_size,
        gst_weights_size,
        gst_pred_config['model'],
        device)

    gst_pred_model_comps, _, _ = ckpt_handler.get_checkpoint(ckpt, model_comps)

    return gst_pred_model_comps


def _get_acoustic_inference_model(config, device):

    exp_spec, input_phonemes, _, _, _ = torch.load(config['acoustic_input_sample'],
                                                   map_location=device,
                                                   weights_only=True)

    with open(config['acoustic_training_cfg'], 'r', encoding='utf-8') as cfg_f:
        acoustic_cfg = json.load(cfg_f)

    logging.info('Loading the acoustic model...')

    acoustic_comps = _load_acoustic_model_components(
        acoustic_cfg, config['acoustic_ckpt'],
        (exp_spec.shape[0], exp_spec.shape[1]),
        (input_phonemes.shape[0], input_phonemes.shape[1]),
        device)

    acoustic_comps.eval()

    logging.info('Loading the vocoder...')

    vocoder = hifigan_bundle.get_vocoder().to(device)

    vocoder.eval()

    logging.info('Composing the inference model...')

    return inference.InferenceAcousticModel(acoustic_comps,
                                            vocoder,
                                            config['deterministic_gst_weight'],
                                            ).to(device)


def _get_gst_predictor_inference_model(config, device):

    logging.info('Loading the GST predictor components...')

    assert config['gst_pred_training_cfg'] is not None
    assert config['gst_pred_ckpt'] is not None
    assert config['gst_pred_input_sample'] is not None

    phoneme_repr, _, bert_embeddings, exp_gst_emb, exp_gst_w = torch.load(
        config['gst_pred_input_sample'],
        map_location=device,
        weights_only=True)
    bert_embeddings = bert_embeddings.unsqueeze(0).to(device)

    with open(config['gst_pred_training_cfg'], 'r', encoding='utf-8') as cfg_f:
        gst_predictor_cfg = json.load(cfg_f)

    gst_pred_comps = _load_gst_pred_components(
        gst_predictor_cfg,
        config['gst_pred_ckpt'],
        (phoneme_repr.shape[0], phoneme_repr.shape[1]),
        exp_gst_emb.shape[0],
        exp_gst_w.shape[0],
        device)

    gst_pred_comps.eval()

    logging.info('Composing the inference model...')

    diff_cfg = gst_predictor_cfg['training']['diffusion']
    diff_handler = diff_utils.DiffusionHandler(
        diff_utils.LinearScheduler(diff_cfg['beta_min'],
                                   diff_cfg['beta_max'],
                                   diff_cfg['n_steps']),
        device
    )

    scaling_values = torch.load(
        os.path.join(
            gst_predictor_cfg['data']['dataset_path'],
            'stats',
            'gst_embedding_stats.pt'),
        map_location=device
    )

    return inference.InferenceGSTPredictor(gst_pred_comps,
                                           diff_handler,
                                           scaling_values,
                                           diff_cfg['guidance_scale']
                                           ).to(device)


def main(config):
    """Loads the model and runs inference."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, input_phonemes, _, p_mask, _ = torch.load(
        config['acoustic_input_sample'],
        map_location=device,
        weights_only=True
    )

    acoustic_inference_model = _get_acoustic_inference_model(config, device)

    if config['gst_pred_training_cfg'] is not None:
        phoneme_repr, phoneme_mask, bert_embeddings, _, _ = torch.load(
            config['gst_pred_input_sample'],
            map_location=device,
            weights_only=True
        )

        gst_pred_inference_model = _get_gst_predictor_inference_model(config, device)
        gst_weights, gst_emb = gst_pred_inference_model(  # pylint:disable=not-callable
            phoneme_repr.unsqueeze(0).to(device),
            bert_embeddings.unsqueeze(0).to(device),
            phoneme_mask.unsqueeze(0).to(device),
        )
    else:
        gst_weights, gst_emb = None, None

    logging.info('Running inference...')

    with torch.no_grad():
        waveform = acoustic_inference_model(  # pylint:disable=not-callable
            input_phonemes.unsqueeze(0).to(device),
            p_mask.unsqueeze(0).to(device),
            gst_weights,
            gst_emb
        )

    logging.info("Saving the output waveform to '%s'", config['output_path'])
    torchaudio.save(config['output_path'], waveform[0].cpu(), 22050)


if __name__ == '__main__':

    logging_utils.setup_logging()

    configuration = scripts_utils.try_obtain_cfg_from_cl(
        'Runs inference of the system.',
        DEFAULT_CONFIG)

    main(configuration)
