# -*- coding: utf-8 -*-
"""Downloads and prepares the dataset for GST Predictor model."""
import logging
import os
import json

import torch

from utilities import logging_utils
from utilities import scripts_utils
from models.acoustic import utils as acoustic_utils
from data.preprocessing import text as text_prep

DEFAULT_CONFIG = {
    'processed_ds_path': scripts_utils.CfgRequired(),
    'output_path': scripts_utils.CfgRequired(),
    'acoustic_model_checkpoint': scripts_utils.CfgRequired(),
    'acoustic_model_cfg': scripts_utils.CfgRequired(),
}


def main(config):
    """Downloads the dataset."""

    metadata_path = os.path.join(config['processed_ds_path'], 'metadata.json')
    with open(metadata_path, 'r', encoding='utf-8') as cfg_f:
        metadata = json.load(cfg_f)

    logging.info('Loading the acoustic model.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    acoustic_model_comps = acoustic_utils.create_model_components(
        (80, metadata['output_spectrogram_length']),
        (metadata['phonemes_sequence_length'], len(text_prep.ENHANCED_MFA_ARP_VOCAB)),
        config['acoustic_model_cfg'],
        device)

    assert (acoustic_model_comps.gst is not None) and (acoustic_model_comps.embedder is not None)

    acoustic_model_comps.eval()

    logging.info('Serializing the GST Predictor\'s dataset.')

    sample_names = filter(lambda path: '.pt' in path, os.listdir(config['processed_ds_path']))
    sample_names = list(sample_names)

    for sample_idx, sample_name in enumerate(sample_names):
        data_sample_path = os.path.join(config['processed_ds_path'], sample_name)
        spectrogram, phonemes, _, _, _ = torch.load(data_sample_path, weights_only=True)

        spectrogram = torch.unsqueeze(spectrogram, dim=0).to(device)
        phonemes = torch.unsqueeze(phonemes, dim=0).to(device)

        with torch.no_grad():
            enhanced_phonemes = acoustic_model_comps.encoder.run_basic_blocks(phonemes)
            gst_embedding = acoustic_model_comps.embedder(spectrogram, acoustic_model_comps.gst())

        enhanced_phonemes = enhanced_phonemes.squeeze(dim=0).to('cpu')
        gst_embedding = gst_embedding.squeeze(dim=0).to('cpu')

        output_path = os.path.join(config['output_path'], sample_name)
        torch.save((enhanced_phonemes, gst_embedding), output_path)

        if (sample_idx + 1) % 1000 == 0:
            logging.debug('Processed %d samples.', sample_idx + 1)

    logging.info('Calculating the dataset statistics.')

    gst_embedding_mean = torch.zeros((config['acoustic_model_cfg']['d_model']))
    gst_embedding_std = torch.zeros((config['acoustic_model_cfg']['d_model']))

    for sample_name in sample_names:
        sample_path = os.path.join(config['output_path'], sample_name)
        _, embedding = torch.load(sample_path, weights_only=True)

        gst_embedding_mean += embedding

    gst_embedding_mean /= len(sample_names)

    for sample_name in sample_names:
        sample_path = os.path.join(config['output_path'], sample_name)
        _, embedding = torch.load(sample_path, weights_only=True)

        gst_embedding_std += (embedding - gst_embedding_mean) ** 2

    gst_embedding_std /= len(sample_names)
    gst_embedding_std = torch.sqrt(gst_embedding_std)

    os.makedirs(os.path.join(config['output_path'], 'stats'), exist_ok=True)
    stats_path = os.path.join(config['output_path'], 'stats', 'gst_embedding_stats.pt')

    torch.save((gst_embedding_mean, gst_embedding_std), stats_path)


if __name__ == '__main__':

    logging_utils.setup_logging()

    configuration = scripts_utils.try_obtain_cfg_from_cl(
        "Prepares the dataset for GST Predictor model.",
        DEFAULT_CONFIG)

    main(configuration)
