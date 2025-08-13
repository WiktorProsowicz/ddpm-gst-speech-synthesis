# -*- coding: utf-8 -*-
"""Downloads and prepares the dataset for GST Predictor model."""
import csv
import json
import logging
import os
import tqdm

import pytorch_pretrained_bert as bert_lib
import torch
import torch_dev_utils as tdu
from torchvision import transforms

from data.preprocessing import text as text_prep
from models.acoustic import utils as acoustic_utils
from utilities import inference as inference_utils
from utilities import logging_utils
from utilities import scripts_utils


DEFAULT_CONFIG = {
    'processed_ds_path': scripts_utils.CfgRequired(),
    'ljspeech_metadata_path': scripts_utils.CfgRequired(),
    'output_path': scripts_utils.CfgRequired(),
    # PAth to the json config used to train the acoustic model.
    'acoustic_model_training_cfg': scripts_utils.CfgRequired(),
    # E.g. 'ckpt_10'
    'acoustic_model_checkpoint': scripts_utils.CfgRequired(),
}


def _get_acoustic_components(config, ds_metadata, device: torch.device):

    with open(config['acoustic_model_training_cfg'], 'r', encoding='utf-8') as cfg_f:
        acoustic_cfg = json.load(cfg_f)

    acoustic_model_comps = acoustic_utils.create_model_components(
        (80, ds_metadata['output_spectrogram_length']),
        (ds_metadata['phonemes_sequence_length'],
         len(text_prep.ENHANCED_MFA_ARP_VOCAB)),
        acoustic_cfg['model'],
        device)

    ckpt_handler = tdu.serialization.ModelCheckpointHandler(
        acoustic_cfg['training']['checkpoints_path'],
        device,
        False)

    ckpt_handler, _, _ = ckpt_handler.get_checkpoint(
        config['acoustic_model_checkpoint'],
        acoustic_model_comps)

    assert acoustic_model_comps.embedder is not None

    acoustic_model_comps.eval()

    return acoustic_model_comps


def _get_samples_transcripts(config):

    with open(config['ljspeech_metadata_path'], 'r', encoding='utf-8') as meta_f:
        ljspeech_metadata = list(csv.reader(meta_f,
                                            delimiter='|',
                                            quoting=csv.QUOTE_NONE))

    sample_to_transcript = {sample[0]: sample[2]
                            for sample in ljspeech_metadata}

    return sample_to_transcript


def _save_ds_stats(config):  # pylint: disable=too-many-locals

    sample_names = filter(lambda path: '.pt' in path,
                          os.listdir(config['processed_ds_path']))
    sample_names = list(sample_names)

    with open(config['acoustic_model_training_cfg'], 'r', encoding='utf-8') as cfg_f:
        acoustic_cfg = json.load(cfg_f)

    emb_maximum = torch.zeros(acoustic_cfg['model']['d_model'])
    emb_minimum = torch.zeros(acoustic_cfg['model']['d_model'])
    w_maximum = torch.zeros(acoustic_cfg['model']['gst']['n_tokens'])
    w_minimum = torch.zeros(acoustic_cfg['model']['gst']['n_tokens'])

    for sample_name in sample_names:
        sample_path = os.path.join(config['output_path'], sample_name)
        _, _, _, embedding, weights = torch.load(sample_path, weights_only=True)

        emb_maximum = torch.maximum(emb_maximum, embedding)
        emb_minimum = torch.minimum(emb_minimum, embedding)
        w_maximum = torch.maximum(w_maximum, weights)
        w_minimum = torch.minimum(w_minimum, weights)

    emb_scale_shift = -emb_minimum
    emb_scale_factor = 1.0 / (emb_maximum - emb_minimum)
    w_scale_shift = -w_minimum
    w_scale_factor = 1.0 / (w_maximum - w_minimum)

    os.makedirs(os.path.join(config['output_path'], 'stats'), exist_ok=True)
    stats_path = os.path.join(
        config['output_path'], 'stats', 'gst_embedding_stats.pt')

    torch.save((emb_scale_factor, emb_scale_shift,
                w_scale_factor, w_scale_shift), stats_path)


def main(config):  # pylint: disable=too-many-locals
    """Downloads the dataset."""

    metadata_path = os.path.join(config['processed_ds_path'], 'metadata.json')
    with open(metadata_path, 'r', encoding='utf-8') as cfg_f:
        metadata = json.load(cfg_f)

    sample_to_transcript = _get_samples_transcripts(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info('Loading the acoustic model.')

    acoustic_model_comps = _get_acoustic_components(config, metadata, device)

    logging.info('Loading the pre-trained BERT model.')

    tokenizer = bert_lib.BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = bert_lib.BertModel.from_pretrained('bert-base-uncased')
    bert_model = bert_model.to(device).eval()

    text_transform = transforms.Compose([
        text_prep.G2PTransform(),
        text_prep.OneHotEncodeTransform(
            text_prep.ENHANCED_MFA_ARP_VOCAB),
    ])

    logging.info('Serializing the GST Predictor\'s dataset.')

    sample_names = filter(lambda path: '.pt' in path,
                          os.listdir(config['processed_ds_path']))
    sample_names = list(sample_names)

    for sample_name in tqdm.tqdm(sample_names, 'Serializing samples', total=len(sample_names)):

        spectrogram, _, _, _, s_mask, = torch.load(
            os.path.join(config['processed_ds_path'], sample_name),
            weights_only=True)

        sample_key = sample_name.split('.')[0]

        bert_embeddings, phonemes = inference_utils.obtain_gst_predictor_inputs(
            sample_to_transcript[sample_key],
            text_transform,
            tokenizer,
            bert_model,
            device)

        n_to_pad = metadata['phonemes_sequence_length'] - bert_embeddings.shape[0]

        bert_embeddings = torch.nn.functional.pad(
            bert_embeddings, (0, 0, 0, n_to_pad))

        phonemes = torch.nn.functional.pad(
            phonemes, (0, 0, 0, n_to_pad))

        phonemes_mask = torch.logical_not(
            inference_utils.create_transcript_mask(phonemes).to(torch.bool))

        spectrogram = torch.unsqueeze(spectrogram, dim=0).to(device)
        phonemes = torch.unsqueeze(phonemes, dim=0).to(device)
        phonemes_mask = torch.unsqueeze(phonemes_mask, dim=0).to(device)
        s_mask = torch.unsqueeze(s_mask, dim=0).to(device)

        with torch.no_grad():
            enhanced_phonemes = acoustic_model_comps.encoder.run_basic_blocks(
                phonemes, phonemes_mask)
            gst_weights = acoustic_model_comps.embedder.obtain_gst_weights(spectrogram,
                                                                           s_mask)
            gst_embedding = acoustic_model_comps.embedder(spectrogram, s_mask)

        sample = (enhanced_phonemes.squeeze(dim=0).to('cpu'),
                  gst_weights.squeeze(dim=0).to('cpu'),
                  phonemes_mask.squeeze(dim=0).to('cpu'),
                  gst_embedding.squeeze(dim=0).to('cpu'))

        torch.save(sample, os.path.join(config['output_path'], sample_name))

    logging.info('Calculating the dataset statistics.')

    _save_ds_stats(config)


if __name__ == '__main__':

    logging_utils.setup_logging()

    configuration = scripts_utils.try_obtain_cfg_from_cl(
        'Prepares the dataset for GST Predictor model.',
        DEFAULT_CONFIG)

    main(configuration)
