# -*- coding: utf-8 -*-
"""Loads trained model components and runs inference on a given input.

The script preprocesses and encodes the input text, loads the acoustic model
and mel2linear converter.

For script's configuration, see `DEFAULT_CONFIG` constant.
"""
import os
import logging
import json

import torch
import torchaudio
from torchvision import transforms
from torchaudio.prototype.pipelines import HIFIGAN_VOCODER_V3_LJSPEECH as hifigan_bundle

from data.preprocessing import text as text_prep
from utilities import logging_utils
from utilities import scripts_utils
from utilities import diffusion as diff_utils
from utilities import inference

DEFAULT_CONFIG = {
    'compiled_model_path': scripts_utils.CfgRequired(),
    'input_text': scripts_utils.CfgRequired(),
    'output_path': scripts_utils.CfgRequired()
}


def _obtain_style_embedding(phoneme_representations: torch.Tensor,
                            gst_predictor_encoder: torch.jit.ScriptModule,
                            gst_predictor_decoder: torch.jit.ScriptModule,
                            config,
                            device: torch.device) -> torch.Tensor:

    param_scheduler = diff_utils.LinearScheduler(config['diff_beta_min'],
                                                 config['diff_beta_max'],
                                                 config['diff_timesteps'])

    diffusion_handler = diff_utils.DiffusionHandler(param_scheduler, device)

    with torch.no_grad():

        phoneme_embedding = gst_predictor_encoder(phoneme_representations)

        noised_style_embedding = torch.randn(1, config['style_embedding_size'], device=device)

        for diff_timestep in reversed(range(config['diff_timesteps'])):

            predicted_noise = gst_predictor_decoder(
                noised_style_embedding,
                torch.tensor([diff_timestep], device=device),
                phoneme_embedding
            )

            noised_style_embedding = diffusion_handler.remove_noise(noised_style_embedding,
                                                                    predicted_noise,
                                                                    diff_timestep)

    return noised_style_embedding


def main(config):  # pylint: disable=too-many-locals
    """Loads the model and runs inference."""

    metadata_path = os.path.join(config['compiled_model_path'], 'metadata.json')
    with open(metadata_path, 'r', encoding='utf-8') as metadata_file:
        metadata = json.load(metadata_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info('Loading the compiled model...')

    acoustic_encoder_path = os.path.join(config['compiled_model_path'], 'acoustic_encoder.pt')
    acoustic_encoder = torch.jit.load(acoustic_encoder_path).to(device)

    acoustic_decoder_path = os.path.join(config['compiled_model_path'], 'acoustic_decoder.pt')
    acoustic_decoder = torch.jit.load(acoustic_decoder_path).to(device)

    acoustic_encoder.eval()
    acoustic_decoder.eval()

    if metadata['gst_predictor_cfg'] is not None:

        logging.info('Loading the GST predictor components...')

        encoder_path = os.path.join(config['compiled_model_path'], 'gst_predictor_encoder.pt')
        decoder_path = os.path.join(config['compiled_model_path'], 'gst_predictor_decoder.pt')

        gst_predictor_encoder = torch.jit.load(encoder_path).to(device)
        gst_predictor_decoder = torch.jit.load(decoder_path).to(device)

        gst_predictor_encoder.eval()
        gst_predictor_decoder.eval()

        def get_style_emb(phoneme_representations):
            return _obtain_style_embedding(
                phoneme_representations,
                gst_predictor_encoder,
                gst_predictor_decoder,
                metadata['gst_predictor_cfg'],
                device
            )

    else:
        def get_style_emb(_):
            return None

    logging.info('Logging HiFi-GAN vocoder...')
    vocoder = hifigan_bundle.get_vocoder().to(device)

    logging.info('Transforming the input text to phonemes...')
    all_input_phonemes = text_prep.G2PTransform()(config['input_text'])

    logging.debug('Input phonemes: %s', all_input_phonemes)

    phonemes_transform = transforms.Compose([
        text_prep.PadSequenceTransform(metadata['input_phonemes_length']),
        text_prep.OneHotEncodeTransform(text_prep.ENHANCED_MFA_ARP_VOCAB)
    ])

    logging.info('Running inference...')

    total_output_waveform = None

    for run_idx in range((len(all_input_phonemes) // metadata['input_phonemes_length']) + 1):
        input_phonemes = all_input_phonemes[
            run_idx * metadata['input_phonemes_length']:
            (run_idx + 1) * metadata['input_phonemes_length']
        ]

        input_phonemes = phonemes_transform(input_phonemes).unsqueeze(0).to(device)

        with torch.no_grad():
            transcript_mask = inference.create_transcript_mask(input_phonemes)
            phoneme_representations = acoustic_encoder(input_phonemes)

            style_embedding = get_style_emb(phoneme_representations)

            if style_embedding is not None:
                output_spec, log_durations = acoustic_decoder((phoneme_representations,
                                                              transcript_mask,
                                                              style_embedding))

            else:
                output_spec, log_durations = acoustic_decoder((phoneme_representations,
                                                              transcript_mask))

            durations_mask = (log_durations > 0).to(torch.int64)
            durations = (torch.pow(2.0, log_durations) + 1e-4).to(torch.int64) * durations_mask
            total_dur = durations.sum()
            output_spec = output_spec[:, :, :total_dur]

            waveform = vocoder(output_spec)

            if total_output_waveform is None:
                total_output_waveform = waveform[0]

            else:
                total_output_waveform = torch.cat([total_output_waveform, waveform[0]], dim=1)

    logging.info("Saving the output waveform to '%s'", config['output_path'])
    torchaudio.save(config['output_path'], total_output_waveform.cpu(), 22050)


if __name__ == '__main__':

    logging_utils.setup_logging()

    configuration = scripts_utils.try_obtain_cfg_from_cl(
        'Runs inference of the system.',
        DEFAULT_CONFIG)

    main(configuration)
