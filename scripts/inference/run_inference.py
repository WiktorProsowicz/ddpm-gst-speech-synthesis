# -*- coding=utf-8 -*-
"""Loads trained model components and runs inference on a given input.

The script preprocesses and encodes the input text, loads the acoustic model
and mel2linear converter.

For script's configuration, see `DEFAULT_CONFIG` constant.
"""

import argparse
import logging

import torch
import torchvision.transforms as transforms
import torchaudio

from utilities import scripts_utils
from utilities import logging_utils
from models.acoustic import utils as acoustic_utils
from models.mel_to_lin_converter import utils as mel2lin_utils
from models import utils as shared_m_utils
from data.preprocessing import text as text_prep

DEFAULT_CONFIG = {
    'compiled_model_path': scripts_utils.CfgRequired(),
    'input_phonemes_length': 20,
    'input_text': scripts_utils.CfgRequired(),
    # Can be one of ('none', 'reference', 'weights')
    'gst_mode': 'none',
    "scale_min": -100.0,
    "scale_max": 41.0,
    'output_path': scripts_utils.CfgRequired(),
    'gst_reference_cfg': {
        # Path to the audio file to use as reference for GST embedding
        'reference_audio_path': None,
        "spectrogram_window_length": 1024,
        "spectrogram_hop_length": 256,
        "n_mels": 80,
        "spec_length": 200,
        "sample_rate": 22050,
    },
    "gst_weights_cfg": {
        "weights_path": None
    }
}


def _scale_spec(spectrogram: torch.Tensor, target_min: float, target_max: float) -> torch.Tensor:
    """Scales the spectrogram to the target range."""

    min_val = spectrogram.min()
    max_val = spectrogram.max()

    return (spectrogram - min_val) / (max_val - min_val) * (target_max - target_min) + target_min


def main(config):
    """Loads the model and runs inference."""

    # if config['gst_weights'] is None and config['reference_audio_path'] is None:
    #     raise ValueError('Either `gst_weights` or `reference_audio_path` must be provided.')

    logging.info("Loading the compiled model...")
    compiled_model = torch.jit.load(config['compiled_model_path'])

    logging.info("Transforming the input text to phonemes...")
    all_input_phonemes = text_prep.G2PTransform()(config['input_text'])

    logging.debug("Input phonemes: %s", all_input_phonemes)

    phonemes_transform = transforms.Compose([
        text_prep.PadSequenceTransform(config['input_phonemes_length']),
        text_prep.OneHotEncodeTransform(text_prep.ENHANCED_MFA_ARP_VOCAB)
    ])

    output_transform = transforms.Compose([
        transforms.Lambda(lambda x: torchaudio.functional.DB_to_amplitude(x, ref=1.0, power=0.5)),
        torchaudio.transforms.GriffinLim(n_fft=1024,
                                         win_length=1024,
                                         hop_length=256,
                                         power=1),
    ])

    logging.info("Running inference...")

    total_output_lin_spec = None

    for run_idx in range((len(all_input_phonemes) // config['input_phonemes_length']) + 1):
        input_phonemes = all_input_phonemes[
            run_idx * config['input_phonemes_length']:
            (run_idx + 1) * config['input_phonemes_length']
        ]

        input_phonemes = phonemes_transform(input_phonemes).unsqueeze(0)

        if config["gst_mode"] == 'reference':

            cfg = config['gst_reference_cfg']

            ref_audio, original_sr = torchaudio.load(cfg['reference_audio_path'])

            ref_speech_transform = transforms.Compose([
                torchaudio.transforms.Resample(original_sr, cfg['sample_rate']),
                torchaudio.transforms.MelSpectrogram(sample_rate=cfg['sample_rate'],
                                                     n_fft=cfg['spectrogram_window_length'],
                                                     win_length=cfg['spectrogram_window_length'],
                                                     hop_length=cfg['spectrogram_hop_length'],
                                                     n_mels=cfg['n_mels']),
                torchaudio.transforms.AmplitudeToDB(),
                transforms.Lambda(lambda x: _scale_spec(x, 0.0, 1.0))
            ])

            ref_speech = ref_speech_transform(ref_audio)

            if (run_idx + 1) * cfg['spec_length'] >= ref_speech.shape[2]:
                ref_speech = ref_speech[:, :,
                                        run_idx * cfg['spec_length']:
                                        (run_idx + 1) * cfg['spec_length']]

            else:
                ref_speech = ref_speech[:, :, :cfg['spec_length']]

            model_input = (input_phonemes, ref_speech)

        elif config["gst_mode"] == 'weights':

            cfg = config['gst_weights_cfg']
            gst_weights = torch.load(cfg['weights_path'], weights_only=True)

            model_input = (input_phonemes, gst_weights.unsqueeze(0))

        else:
            model_input = (input_phonemes,)

        with torch.no_grad():
            output_lin_spec, log_durations = compiled_model(model_input)

            durations_mask = (log_durations > 0).to(torch.int64)
            durations = (torch.pow(2.0, log_durations) + 1e-4).to(torch.int64) * durations_mask
            total_dur = durations.sum()
            output_lin_spec = output_lin_spec[:, :, :total_dur]

            if total_output_lin_spec is None:
                total_output_lin_spec = output_lin_spec

            else:
                total_output_lin_spec = torch.cat([total_output_lin_spec, output_lin_spec], dim=2)

    total_output_lin_spec = _scale_spec(
        total_output_lin_spec,
        config['scale_min'],
        config['scale_max'])

    waveform = output_transform(total_output_lin_spec)

    logging.info("Saving the output waveform to '%s'", config['output_path'])
    torchaudio.save(config['output_path'], waveform, 22050)


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
