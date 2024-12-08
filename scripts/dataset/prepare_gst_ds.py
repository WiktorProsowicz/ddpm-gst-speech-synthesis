# -*- coding=utf-8 -*-
"""Downloads and prepares the dataset for GST Predictor model."""

import argparse
import logging
import subprocess
import os

import gdown

from utilities import logging_utils


GST_URL = 'https://drive.google.com/uc?id=1VMJnPZokbjHGPP5Fv5dmKOi59UDlPLKB'


def main(output_path: str):
    """Downloads the dataset."""

    if os.path.exists(output_path):
        logging.info('Output path %s already exists.', output_path)
        return

    os.makedirs(output_path, exist_ok=True)

    logging.info('Downloading the dataset...')

    output_arch = os.path.join(output_path, 'ljpseech_1.1_gst_32_phonemes_20x73.tar.bz2')
    gdown.download(GST_URL, output_arch, quiet=False)

    subprocess.run(['bzip2', '-d', output_arch], check=True)

    tar_path = output_arch[:-4]

    subprocess.run(['tar', '-xf', tar_path, '-C', output_path], check=True)

    subprocess.run(['rm', tar_path], check=True)

    logging.info('Dataset written to %s', output_path)


def _get_cl_args() -> argparse.Namespace:

    arg_parser = argparse.ArgumentParser(
        description='Downloads the dataset for GST Predictor model.')

    arg_parser.add_argument(
        '--output_path',
        type=str,
        help='Directory the dataset shall be written to.'
    )

    return arg_parser.parse_args()


if __name__ == '__main__':

    logging_utils.setup_logging()

    args = _get_cl_args()

    main(args.output_path)
