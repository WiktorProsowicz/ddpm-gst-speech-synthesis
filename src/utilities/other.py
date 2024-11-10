# -*- coding: utf-8 -*-
"""Contains utility functions for various purposes."""

import subprocess
import os
import logging
import requests
import gdown


GDRIVE_DOWNLOAD_URL = 'https://drive.google.com/uc?id=1WGdcR7cBiv5md0e7VXEXe3mtox5xKIE_'


def download_phoneme_alignments(destination_path: str):
    """Downloads the phoneme alignments for the LJSpeech dataset.

    See data.preprocessing.alignments for more details.

    Args:
        destination_path: Path to the directory where the alignments should be saved.
    """

    arch_path = os.path.join(destination_path, 'ljspeech_1.1_alignments.tar.bz2')

    gdown.download(GDRIVE_DOWNLOAD_URL, arch_path, quiet=False)

    try:
        subprocess.run(['bzip2', '-d', arch_path], check=True)

        tar_path = arch_path[:-4]

        subprocess.run(['tar', '-xf', tar_path, '-C', destination_path], check=True)

        subprocess.run(['rm', tar_path], check=True)

    except subprocess.CalledProcessError as proc_err:
        logging.critical("Failed to download the phoneme alignments: %s", proc_err)
