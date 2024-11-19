# -*- coding: utf-8 -*-
"""Contains definition of custom logging tools based on logging lib."""
import logging.config
import os
import pathlib
from typing import Any
from typing import Dict

import yaml  # type: ignore

UTILITIES_HOME = pathlib.Path(__file__).absolute().parent.as_posix()
LOGGING_CONFIG_PATH = os.path.join(UTILITIES_HOME, 'res', 'logging_cfg.yaml')


def setup_logging() -> None:
    """Sets up project-wide logging configuration.

    This function should be called at the
    beginning of the scripts run from the console.
    """

    logging_config = _get_logging_config()

    logging.config.dictConfig(logging_config)


def _get_logging_config() -> Dict[str, Any]:
    """Creates a global logging configuration.

    Returns:
        Compiled configuration ready to be loaded as a configuration
        dictionary to the logging module.
    """

    custom_formatters = {'_ColorFormatter': _ColorFormatter}

    with open(LOGGING_CONFIG_PATH, 'r', encoding='utf-8') as config_file:
        config_dict = yaml.safe_load(config_file.read())

    for _, formatter in config_dict['formatters'].items():
        if formatter['()'] in custom_formatters:
            formatter['()'] = custom_formatters[formatter['()']]

    return config_dict


class _ColorFormatter(logging.Formatter):
    """Adds color to the log messages.

    This is the default formatter used by the sound-processing modules.
    """

    _COLORS = {
        'DEBUG': '\033[32m',
        'INFO': '\033[36m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[31;1m',
    }

    _END_COLOR = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        """Overrides the default method of `Formatter` class."""

        pre_formatted = super().format(record)

        return f'{self._COLORS[record.levelname]}{pre_formatted}{self._END_COLOR}'


if __name__ == '__main__':

    setup_logging()

    logging.debug('This is a debug message.')
    logging.info('This is an info message.')
    logging.warning('This is a warning message.')
    logging.error('This is an error message.')
    logging.critical('This is a critical message.')
