# -*- coding: utf-8 -*-
"""Contains common utilities used by the scripts."""

from typing import Optional, Dict
import logging
import json
import sys


class CfgRequired:
    """Indicates that a field is required in the configuration.

    If it is encountered in the script's configuration, the script
    caller should be notified and asked to provide the proper configuration.
    """


def _merge_configs(default_config: Dict, user_config: Dict) -> Dict:
    """Merges the default configuration with the user configuration."""

    config = {key: default_config[key] for key in default_config}

    for key in default_config:

        if isinstance(default_config[key], dict):
            config[key] = _merge_configs(default_config[key], user_config.get(key, {}))

        elif key in user_config:
            config[key] = user_config[key]

    return config


def _validate_config(config: Dict):
    """Validates the configuration.

    The function checks if all required fields are present in the configuration.
    """

    for key in config:
        if isinstance(config[key], CfgRequired):
            logging.critical(
                "The field '%s' is required in the configuration. Please provide it.",
                key)
            sys.exit(1)

        elif isinstance(config[key], dict):
            _validate_config(config[key])


def try_load_user_config(config_path: Optional[str], default_config: Dict):
    """Tries to load the user configuration from the given path.

    If the configuration file does not exist, the default configuration is used. Otherwise
    the function tries to merge the provided config with the default one. The configuration
    is validated.

    Args:
        config_path: Path to the user configuration file.
        default_config: Configuration with default values.
    """

    config = default_config

    if config_path:

        try:
            with open(config_path, 'r') as config_file:
                config = _merge_configs(config, json.load(config_file))

        except FileNotFoundError:
            logging.critical(
                "The configuration file was not found at the provided path: %s",
                config_path)
            sys.exit(1)

    _validate_config(config)

    return config
