# -*- coding: utf-8 -*-
"""Contains common utilities used by the scripts."""
import argparse
import json
import logging
import sys
from dataclasses import dataclass
from typing import Dict
from typing import Optional
from typing import Any


class CfgRequired:
    """Indicates that a field is required in the configuration.

    If it is encountered in the script's configuration, the script caller should be notified and
    asked to provide the proper configuration.
    """


@dataclass
class CfgOptional:
    """Indicates that a field is optional in the configuration.

    Optional fields carry a default value but it is allowed not to provide them in a configuration. 
    """

    default_value: Any


def _merge_configs(default_config: Dict, user_config: Dict) -> Dict:
    """Merges the default configuration with the user configuration."""

    config = {key: default_config[key] for key in default_config}

    for key in default_config:

        if isinstance(default_config[key], dict):
            config[key] = _merge_configs(default_config[key], user_config.get(key, {}))

        elif isinstance(default_config[key], CfgOptional):

            if key not in user_config:
                config[key] = default_config[key].default_value

            elif user_config[key] is None:
                config[key] = None

            else:
                config[key] = user_config[key]

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

        elif isinstance(config[key], CfgOptional):
            config[key] = config[key].default_value

            if isinstance(config[key], dict):
                _validate_config(config[key])

        elif isinstance(config[key], dict):
            _validate_config(config[key])


def _sanitize_config_to_dump(config: Dict) -> Dict:

    sanitized_config = {}

    for key in config:
        if isinstance(config[key], CfgOptional):
            sanitized_config[key] = config[key].default_value

            if isinstance(sanitized_config[key], dict):
                sanitized_config[key] = _sanitize_config_to_dump(sanitized_config[key])

        elif isinstance(config[key], dict):
            sanitized_config[key] = _sanitize_config_to_dump(config[key])

        elif isinstance(config[key], CfgRequired):
            sanitized_config[key] = "REQUIRED"

        else:
            sanitized_config[key] = config[key]

    return sanitized_config


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
            with open(config_path, 'r', encoding='utf-8') as config_file:
                config = _merge_configs(config, json.load(config_file))

        except FileNotFoundError:
            logging.critical(
                'The configuration file was not found at the provided path: %s',
                config_path)
            sys.exit(1)

    _validate_config(config)

    return config


def try_obtain_cfg_from_cl(script_description: str, default_config: Dict) -> Dict:
    """Composes a script configuration from the provided CL args.

    The expected arguments and their role are unified across the scripts.

    Args:
        script_description: Description of the script's purpose.
        default_config: Default configuration for the script.
    """

    arg_parser = argparse.ArgumentParser(
        description=script_description)

    arg_parser.add_argument(
        '--config_path',
        type=str,
        help='Path to the folder containing configuration files.'
    )

    arg_parser.add_argument(
        '--dump_default_cfg',
        type=str,
        help='Dump the default configuration to the provided path and exit.',
        required=False
    )

    args = arg_parser.parse_args()

    if args.dump_default_cfg:
        with open(args.dump_default_cfg, 'w', encoding='utf-8') as config_file:
            json.dump(_sanitize_config_to_dump(default_config), config_file, indent=4)
        sys.exit(0)

    return try_load_user_config(args.config_path, default_config)
