# -*- coding: utf-8 -*-
"""Script containing functions for managing the project workspace.

Available functions and corresponding arguments are described in the
main function as well as in the doc strings of the functions.
"""
import argparse
import logging
import os
import pathlib
import subprocess
import sys
import venv
from os import path

HOME_PATH = pathlib.Path(__file__).absolute().parent.as_posix()


def run_repository_checks():
    """Runs static checks on the files found in the repository.

    The purpose is to determine whether the current code is qualifies to be
    merged into main branch. This together with unit tests is an obligatory
    condition so that the code can be published on the main project branch.

    The checks include:
        - running pre-commit hooks specified in .pre-commit-config.yaml
    """

    try:

        logging.info('Running pre-commit hooks.')

        subprocess.run(['pre-commit', 'run', '--all-files'], check=True)

    except subprocess.CalledProcessError as proc_error:
        logging.critical('Static checks failed: %s', proc_error)


def setup_venv() -> None:
    """Sets up the virtual environment."""

    venv_path = path.join(HOME_PATH, 'venv')

    if os.path.exists(venv_path):
        logging.warning(
            "Directory '%s' already exists. If you are sure you want to" +
            ' replace it with a new environment, delete it and run again.', venv_path)
        return

    venv.create(venv_path, with_pip=True, upgrade_deps=True, clear=False, system_site_packages=True)

    logging.info(
        "Successfully created a virtual environment at directory '%s'", venv_path)
    logging.info(
        "You can now activate the environment with 'source ./venv/bin/activate'.")
    logging.info(
        "Then type 'python3 -m pip install -r requirements.txt' to install dependencies.")
    logging.info("Then type 'deactivate' to deactivate the environment.")


def _get_arg_parser() -> argparse.ArgumentParser:
    """Returns an argument parser for the script."""

    functions_descriptions = '\n'.join(
        [f'{func.__name__}: {func.__doc__.splitlines()[0]}'
         for func in _get_available_functions()])

    program_desc = ('Script contains functions helping with project management.\n' +
                    'Available functions:\n\n' +
                    f'{functions_descriptions}')

    arg_parser = argparse.ArgumentParser(
        description=program_desc, formatter_class=argparse.RawDescriptionHelpFormatter)

    arg_parser.add_argument(
        'function_name', help='name of the function to be used')
    arg_parser.add_argument(
        'args', nargs='*', help='positional arguments for the function')

    return arg_parser


def main(function: str, *args) -> None:
    """Main function delegating the flow to other ones.

    Args:
        function (str): Name of the function to be called.
    """

    for available_func in _get_available_functions():
        if available_func.__name__ == function:
            available_func(*args)  # type: ignore
            return

    logging.error("Couldn't find the function '%s'.", function)


def _is_run_from_venv():
    """Tells whether the user runs the script from a vierual environment."""

    return sys.prefix != sys.base_prefix


def _get_available_functions():
    """Returns a list of callable setup functions.

    The set of available functions depends on whether the file is run
    inside of a virtual environment or globally (i.e. first time in
    order to setup a venv).
    """

    available_functions_env = [run_repository_checks]

    available_functions_glob = [setup_venv]

    if _is_run_from_venv():
        return available_functions_env

    return available_functions_glob


if __name__ == '__main__':

    sys.path.append(os.path.join(HOME_PATH, 'src'))

    parser = _get_arg_parser()
    arguments = parser.parse_args()

    if _is_run_from_venv():
        from utilities import logging_utils
        logging_utils.setup_logging()

    main(arguments.function_name, *arguments.args)
