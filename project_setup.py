# -*- coding: utf-8 -*-
"""Script containing functions for managing the project workspace.

Available functions and corresponding arguments are described in the
main function as well as in the doc strings of the functions.
"""
import argparse
import dataclasses
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import venv
from os import environ
from os import path

HOME_PATH = pathlib.Path(__file__).absolute().parent.as_posix()


@dataclasses.dataclass(frozen=True)
class _TestsRunParams:
    """Contains configuration used while running tests."""

    tests_path: str  # contains test files
    resources_path: str  # contains resources used by tests
    results_path: str  # shall contain test results dump


def _run_tests(config: _TestsRunParams) -> None:
    """Runs a specific kind of tests.

    Args:
        config: Configuration of the tests to be run.
    """

    src_path = path.join(HOME_PATH, 'src')

    coverage_data_file = path.join(config.results_path, '.coverage')
    coverage_stats_dir = path.join(config.results_path, 'coverage_stats')
    tests_report_file = path.join(config.results_path, 'tests_report.xml')

    if not os.path.exists(config.results_path):
        os.makedirs(config.results_path)

    logging.info('Cleaning test results directory...')

    for root, dirs, files in os.walk(config.results_path):

        for file in files:
            os.remove(os.path.join(root, file))
        for directory in dirs:
            shutil.rmtree(os.path.join(root, directory))

    current_env = environ.copy()
    current_env['PYTHONPATH'] = (src_path +
                                 ':' + current_env.get('PYTHONPATH', ''))
    current_env['TEST_RESOURCES'] = config.resources_path
    current_env['TEST_RESULTS'] = config.results_path

    logging.info('Running tests...')

    command = (f'python3 -m coverage run --data-file={coverage_data_file} --source={src_path}'
               f' -m pytest --import-mode=prepend -s {config.tests_path} --tb=short'
               f' --junitxml={tests_report_file} -W ignore::DeprecationWarning'
               f' --rootdir={config.tests_path}')

    subprocess.run(command.split(), check=False, env=current_env)

    logging.info('Generating coverage report...')

    command = (f'python3 -m coverage html --data-file={coverage_data_file}'
               f' --directory={coverage_stats_dir} --omit=*/__init__.py')

    subprocess.run(command.split(), check=False, env=current_env)

    os.remove(coverage_data_file)


def run_unit_tests() -> None:
    """Run available unit tests from the 'tests/unit' directory."""

    tests_path = path.join(HOME_PATH, 'tests')
    resources_path = path.join(HOME_PATH, 'tests', 'res')
    results_path = path.join(HOME_PATH, 'test_results')

    tests_config = _TestsRunParams(tests_path, resources_path, results_path)

    logging.info('Tests configuration:')
    logging.info('\ttests_path:      %s', tests_config.tests_path)
    logging.info('\tresources_path:  %s', tests_config.resources_path)
    logging.info('\tresults_path:    %s', tests_config.results_path)

    _run_tests(tests_config)


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

    venv.create(venv_path, with_pip=True, upgrade_deps=True, clear=False)

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

    available_functions_env = [run_unit_tests, run_repository_checks]

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
