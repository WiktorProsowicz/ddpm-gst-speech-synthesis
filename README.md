## About the project

This project contains the implementation of a system performing human speech synthesis task. The system is intended to generate expressive, controllable, natural human speech using modern approaches to the GenerativeAI task.

### Project structure

```yaml
- doc              # All resources related to the research behind the system
- .devcontainer    # Configuration of the docker environment (see the 'Setup' chapter)
```

### Setup

The project is intended to be run within a proper Docker container. There are two setup options:

1. Development
   - the project should be edited preferably within a VSCode's DevContainer
      - this automatically runs the setup scripts for the environment
      - all VSCode's extensions, that come in handy, are configured for the DevContainer
   - the environment uses configurations from the `.devcontainer` folder
   - `project_setup.py` contains several functions for CI purposes (check `project_setup.py --help`)

2. Runtime
   - the project should be opened within a Docker container compatible with the Docker image the `.devcontainer/Dockerfile` is based on
   - the project requires dependencies from `requirements.txt` to be installed

It is recommended to use all project's functionalities within a virtual environment, no matter which setup option has been chosen.

```
python3.11 project_setup.py setup_venv && source venv/bin/activate
pip install -r requirements.txt
```

### Usage

To run the scripts from the `scripts` directory one should first ensure the `PYTHONPATH` environment variable is correctly set. This is crucial for the source code to be seen by the scripts without the need to install it as a library. For example:

```
export PYTHONPATH=$PYTHONPATH:/home/devcontainer/workspace/src
python scripts/train_model.py
```

## Changelog

### 0.1.0

- Established the initial project structure
- Added document with project's overview

### 0.2.0

- Introduced Docker environment setup
- Added project setup script
- Added pre-commit hooks
- Added utilities for logging
- Added project's dependencies config

### 0.3.0

- Added script and tools for downloading and preprocessing input data

### 0.4.0

- Added tools for loading the preprocessed data
- Added support for Tensorboard
- Added utilities for handling scripts configuration

### 0.4.1

- Added deterministic dataset splitting
- Fixed preparing input phonemes and durations