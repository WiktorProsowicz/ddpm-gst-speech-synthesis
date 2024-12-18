## About the project

This project contains the implementation of a system performing human speech synthesis task. The system is intended to generate expressive, controllable, natural human speech using modern approaches to the GenerativeAI task.

### Project structure

```yaml
- doc              # All resources related to the research behind the system
- .devcontainer    # Configuration of the docker environment (see the 'Setup' chapter)
- src:             # Source code
   - data          # Dataset downloading & preprocessing tools
   - layers        # Neural modules used in models' architecture
   - models        # Models' API, training tools, serialization etc
   - utilities     # Utility functions & classes for inference, preprocessing etc
- scripts          # Scripts to be run by the user
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

See the Changelog.md file for the project's changes.
