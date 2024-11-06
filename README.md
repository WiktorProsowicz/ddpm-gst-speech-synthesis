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
   - the environment uses configurations from the `.devcontainer` folder
   - `project_setup.py` contains several functions for CI purposes (check `project_setup.py --help`)

2. Runtime
   - the project should be opened within a Docker container compatible with `.devcontainer/Dockerfile`
   - the project requires dependencies from `requirements.txt` to be installed

## Changelog

### 0.0.1

- Established the initial project structure
- Added document with project's overview

### 0.0.2

- Introduced Docker environment setup
- Added project setup script
- Added pre-commit hooks
- Added utilities for logging
- Added project's dependencies config