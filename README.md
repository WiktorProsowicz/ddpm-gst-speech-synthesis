# About the project

This repository contains the implementation of a system performing human speech synthesis task. The system is intended to generate expressive, controllable, natural human speech using modern approaches to the GenerativeAI task. In particular, it leverages the concept of Global Style Tokens to learn the style representation in an unsupervised manner. During inference, the style features are predicted using a combination of deterministic and diffusion-based models, which allows for smooth control of the diversity of the generated speech's style.

I strongly recommend you to check out our [paper](https://www.mdpi.com/2079-9292/14/23/4759), which describes the details behind the work. For those who would like to listen to our speech samples, they are available on the [page](https://wiktorprosowicz.github.io/ddpm-gst-speech-synthesis/) dedicated to this project.

## Project structure

```yaml
- docs              # All resources related to the research behind the system
- .devcontainer    # Configuration of the docker environment (see the 'Setup' chapter)
- src:             # Source code
   - data          # Dataset downloading & preprocessing tools
   - layers        # Neural modules used in models' architecture
   - models        # Models' API, training tools, serialization etc
   - utilities     # Utility functions & classes for inference, preprocessing etc
- scripts          # Scripts to be run by the user. In practice, this is the project's public API
```

## Setup

The project is intended to be run within a proper Docker container. There are two setup options:

1. Development
   - the project should be edited preferably within a VSCode's DevContainer
      - this automatically runs the setup scripts for the environment
      - all useful VSCode's extensions are configured for the DevContainer
   - the environment uses configurations from the `.devcontainer` folder
   - `project_setup.py` contains several functions for CI purposes (check `project_setup.py --help`)

2. Runtime
   - the project should be opened within a Docker container compatible with the Docker image the `.devcontainer/Dockerfile` is based on
   - the project requires dependencies from `requirements.txt` to be installed

It is recommended to use all project's functionalities within a virtual environment, no matter which setup option has been chosen.

```
python3.11 project_setup.py setup_venv && source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Usage

To run the scripts from the `scripts` directory one should first ensure the `PYTHONPATH` environment variable is correctly set. This is crucial for the source code to be seen by the scripts without the need to install it as a library. For example:

```
export PYTHONPATH=$PYTHONPATH:/home/devcontainer/workspace/src
python scripts/dataset/prepare_dataset.py
```

## Tutorial

In this chapter, we present several typical use cases so that the user can quickly grasp proper understanding of the project.

It is recommended to keep all experiment runs in a directory of choice, e.g. `tmp`. The tensorboard logs are going to be saved in the `runs` directory.

Each script, if the `--dump_default_cfg` argument is passed, saves its default configuration to the specified file. This can be subsequently edited according to the user's needs.

### Preparing the dataset

1. Create the output directory e.g. `tmp/ljspeech/`
2. Prepare the configuration e.g. `tmp/ljspeech/cfg.json`
3. Run

```
python scripts/dataset/prepare_dataset.py --config_path tmp/ljspeech/cfg.json
```

4. The script creates the `raw`, `processed` and `alignments` dirs. The `processed` directory is used in the training scripts.

### Training the acoustic model

1. Create the output directory, e.g. `tmp/acoustic_model/`.
2. Prepare the configuration (e.g. `tmp/acoustic_model/cfg.json`) and the output path for the checkpoints (e.g. `tmp/acoustic_model/checkpoints/`).
3. In the configuration, pay special attention to the `checkpoints_path`, `run_label` and `dataset_path` parameters.
4. Run:

```
python scripts/training/train_acoustic_model --config_path tmp/acoustic_model/cfg.json
```

### Preparing the gst dataset

1. Create the output directory, e.g. `tmp/ljspeech/gst/`.
2. Prepare the configuration, e.g. `tmp/ljspeech/gst_cfg.json`.
3. Run

```
python scripts/dataset/prepare_gst_ds.py --config_path tmp/ljspeech/gst_cfg.json
```

### Training the GST Predictor model

1. Create the output directory, configuration and checkpoints path, e.g. in `tmp/gst_predictor/`.
2. In the configuration, path special attention to the `dataset_path`. In this case, this should be the `tmp/ljspeech/gst` directory.
3. Run the `scripts/train_gst_predictor.py` script.

### Running the inference

An example configuration:

```json
{
    "acoustic_training_cfg": "tmp/acoustic_model/cfg.json",
    "acoustic_ckpt": "ckpt_5",
    "acoustic_input_sample": "tmp/ljspeech/processed/0001-001.pt",
    "gst_pred_training_cfg": "tmp/gst_predictor/cfg.json",
    "gst_pred_ckpt": "ckpt_20",
    "gst_pred_input_sample": "tmp/ljspeech/gst/0001-001.pt",
    "deterministic_gst_weight": 0.2,

    "output_path": "tmp/output.wav"
}
```

## Changelog

See the Changelog.md file for the project's changes.
