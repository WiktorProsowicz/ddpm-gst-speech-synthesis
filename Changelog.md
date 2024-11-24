# Changelog

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

#### 0.4.1

- Added deterministic dataset splitting
- Fixed preparing input phonemes and durations

#### 0.4.2

- Added debug statements to dataset preparation script

### 0.5.0

- Implemented base components of the DDPM-GST-SPEECH_GEN
   - encoder
   - decoder
   - duration predictor
   - length regulator
- Implemented training pipeline with logging, profiling and validation
- Implemented the model training script

#### 0.5.1

- Added GPU support to the DevContainer environment

#### 0.5.2

- Added running full backward diffusion process to the training pipeline
- Added debug statements abound training time

## 1.0.0

- Verified the user scripts
- Added the DDPM-GST-SPEECH-GEN model's overview

#### 1.0.1

- Changed the internal project structure inside the `src` directory
- Switched the profiling in the `train_model` script to be disabled by default
- Fixed a bug related to total predicted phonemes duration exceeding the maximum allowed value
- Enhanced visualization in logging

### 1.1.0

- Enhanced the model's internal structure
- Added small optimizations to the training/validation pass
- Added new controllable parameters to the 'train_model' script

### 1.2.0

- Added dropout to the Encoder module
- Made the dropout in each layer controllable
- Added new options to 'train_model' script
- Extended the full backward diffusion at validation with an additional run for a sample from the train dataset

### 1.3.0

- Added script for splitting processed samples into chunks
- Made spectrograms normalizing optional
- Modified Encoder's architecture
- Fixed bug with fixed Decoder's parameters

#### 1.3.1

- Fixed bug related to serializing split data chunks (order of elements in serialized tuple)
