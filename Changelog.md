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