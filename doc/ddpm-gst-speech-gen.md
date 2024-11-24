# DDPM-GST-SPEECH-GEN

The DDPM-GST-Speech-Gen model performs the TTS task based on the Denoising Diffusion Probabilistic Models concept. It transforms a given input text (encoded as ARPA phonemes with additional special tokens) into a mel-spectrogram.

## Model description

The model components are based on the [Diff-TTS](https://arxiv.org/abs/2104.01409) model. They involve:

- Encoder
    - converts the input phonemes into enriched representations
    - extracts the inter-phoneme relationships to support the generation of spectrogram frames
- Duration Predictor:
    - consumes the Encoder's output
    - predicts the log-scale durations of particular phonemes
    - is trained with an additional loss comparing the output with the Ground Truth phoneme durations
- Length Regulator:
    - stretches the Encoder's output frames according to the phoneme durations
    - during training takes the Ground Truth phoneme durations
    - during inference takes the predicted phoneme durations
- Decoder:
    - predicts the noise that has been added to the GT spectrogram based on the input noised sample
    - the generation is conditioned with the stretched Encoder's output and the diffusion time step


## Changelog

### 1.0.0

The model together with its training pipeline and utilities have been implemented. The current model does not support GST and therefore enabled only the diffusion process conditioned by the input phonemes and diffusion timestep.

### 1.1.0

The duration predictor has now residual connections and additional dropout.

The decoder avoids using convolutions with kernel's size equal to 1. Additionally its residual blocks use internal number channels other than the number of channels in the input spectrogram. Each convolutional layer is equipped with activation and dropout to alleviate overfitting.

### 1.2.0

The Encoder has now dropout.

### 1.3.0

Got rid of dilation in Encoder.
