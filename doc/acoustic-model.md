# Acoustic Model

The Acoustic Model is an alternative acoustic model (!) for the DDPM-GST-Speech-Gen. It maps the input encoded phonemes into a mel-spectrogram and is able to leverage the style embedding from the GST module.

## Model description

The architecture of the backbone of the model is largely inspired by the [FastSpeech](https://arxiv.org/abs/1905.09263) and [TransformerTTS](https://arxiv.org/abs/1809.08895). Its primary components are:

- Encoder:
    - creates phoneme embeddings and than extracts enhanced representation from them using Self-Attention mechanism
    - conditions the representations extraction on the provided GST embedding
- Duration Predictor, Length Regulator
- Decoder:
    - decodes the stretched phoneme representations into mel-spectrogram frames
- GST provider:
    - initializes a set Global Style Tokens
- Reference Embedder:
    - encodes the reference audio and runs it through the GST attention module

## Changelog

### 1.0.0

Both the model training pipeline and components are implemented. The model supports GST conditioning.
