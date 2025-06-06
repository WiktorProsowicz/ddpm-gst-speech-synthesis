
from typing import Tuple

from layers.shared import fft_block

import torch

BERT_EMBEDDING_SIZE = 768


class DeterministicWeightsPred(torch.nn.Module):
    """Predicts GST weights in a deterministic manner."""

    def __init__(self,
                 input_phonemes_shape: Tuple[int, int],
                 output_weights_size: int,
                 n_blocks: int,
                 fft_conv_channels: int,
                 internal_dim: int,
                 dropout_rate: float):
        """Initializes the GST weights predictor."""

        super().__init__()

        self._prenet = torch.nn.Sequential(
            torch.nn.Linear(input_phonemes_shape[1] + BERT_EMBEDDING_SIZE, internal_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate)
        )

        self._blocks = torch.nn.ModuleList(
            [
                fft_block.FFTBlock((input_phonemes_shape[0],
                                    internal_dim),
                                   16,
                                   dropout_rate,
                                   fft_conv_channels) for _ in range(n_blocks)
            ]
        )

        self._pool_query = torch.nn.Parameter(
            torch.empty(1, 1, internal_dim),
            requires_grad=True
        )

        torch.nn.init.xavier_uniform_(self._pool_query)

        self._pool_att = torch.nn.MultiheadAttention(
            embed_dim=internal_dim,
            num_heads=16,
            dropout=dropout_rate,
            batch_first=True
        )

        self._post_net = torch.nn.Sequential(
            torch.nn.Linear(internal_dim,  output_weights_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate)
        )

    def forward(self,
                phoneme_representations: torch.Tensor,
                bert_embeddings: torch.Tensor,
                phoneme_mask: torch.Tensor) -> torch.Tensor:
        """Predicts GST weights."""

        non_padding_mask = torch.logical_not(phoneme_mask).unsqueeze(-1)

        output = torch.cat(
            [phoneme_representations, bert_embeddings], dim=2
        )

        output = self._prenet(output)

        for block in self._blocks:
            output = block(output, phoneme_mask, non_padding_mask)

        attention_query = self._pool_query.expand(
            phoneme_representations.size(0), -1, -1)
        attention_output, _ = self._pool_att(attention_query,
                                             output,
                                             output,
                                             key_padding_mask=phoneme_mask)

        attention_output = attention_output.squeeze(1)

        return self._post_net(attention_output)
