"""Temporal Transformer Encoder regression model for four-score cognitive estimation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from cognitive_state.data.constants import SCORE_NAMES
from cognitive_state.models.fusion import ModalityFusion

NUM_SCORES: int = len(SCORE_NAMES)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017).

    Adds fixed position signals to token embeddings so the encoder can
    distinguish sequence positions without trainable parameters.

    Input shape:  ``(batch, seq, d_model)``
    Output shape: ``(batch, seq, d_model)``
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self._drop = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        # Slice div_term to handle both even and odd d_model.
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional signals and apply dropout.

        Args:
            x: Float tensor ``(batch, seq, d_model)``.

        Returns:
            Float tensor ``(batch, seq, d_model)``.
        """
        x = x + self.pe[:, : x.size(1)]  # type: ignore[index]
        return self._drop(x)


class TemporalTransformer(nn.Module):
    """Temporal Transformer Encoder for four-score cognitive state regression.

    Architecture:
        1. ``ModalityFusion``  — projects raw features to *d_model*.
        2. ``PositionalEncoding`` — adds sinusoidal position signals.
        3. ``TransformerEncoder`` — *num_layers* self-attention + FFN layers.
        4. Temporal mean pooling — aggregates the sequence to one vector.
        5. Regression head — two-layer MLP + Sigmoid bounded to ``[0, 1]``.

    Input shape:  ``(batch, seq, input_features)``
    Output shape: ``(batch, NUM_SCORES)`` — each value in ``[0.0, 1.0]``
    """

    def __init__(
        self,
        input_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_scores: int = NUM_SCORES,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self._fusion = ModalityFusion(input_features, d_model, dropout)
        self._pos_enc = PositionalEncoding(d_model, dropout, max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self._encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self._head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_scores),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run the full encoder and regression pipeline.

        Args:
            x: Float tensor ``(batch, seq, input_features)``.

        Returns:
            Float tensor ``(batch, num_scores)`` with all values in ``[0, 1]``.
        """
        x = self._fusion(x)     # (batch, seq, d_model)
        x = self._pos_enc(x)    # (batch, seq, d_model)
        x = self._encoder(x)    # (batch, seq, d_model)
        x = x.mean(dim=1)       # (batch, d_model)
        return self._head(x)    # (batch, num_scores)
