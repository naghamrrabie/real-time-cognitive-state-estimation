"""Modality fusion layer: projects concatenated features into a shared d_model space."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class ModalityFusion(nn.Module):
    """Projects concatenated multimodal features to a unified d_model representation.

    Applies a learned linear projection followed by layer normalisation and
    dropout.  Accepts any sequence length and does not add positional
    information (that is handled by the encoder).

    Input shape:  ``(batch, timesteps, input_features)``
    Output shape: ``(batch, timesteps, d_model)``
    """

    def __init__(
        self,
        input_features: int,
        d_model: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._proj = nn.Linear(input_features, d_model)
        self._norm = nn.LayerNorm(d_model)
        self._drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Project and normalise a batch of feature sequences.

        Args:
            x: Float tensor ``(batch, timesteps, input_features)``.

        Returns:
            Float tensor ``(batch, timesteps, d_model)``.
        """
        return self._drop(self._norm(self._proj(x)))
