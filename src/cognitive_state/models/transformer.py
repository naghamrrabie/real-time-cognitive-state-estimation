"""Temporal Transformer Encoder regression model.

Minimal shape-contract stub. T030 (fusion) and T031 (full transformer encoder)
will replace the body while keeping this constructor and forward signature.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from cognitive_state.data.constants import SCORE_NAMES

NUM_SCORES: int = len(SCORE_NAMES)


class TemporalTransformer(nn.Module):
    """Temporal regression model: (batch, timesteps, features) -> (batch, NUM_SCORES).

    Constructor only requires *input_features*. T031 will add nhead, num_layers,
    and d_model without changing this signature.
    """

    def __init__(
        self,
        input_features: int,
        num_scores: int = NUM_SCORES,
    ) -> None:
        super().__init__()
        self._head = nn.Linear(input_features, num_scores)

    def forward(self, x: Tensor) -> Tensor:
        """Pool over the time axis and project to regression scores.

        Args:
            x: Float tensor of shape ``(batch, timesteps, input_features)``.

        Returns:
            Float tensor of shape ``(batch, num_scores)``.
        """
        return self._head(x.mean(dim=1))
