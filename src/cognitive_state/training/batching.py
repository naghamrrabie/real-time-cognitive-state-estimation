"""Training batch collation for grouped modalities (minimal stub for T052)."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from cognitive_state.data.dataset import DatasetSample


def collate_batch(samples: list[DatasetSample]) -> tuple[Tensor, Tensor]:
    """Collate a list of ``DatasetSample`` objects into (X, y) tensors.

    Args:
        samples: Non-empty list of dataset samples.  All samples must have
            the same ``features`` shape.

    Returns:
        ``(X, y)`` where:

        - ``X``: float32 tensor of shape ``(batch_size, window_size, feature_dim)``
        - ``y``: float32 tensor of shape ``(batch_size, 4)`` with score values
          in the order ``fatigue, attention, stress, engagement``.
    """
    feature_stack = np.stack([s.features for s in samples], axis=0)
    label_stack = np.array(
        [
            [s.label.fatigue, s.label.attention, s.label.stress, s.label.engagement]
            for s in samples
        ],
        dtype=np.float32,
    )
    x = torch.from_numpy(feature_stack.astype(np.float32))
    y = torch.from_numpy(label_stack)
    return x, y
