"""Synthetic/demo label generator with smoke-label metadata (minimal stub for T051)."""

from __future__ import annotations

import numpy as np

from cognitive_state.data.schemas import CognitiveScoreVector

SYNTHETIC_LABEL_SOURCE: str = "synthetic"

SYNTHETIC_LABEL_WARNING: str = (
    "Synthetic labels are for pipeline validation only and are not "
    "evidence of real cognitive-state accuracy."
)


def generate_synthetic_labels(
    n: int,
    *,
    seed: int = 0,
) -> list[CognitiveScoreVector]:
    """Generate n synthetic cognitive score label vectors.

    Values are sampled uniformly in [0.0, 1.0].  Results are deterministic
    for a given *seed*.

    Args:
        n: Number of label vectors to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of n ``CognitiveScoreVector`` instances.
    """
    rng = np.random.default_rng(seed)
    values = rng.uniform(0.0, 1.0, size=(n, 4))
    return [
        CognitiveScoreVector(
            fatigue=float(row[0]),
            attention=float(row[1]),
            stress=float(row[2]),
            engagement=float(row[3]),
        )
        for row in values
    ]
