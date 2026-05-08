"""Synthetic/demo label generator with smoke-label metadata (T051)."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from cognitive_state.data.schemas import CognitiveScoreVector

SYNTHETIC_LABEL_SOURCE: str = "synthetic"
PLACEHOLDER_LABEL_SOURCE: str = "placeholder"

SYNTHETIC_LABEL_WARNING: str = (
    "Synthetic labels are for pipeline validation only and are not "
    "evidence of real cognitive-state accuracy."
)


class SyntheticLabelBatch(NamedTuple):
    """Labels plus provenance metadata for smoke/demo runs.

    Attributes:
        labels: One ``CognitiveScoreVector`` per window.
        label_source: Always ``"synthetic"`` for this generator.
        warning: Human-readable provenance disclaimer.
    """

    labels: list[CognitiveScoreVector]
    label_source: str
    warning: str


def generate_synthetic_labels(
    n: int,
    *,
    seed: int = 0,
) -> list[CognitiveScoreVector]:
    """Generate *n* synthetic cognitive score label vectors.

    Values are sampled uniformly in [0.0, 1.0].  Results are deterministic
    for a given *seed*.

    Args:
        n: Number of label vectors to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of *n* ``CognitiveScoreVector`` instances.
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


def generate_synthetic_label_batch(
    n: int,
    *,
    seed: int = 0,
) -> SyntheticLabelBatch:
    """Like :func:`generate_synthetic_labels` but returns provenance metadata.

    Args:
        n: Number of label vectors to generate.
        seed: Random seed for reproducibility.

    Returns:
        :class:`SyntheticLabelBatch` carrying labels, source tag, and warning.
    """
    return SyntheticLabelBatch(
        labels=generate_synthetic_labels(n, seed=seed),
        label_source=SYNTHETIC_LABEL_SOURCE,
        warning=SYNTHETIC_LABEL_WARNING,
    )


#: Alias — identical to :func:`generate_synthetic_labels` with source "placeholder".
def generate_placeholder_labels(
    n: int,
    *,
    seed: int = 0,
) -> list[CognitiveScoreVector]:
    """Generate *n* placeholder label vectors (uniform random, for demo use only)."""
    return generate_synthetic_labels(n, seed=seed)
