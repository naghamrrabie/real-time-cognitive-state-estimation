"""Window-level dataset loader and validation (minimal stub for T050)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cognitive_state.data.constants import LABEL_SOURCES
from cognitive_state.data.exceptions import LabelError, ValidationError
from cognitive_state.data.schemas import CognitiveScoreVector


@dataclass
class DatasetSample:
    """A feature window paired with a window-level label vector.

    Attributes:
        window_id: Unique identifier for the source window.
        features: Float32 array of shape ``(window_size, feature_dim)``.
        label: Four-score label vector for this window.
        label_source: One of ``LABEL_SOURCES`` — ``"synthetic"``,
            ``"placeholder"``, ``"real"``, or ``"missing"``.
        split: Optional dataset split assignment.
    """

    window_id: str
    features: np.ndarray
    label: CognitiveScoreVector
    label_source: str = "synthetic"
    split: str | None = field(default=None)


class WindowDataset:
    """Iterable dataset of ``DatasetSample`` objects aligned to feature windows.

    Each sample pairs one window's feature array with one label vector.
    Label count must exactly match the number of feature windows.

    Args:
        features: Float32 array of shape ``(n_windows, window_size, feature_dim)``.
        labels: Exactly ``n_windows`` label vectors.
        label_source: Source tag applied to every sample.
        window_ids: Optional explicit window identifiers.

    Raises:
        LabelError: If ``len(labels) != features.shape[0]``.
        ValidationError: If ``label_source`` is not in ``LABEL_SOURCES``.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: list[CognitiveScoreVector],
        *,
        label_source: str = "synthetic",
        window_ids: list[str] | None = None,
    ) -> None:
        if features.shape[0] != len(labels):
            raise LabelError(
                f"Feature window count ({features.shape[0]}) does not match "
                f"label count ({len(labels)})."
            )
        if label_source not in LABEL_SOURCES:
            raise ValidationError(
                f"Unknown label_source {label_source!r}. "
                f"Must be one of {LABEL_SOURCES}."
            )
        self._features = features.astype(np.float32)
        self._labels = labels
        self._label_source = label_source
        self._window_ids = (
            window_ids if window_ids is not None
            else [f"w{i}" for i in range(len(labels))]
        )

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> DatasetSample:
        return DatasetSample(
            window_id=self._window_ids[idx],
            features=self._features[idx],
            label=self._labels[idx],
            label_source=self._label_source,
        )

    def __iter__(self):  # type: ignore[override]
        for i in range(len(self)):
            yield self[i]
