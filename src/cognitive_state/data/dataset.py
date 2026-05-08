"""Window-level dataset loader and validation (T050)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from cognitive_state.data.constants import LABEL_SOURCES
from cognitive_state.data.exceptions import LabelError, ValidationError
from cognitive_state.data.schemas import CognitiveScoreVector

if TYPE_CHECKING:
    from cognitive_state.data.windowing import FeatureWindowBatch


@dataclass
class DatasetSample:
    """A feature window paired with a window-level label vector.

    Attributes:
        window_id: Unique identifier for the source window.
        features: Float32 array of shape ``(window_size, feature_dim)``.
        label: Four-score label vector for this window.
        label_source: One of ``LABEL_SOURCES``.
        split: Optional dataset split tag (e.g. ``"train"``, ``"val"``).
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
        split: Optional split tag propagated to every sample.

    Raises:
        LabelError: If ``len(labels) != features.shape[0]``.
        ValidationError: If ``features`` is not 3-dimensional or
            ``label_source`` is not in ``LABEL_SOURCES``.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: list[CognitiveScoreVector],
        *,
        label_source: str = "synthetic",
        window_ids: list[str] | None = None,
        split: str | None = None,
    ) -> None:
        if features.ndim != 3:
            raise ValidationError(
                f"features must be a 3-D array (n_windows, window_size, feature_dim); "
                f"got ndim={features.ndim}."
            )
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
        self._split = split
        self._window_ids = (
            window_ids if window_ids is not None
            else [f"w{i}" for i in range(len(labels))]
        )

    # ------------------------------------------------------------------
    # Convenience constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_feature_batch(
        cls,
        batch: "FeatureWindowBatch",
        labels: list[CognitiveScoreVector],
        *,
        label_source: str = "synthetic",
        split: str | None = None,
    ) -> "WindowDataset":
        """Construct a dataset directly from a :class:`FeatureWindowBatch`.

        Args:
            batch: Pre-built window batch produced by :func:`build_windows`.
            labels: One label per window in ``batch``.
            label_source: Source tag for all samples.
            split: Optional split tag.
        """
        return cls(
            batch.features,
            labels,
            label_source=label_source,
            split=split,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def label_source(self) -> str:
        """Source tag for every sample in this dataset."""
        return self._label_source

    # ------------------------------------------------------------------
    # Sequence protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> DatasetSample:
        return DatasetSample(
            window_id=self._window_ids[idx],
            features=self._features[idx],
            label=self._labels[idx],
            label_source=self._label_source,
            split=self._split,
        )

    def __iter__(self):  # type: ignore[override]
        for i in range(len(self)):
            yield self[i]
