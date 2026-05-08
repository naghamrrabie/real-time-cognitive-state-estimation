"""Temporal feature window builder for inference and dataset preparation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cognitive_state.data.exceptions import ValidationError, WindowShapeError
from cognitive_state.data.feature_csv import FEATURE_CSV_COLUMNS

# Columns used only as window provenance metadata, excluded from model arrays.
_METADATA_COLUMNS: frozenset[str] = frozenset({"frame_index", "timestamp_seconds"})

# Default ordered feature columns fed into the model (all non-metadata columns).
WINDOW_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    col for col in FEATURE_CSV_COLUMNS if col not in _METADATA_COLUMNS
)


@dataclass(frozen=True)
class WindowMetadata:
    """Frame and timing provenance for one temporal window."""

    window_index: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float


@dataclass
class FeatureWindowBatch:
    """All temporal windows produced from a single ordered feature sequence.

    Attributes:
        features: Float32 array of shape ``(n_windows, window_size, feature_dim)``.
        metadata: Per-window provenance records, one per window, in order.
        feature_names: Ordered column names matching the last axis of *features*.
        window_size: Number of timesteps per window.
        stride: Frame advance between consecutive window start positions.
    """

    features: np.ndarray
    metadata: list[WindowMetadata]
    feature_names: tuple[str, ...]
    window_size: int
    stride: int

    @property
    def n_windows(self) -> int:
        """Number of windows in this batch."""
        return len(self.metadata)

    @property
    def feature_dim(self) -> int:
        """Number of features per timestep."""
        return int(self.features.shape[2]) if self.features.ndim == 3 else 0


def build_windows(
    rows: list[dict[str, float | int]],
    *,
    window_size: int,
    stride: int = 1,
    feature_names: tuple[str, ...] | None = None,
) -> FeatureWindowBatch:
    """Build sliding temporal windows from an ordered sequence of feature rows.

    Args:
        rows: Per-frame feature records from ``extract_frame_features`` or
              ``read_feature_csv``.  Each dict must contain the columns named
              in *feature_names* as well as ``frame_index`` and
              ``timestamp_seconds`` for provenance metadata.
        window_size: Number of frames per window.  Must be >= 1.
        stride: Number of frames to advance between windows.  Must be >= 1.
        feature_names: Feature columns to include in model arrays.  Defaults
                       to ``WINDOW_FEATURE_COLUMNS`` (all non-metadata columns
                       from ``FEATURE_CSV_COLUMNS``).

    Returns:
        ``FeatureWindowBatch`` whose ``features`` has shape
        ``(n_windows, window_size, feature_dim)``.

    Raises:
        WindowShapeError: If ``window_size < 1``, ``stride < 1``, or if
            ``len(rows) < window_size``.
        ValidationError: If any column in *feature_names* is absent from the
            first row of *rows*.
    """
    if window_size < 1:
        raise WindowShapeError(
            f"window_size must be >= 1; got {window_size}"
        )
    if stride < 1:
        raise WindowShapeError(
            f"stride must be >= 1; got {stride}"
        )
    if len(rows) < window_size:
        raise WindowShapeError(
            f"Need at least {window_size} rows for window_size={window_size}; "
            f"got {len(rows)}"
        )

    names: tuple[str, ...] = (
        feature_names if feature_names is not None else WINDOW_FEATURE_COLUMNS
    )

    if rows:
        missing_cols = [n for n in names if n not in rows[0]]
        if missing_cols:
            raise ValidationError(
                f"Missing feature columns in rows: {missing_cols}"
            )

    feature_matrix = np.array(
        [[float(row.get(name, 0.0)) for name in names] for row in rows],
        dtype=np.float32,
    )
    timestamps = [float(row.get("timestamp_seconds", 0.0)) for row in rows]
    frame_indices = [int(row.get("frame_index", i)) for i, row in enumerate(rows)]

    n_rows = len(rows)
    windows: list[np.ndarray] = []
    meta: list[WindowMetadata] = []

    start = 0
    window_idx = 0
    while start + window_size <= n_rows:
        end = start + window_size
        windows.append(feature_matrix[start:end])
        meta.append(
            WindowMetadata(
                window_index=window_idx,
                start_frame=frame_indices[start],
                end_frame=frame_indices[end - 1],
                start_time=timestamps[start],
                end_time=timestamps[end - 1],
            )
        )
        start += stride
        window_idx += 1

    features = (
        np.stack(windows, axis=0)
        if windows
        else np.empty((0, window_size, len(names)), dtype=np.float32)
    )

    return FeatureWindowBatch(
        features=features,
        metadata=meta,
        feature_names=names,
        window_size=window_size,
        stride=stride,
    )
