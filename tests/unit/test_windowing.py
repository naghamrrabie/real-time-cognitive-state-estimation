"""Feature window validation tests (T037)."""

from __future__ import annotations

import numpy as np
import pytest

from cognitive_state.data.exceptions import ValidationError, WindowShapeError
from cognitive_state.data.feature_csv import FEATURE_CSV_COLUMNS
from cognitive_state.data.windowing import (
    WINDOW_FEATURE_COLUMNS,
    FeatureWindowBatch,
    WindowMetadata,
    WindowMissingSummary,
    build_windows,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rows(n: int, *, value: float = 0.5) -> list[dict[str, float | int]]:
    """Return n synthetic feature rows with all FEATURE_CSV_COLUMNS populated."""
    rows: list[dict[str, float | int]] = []
    for i in range(n):
        row: dict[str, float | int] = {col: value for col in FEATURE_CSV_COLUMNS}
        row["frame_index"] = i
        row["timestamp_seconds"] = i / 30.0
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# WINDOW_FEATURE_COLUMNS constant invariants
# ---------------------------------------------------------------------------

def test_window_feature_columns_excludes_frame_index() -> None:
    assert "frame_index" not in WINDOW_FEATURE_COLUMNS


def test_window_feature_columns_excludes_timestamp_seconds() -> None:
    assert "timestamp_seconds" not in WINDOW_FEATURE_COLUMNS


def test_window_feature_columns_is_non_empty() -> None:
    assert len(WINDOW_FEATURE_COLUMNS) > 0


def test_window_feature_columns_is_subset_of_feature_csv_columns() -> None:
    assert set(WINDOW_FEATURE_COLUMNS).issubset(set(FEATURE_CSV_COLUMNS))


def test_window_feature_columns_is_a_tuple_of_strings() -> None:
    assert isinstance(WINDOW_FEATURE_COLUMNS, tuple)
    assert all(isinstance(col, str) for col in WINDOW_FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Window count arithmetic
# ---------------------------------------------------------------------------

def test_build_windows_correct_count_60_rows_ws30_stride15() -> None:
    rows = _make_rows(60)
    batch = build_windows(rows, window_size=30, stride=15)
    # (60 - 30) // 15 + 1 = 3
    assert batch.n_windows == 3


def test_build_windows_single_window_when_rows_equal_window_size() -> None:
    rows = _make_rows(30)
    batch = build_windows(rows, window_size=30, stride=1)
    assert batch.n_windows == 1


def test_build_windows_stride_one_produces_maximum_windows() -> None:
    rows = _make_rows(10)
    batch = build_windows(rows, window_size=5, stride=1)
    # (10 - 5) // 1 + 1 = 6
    assert batch.n_windows == 6


def test_build_windows_no_partial_windows_at_end() -> None:
    # 11 rows, window=5, stride=3: starts 0,3,6 → 9 is the last valid end
    rows = _make_rows(11)
    batch = build_windows(rows, window_size=5, stride=3)
    assert batch.n_windows == 3


# ---------------------------------------------------------------------------
# Feature array shape and dtype
# ---------------------------------------------------------------------------

def test_build_windows_features_shape_is_n_windows_ws_dim() -> None:
    rows = _make_rows(60)
    batch = build_windows(rows, window_size=30, stride=15)
    assert batch.features.shape == (3, 30, len(WINDOW_FEATURE_COLUMNS))


def test_build_windows_feature_dim_property_matches_array() -> None:
    rows = _make_rows(60)
    batch = build_windows(rows, window_size=30, stride=15)
    assert batch.feature_dim == len(WINDOW_FEATURE_COLUMNS)


def test_build_windows_features_dtype_is_float32() -> None:
    rows = _make_rows(30)
    batch = build_windows(rows, window_size=30, stride=1)
    assert batch.features.dtype == np.float32


# ---------------------------------------------------------------------------
# WindowMetadata correctness
# ---------------------------------------------------------------------------

def test_build_windows_metadata_count_equals_n_windows() -> None:
    rows = _make_rows(60)
    batch = build_windows(rows, window_size=30, stride=15)
    assert len(batch.metadata) == batch.n_windows


def test_build_windows_first_window_start_frame_is_zero() -> None:
    rows = _make_rows(60)
    batch = build_windows(rows, window_size=30, stride=15)
    assert batch.metadata[0].start_frame == 0


def test_build_windows_first_window_end_frame_is_window_size_minus_one() -> None:
    rows = _make_rows(60)
    batch = build_windows(rows, window_size=30, stride=15)
    assert batch.metadata[0].end_frame == 29


def test_build_windows_second_window_start_frame_equals_stride() -> None:
    rows = _make_rows(60)
    batch = build_windows(rows, window_size=30, stride=15)
    assert batch.metadata[1].start_frame == 15


def test_build_windows_window_indices_are_sequential() -> None:
    rows = _make_rows(60)
    batch = build_windows(rows, window_size=30, stride=15)
    for i, meta in enumerate(batch.metadata):
        assert meta.window_index == i


def test_build_windows_first_start_time_is_zero() -> None:
    rows = _make_rows(60)
    batch = build_windows(rows, window_size=30, stride=15)
    assert batch.metadata[0].start_time == pytest.approx(0.0)


def test_build_windows_end_time_matches_last_frame_timestamp() -> None:
    rows = _make_rows(60)
    batch = build_windows(rows, window_size=30, stride=15)
    # window 0 covers frames 0..29; frame 29 timestamp = 29/30
    assert batch.metadata[0].end_time == pytest.approx(29 / 30.0)


def test_build_windows_metadata_records_are_window_metadata_instances() -> None:
    rows = _make_rows(30)
    batch = build_windows(rows, window_size=30, stride=1)
    assert all(isinstance(m, WindowMetadata) for m in batch.metadata)


# ---------------------------------------------------------------------------
# feature_names
# ---------------------------------------------------------------------------

def test_build_windows_default_feature_names_are_window_feature_columns() -> None:
    rows = _make_rows(30)
    batch = build_windows(rows, window_size=30, stride=1)
    assert batch.feature_names == WINDOW_FEATURE_COLUMNS


def test_build_windows_custom_feature_names_are_preserved() -> None:
    rows = _make_rows(30)
    custom = WINDOW_FEATURE_COLUMNS[:3]
    batch = build_windows(rows, window_size=30, stride=1, feature_names=custom)
    assert batch.feature_names == custom


def test_build_windows_custom_feature_names_affect_array_dim() -> None:
    rows = _make_rows(30)
    custom = WINDOW_FEATURE_COLUMNS[:5]
    batch = build_windows(rows, window_size=30, stride=1, feature_names=custom)
    assert batch.features.shape == (1, 30, 5)


# ---------------------------------------------------------------------------
# missing_summary field present on batch
# ---------------------------------------------------------------------------

def test_build_windows_missing_summary_has_one_entry_per_window() -> None:
    rows = _make_rows(60)
    batch = build_windows(rows, window_size=30, stride=15)
    assert len(batch.missing_summary) == batch.n_windows


def test_build_windows_missing_summary_entries_are_window_missing_summary() -> None:
    rows = _make_rows(30)
    batch = build_windows(rows, window_size=30, stride=1)
    assert all(isinstance(s, WindowMissingSummary) for s in batch.missing_summary)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_build_windows_raises_window_shape_error_on_window_size_zero() -> None:
    with pytest.raises(WindowShapeError):
        build_windows(_make_rows(30), window_size=0, stride=1)


def test_build_windows_raises_window_shape_error_on_window_size_negative() -> None:
    with pytest.raises(WindowShapeError):
        build_windows(_make_rows(30), window_size=-5, stride=1)


def test_build_windows_raises_window_shape_error_on_stride_zero() -> None:
    with pytest.raises(WindowShapeError):
        build_windows(_make_rows(30), window_size=10, stride=0)


def test_build_windows_raises_window_shape_error_on_stride_negative() -> None:
    with pytest.raises(WindowShapeError):
        build_windows(_make_rows(30), window_size=10, stride=-1)


def test_build_windows_raises_window_shape_error_when_rows_fewer_than_window_size() -> None:
    with pytest.raises(WindowShapeError):
        build_windows(_make_rows(5), window_size=10, stride=1)


def test_build_windows_raises_validation_error_on_missing_feature_column() -> None:
    rows = [{"frame_index": i, "timestamp_seconds": float(i)} for i in range(30)]
    with pytest.raises(ValidationError):
        build_windows(rows, window_size=30, stride=1)


# ---------------------------------------------------------------------------
# FeatureWindowBatch backward compatibility
# ---------------------------------------------------------------------------

def test_feature_window_batch_can_be_constructed_without_missing_summary() -> None:
    features = np.zeros((1, 5, 3), dtype=np.float32)
    meta = [WindowMetadata(0, 0, 4, 0.0, 4.0)]
    batch = FeatureWindowBatch(
        features=features,
        metadata=meta,
        feature_names=("a", "b", "c"),
        window_size=5,
        stride=1,
    )
    assert batch.missing_summary == []


def test_feature_window_batch_window_size_and_stride_are_preserved() -> None:
    rows = _make_rows(30)
    batch = build_windows(rows, window_size=30, stride=5)
    assert batch.window_size == 30
    assert batch.stride == 5
