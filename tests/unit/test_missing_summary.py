"""Missing-landmark summary tests (T038)."""

from __future__ import annotations

import pytest

from cognitive_state.data.windowing import WindowMissingSummary
from cognitive_state.features.missing import (
    summarize_batch_missing,
    summarize_window_missing,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rows_all_present(n: int) -> list[dict[str, float | int]]:
    return [{"face_detected": 1.0, "pose_detected": 1.0} for _ in range(n)]


def _rows_all_absent(n: int) -> list[dict[str, float | int]]:
    return [{"face_detected": 0.0, "pose_detected": 0.0} for _ in range(n)]


def _rows_mixed(
    n: int, *, n_face_missing: int, n_pose_missing: int
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for i in range(n):
        rows.append(
            {
                "face_detected": 0.0 if i < n_face_missing else 1.0,
                "pose_detected": 0.0 if i < n_pose_missing else 1.0,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# WindowMissingSummary — frozen dataclass contract
# ---------------------------------------------------------------------------

def test_window_missing_summary_is_frozen() -> None:
    s = WindowMissingSummary(
        window_index=0, total_frames=10,
        face_missing_frames=2, pose_missing_frames=3,
    )
    with pytest.raises((AttributeError, TypeError)):
        s.window_index = 99  # type: ignore[misc]


def test_window_missing_summary_stores_all_fields() -> None:
    s = WindowMissingSummary(
        window_index=1, total_frames=20,
        face_missing_frames=4, pose_missing_frames=6,
    )
    assert s.window_index == 1
    assert s.total_frames == 20
    assert s.face_missing_frames == 4
    assert s.pose_missing_frames == 6


# ---------------------------------------------------------------------------
# face_missing_rate property
# ---------------------------------------------------------------------------

def test_face_missing_rate_half() -> None:
    s = WindowMissingSummary(
        window_index=0, total_frames=10,
        face_missing_frames=5, pose_missing_frames=0,
    )
    assert s.face_missing_rate == pytest.approx(0.5)


def test_face_missing_rate_zero_when_none_missing() -> None:
    s = WindowMissingSummary(
        window_index=0, total_frames=10,
        face_missing_frames=0, pose_missing_frames=0,
    )
    assert s.face_missing_rate == pytest.approx(0.0)


def test_face_missing_rate_one_when_all_missing() -> None:
    s = WindowMissingSummary(
        window_index=0, total_frames=10,
        face_missing_frames=10, pose_missing_frames=0,
    )
    assert s.face_missing_rate == pytest.approx(1.0)


def test_face_missing_rate_zero_when_total_frames_zero() -> None:
    s = WindowMissingSummary(
        window_index=0, total_frames=0,
        face_missing_frames=0, pose_missing_frames=0,
    )
    assert s.face_missing_rate == 0.0


# ---------------------------------------------------------------------------
# pose_missing_rate property
# ---------------------------------------------------------------------------

def test_pose_missing_rate_computed_correctly() -> None:
    s = WindowMissingSummary(
        window_index=0, total_frames=10,
        face_missing_frames=0, pose_missing_frames=2,
    )
    assert s.pose_missing_rate == pytest.approx(0.2)


def test_pose_missing_rate_zero_when_total_frames_zero() -> None:
    s = WindowMissingSummary(
        window_index=0, total_frames=0,
        face_missing_frames=0, pose_missing_frames=0,
    )
    assert s.pose_missing_rate == 0.0


# ---------------------------------------------------------------------------
# summarize_window_missing — basic correctness
# ---------------------------------------------------------------------------

def test_summarize_window_missing_all_present_has_zero_missing() -> None:
    rows = _rows_all_present(10)
    s = summarize_window_missing(rows, window_index=0)
    assert s.face_missing_frames == 0
    assert s.pose_missing_frames == 0


def test_summarize_window_missing_all_absent_has_all_missing() -> None:
    rows = _rows_all_absent(10)
    s = summarize_window_missing(rows, window_index=0)
    assert s.face_missing_frames == 10
    assert s.pose_missing_frames == 10


def test_summarize_window_missing_mixed_counts() -> None:
    rows = _rows_mixed(10, n_face_missing=3, n_pose_missing=5)
    s = summarize_window_missing(rows, window_index=0)
    assert s.face_missing_frames == 3
    assert s.pose_missing_frames == 5


def test_summarize_window_missing_records_total_frames() -> None:
    rows = _rows_all_present(15)
    s = summarize_window_missing(rows, window_index=0)
    assert s.total_frames == 15


def test_summarize_window_missing_records_window_index() -> None:
    rows = _rows_all_present(5)
    s = summarize_window_missing(rows, window_index=7)
    assert s.window_index == 7


def test_summarize_window_missing_returns_window_missing_summary() -> None:
    rows = _rows_all_present(5)
    s = summarize_window_missing(rows, window_index=0)
    assert isinstance(s, WindowMissingSummary)


def test_summarize_window_missing_absent_key_defaults_to_present() -> None:
    rows = [{} for _ in range(5)]
    s = summarize_window_missing(rows, window_index=0)
    assert s.face_missing_frames == 0
    assert s.pose_missing_frames == 0


# ---------------------------------------------------------------------------
# summarize_batch_missing — window count and alignment
# ---------------------------------------------------------------------------

def test_summarize_batch_missing_one_summary_per_window() -> None:
    rows = _rows_all_present(60)
    summaries = summarize_batch_missing(rows, window_size=30, stride=15)
    # (60 - 30) // 15 + 1 = 3
    assert len(summaries) == 3


def test_summarize_batch_missing_window_indices_are_sequential() -> None:
    rows = _rows_all_present(60)
    summaries = summarize_batch_missing(rows, window_size=30, stride=15)
    for i, s in enumerate(summaries):
        assert s.window_index == i


def test_summarize_batch_missing_empty_when_rows_fewer_than_window_size() -> None:
    rows = _rows_all_present(5)
    summaries = summarize_batch_missing(rows, window_size=30, stride=15)
    assert summaries == []


def test_summarize_batch_missing_total_frames_equals_window_size() -> None:
    rows = _rows_all_present(60)
    summaries = summarize_batch_missing(rows, window_size=30, stride=15)
    for s in summaries:
        assert s.total_frames == 30


def test_summarize_batch_missing_counts_only_frames_in_that_window() -> None:
    # First 30 rows have face_detected=0.0, remaining 30 have 1.0
    rows = _rows_all_absent(30) + _rows_all_present(30)
    summaries = summarize_batch_missing(rows, window_size=30, stride=30)
    assert summaries[0].face_missing_frames == 30
    assert summaries[1].face_missing_frames == 0
