import pytest

from cognitive_state.video import (
    elapsed_timestamp_seconds,
    format_source_progress,
    timestamp_from_frame_index,
    timestamp_from_position,
)


def test_timestamp_from_frame_index_uses_fps() -> None:
    assert timestamp_from_frame_index(frame_index=3, fps=2.0) == 1.5


def test_timestamp_from_frame_index_rejects_invalid_fps() -> None:
    with pytest.raises(ValueError, match="fps"):
        timestamp_from_frame_index(frame_index=1, fps=0.0)


def test_timestamp_from_position_prefers_capture_position() -> None:
    assert timestamp_from_position(
        frame_index=3,
        fps=2.0,
        position_milliseconds=2500.0,
    ) == 2.5


def test_timestamp_from_position_falls_back_to_frame_index() -> None:
    assert timestamp_from_position(
        frame_index=3,
        fps=2.0,
        position_milliseconds=0.0,
    ) == 1.5


def test_elapsed_timestamp_seconds_is_relative_to_start() -> None:
    assert elapsed_timestamp_seconds(start_seconds=10.0, current_seconds=12.25) == 2.25


def test_format_source_progress_includes_known_totals() -> None:
    progress = format_source_progress(
        frame_index=4,
        timestamp_seconds=2.0,
        total_frames=10,
        duration_seconds=5.0,
    )

    assert progress == "frame 5/10, 2.00s/5.00s"
