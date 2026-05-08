"""Frame timing helpers for webcam and recorded video sources."""

from __future__ import annotations

import math

DEFAULT_FPS = 30.0


def normalize_fps(value: float | int | None, fallback: float = DEFAULT_FPS) -> float:
    """Return a positive finite FPS value using fallback when needed."""

    if value is None:
        return fallback
    fps = float(value)
    if not math.isfinite(fps) or fps <= 0:
        return fallback
    return fps


def timestamp_from_frame_index(*, frame_index: int, fps: float) -> float:
    """Compute timestamp seconds from a zero-based frame index and FPS."""

    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a positive finite value")
    return frame_index / fps


def timestamp_from_position(
    *,
    frame_index: int,
    fps: float,
    position_milliseconds: float | None,
) -> float:
    """Prefer OpenCV position metadata, then fall back to frame index timing."""

    if position_milliseconds is not None:
        position = float(position_milliseconds)
        if math.isfinite(position) and position > 0:
            return position / 1000.0
    return timestamp_from_frame_index(frame_index=frame_index, fps=fps)


def elapsed_timestamp_seconds(*, start_seconds: float, current_seconds: float) -> float:
    """Compute non-negative elapsed seconds for live webcam timestamps."""

    elapsed = current_seconds - start_seconds
    return max(0.0, elapsed)


def duration_from_frame_count(
    *,
    frame_count: int | float | None,
    fps: float | None,
) -> float | None:
    """Compute video duration when frame count and FPS are available."""

    if frame_count is None or fps is None:
        return None
    count = int(frame_count)
    if count <= 0 or not math.isfinite(fps) or fps <= 0:
        return None
    return count / fps


def format_source_progress(
    *,
    frame_index: int,
    timestamp_seconds: float,
    total_frames: int | None = None,
    duration_seconds: float | None = None,
) -> str:
    """Format compact source progress for later console output."""

    frame_number = frame_index + 1
    frame_part = f"frame {frame_number}"
    if total_frames is not None and total_frames > 0:
        frame_part = f"{frame_part}/{total_frames}"

    time_part = f"{timestamp_seconds:.2f}s"
    if duration_seconds is not None and duration_seconds > 0:
        time_part = f"{time_part}/{duration_seconds:.2f}s"

    return f"{frame_part}, {time_part}"
