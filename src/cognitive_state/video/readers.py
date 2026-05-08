"""OpenCV-backed frame readers for video files and webcams."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

import cv2
import numpy as np

from cognitive_state.video.sources import SourceType, VideoOpenError, VideoSource
from cognitive_state.video.timing import (
    duration_from_frame_count,
    elapsed_timestamp_seconds,
    normalize_fps,
    timestamp_from_position,
)

CaptureFactory = Callable[[str | int], Any]
Clock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class VideoFrame:
    """One decoded OpenCV frame with source-relative timing metadata."""

    frame_index: int
    timestamp_seconds: float
    image: np.ndarray


def iter_video_file(
    path: str | Path,
    *,
    capture_factory: CaptureFactory = cv2.VideoCapture,
) -> Iterator[VideoFrame]:
    """Yield frames from a recorded video file in offline/as-fast-as-practical mode."""

    yield from read_frames(VideoSource.video_file(path), capture_factory=capture_factory)


def iter_webcam(
    index: int = 0,
    *,
    capture_factory: CaptureFactory = cv2.VideoCapture,
    clock: Clock = monotonic,
) -> Iterator[VideoFrame]:
    """Yield frames from a webcam device using elapsed live timestamps."""

    yield from read_frames(
        VideoSource.webcam(index),
        capture_factory=capture_factory,
        clock=clock,
    )


def read_frames(
    source: VideoSource,
    *,
    capture_factory: CaptureFactory = cv2.VideoCapture,
    clock: Clock = monotonic,
) -> Iterator[VideoFrame]:
    """Yield decoded frames for the configured source."""

    capture = _open_capture(source, capture_factory)
    try:
        fps = normalize_fps(_capture_float(capture, cv2.CAP_PROP_FPS))
        start_seconds = clock()

        frame_index = 0
        while True:
            read_ok, frame = capture.read()
            if not read_ok:
                break
            if frame is None:
                break

            if source.source_type is SourceType.WEBCAM:
                timestamp_seconds = elapsed_timestamp_seconds(
                    start_seconds=start_seconds,
                    current_seconds=clock(),
                )
            else:
                timestamp_seconds = timestamp_from_position(
                    frame_index=frame_index,
                    fps=fps,
                    position_milliseconds=_capture_float(
                        capture,
                        cv2.CAP_PROP_POS_MSEC,
                    ),
                )

            yield VideoFrame(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                image=frame,
            )
            frame_index += 1
    finally:
        capture.release()


def describe_open_source(
    source: VideoSource,
    *,
    capture_factory: CaptureFactory = cv2.VideoCapture,
) -> VideoSource:
    """Open a source long enough to return capture metadata."""

    capture = _open_capture(source, capture_factory)
    try:
        fps = normalize_fps(_capture_float(capture, cv2.CAP_PROP_FPS))
        frame_count = _capture_float(capture, cv2.CAP_PROP_FRAME_COUNT)
        return source.with_open_metadata(
            fps=fps,
            width=_capture_int(capture, cv2.CAP_PROP_FRAME_WIDTH),
            height=_capture_int(capture, cv2.CAP_PROP_FRAME_HEIGHT),
            duration_seconds=duration_from_frame_count(
                frame_count=frame_count,
                fps=fps,
            ),
        )
    finally:
        capture.release()


def _open_capture(source: VideoSource, capture_factory: CaptureFactory) -> Any:
    if source.source_type is SourceType.VIDEO_FILE:
        path = Path(str(source.uri))
        if not path.is_file():
            raise VideoOpenError(f"Video file does not exist: {path}")
        capture_arg: str | int = str(path)
        failed_target = f"video file {path}"
    else:
        capture_arg = int(source.uri)
        failed_target = f"webcam device {capture_arg}"

    try:
        capture = capture_factory(capture_arg)
    except Exception as exc:  # pragma: no cover - defensive wrapper
        raise VideoOpenError(f"Could not open {failed_target}: {exc}") from exc

    if capture is None or not capture.isOpened():
        raise VideoOpenError(f"Could not open {failed_target}")

    return capture


def _capture_float(capture: Any, prop_id: int) -> float:
    try:
        return float(capture.get(prop_id))
    except Exception:
        return 0.0


def _capture_int(capture: Any, prop_id: int) -> int | None:
    value = _capture_float(capture, prop_id)
    if value <= 0:
        return None
    return int(value)
