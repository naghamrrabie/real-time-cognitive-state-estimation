"""OpenCV live webcam preview for streaming inference."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from cognitive_state.data.schemas import ModelPrediction
from cognitive_state.features.landmarks import (
    LandmarkPoint,
    LandmarkSet,
    MediaPipeLandmarkExtractor,
)
from cognitive_state.features.pipeline import extract_frame_features
from cognitive_state.inference.pipeline import run_inference
from cognitive_state.models.transformer import TemporalTransformer
from cognitive_state.video import VideoFrame, iter_webcam


class LandmarkExtractor(Protocol):
    """Protocol for a closeable frame landmark extractor."""

    def extract(self, frame: VideoFrame) -> LandmarkSet:
        """Extract landmarks from one frame."""

    def close(self) -> None:
        """Release extractor resources."""


FrameReader = Callable[[int], Iterable[VideoFrame]]
ExtractorFactory = Callable[[], LandmarkExtractor]
PreviewFrameCallback = Callable[[str, np.ndarray], int]
ScoreCallback = Callable[[ModelPrediction], None]

_WINDOW_NAME = "Cognitive State Live Preview"
_QUIT_KEYS = {27, ord("q"), ord("Q")}


@dataclass(frozen=True)
class LivePreviewResult:
    """Rows and predictions collected during one live preview run."""

    rows: list[dict[str, float | int]]
    predictions: list[ModelPrediction]


def run_webcam_preview(
    webcam_index: int,
    model: TemporalTransformer,
    *,
    window_size: int,
    stride: int,
    target_windows: int | None = None,
    window_name: str = _WINDOW_NAME,
    frame_reader: FrameReader = iter_webcam,
    extractor_factory: ExtractorFactory = MediaPipeLandmarkExtractor,
    show_frame: PreviewFrameCallback | None = None,
    close_windows: Callable[[], None] | None = None,
    emit_score: ScoreCallback | None = None,
) -> LivePreviewResult:
    """Run webcam inference while displaying an OpenCV preview window.

    The preview updates every frame. Scores are updated whenever a complete
    temporal window is available. Press Q or Esc to stop early.
    """

    preview = show_frame if show_frame is not None else _show_frame
    closer = close_windows if close_windows is not None else cv2.destroyAllWindows

    rows: list[dict[str, float | int]] = []
    predictions: list[ModelPrediction] = []
    previous_landmarks: LandmarkSet | None = None
    last_prediction: ModelPrediction | None = None

    frame_iterator = iter(frame_reader(webcam_index))
    extractor = extractor_factory()

    try:
        for frame in frame_iterator:
            landmarks = extractor.extract(frame)
            row = extract_frame_features(
                landmarks,
                previous_landmarks=previous_landmarks,
            )
            row["face_detected"] = 1.0 if landmarks.face_detected else 0.0
            row["pose_detected"] = 1.0 if landmarks.pose_detected else 0.0
            rows.append(row)
            previous_landmarks = landmarks

            next_start = len(predictions) * stride
            if len(rows) >= next_start + window_size:
                window_rows = rows[next_start : next_start + window_size]
                new_predictions = run_inference(
                    window_rows,
                    model,
                    window_size=window_size,
                    stride=stride,
                )
                if new_predictions:
                    window_index = len(predictions)
                    last_prediction = new_predictions[-1]
                    last_prediction.window_id = f"w{window_index}"
                    last_prediction.source_progress = _format_progress(
                        window_index,
                        target_windows,
                    )
                    predictions.append(last_prediction)
                    if emit_score is not None:
                        emit_score(last_prediction)

            display_frame = _draw_preview_overlay(
                frame.image,
                landmarks,
                last_prediction=last_prediction,
                collected_frames=len(rows),
                window_size=window_size,
            )
            key = preview(window_name, display_frame)
            if key in _QUIT_KEYS:
                break
            if target_windows is not None and len(predictions) >= target_windows:
                break
    finally:
        extractor.close()
        if hasattr(frame_iterator, "close"):
            frame_iterator.close()
        closer()

    return LivePreviewResult(rows=rows, predictions=predictions)


def _format_progress(window_index: int, target_windows: int | None) -> str:
    current = window_index + 1
    if target_windows is None:
        return str(current)
    return f"{current}/{target_windows}"


def _show_frame(window_name: str, frame: np.ndarray) -> int:
    cv2.imshow(window_name, frame)
    return cv2.waitKey(1) & 0xFF


def _draw_preview_overlay(
    frame: np.ndarray,
    landmarks: LandmarkSet,
    *,
    last_prediction: ModelPrediction | None,
    collected_frames: int,
    window_size: int,
) -> np.ndarray:
    display = frame.copy()
    _draw_landmark_points(display, landmarks)

    panel_height = 150 if last_prediction is None else 180
    panel = display.copy()
    cv2.rectangle(panel, (0, 0), (display.shape[1], panel_height), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.65, display, 0.35, 0, display)

    _put_text(display, "Cognitive State Live Preview", 12, 30, (255, 255, 255), 0.75, 2)
    face_status = "face: detected" if landmarks.face_detected else "face: missing"
    pose_status = "pose: detected" if landmarks.pose_detected else "pose: missing"
    status_color = (80, 220, 120) if landmarks.face_detected else (80, 80, 255)
    _put_text(
        display,
        f"{face_status}   {pose_status}   frame={landmarks.frame_index}",
        12,
        62,
        status_color,
        0.6,
        2,
    )

    if last_prediction is None:
        ready = min(collected_frames, window_size)
        _put_text(
            display,
            f"collecting frames: {ready}/{window_size}",
            12,
            98,
            (235, 235, 235),
            0.6,
            2,
        )
    else:
        scores = last_prediction.scores
        _put_text(
            display,
            f"fatigue={scores.fatigue:.3f}   attention={scores.attention:.3f}",
            12,
            100,
            (235, 235, 235),
            0.6,
            2,
        )
        _put_text(
            display,
            f"stress={scores.stress:.3f}   engagement={scores.engagement:.3f}",
            12,
            132,
            (235, 235, 235),
            0.6,
            2,
        )

    _put_text(display, "press Q or Esc to stop", 12, panel_height - 18, (210, 210, 210), 0.52, 1)
    return display


def _draw_landmark_points(display: np.ndarray, landmarks: LandmarkSet) -> None:
    if landmarks.face_landmarks:
        _draw_points(display, landmarks.face_landmarks, color=(40, 220, 255), step=8, radius=1)
    if landmarks.pose_landmarks:
        _draw_points(display, landmarks.pose_landmarks, color=(80, 255, 120), step=1, radius=3)


def _draw_points(
    display: np.ndarray,
    points: tuple[LandmarkPoint, ...],
    *,
    color: tuple[int, int, int],
    step: int,
    radius: int,
) -> None:
    height, width = display.shape[:2]
    for point in points[::step]:
        if not 0.0 <= point.x <= 1.0 or not 0.0 <= point.y <= 1.0:
            continue
        x = int(point.x * width)
        y = int(point.y * height)
        cv2.circle(display, (x, y), radius, color, -1)


def _put_text(
    display: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    scale: float,
    thickness: int,
) -> None:
    cv2.putText(
        display,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
