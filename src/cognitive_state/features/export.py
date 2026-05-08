"""Feature extraction export service for video files."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from itertools import chain, islice
from pathlib import Path
from typing import Protocol

from cognitive_state.data.feature_csv import write_feature_csv
from cognitive_state.features.landmarks import LandmarkSet, MediaPipeLandmarkExtractor
from cognitive_state.features.pipeline import extract_frame_features
from cognitive_state.video import VideoFrame, iter_video_file


class LandmarkExtractor(Protocol):
    """Protocol for MediaPipe-compatible landmark extractors."""

    def extract(self, frame: VideoFrame) -> LandmarkSet:
        """Extract landmarks from one frame."""

    def close(self) -> None:
        """Release extractor resources."""


FrameReader = Callable[[str | Path], Iterable[VideoFrame]]
ExtractorFactory = Callable[[], LandmarkExtractor]


@dataclass(frozen=True, slots=True)
class FeatureExportSummary:
    """Summary for one feature CSV export run."""

    video_path: Path
    output_csv: Path
    frames_processed: int
    rows_written: int
    face_detected_frames: int
    pose_detected_frames: int


class FeatureExportError(RuntimeError):
    """Raised when feature export cannot produce rows."""


def extract_features_from_video(
    video_path: str | Path,
    output_csv: str | Path,
    *,
    max_frames: int | None = None,
    frame_reader: FrameReader = iter_video_file,
    extractor_factory: ExtractorFactory = MediaPipeLandmarkExtractor,
) -> FeatureExportSummary:
    """Extract per-frame landmark-derived features from a video file to CSV."""

    source_path = Path(video_path)
    output_path = Path(output_csv)
    rows = _iter_feature_rows(
        source_path,
        max_frames=max_frames,
        frame_reader=frame_reader,
        extractor_factory=extractor_factory,
    )
    try:
        first_row = next(rows)
    except StopIteration as exc:
        raise FeatureExportError(f"No readable frames found in video: {source_path}") from exc

    counted_rows = _counting_rows(chain((first_row,), rows))
    rows_written = write_feature_csv(counted_rows, output_path)

    return FeatureExportSummary(
        video_path=source_path,
        output_csv=output_path,
        frames_processed=counted_rows.frames_processed,
        rows_written=rows_written,
        face_detected_frames=counted_rows.face_detected_frames,
        pose_detected_frames=counted_rows.pose_detected_frames,
    )


def _iter_feature_rows(
    video_path: Path,
    *,
    max_frames: int | None,
    frame_reader: FrameReader,
    extractor_factory: ExtractorFactory,
) -> Iterator[dict[str, float | int]]:
    frame_iterator = iter(frame_reader(video_path))
    if max_frames is not None:
        frame_iterator = islice(frame_iterator, max_frames)

    try:
        first_frame = next(frame_iterator)
    except StopIteration:
        return

    extractor = extractor_factory()
    previous_landmarks: LandmarkSet | None = None

    try:
        for frame in chain((first_frame,), frame_iterator):
            landmarks = extractor.extract(frame)
            row = extract_frame_features(
                landmarks,
                previous_landmarks=previous_landmarks,
            )
            row["face_detected"] = 1.0 if landmarks.face_detected else 0.0
            row["pose_detected"] = 1.0 if landmarks.pose_detected else 0.0
            previous_landmarks = landmarks
            yield row
    finally:
        extractor.close()


class _CountingRows:
    def __init__(self, rows: Iterator[dict[str, float | int]]) -> None:
        self._rows = rows
        self.frames_processed = 0
        self.face_detected_frames = 0
        self.pose_detected_frames = 0

    def __iter__(self) -> Iterator[dict[str, float | int]]:
        for row in self._rows:
            self.frames_processed += 1
            self.face_detected_frames += int(row["face_detected"])
            self.pose_detected_frames += int(row["pose_detected"])
            yield row


def _counting_rows(rows: Iterator[dict[str, float | int]]) -> _CountingRows:
    return _CountingRows(rows)
