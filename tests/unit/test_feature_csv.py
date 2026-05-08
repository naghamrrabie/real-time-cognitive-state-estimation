from pathlib import Path

import numpy as np

from cognitive_state.data.feature_csv import (
    FEATURE_CSV_COLUMNS,
    read_feature_csv,
    write_feature_csv,
)
from cognitive_state.features import LandmarkSet
from cognitive_state.features.export import extract_features_from_video
from cognitive_state.video import VideoFrame


class FakeExtractor:
    def __init__(self, results: list[LandmarkSet]) -> None:
        self.results = list(results)
        self.closed = False

    def extract(self, _frame: VideoFrame) -> LandmarkSet:
        return self.results.pop(0)

    def close(self) -> None:
        self.closed = True


def _landmark_set(
    *,
    frame_index: int,
    timestamp_seconds: float,
    face_detected: bool,
    pose_detected: bool,
) -> LandmarkSet:
    return LandmarkSet(
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        face_landmarks=None,
        pose_landmarks=None,
        face_detected=face_detected,
        pose_detected=pose_detected,
    )


def test_feature_csv_writer_creates_parent_and_stable_header(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "features.csv"
    row = {column: 0.0 for column in FEATURE_CSV_COLUMNS}
    row["frame_index"] = 1
    row["timestamp_seconds"] = 0.25
    row["face_detected"] = 1.0

    written = write_feature_csv([row], output_path)

    assert written == 1
    assert output_path.exists()
    header = output_path.read_text().splitlines()[0].split(",")
    assert tuple(header) == FEATURE_CSV_COLUMNS


def test_feature_csv_reader_returns_numeric_rows(tmp_path: Path) -> None:
    output_path = tmp_path / "features.csv"
    row = {column: 0.0 for column in FEATURE_CSV_COLUMNS}
    row["frame_index"] = 4
    row["timestamp_seconds"] = 2.5
    row["pose_detected"] = 1.0
    write_feature_csv([row], output_path)

    rows = read_feature_csv(output_path)

    assert rows[0]["frame_index"] == 4
    assert rows[0]["timestamp_seconds"] == 2.5
    assert rows[0]["pose_detected"] == 1.0


def test_export_service_reads_frames_extracts_features_and_writes_csv(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "sample.mp4"
    output_path = tmp_path / "out" / "features.csv"
    frames = [
        VideoFrame(0, 0.0, np.zeros((2, 2, 3), dtype=np.uint8)),
        VideoFrame(1, 0.5, np.zeros((2, 2, 3), dtype=np.uint8)),
    ]
    extractor = FakeExtractor(
        [
            _landmark_set(
                frame_index=0,
                timestamp_seconds=0.0,
                face_detected=True,
                pose_detected=False,
            ),
            _landmark_set(
                frame_index=1,
                timestamp_seconds=0.5,
                face_detected=False,
                pose_detected=True,
            ),
        ]
    )

    summary = extract_features_from_video(
        video_path,
        output_path,
        frame_reader=lambda _path: iter(frames),
        extractor_factory=lambda: extractor,
    )

    rows = read_feature_csv(output_path)
    assert summary.frames_processed == 2
    assert summary.rows_written == 2
    assert summary.face_detected_frames == 1
    assert summary.pose_detected_frames == 1
    assert rows[0]["face_detected"] == 1.0
    assert rows[0]["pose_detected"] == 0.0
    assert rows[1]["face_detected"] == 0.0
    assert rows[1]["pose_detected"] == 1.0
    assert extractor.closed is True
