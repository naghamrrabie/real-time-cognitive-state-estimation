"""CSV schema and I/O helpers for per-frame feature exports."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path

from cognitive_state.features.pipeline import FRAME_FEATURE_NAMES

DETECTION_COLUMN_NAMES = ("face_detected", "pose_detected")
FEATURE_CSV_COLUMNS = (
    "frame_index",
    "timestamp_seconds",
    *DETECTION_COLUMN_NAMES,
    *(name for name in FRAME_FEATURE_NAMES if name not in {"frame_index", "timestamp_seconds"}),
)


def write_feature_csv(
    rows: Iterable[Mapping[str, float | int]],
    output_path: str | Path,
) -> int:
    """Write per-frame feature rows with a stable schema."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FEATURE_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(_stable_row(row))
            count += 1

    return count


def read_feature_csv(input_path: str | Path) -> list[dict[str, float | int]]:
    """Read feature rows from CSV and coerce numeric values."""

    with Path(input_path).open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return [_parse_row(row) for row in reader]


def _stable_row(row: Mapping[str, float | int]) -> dict[str, float | int]:
    return {column: row.get(column, 0.0) for column in FEATURE_CSV_COLUMNS}


def _parse_row(row: Mapping[str, str]) -> dict[str, float | int]:
    parsed: dict[str, float | int] = {}
    for column in FEATURE_CSV_COLUMNS:
        value = row.get(column, "0")
        parsed[column] = int(float(value)) if column == "frame_index" else float(value)
    return parsed
