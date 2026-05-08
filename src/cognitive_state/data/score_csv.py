"""Score CSV writer and schema validation (T061)."""

from __future__ import annotations

import csv
from pathlib import Path

from cognitive_state.data.constants import SCORE_CSV_COLUMNS
from cognitive_state.data.schemas import ModelPrediction


def write_score_csv(
    predictions: list[ModelPrediction],
    path: str | Path,
) -> int:
    """Write per-window score predictions to a CSV file.

    The output schema is always ``SCORE_CSV_COLUMNS``:
    ``timestamp_seconds, source_progress, fatigue, attention, stress, engagement``.

    Args:
        predictions: Ordered list of model predictions to write.
        path: Destination CSV path.  Parent directories are created.

    Returns:
        Number of data rows written (excluding the header).
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(SCORE_CSV_COLUMNS))
        writer.writeheader()
        for p in predictions:
            writer.writerow({
                "timestamp_seconds": p.timestamp_seconds,
                "source_progress": p.source_progress,
                "fatigue": p.scores.fatigue,
                "attention": p.scores.attention,
                "stress": p.scores.stress,
                "engagement": p.scores.engagement,
            })
            count += 1
    return count


def read_score_csv(path: str | Path) -> list[dict[str, float | str]]:
    """Read a score CSV written by :func:`write_score_csv`.

    Score columns (fatigue, attention, stress, engagement,
    timestamp_seconds) are cast to ``float``; remaining columns are kept
    as strings.

    Args:
        path: Path to an existing score CSV.

    Returns:
        List of dicts, one per data row.
    """
    _FLOAT_COLS = frozenset(
        {"timestamp_seconds", "fatigue", "attention", "stress", "engagement"}
    )
    rows: list[dict[str, float | str]] = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            parsed: dict[str, float | str] = {}
            for key, val in row.items():
                parsed[key] = float(val) if key in _FLOAT_COLS else val
            rows.append(parsed)
    return rows
