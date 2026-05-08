"""Score CSV writer and schema validation tests (T057)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from cognitive_state.data.constants import SCORE_CSV_COLUMNS
from cognitive_state.data.schemas import CognitiveScoreVector, ModelPrediction
from cognitive_state.data.score_csv import read_score_csv, write_score_csv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_predictions(n: int = 3) -> list[ModelPrediction]:
    return [
        ModelPrediction(
            window_id=f"w{i}",
            timestamp_seconds=float(i),
            scores=CognitiveScoreVector(
                fatigue=0.1 * i % 1.0,
                attention=0.2 * i % 1.0,
                stress=0.3 * i % 1.0,
                engagement=0.4 * i % 1.0,
            ),
            source_progress=f"{i}/{n}",
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# write_score_csv — output file creation
# ---------------------------------------------------------------------------

def test_write_score_csv_creates_file(tmp_path: Path) -> None:
    out = tmp_path / "scores.csv"
    write_score_csv(_make_predictions(), out)
    assert out.exists()


def test_write_score_csv_returns_row_count(tmp_path: Path) -> None:
    preds = _make_predictions(5)
    n = write_score_csv(preds, tmp_path / "scores.csv")
    assert n == 5


def test_write_score_csv_zero_predictions_writes_header_only(tmp_path: Path) -> None:
    out = tmp_path / "scores.csv"
    write_score_csv([], out)
    with out.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows == []


# ---------------------------------------------------------------------------
# write_score_csv — schema
# ---------------------------------------------------------------------------

def test_write_score_csv_has_all_required_columns(tmp_path: Path) -> None:
    out = tmp_path / "scores.csv"
    write_score_csv(_make_predictions(), out)
    with out.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        row = next(reader)
    for col in SCORE_CSV_COLUMNS:
        assert col in row, f"Missing column: {col!r}"


def test_write_score_csv_score_values_are_bounded(tmp_path: Path) -> None:
    out = tmp_path / "scores.csv"
    write_score_csv(_make_predictions(10), out)
    with out.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for score in ("fatigue", "attention", "stress", "engagement"):
                val = float(row[score])
                assert 0.0 <= val <= 1.0, f"{score}={val} out of [0,1]"


def test_write_score_csv_timestamp_is_numeric(tmp_path: Path) -> None:
    out = tmp_path / "scores.csv"
    write_score_csv(_make_predictions(3), out)
    with out.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            float(row["timestamp_seconds"])  # must not raise


def test_write_score_csv_creates_parent_directories(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "dir" / "scores.csv"
    write_score_csv(_make_predictions(), out)
    assert out.exists()


# ---------------------------------------------------------------------------
# read_score_csv
# ---------------------------------------------------------------------------

def test_read_score_csv_returns_correct_count(tmp_path: Path) -> None:
    out = tmp_path / "scores.csv"
    preds = _make_predictions(4)
    write_score_csv(preds, out)
    rows = read_score_csv(out)
    assert len(rows) == 4


def test_read_score_csv_rows_are_dicts(tmp_path: Path) -> None:
    out = tmp_path / "scores.csv"
    write_score_csv(_make_predictions(2), out)
    rows = read_score_csv(out)
    for row in rows:
        assert isinstance(row, dict)


def test_read_score_csv_has_all_columns(tmp_path: Path) -> None:
    out = tmp_path / "scores.csv"
    write_score_csv(_make_predictions(2), out)
    rows = read_score_csv(out)
    for row in rows:
        for col in SCORE_CSV_COLUMNS:
            assert col in row


def test_read_score_csv_score_values_are_float(tmp_path: Path) -> None:
    out = tmp_path / "scores.csv"
    write_score_csv(_make_predictions(3), out)
    rows = read_score_csv(out)
    for row in rows:
        for score in ("fatigue", "attention", "stress", "engagement"):
            assert isinstance(row[score], float)


def test_read_score_csv_roundtrip_preserves_values(tmp_path: Path) -> None:
    out = tmp_path / "scores.csv"
    preds = _make_predictions(3)
    write_score_csv(preds, out)
    rows = read_score_csv(out)
    for pred, row in zip(preds, rows):
        assert row["fatigue"] == pytest.approx(pred.scores.fatigue, abs=1e-6)
        assert row["attention"] == pytest.approx(pred.scores.attention, abs=1e-6)
