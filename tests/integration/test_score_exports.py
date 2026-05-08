"""Integration test: inference export — scores CSV and plot (T059)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from cognitive_state.data.constants import SCORE_CSV_COLUMNS
from cognitive_state.data.feature_csv import FEATURE_CSV_COLUMNS
from cognitive_state.data.schemas import CognitiveScoreVector, ModelPrediction
from cognitive_state.data.score_csv import read_score_csv, write_score_csv
from cognitive_state.inference.plotting import plot_scores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_features_csv(path: Path, n_rows: int = 60) -> None:
    import csv as _csv
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=list(FEATURE_CSV_COLUMNS))
        writer.writeheader()
        for i in range(n_rows):
            row: dict = {col: 0.5 for col in FEATURE_CSV_COLUMNS}
            row["frame_index"] = i
            row["timestamp_seconds"] = i / 30.0
            writer.writerow(row)


def _make_predictions(n: int = 4) -> list[ModelPrediction]:
    return [
        ModelPrediction(
            window_id=f"w{i}",
            timestamp_seconds=float(i),
            scores=CognitiveScoreVector(fatigue=0.2, attention=0.5, stress=0.3, engagement=0.7),
            source_progress=f"{i}/{n}",
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# score CSV round-trip
# ---------------------------------------------------------------------------

def test_write_then_read_score_csv_preserves_row_count(tmp_path: Path) -> None:
    preds = _make_predictions(5)
    out = tmp_path / "scores.csv"
    write_score_csv(preds, out)
    rows = read_score_csv(out)
    assert len(rows) == 5


def test_score_csv_columns_match_schema(tmp_path: Path) -> None:
    out = tmp_path / "scores.csv"
    write_score_csv(_make_predictions(2), out)
    with out.open(newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    assert list(SCORE_CSV_COLUMNS) == header


def test_score_csv_score_values_in_unit_range(tmp_path: Path) -> None:
    out = tmp_path / "scores.csv"
    write_score_csv(_make_predictions(4), out)
    rows = read_score_csv(out)
    for row in rows:
        for score in ("fatigue", "attention", "stress", "engagement"):
            assert 0.0 <= float(row[score]) <= 1.0


# ---------------------------------------------------------------------------
# plot integration
# ---------------------------------------------------------------------------

def test_plot_scores_integration_creates_file(tmp_path: Path) -> None:
    out = tmp_path / "scores.png"
    plot_scores(_make_predictions(), out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_scores_integration_accepts_predictions_list(tmp_path: Path) -> None:
    out = tmp_path / "scores.png"
    preds = _make_predictions(10)
    result = plot_scores(preds, out)
    assert result == out


# ---------------------------------------------------------------------------
# run_inference_from_csv with scores_csv_path
# ---------------------------------------------------------------------------

def test_run_inference_from_csv_writes_scores_csv(tmp_path: Path) -> None:
    from cognitive_state.inference.pipeline import load_model, run_inference_from_csv
    from cognitive_state.data.windowing import WINDOW_FEATURE_COLUMNS

    feat_csv = tmp_path / "features.csv"
    scores_out = tmp_path / "scores.csv"
    _write_features_csv(feat_csv)

    model = load_model(input_features=len(WINDOW_FEATURE_COLUMNS))
    run_inference_from_csv(feat_csv, model, scores_csv_path=scores_out)

    assert scores_out.exists()
    rows = read_score_csv(scores_out)
    assert len(rows) >= 1
