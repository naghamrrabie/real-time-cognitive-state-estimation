"""Integration test: end-to-end inference pipeline from feature rows to predictions (T019)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from cognitive_state.data.exceptions import WindowShapeError
from cognitive_state.data.feature_csv import FEATURE_CSV_COLUMNS
from cognitive_state.data.schemas import ModelPrediction
from cognitive_state.data.windowing import WINDOW_FEATURE_COLUMNS
from cognitive_state.inference.pipeline import (
    DEFAULT_STRIDE,
    DEFAULT_WINDOW_SIZE,
    load_model,
    run_inference,
    run_inference_from_csv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_feature_rows(n: int) -> list[dict[str, float | int]]:
    """Return n synthetic per-frame feature rows with all required columns."""
    rows: list[dict[str, float | int]] = []
    for i in range(n):
        row: dict[str, float | int] = {col: 0.5 for col in FEATURE_CSV_COLUMNS}
        row["frame_index"] = i
        row["timestamp_seconds"] = i / 30.0
        rows.append(row)
    return rows


def _write_feature_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FEATURE_CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------

def test_load_model_returns_model_in_eval_mode() -> None:
    model = load_model(input_features=len(WINDOW_FEATURE_COLUMNS))
    assert not model.training


def test_load_model_accepts_window_feature_column_count() -> None:
    model = load_model(input_features=len(WINDOW_FEATURE_COLUMNS))
    import torch
    x = torch.randn(1, DEFAULT_WINDOW_SIZE, len(WINDOW_FEATURE_COLUMNS))
    out = model(x)
    assert out.shape == (1, 4)


# ---------------------------------------------------------------------------
# run_inference — window count
# ---------------------------------------------------------------------------

def test_run_inference_produces_one_prediction_per_window() -> None:
    rows = _make_feature_rows(60)
    model = load_model(input_features=len(WINDOW_FEATURE_COLUMNS))
    predictions = run_inference(rows, model, window_size=30, stride=15)
    # windows: (60 - 30) // 15 + 1 = 3
    assert len(predictions) == 3


def test_run_inference_uses_default_window_and_stride() -> None:
    rows = _make_feature_rows(DEFAULT_WINDOW_SIZE + DEFAULT_STRIDE * 2)
    model = load_model(input_features=len(WINDOW_FEATURE_COLUMNS))
    predictions = run_inference(rows, model)
    assert len(predictions) >= 1


# ---------------------------------------------------------------------------
# run_inference — score bounds
# ---------------------------------------------------------------------------

def test_run_inference_all_scores_within_unit_range() -> None:
    rows = _make_feature_rows(60)
    model = load_model(input_features=len(WINDOW_FEATURE_COLUMNS))
    predictions = run_inference(rows, model, window_size=30, stride=15)
    for p in predictions:
        assert 0.0 <= p.scores.fatigue <= 1.0
        assert 0.0 <= p.scores.attention <= 1.0
        assert 0.0 <= p.scores.stress <= 1.0
        assert 0.0 <= p.scores.engagement <= 1.0


def test_run_inference_returns_model_prediction_objects() -> None:
    rows = _make_feature_rows(60)
    model = load_model(input_features=len(WINDOW_FEATURE_COLUMNS))
    predictions = run_inference(rows, model, window_size=30, stride=15)
    for p in predictions:
        assert isinstance(p, ModelPrediction)


# ---------------------------------------------------------------------------
# run_inference — prediction metadata
# ---------------------------------------------------------------------------

def test_run_inference_source_progress_format_is_fraction() -> None:
    rows = _make_feature_rows(60)
    model = load_model(input_features=len(WINDOW_FEATURE_COLUMNS))
    predictions = run_inference(rows, model, window_size=30, stride=15)
    total = len(predictions)
    for i, p in enumerate(predictions):
        assert p.source_progress == f"{i + 1}/{total}"


def test_run_inference_window_ids_are_sequential() -> None:
    rows = _make_feature_rows(60)
    model = load_model(input_features=len(WINDOW_FEATURE_COLUMNS))
    predictions = run_inference(rows, model, window_size=30, stride=15)
    for i, p in enumerate(predictions):
        assert p.window_id == f"w{i}"


def test_run_inference_timestamps_are_non_negative() -> None:
    rows = _make_feature_rows(60)
    model = load_model(input_features=len(WINDOW_FEATURE_COLUMNS))
    predictions = run_inference(rows, model, window_size=30, stride=15)
    for p in predictions:
        assert p.timestamp_seconds >= 0.0


# ---------------------------------------------------------------------------
# run_inference — error cases
# ---------------------------------------------------------------------------

def test_run_inference_raises_window_shape_error_when_rows_too_few() -> None:
    rows = _make_feature_rows(5)
    model = load_model(input_features=len(WINDOW_FEATURE_COLUMNS))
    with pytest.raises(WindowShapeError):
        run_inference(rows, model, window_size=30, stride=15)


# ---------------------------------------------------------------------------
# run_inference_from_csv
# ---------------------------------------------------------------------------

def test_run_inference_from_csv_produces_predictions(tmp_path: Path) -> None:
    rows = _make_feature_rows(60)
    csv_path = tmp_path / "features.csv"
    _write_feature_csv(csv_path, rows)
    model = load_model(input_features=len(WINDOW_FEATURE_COLUMNS))
    predictions = run_inference_from_csv(csv_path, model, window_size=30, stride=15)
    assert len(predictions) == 3


def test_run_inference_from_csv_matches_run_inference(tmp_path: Path) -> None:
    rows = _make_feature_rows(60)
    csv_path = tmp_path / "features.csv"
    _write_feature_csv(csv_path, rows)
    model = load_model(input_features=len(WINDOW_FEATURE_COLUMNS))
    direct = run_inference(rows, model, window_size=DEFAULT_WINDOW_SIZE, stride=DEFAULT_STRIDE)
    from_csv = run_inference_from_csv(
        csv_path, model, window_size=DEFAULT_WINDOW_SIZE, stride=DEFAULT_STRIDE
    )
    assert len(direct) == len(from_csv)
    for a, b in zip(direct, from_csv):
        assert a.window_id == b.window_id
        assert a.scores.fatigue == pytest.approx(b.scores.fatigue, abs=1e-5)
        assert a.scores.attention == pytest.approx(b.scores.attention, abs=1e-5)
        assert a.scores.stress == pytest.approx(b.scores.stress, abs=1e-5)
        assert a.scores.engagement == pytest.approx(b.scores.engagement, abs=1e-5)


def test_run_inference_from_csv_scores_are_bounded(tmp_path: Path) -> None:
    rows = _make_feature_rows(60)
    csv_path = tmp_path / "features.csv"
    _write_feature_csv(csv_path, rows)
    model = load_model(input_features=len(WINDOW_FEATURE_COLUMNS))
    predictions = run_inference_from_csv(csv_path, model)
    for p in predictions:
        assert 0.0 <= p.scores.fatigue <= 1.0
        assert 0.0 <= p.scores.attention <= 1.0
        assert 0.0 <= p.scores.stress <= 1.0
        assert 0.0 <= p.scores.engagement <= 1.0
