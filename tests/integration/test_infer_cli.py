"""Integration test: CLI contract for `cognitive-state infer` (T020)."""

from __future__ import annotations

import csv
from importlib import import_module
from pathlib import Path

import pytest

from cognitive_state.data.constants import SCORE_CSV_COLUMNS
from cognitive_state.data.feature_csv import FEATURE_CSV_COLUMNS

cli_module = import_module("cognitive_state.cli.main")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_synthetic_features_csv(path: Path, n_rows: int = 60) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FEATURE_CSV_COLUMNS))
        writer.writeheader()
        for i in range(n_rows):
            row: dict[str, float | int] = {col: 0.5 for col in FEATURE_CSV_COLUMNS}
            row["frame_index"] = i
            row["timestamp_seconds"] = i / 30.0
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Basic success path
# ---------------------------------------------------------------------------

def test_infer_from_features_csv_exits_zero(tmp_path: Path, capsys) -> None:
    csv_path = tmp_path / "features.csv"
    _write_synthetic_features_csv(csv_path)
    exit_code = cli_module.main(["infer", "--features-csv", str(csv_path)])
    assert exit_code == 0


def test_infer_from_features_csv_prints_all_score_names(tmp_path: Path, capsys) -> None:
    csv_path = tmp_path / "features.csv"
    _write_synthetic_features_csv(csv_path)
    cli_module.main(["infer", "--features-csv", str(csv_path)])
    captured = capsys.readouterr()
    for name in ("fatigue", "attention", "stress", "engagement"):
        assert name in captured.out, f"Score name missing from stdout: {name!r}"


def test_infer_from_features_csv_prints_at_least_one_score_line(
    tmp_path: Path, capsys
) -> None:
    csv_path = tmp_path / "features.csv"
    _write_synthetic_features_csv(csv_path)
    cli_module.main(["infer", "--features-csv", str(csv_path)])
    captured = capsys.readouterr()
    score_lines = [ln for ln in captured.out.splitlines() if "fatigue=" in ln]
    assert len(score_lines) >= 1


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_infer_no_source_returns_exit_2(capsys) -> None:
    exit_code = cli_module.main(["infer"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "error:" in captured.err


def test_infer_missing_features_csv_returns_exit_2(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "nonexistent.csv"
    exit_code = cli_module.main(["infer", "--features-csv", str(missing)])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "file:" in captured.err


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------

def test_infer_without_checkpoint_emits_warning(tmp_path: Path, capsys) -> None:
    csv_path = tmp_path / "features.csv"
    _write_synthetic_features_csv(csv_path)
    cli_module.main(["infer", "--features-csv", str(csv_path)])
    captured = capsys.readouterr()
    assert "warning" in captured.err.lower()
    assert "checkpoint" in captured.err


def test_infer_plot_flag_emits_warning_and_exits_zero(tmp_path: Path, capsys) -> None:
    csv_path = tmp_path / "features.csv"
    _write_synthetic_features_csv(csv_path)
    exit_code = cli_module.main(
        ["infer", "--features-csv", str(csv_path), "--plot"]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "plot" in captured.err.lower()


# ---------------------------------------------------------------------------
# --scores-csv output
# ---------------------------------------------------------------------------

def test_infer_scores_csv_is_created(tmp_path: Path) -> None:
    csv_path = tmp_path / "features.csv"
    scores_path = tmp_path / "scores.csv"
    _write_synthetic_features_csv(csv_path)
    exit_code = cli_module.main(
        ["infer", "--features-csv", str(csv_path), "--scores-csv", str(scores_path)]
    )
    assert exit_code == 0
    assert scores_path.exists()


def test_infer_scores_csv_has_correct_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "features.csv"
    scores_path = tmp_path / "scores.csv"
    _write_synthetic_features_csv(csv_path)
    cli_module.main(
        ["infer", "--features-csv", str(csv_path), "--scores-csv", str(scores_path)]
    )
    with scores_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert len(rows) >= 1
    for col in SCORE_CSV_COLUMNS:
        assert col in rows[0], f"Missing column in scores CSV: {col!r}"


def test_infer_scores_csv_values_are_bounded(tmp_path: Path) -> None:
    csv_path = tmp_path / "features.csv"
    scores_path = tmp_path / "scores.csv"
    _write_synthetic_features_csv(csv_path)
    cli_module.main(
        ["infer", "--features-csv", str(csv_path), "--scores-csv", str(scores_path)]
    )
    with scores_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for col in ("fatigue", "attention", "stress", "engagement"):
                value = float(row[col])
                assert 0.0 <= value <= 1.0, f"{col}={value} out of [0, 1]"


# ---------------------------------------------------------------------------
# --max-windows
# ---------------------------------------------------------------------------

def test_infer_max_windows_limits_score_lines(tmp_path: Path, capsys) -> None:
    csv_path = tmp_path / "features.csv"
    _write_synthetic_features_csv(csv_path, n_rows=120)
    cli_module.main(
        ["infer", "--features-csv", str(csv_path), "--max-windows", "1"]
    )
    captured = capsys.readouterr()
    score_lines = [ln for ln in captured.out.splitlines() if "fatigue=" in ln]
    assert len(score_lines) == 1


# ---------------------------------------------------------------------------
# --video path (monkeypatched)
# ---------------------------------------------------------------------------

def test_infer_video_path_dispatches_to_iter_feature_rows(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    def fake_iter_rows(video_path, **kwargs):  # type: ignore[override]
        for i in range(60):
            row: dict[str, float | int] = {col: 0.5 for col in FEATURE_CSV_COLUMNS}
            row["frame_index"] = i
            row["timestamp_seconds"] = i / 30.0
            yield row

    monkeypatch.setattr(cli_module, "iter_feature_rows_from_video", fake_iter_rows)
    exit_code = cli_module.main(["infer", "--video", "sample.mp4"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "fatigue" in captured.out
