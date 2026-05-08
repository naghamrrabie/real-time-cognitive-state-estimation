"""Integration test: CLI contract for `cognitive-state plot-scores` (T060)."""

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

def _write_scores_csv(path: Path, n_rows: int = 5) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(SCORE_CSV_COLUMNS))
        writer.writeheader()
        for i in range(n_rows):
            writer.writerow({
                "timestamp_seconds": float(i),
                "source_progress": f"{i}/{n_rows}",
                "fatigue": 0.2,
                "attention": 0.5,
                "stress": 0.3,
                "engagement": 0.7,
            })


# ---------------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------------

def test_plot_scores_help_exits_zero(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli_module.main(["plot-scores", "--help"])
    assert exc.value.code == 0


def test_plot_scores_help_mentions_scores_csv(capsys) -> None:
    with pytest.raises(SystemExit):
        cli_module.main(["plot-scores", "--help"])
    captured = capsys.readouterr()
    assert "--scores-csv" in captured.out


def test_plot_scores_help_mentions_output(capsys) -> None:
    with pytest.raises(SystemExit):
        cli_module.main(["plot-scores", "--help"])
    captured = capsys.readouterr()
    assert "--output" in captured.out


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------

def test_plot_scores_exits_zero(tmp_path: Path) -> None:
    scores = tmp_path / "scores.csv"
    out = tmp_path / "plot.png"
    _write_scores_csv(scores)
    exit_code = cli_module.main(
        ["plot-scores", "--scores-csv", str(scores), "--output", str(out)]
    )
    assert exit_code == 0


def test_plot_scores_creates_output_file(tmp_path: Path) -> None:
    scores = tmp_path / "scores.csv"
    out = tmp_path / "plot.png"
    _write_scores_csv(scores)
    cli_module.main(
        ["plot-scores", "--scores-csv", str(scores), "--output", str(out)]
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_scores_prints_output_path(tmp_path: Path, capsys) -> None:
    scores = tmp_path / "scores.csv"
    out = tmp_path / "plot.png"
    _write_scores_csv(scores)
    cli_module.main(
        ["plot-scores", "--scores-csv", str(scores), "--output", str(out)]
    )
    captured = capsys.readouterr()
    assert "plot" in captured.out.lower() or str(out) in captured.out


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_plot_scores_missing_scores_csv_returns_exit_2(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "nonexistent.csv"
    out = tmp_path / "plot.png"
    exit_code = cli_module.main(
        ["plot-scores", "--scores-csv", str(missing), "--output", str(out)]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "scores-csv" in captured.err


def test_plot_scores_no_args_exits_nonzero(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli_module.main(["plot-scores"])
    assert exc.value.code != 0
