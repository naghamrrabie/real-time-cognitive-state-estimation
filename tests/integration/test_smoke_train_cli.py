"""Integration test: CLI contract for `cognitive-state smoke-train` (T049)."""

from __future__ import annotations

import csv
from importlib import import_module
from pathlib import Path

import pytest

from cognitive_state.data.feature_csv import FEATURE_CSV_COLUMNS

cli_module = import_module("cognitive_state.cli.main")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_synthetic_features_csv(path: Path, n_rows: int = 60) -> None:
    """Write a minimal feature CSV with enough rows for at least one window."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FEATURE_CSV_COLUMNS))
        writer.writeheader()
        for i in range(n_rows):
            row: dict[str, float | int] = {col: 0.5 for col in FEATURE_CSV_COLUMNS}
            row["frame_index"] = i
            row["timestamp_seconds"] = i / 30.0
            writer.writerow(row)


# ---------------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------------

def test_smoke_train_help_exits_zero(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli_module.main(["smoke-train", "--help"])
    assert exc.value.code == 0


def test_smoke_train_help_mentions_features_csv(capsys) -> None:
    with pytest.raises(SystemExit):
        cli_module.main(["smoke-train", "--help"])
    captured = capsys.readouterr()
    assert "--features-csv" in captured.out


def test_smoke_train_help_mentions_synthetic_labels(capsys) -> None:
    with pytest.raises(SystemExit):
        cli_module.main(["smoke-train", "--help"])
    captured = capsys.readouterr()
    assert "--synthetic-labels" in captured.out


# ---------------------------------------------------------------------------
# Basic success path
# ---------------------------------------------------------------------------

def test_smoke_train_exits_zero_with_synthetic_labels(tmp_path: Path, capsys) -> None:
    csv_path = tmp_path / "features.csv"
    _write_synthetic_features_csv(csv_path)
    exit_code = cli_module.main(
        ["smoke-train", "--features-csv", str(csv_path), "--synthetic-labels"]
    )
    assert exit_code == 0


def test_smoke_train_prints_sample_count(tmp_path: Path, capsys) -> None:
    csv_path = tmp_path / "features.csv"
    _write_synthetic_features_csv(csv_path)
    cli_module.main(
        ["smoke-train", "--features-csv", str(csv_path), "--synthetic-labels"]
    )
    captured = capsys.readouterr()
    assert "samples" in captured.out


def test_smoke_train_prints_input_shape(tmp_path: Path, capsys) -> None:
    csv_path = tmp_path / "features.csv"
    _write_synthetic_features_csv(csv_path)
    cli_module.main(
        ["smoke-train", "--features-csv", str(csv_path), "--synthetic-labels"]
    )
    captured = capsys.readouterr()
    assert "input shape" in captured.out or "shape" in captured.out


def test_smoke_train_prints_output_shape(tmp_path: Path, capsys) -> None:
    csv_path = tmp_path / "features.csv"
    _write_synthetic_features_csv(csv_path)
    cli_module.main(
        ["smoke-train", "--features-csv", str(csv_path), "--synthetic-labels"]
    )
    captured = capsys.readouterr()
    assert "output shape" in captured.out or "shape" in captured.out


def test_smoke_train_prints_mae(tmp_path: Path, capsys) -> None:
    csv_path = tmp_path / "features.csv"
    _write_synthetic_features_csv(csv_path)
    cli_module.main(
        ["smoke-train", "--features-csv", str(csv_path), "--synthetic-labels"]
    )
    captured = capsys.readouterr()
    assert "MAE" in captured.out or "mae" in captured.out.lower()


def test_smoke_train_prints_rmse(tmp_path: Path, capsys) -> None:
    csv_path = tmp_path / "features.csv"
    _write_synthetic_features_csv(csv_path)
    cli_module.main(
        ["smoke-train", "--features-csv", str(csv_path), "--synthetic-labels"]
    )
    captured = capsys.readouterr()
    assert "RMSE" in captured.out or "rmse" in captured.out.lower()


# ---------------------------------------------------------------------------
# Synthetic-label warning
# ---------------------------------------------------------------------------

def test_smoke_train_emits_synthetic_label_warning_to_stderr(
    tmp_path: Path, capsys
) -> None:
    csv_path = tmp_path / "features.csv"
    _write_synthetic_features_csv(csv_path)
    cli_module.main(
        ["smoke-train", "--features-csv", str(csv_path), "--synthetic-labels"]
    )
    captured = capsys.readouterr()
    assert "warning" in captured.err.lower()
    assert "synthetic" in captured.err.lower()


def test_smoke_train_smoke_metric_warning_in_output(tmp_path: Path, capsys) -> None:
    csv_path = tmp_path / "features.csv"
    _write_synthetic_features_csv(csv_path)
    cli_module.main(
        ["smoke-train", "--features-csv", str(csv_path), "--synthetic-labels"]
    )
    captured = capsys.readouterr()
    combined = (captured.out + captured.err).lower()
    assert "synthetic" in combined or "placeholder" in combined


# ---------------------------------------------------------------------------
# --checkpoint-out
# ---------------------------------------------------------------------------

def test_smoke_train_saves_checkpoint_when_requested(tmp_path: Path) -> None:
    csv_path = tmp_path / "features.csv"
    ckpt_path = tmp_path / "smoke.pt"
    _write_synthetic_features_csv(csv_path)
    exit_code = cli_module.main([
        "smoke-train",
        "--features-csv", str(csv_path),
        "--synthetic-labels",
        "--checkpoint-out", str(ckpt_path),
    ])
    assert exit_code == 0
    assert ckpt_path.exists()


def test_smoke_train_checkpoint_is_loadable(tmp_path: Path) -> None:
    import torch
    csv_path = tmp_path / "features.csv"
    ckpt_path = tmp_path / "smoke.pt"
    _write_synthetic_features_csv(csv_path)
    cli_module.main([
        "smoke-train",
        "--features-csv", str(csv_path),
        "--synthetic-labels",
        "--checkpoint-out", str(ckpt_path),
    ])
    state = torch.load(ckpt_path, map_location="cpu")
    assert isinstance(state, dict)
    assert len(state) > 0


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_smoke_train_no_labels_source_returns_exit_2(tmp_path: Path, capsys) -> None:
    csv_path = tmp_path / "features.csv"
    _write_synthetic_features_csv(csv_path)
    exit_code = cli_module.main(
        ["smoke-train", "--features-csv", str(csv_path)]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "error:" in captured.err


def test_smoke_train_missing_features_csv_returns_exit_2(
    tmp_path: Path, capsys
) -> None:
    missing = tmp_path / "nonexistent.csv"
    exit_code = cli_module.main([
        "smoke-train",
        "--features-csv", str(missing),
        "--synthetic-labels",
    ])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "features-csv:" in captured.err


def test_smoke_train_too_few_rows_returns_exit_2(tmp_path: Path, capsys) -> None:
    csv_path = tmp_path / "features.csv"
    _write_synthetic_features_csv(csv_path, n_rows=3)  # fewer than DEFAULT_WINDOW_SIZE
    exit_code = cli_module.main([
        "smoke-train",
        "--features-csv", str(csv_path),
        "--synthetic-labels",
    ])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "windows:" in captured.err
