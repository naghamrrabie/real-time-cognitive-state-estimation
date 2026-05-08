from pathlib import Path
from importlib import import_module

from cognitive_state.features.export import FeatureExportSummary

cli_module = import_module("cognitive_state.cli.main")


def test_extract_features_cli_writes_summary(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    output_path = tmp_path / "features.csv"

    def fake_extract_features_from_video(
        video_path: str | Path,
        output_csv: str | Path,
        *,
        max_frames: int | None = None,
    ) -> FeatureExportSummary:
        assert Path(video_path) == Path("sample.mp4")
        assert Path(output_csv) == output_path
        assert max_frames == 3
        return FeatureExportSummary(
            video_path=Path(video_path),
            output_csv=Path(output_csv),
            frames_processed=3,
            rows_written=3,
            face_detected_frames=2,
            pose_detected_frames=1,
        )

    monkeypatch.setattr(cli_module, "extract_features_from_video", fake_extract_features_from_video)

    exit_code = cli_module.main(
        [
            "extract-features",
            "--video",
            "sample.mp4",
            "--output",
            str(output_path),
            "--max-frames",
            "3",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Extracted 3 frames" in captured.out
    assert "face detections: 2" in captured.out
    assert "pose detections: 1" in captured.out


def test_extract_features_cli_handles_invalid_video_path(
    tmp_path: Path,
    capsys,
) -> None:
    missing_video = tmp_path / "missing.mp4"
    output_path = tmp_path / "features.csv"

    exit_code = cli_module.main(
        [
            "extract-features",
            "--video",
            str(missing_video),
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code != 0
    assert "video:" in captured.err
    assert "Video file does not exist" in captured.err
