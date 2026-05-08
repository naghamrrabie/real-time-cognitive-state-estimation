from cognitive_state import __version__
from cognitive_state.cli.main import main
from cognitive_state.config import DEFAULT_CONFIG


def test_package_version_is_available() -> None:
    assert __version__


def test_default_config_has_artifact_directory() -> None:
    assert DEFAULT_CONFIG.artifacts_dir.name == "artifacts"


def test_cli_placeholder_runs() -> None:
    assert main([]) == 0
